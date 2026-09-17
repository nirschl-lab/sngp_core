"""Fit a trained checkpoint's single post-hoc calibration knob on validation NLL and write
the fitted value back into a sibling checkpoint -- no retraining, no config edit.

Every model family has exactly one inference-only scalar that shapes calibration but
cannot move the training fit (docs/HPO_GUIDE.md):

  * `baseline_classifier` / `deep_ensemble`: a temperature `T` -- `logits / T`.
  * `sngp_classifier`: the mean-field factor -- `raw_logits / sqrt(1 + factor * variance)`.
    This is Liu et al.'s kernel amplitude sigma, which the reference implementation
    collapses into `gp_mean_field_factor`, and the paper's own recommendation is to
    estimate it on held-out data by minimizing the log score.

What each metric can actually respond to
----------------------------------------
Both knobs divide every logit of an example by the same positive scalar, so:

  * accuracy and macro-F1 are **exactly invariant** -- the argmax cannot change (this
    script asserts it);
  * macro-AUPRC / AUROC move only through the per-example reordering that per-example
    denominators induce (SNGP), which is noise next to seed variance;
  * NLL and ECE genuinely respond, and have an interior optimum.

So select on **validation NLL** (a proper scoring rule; unlike ECE it has no bin-count
artifact), then report ECE and everything else on the *test* split. Selecting and
reporting on the same split would make the calibration numbers circular, which is why
`--split test` is refused unless `--report-only`.

Because the fitted value is a constructor argument in the net's `net_spec`, it is baked
into the checkpoint via `src.checkpointing.io.write_checkpoint_with_net_spec`: inference
(checkpoint-authoritative) picks it up with no change anywhere else, `read_meta` shows
it, and `sngp_core.calibration` records the provenance. For SNGP the covariance is
recomputed lazily from the persisted `precision_accum`, so a `--ridge` change is honoured
the same way. The source checkpoint is never modified.

Usage
-----
    uv run scripts/checkpoints/calibrate_checkpoint.py \\
        --ckpt /path/to/best.ckpt --experiment baseline_acevedo --split val
    # -> writes /path/to/best.calibrated.ckpt (use --out to choose, --force to replace)

    uv run scripts/checkpoints/calibrate_checkpoint.py --ckpt ... --experiment sngp_acevedo \\
        --ridge 1e-3 1.0           # SNGP only: also choose the ridge (one extra pass per value)
    uv run scripts/checkpoints/calibrate_checkpoint.py --ckpt ... --experiment ... --dry-run
    uv run scripts/checkpoints/calibrate_checkpoint.py --ckpt best.calibrated.ckpt \\
        --experiment ... --split test --report-only    # report the already-fitted knob on test
"""
import argparse
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import rootutils
import torch
from hydra import compose, initialize
from sklearn.metrics import average_precision_score, f1_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.checkpointing.io import load_net, read_meta, write_checkpoint_with_net_spec  # noqa: E402
from src.metrics.calibration_variants import smooth_ece  # noqa: E402
from src.metrics.posthoc_calibration import (  # noqa: E402
    KNOB_MEAN_FIELD,
    KNOB_TEMPERATURE,
    apply_knob,
    fit_knob,
    grid_for,
    nll,
)

FAMILY_KNOB = {
    "sngp_classifier": KNOB_MEAN_FIELD,
    "baseline_classifier": KNOB_TEMPERATURE,
    "deep_ensemble": KNOB_TEMPERATURE,
}


@dataclass(frozen=True)
class Collected:
    """One forward pass over a split, with the current knob undone."""

    base_logits: torch.Tensor  # knob-free logits: SNGP raw_logits; temperature families logits * T_current
    variance: Optional[torch.Tensor]  # SNGP [N, 1]; None otherwise
    targets: torch.Tensor


def knob_for(net_spec: Dict[str, Any]) -> str:
    name = net_spec.get("name")
    if name not in FAMILY_KNOB:
        raise ValueError(f"No post-hoc calibration knob is defined for net {name!r}; known: {sorted(FAMILY_KNOB)}")
    return FAMILY_KNOB[name]


def current_value(net: torch.nn.Module, knob: str) -> float:
    return float(net.temperature if knob == KNOB_TEMPERATURE else net.mean_field_factor)


def default_calibrated_path(src: Path) -> Path:
    """best.ckpt -> best.calibrated.ckpt, next to the source."""
    return src.with_name(f"{src.stem}.calibrated{src.suffix}")


@torch.no_grad()
def collect_outputs(net: torch.nn.Module, loader, device: str, *, knob: str, current: float) -> Collected:
    base, variances, targets = [], [], []
    net.eval()
    for batch in loader:
        _, x, y, _ = batch
        out = net(x.to(device))
        if knob == KNOB_MEAN_FIELD:
            if out.raw_logits is None or out.variance is None:
                raise ValueError("Model reported no raw_logits/variance -- is this an SNGP checkpoint in eval mode?")
            base.append(out.raw_logits.float().cpu())
            variances.append(out.variance.float().cpu())
        else:
            # Undo the temperature currently baked into the net so the fit starts from
            # knob-free logits (exact at T=1, which is what an untuned checkpoint has).
            base.append((out.logits.float() * current).cpu())
        targets.append(y.cpu())
    return Collected(
        base_logits=torch.cat(base),
        variance=torch.cat(variances) if variances else None,
        targets=torch.cat(targets),
    )


def logits_at(collected: Collected, knob: str, value: float) -> torch.Tensor:
    return apply_knob(knob, value, collected.base_logits, raw_logits=collected.base_logits, variance=collected.variance)


def score(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    n, k = probs.shape
    onehot = torch.zeros(n, k).scatter_(1, targets.unsqueeze(1), 1.0)
    return {
        "nll": nll(logits, targets),
        "smooth_ece": float(smooth_ece(probs, targets)),
        "acc": (preds == targets).float().mean().item(),
        "macro_f1": float(f1_score(targets.numpy(), preds.numpy(), average="macro")),
        "macro_auprc": float(average_precision_score(onehot.numpy(), probs.numpy(), average="macro")),
    }


def fit(collected: Collected, knob: str) -> Tuple[float, float]:
    return fit_knob(knob, collected.base_logits, collected.targets, raw_logits=collected.base_logits, variance=collected.variance)


def print_table(collected: Collected, knob: str, grid: Sequence[float], fitted: Optional[float]) -> None:
    rows = list(grid) + ([fitted] if fitted is not None and fitted not in grid else [])
    print(f"{knob:>18} {'NLL':>9} {'smECE':>9} {'acc':>9} {'macroF1':>9} {'mAUPRC':>9}")
    for value in sorted(rows):
        m = score(logits_at(collected, knob, value), collected.targets)
        marker = "  <- fitted" if fitted is not None and value == fitted else ""
        print(f"{value:>18.5g} {m['nll']:>9.5f} {m['smooth_ece']:>9.5f} {m['acc']:>9.5f} {m['macro_f1']:>9.5f} {m['macro_auprc']:>9.5f}{marker}")


def build_calibration_record(
    *,
    knob: str,
    value: float,
    previous_value: float,
    split: str,
    experiment: Optional[str],
    dataset_name: Optional[str],
    n_samples: int,
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    source_ckpt: Path,
    ridge_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "knob": knob,
        "value": float(value),
        "previous_value": float(previous_value),
        "objective": "nll",
        "split": split,
        "experiment": experiment,
        "dataset_name": dataset_name,
        "n_samples": int(n_samples),
        "metrics_before": {k: float(v) for k, v in metrics_before.items()},
        "metrics_after": {k: float(v) for k, v in metrics_after.items()},
        "source_ckpt": str(source_ckpt),
        "calibrated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if ridge_penalty is not None:
        record["ridge_penalty"] = float(ridge_penalty)
    return record


def calibrate(
    ckpt_path: Path,
    loader,
    *,
    device: str = "cpu",
    grid: Optional[Sequence[float]] = None,
    out: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
    report_only: bool = False,
    split: str = "val",
    experiment: Optional[str] = None,
    dataset_name: Optional[str] = None,
    ridges: Optional[Sequence[float]] = None,
) -> Tuple[float, Optional[Path]]:
    """Fit (or just report) the knob. Returns `(value, written_path_or_None)`."""
    ckpt_path = Path(ckpt_path)
    meta = read_meta(ckpt_path)
    knob = knob_for(meta.net_spec)
    if knob == KNOB_MEAN_FIELD and not meta.net_spec.get("mean_field", True):
        raise ValueError("This SNGP checkpoint has mean_field=False, so mean_field_factor is inert -- nothing to fit.")
    if ridges and knob != KNOB_MEAN_FIELD:
        raise ValueError("--ridge only applies to SNGP checkpoints")

    net = load_net(ckpt_path, device=device)
    previous = current_value(net, knob)
    grid = list(grid) if grid is not None else list(grid_for(knob))
    print(f"{ckpt_path}\n  net={meta.net_spec.get('name')}  knob={knob}  current={previous:g}  split={split}")

    best: Tuple[float, float, Optional[float], Dict[str, float], Dict[str, float], int] = (float("inf"), previous, None, {}, {}, 0)
    for ridge in list(ridges) if ridges else [None]:
        if ridge is not None:
            # `covariance` is recomputed lazily from `precision_accum`, so changing the
            # ridge needs no reload -- just invalidate the cache.
            net.gp_head.ridge = float(ridge)
            net.gp_head._cov_stale.fill_(True)
            print(f"\nridge_penalty={ridge:g}")

        collected = collect_outputs(net, loader, device, knob=knob, current=previous)
        before = score(logits_at(collected, knob, previous), collected.targets)
        if report_only:
            print_table(collected, knob, grid, None)
            print(f"\nAt the checkpoint's current {knob}={previous:g}: " + "  ".join(f"{k}={v:.5f}" for k, v in before.items()))
            best = (before["nll"], previous, ridge, before, before, len(collected.targets))
            continue

        fitted, fitted_nll = fit(collected, knob)
        after = score(logits_at(collected, knob, fitted), collected.targets)
        print_table(collected, knob, grid, fitted)
        # Sanity: the knob divides each example's logits by one positive scalar, so the
        # argmax -- and hence accuracy / F1 -- must be exactly unchanged.
        if not torch.equal(logits_at(collected, knob, previous).argmax(1), logits_at(collected, knob, fitted).argmax(1)):
            raise RuntimeError("argmax changed between the current and the fitted knob -- something other than the calibration knob is varying")
        print(f"\nfitted {knob}={fitted:g}  (was {previous:g}):  NLL {before['nll']:.5f} -> {after['nll']:.5f},  "
              f"smECE {before['smooth_ece']:.5f} -> {after['smooth_ece']:.5f}")
        if fitted_nll < best[0]:
            best = (fitted_nll, fitted, ridge, before, after, len(collected.targets))

    _, value, ridge, before, after, n = best
    if report_only or dry_run:
        if dry_run:
            print(f"\n--dry-run: would write {knob}={value:g}" + (f", ridge_penalty={ridge:g}" if ridge is not None else "") + " -- nothing written.")
        return value, None

    updates: Dict[str, Any] = {knob: float(value)}
    if ridge is not None:
        updates["ridge_penalty"] = float(ridge)
    record = build_calibration_record(
        knob=knob, value=value, previous_value=previous, split=split, experiment=experiment,
        dataset_name=dataset_name, n_samples=n, metrics_before=before, metrics_after=after,
        source_ckpt=ckpt_path, ridge_penalty=ridge,
    )
    out_path = Path(out) if out is not None else default_calibrated_path(ckpt_path)
    write_checkpoint_with_net_spec(ckpt_path, out_path, updates, calibration=record, overwrite=force)

    written = read_meta(out_path)
    print(f"\nwrote {out_path}")
    print(f"  net_spec.{knob}={written.net_spec[knob]!r}" + (f"  net_spec.ridge_penalty={written.net_spec['ridge_penalty']!r}" if ridge is not None else ""))
    print(f"  calibration: split={record['split']} n={record['n_samples']} NLL {before['nll']:.5f} -> {after['nll']:.5f}")
    print("Report from this checkpoint (checkpoint-authoritative inference):")
    print(f"  uv run src/inference/infer.py ckpt_path={out_path} data=<dataset> fold=test save_path=<dir>")
    return value, out_path


def _build_loader(experiment: str, split: str, batch_size: Optional[int]):
    """The dataloader for `split`, preprocessed exactly as training preprocessed it.

    The augmentation pipeline must be passed under the kwarg matching the loader we then
    ask for: `BaseImageDataModule` falls back to `_default_transform` (a bare
    ToTensor+Normalize, no Resize/CenterCrop) for any pipeline it isn't given, and that
    fallback is silent apart from a log line. Feeding the net full-resolution images
    instead of 224x224 centre crops does not raise -- it just collapses accuracy and
    makes the fitted knob meaningless, so `--split val` must get `val_augmentations`,
    not `test_augmentations`.
    """
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
    key = "val" if split == "val" else "test"
    augmentations = cfg.data.get("img_augmentations")
    aug = hydra.utils.instantiate(augmentations[key]) if augmentations and augmentations.get(key) else None
    overrides = {"batch_size": batch_size} if batch_size else {}
    dm = hydra.utils.instantiate(cfg.data.datamodule, **{f"{key}_augmentations": aug}, **overrides)
    dm.setup("fit" if split == "val" else "test")
    loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()
    return loader, getattr(dm, "dataset_name", None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="checkpoint to calibrate (never modified)")
    parser.add_argument("--experiment", required=True, help="experiment config naming the dataset, e.g. baseline_acevedo")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="fit on val; report on test (default: val)")
    parser.add_argument("--grid", type=float, nargs="+", default=None, help="values to tabulate (default: the family's grid)")
    parser.add_argument("--ridge", type=float, nargs="+", default=None, help="SNGP only: ridge_penalty values to try (one extra pass each)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None, help="output checkpoint (default: <ckpt>.calibrated.ckpt next to the source)")
    parser.add_argument("--force", action="store_true", help="replace an existing --out")
    parser.add_argument("--dry-run", action="store_true", help="fit and print, write nothing")
    parser.add_argument("--report-only", action="store_true", help="score the checkpoint's current knob, no fit, no write")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.report_only:
        print("Refusing to FIT on the test split: that makes the reported calibration circular. "
              "Fit with --split val, then report with --split test --report-only.")
        return 2

    loader, dataset_name = _build_loader(args.experiment, args.split, args.batch_size)
    calibrate(
        Path(args.ckpt), loader, device=args.device, grid=args.grid, out=args.out, force=args.force,
        dry_run=args.dry_run, report_only=args.report_only, split=args.split, experiment=args.experiment,
        dataset_name=dataset_name, ridges=args.ridge,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
