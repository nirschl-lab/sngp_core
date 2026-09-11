"""Tune an SNGP checkpoint's inference-only uncertainty knobs, without retraining.

`mean_field_factor` and `ridge_penalty` do not affect the training fit under canonical
SNGP -- the CE loss sees raw logits, and `ridge_penalty` only seeds the precision matrix.
They are therefore *not* in `configs/hparams_search/sngp.yaml`: they cannot move the
macro-AUPRC objective an Optuna sweep selects on. They still need choosing, and because
`precision_accum` is stored in the checkpoint, that costs one forward pass rather than a
training run.

What each metric can actually respond to
----------------------------------------
`mean_field_factor` divides every logit of an example by the same positive scalar, so:

  * accuracy and macro-F1 are **exactly invariant** -- the argmax cannot change;
  * macro-AUPRC / AUROC move only through the per-example reordering that per-example
    denominators induce, which is a few tenths of a percent -- noise next to seed
    variance;
  * NLL and ECE genuinely respond, and have an interior optimum.

So select on **validation NLL** (a proper scoring rule, and unlike ECE it has no bin-count
artifact), then report ECE and everything else on the *test* split. Selecting and
reporting on the same split would make the calibration numbers circular -- see the
protocol note in docs/HPO_GUIDE.md. Since this knob cannot move accuracy, F1, or OOD
separation, it cannot inflate discriminative or OOD claims either way.

Usage
-----
    uv run python scripts/checkpoints/tune_sngp_mean_field.py \\
        --ckpt /path/to/best.ckpt --experiment sngp_acevedo --split val

    # also sweep the ridge (one extra forward pass per value)
    uv run python scripts/checkpoints/tune_sngp_mean_field.py \\
        --ckpt ... --experiment sngp_acevedo --ridge 1e-3 1.0
"""
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import rootutils
import torch
from hydra import compose, initialize
from sklearn.metrics import average_precision_score, f1_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.checkpointing.io import load_net  # noqa: E402
from src.metrics.calibration_variants import smooth_ece  # noqa: E402

DEFAULT_FACTORS = (0.0, 0.3927, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def _collect(net: torch.nn.Module, loader, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass over the split. Returns (raw_logits, variance, targets)."""
    raw_logits, variances, targets = [], [], []
    net.eval()
    with torch.no_grad():
        for batch in loader:
            _, x, y, _ = batch
            out = net(x.to(device))
            if out.variance is None:
                raise ValueError("Model reported no variance -- is this an SNGP checkpoint?")
            raw_logits.append(out.raw_logits.float().cpu())
            variances.append(out.variance.float().cpu())
            targets.append(y.cpu())
    return torch.cat(raw_logits), torch.cat(variances), torch.cat(targets)


def _score(raw_logits: torch.Tensor, variance: torch.Tensor, targets: torch.Tensor, factor: float) -> Dict[str, float]:
    logits = raw_logits / torch.sqrt(1.0 + factor * variance)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    n, k = probs.shape
    nll = torch.nn.functional.cross_entropy(logits, targets).item()
    onehot = torch.zeros(n, k).scatter_(1, targets.unsqueeze(1), 1.0)
    return {
        "nll": nll,
        "smooth_ece": smooth_ece(probs, targets),
        "acc": (preds == targets).float().mean().item(),
        "macro_f1": f1_score(targets.numpy(), preds.numpy(), average="macro"),
        "macro_auprc": average_precision_score(onehot.numpy(), probs.numpy(), average="macro"),
    }


def _build_loader(experiment: str, split: str, batch_size: Optional[int]):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
    test_aug = hydra.utils.instantiate(cfg.data.img_augmentations.test) if cfg.data.get("img_augmentations") else None
    overrides = {"batch_size": batch_size} if batch_size else {}
    dm = hydra.utils.instantiate(cfg.data.datamodule, test_augmentations=test_aug, **overrides)
    dm.setup("fit" if split == "val" else "test")
    return dm.val_dataloader() if split == "val" else dm.test_dataloader()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="SNGP checkpoint to tune")
    parser.add_argument("--experiment", required=True, help="experiment config naming the dataset, e.g. sngp_acevedo")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="select on val; report on test (default: val)")
    parser.add_argument("--factors", type=float, nargs="+", default=list(DEFAULT_FACTORS))
    parser.add_argument("--ridge", type=float, nargs="+", default=None,
                        help="optional ridge_penalty values; each costs one extra forward pass")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.split == "test":
        print("NOTE: selecting on the test split makes the resulting calibration numbers "
              "circular. Use --split val to choose, then report on test.\n")

    loader = _build_loader(args.experiment, args.split, args.batch_size)
    net = load_net(args.ckpt, device=args.device)
    ridges: List[Optional[float]] = list(args.ridge) if args.ridge else [None]

    best: Tuple[float, Optional[float], float] = (float("inf"), None, 0.0)
    for ridge in ridges:
        if ridge is not None:
            # `covariance` is recomputed lazily from `precision_accum`, so changing the
            # ridge needs no reload -- just invalidate the cache.
            net.gp_head.ridge = ridge
            net.gp_head._cov_stale.fill_(True)

        raw_logits, variance, targets = _collect(net, loader, args.device)
        header = f"ridge={ridge:g}" if ridge is not None else f"ridge={net.gp_head.ridge:g} (as trained)"
        print(f"\n{header}   variance mean={variance.mean():.5f}   n={len(targets)}")
        print(f"{'factor':>8} {'denom':>8} {'NLL':>9} {'smECE':>9} {'acc':>9} {'macroF1':>9} {'mAUPRC':>9}")

        for factor in args.factors:
            m = _score(raw_logits, variance, targets, factor)
            denom = torch.sqrt(1.0 + factor * variance).mean().item()
            print(f"{factor:>8.4g} {denom:>8.4f} {m['nll']:>9.5f} {m['smooth_ece']:>9.5f} "
                  f"{m['acc']:>9.5f} {m['macro_f1']:>9.5f} {m['macro_auprc']:>9.5f}")
            if m["nll"] < best[0]:
                best = (m["nll"], ridge, factor)

    _, best_ridge, best_factor = best
    print(f"\nBest by {args.split} NLL: mean_field_factor={best_factor:g}"
          + (f", ridge_penalty={best_ridge:g}" if best_ridge is not None else ""))
    print("Set it in the experiment config under `model.net.mean_field_factor`.")
    print("Accuracy and macro-F1 columns should be identical across every row -- if they "
          "are not, something other than the mean-field correction is varying.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
