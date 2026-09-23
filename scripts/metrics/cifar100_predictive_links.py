"""Does swapping SNGP's softmax mean-field predictive for a normalized sigmoid / normCDF
link improve this project's CIFAR-100 numbers?

Background: Mucsanyi, Da Costa & Hennig (arXiv:2502.03366) replace softmax with an
element-wise normCDF or sigmoid that is then normalized, which gives the logit-Gaussian
pushforward in closed form. The transform consumes only `(mean, variance)` -- exactly what
the SNGP prediction CSVs already persist as `raw_logits` and `uncertainty` -- so every
alternative predictive is recomputable offline, with no GPU and no re-run of inference.

What the paper does *not* do is apply these links post-hoc: it trains with matched losses
(`NormedSigmoidNLLLoss` / `NormedNdtrNLLLoss`) so the activation is baked into the network.
That matters because softmax is shift-invariant and the two normalized links are not, so a
cross-entropy-trained network's absolute logit level -- a quantity its loss never constrained
-- becomes observable the moment the link changes. This script therefore does not just swap
the link; it measures how much of any movement is attributable to that free parameter.

Three passes:

  (a) `grid`   -- predictive x mean-field factor, with lambda varied independently of the
                  link so a win cannot be confounded with our pinned `lambda = 7.5`.
  (b) `shift`  -- a global logit offset sweep. Softmax must be exactly flat across it; the
                  size of the other two links' movement is the measurement of how
                  ill-posed the post-hoc swap is.
  (c) `best`   -- each predictive given its best achievable `(lambda, T, offset)`.

Pass (c) fits on the test set on purpose. That is not a calibration protocol anyone should
report -- it is a deliberately optimistic upper bound for the *new* links, so that a null
result there is conclusive rather than a tuning artifact.

Usage:

    uv run python scripts/metrics/cifar100_predictive_links.py \\
        --tag overnight_2026-09-20_21-38-42 --include-seed-12345 --out SUMMARY_LINKS.md
"""
import argparse
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Same directory (sys.path[0] when run as a script), so this resolves without packaging.
# Importing the arm mapping rather than restating it is the point: these scripts must not
# drift on which run directory belongs to which arm.
sys.path.insert(0, str(Path(__file__).parent))
from cifar100_overnight_report import ARMS, LABEL_COLS, LEGACY_ARMS, _infer_dir  # noqa: E402

from src.metrics.auc import AUROC  # noqa: E402
from src.metrics.brier import brier_score  # noqa: E402
from src.metrics.dispersion import per_sample_nll  # noqa: E402
from src.metrics.gaussian_predictives import (  # noqa: E402
    LAMBDA_PROBIT,
    PREDICTIVE_NAMES,
    log_predictive,
    predictive,
    saturation_fraction,
)
from src.metrics.io import load_predictions, raw_logits_array, uncertainty_array  # noqa: E402
from src.metrics.posthoc_calibration import minimize_scalar_log  # noqa: E402
from src.metrics.smooth_ece import smECE_fast_compat  # noqa: E402

# The pinned production value (configs/experiment/sngp_cifar100.yaml), the two constants the
# paper pairs its links with, the family default, and 0 = no correction at all.
LAMBDA_GRID: Tuple[float, ...] = (0.0, LAMBDA_PROBIT, 1.0, 7.5, 20.0, 50.0)
PRODUCTION_LAMBDA = 7.5
OFFSET_GRID: Tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0)
OOD_DATASETS: Tuple[str, ...] = ("cifar10", "svhn")

# Baseline has neither `raw_logits` nor `uncertainty` -- no logit Gaussian, nothing to push
# forward. Only the SNGP-family groups are eligible.
SKIP_GROUPS = frozenset({"baseline"})


class ArmData:
    """One arm's `(raw_logits, variance, targets)` for the ID set and each OOD set."""

    def __init__(self, label: str, display: str, group: str) -> None:
        self.label = label
        self.display = display
        self.group = group
        self.id: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.ood: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.published_probs: Optional[np.ndarray] = None


def _load_split(path: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]]:
    """`(raw_logits [N, C], variance [N, 1], targets [N], published class_probs [N, C])`."""
    try:
        frame = load_predictions(str(path / "predictions.csv"), fold="test")
        raw = torch.from_numpy(raw_logits_array(frame)).to(torch.float64)
        var = torch.from_numpy(uncertainty_array(frame)).to(torch.float64).unsqueeze(1)
    except Exception:  # noqa: BLE001 - a missing or Baseline arm must not kill the sweep
        return None
    df = frame.df
    col = next((c for c in LABEL_COLS if c in df.columns), None)
    targets = df[col].to_numpy().astype(int) if col else np.zeros(len(df), dtype=int)
    from src.metrics.io import probs_array

    return raw, var, targets, probs_array(frame)


def _id_metrics(
    raw: torch.Tensor,
    var: torch.Tensor,
    targets: torch.Tensor,
    name: str,
    lam: float,
    *,
    temperature: float = 1.0,
    offset: float = 0.0,
) -> Dict[str, float]:
    probs = predictive(name, raw, var, lam, temperature=temperature, logit_offset=offset)
    preds = probs.argmax(dim=1)
    correct = (preds == targets).to(torch.float64)
    conf = probs.max(dim=1).values
    return {
        "acc": float(correct.mean()),
        "nll": float(per_sample_nll(probs, targets).mean()),
        "brier": brier_score(probs, targets, probs.shape[1]),
        "smece": float(smECE_fast_compat(conf.numpy(), correct.numpy())),
    }


def _msp(
    raw: torch.Tensor, var: torch.Tensor, name: str, lam: float, *, temperature: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    probs = predictive(name, raw, var, lam, temperature=temperature, logit_offset=offset)
    return probs.max(dim=1).values.numpy()


def _nll_of(
    raw: torch.Tensor, var: torch.Tensor, targets: torch.Tensor, name: str, lam: float,
    temperature: float, offset: float,
) -> float:
    """NLL straight from log-space -- no exp/log round trip on 100-class probabilities."""
    log_p = log_predictive(name, raw, var, lam, temperature=temperature, logit_offset=offset)
    return float(-log_p[torch.arange(targets.shape[0]), targets].mean())


def _fit_best(
    raw: torch.Tensor, var: torch.Tensor, targets: torch.Tensor, name: str
) -> Tuple[float, float, float, float]:
    """Best `(lambda, T, offset, nll)` for one predictive, by NLL on the rows given.

    The caller passes the *fit* half of the split; the numbers reported are then measured on
    the other half, so this is an honest held-out fit rather than a self-graded one.

    Softmax ignores `offset` by construction, so its sweep collapses to the single value --
    skipping the redundant work rather than re-deriving the same number seven times. Note
    this leaves the two normalized links with one free parameter more than softmax, which
    is a real and deliberate advantage to them: `offset` is only a knob at all because they
    broke the shift-invariance that made it meaningless.
    """
    offsets = (0.0,) if name == "softmax" else OFFSET_GRID
    best = (1.0, 1.0, 0.0, math.inf)
    for lam in LAMBDA_GRID:
        for offset in offsets:
            t, value = minimize_scalar_log(
                lambda t: _nll_of(raw, var, targets, name, lam, t, offset), 0.05, 20.0
            )
            if value < best[3]:
                best = (lam, t, offset, value)
    return best


def _halves(n: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """A fixed 50/50 row split: fit the knobs on one half, report on the other.

    CIFAR-100's test set is the only labelled split these CSVs contain, so a held-out fit
    has to come from inside it. 5,000 rows is ample for three scalars.
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return perm[: n // 2], perm[n // 2 :]


def _fmt(values: Sequence[Optional[float]], places: int = 4) -> str:
    """Mean ± std across *training seeds* -- the only thing ± means in this family of
    reports (see `cifar100_overnight_report._fmt`, whose convention this matches)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{places}f}"
    return f"{statistics.mean(vals):.{places}f} ± {statistics.stdev(vals):.{places}f}"


def _table(rows: List[List[str]], header: List[str]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def _collect(root: Path, tag: str, label: str, display: str, group: str) -> Optional[ArmData]:
    arm = ArmData(label, display, group)
    id_split = _load_split(_infer_dir(root, tag, label, "cifar100"))
    if id_split is None:
        return None
    raw, var, targets, published = id_split
    arm.id = (raw, var, torch.from_numpy(targets).long())
    arm.published_probs = published
    for ds in OOD_DATASETS:
        ood = _load_split(_infer_dir(root, tag, label, ds))
        if ood is not None:
            arm.ood[ds] = (ood[0], ood[1])
    return arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--infer-root", default=None, help="defaults to $EXPERIMENTS_HOME/$PROJECT_NAME/infer")
    ap.add_argument("--include-seed-12345", action="store_true")
    ap.add_argument("--csv", default=None, help="also write the per-configuration rows here")
    args = ap.parse_args()

    root = Path(
        args.infer_root or Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "infer"
    )

    entries: List[Tuple[str, str, str, str]] = [
        (args.tag, label, display, group)
        for label, (display, group) in ARMS.items()
        if group not in SKIP_GROUPS
    ]
    if args.include_seed_12345:
        entries += [
            ("", label, display, group)
            for label, (display, group) in LEGACY_ARMS.items()
            if group not in SKIP_GROUPS
        ]

    arms = [a for a in (_collect(root, *e) for e in entries) if a is not None]
    if not arms:
        raise SystemExit(f"No SNGP-family arms found under {root}. Wrong --tag or --infer-root?")

    groups: Dict[str, List[ArmData]] = {}
    for arm in arms:
        groups.setdefault(arm.group, []).append(arm)
    group_order = [g for g in ("sngp", "sngp_c41", "specreg", "literal") if g in groups]

    rows_csv: List[Dict[str, object]] = []
    lines: List[str] = [
        "# CIFAR-100: normalized sigmoid / normCDF vs the SNGP mean-field predictive",
        "",
        "Offline re-scoring of the existing prediction CSVs — `raw_logits` and the per-sample",
        "GP variance are both persisted, so no inference was re-run and no GPU was used.",
        "`±` is the spread across **training seeds**, never a bootstrap.",
        "",
        f"Arms: {', '.join(f'{g} (n={len(groups[g])})' for g in group_order)}",
        "",
    ]

    # ---- sanity gate: reproduce the published softmax numbers from raw logits ----
    gate: List[List[str]] = []
    for g in group_order:
        deltas = []
        for arm in groups[g]:
            raw, var, targets = arm.id
            ours = predictive("softmax", raw, var, PRODUCTION_LAMBDA)
            deltas.append(float(np.abs(ours.numpy() - arm.published_probs).max()))
        gate.append([groups[g][0].display, f"{max(deltas):.2e}"])
    lines += [
        "## Gate: the offline path reproduces what inference wrote",
        "",
        "Max absolute difference between `softmax(raw_logits / sqrt(1 + 7.5 var))` recomputed",
        "here and the `class_probs` column the live pipeline wrote. If this is not ~0 nothing",
        "below means anything.",
        "",
        *_table(gate, ["Arm", "max |Δ prob|"]),
        "",
    ]

    # ---- (a) grid: predictive x lambda ----
    lines += [
        "## (a) Predictive × mean-field factor",
        "",
        "λ is varied independently of the link, so a win cannot be confounded with our pinned",
        f"λ = {PRODUCTION_LAMBDA}. λ = {LAMBDA_PROBIT:.4f} is π/8, the constant the paper pairs with its",
        "sigmoid link.",
        "",
    ]
    for g in group_order:
        grid_rows: List[List[str]] = []
        for name in PREDICTIVE_NAMES:
            for lam in LAMBDA_GRID:
                per_seed = [_id_metrics(*arm.id, name, lam) for arm in groups[g]]
                auroc: Dict[str, List[float]] = {ds: [] for ds in OOD_DATASETS}
                for arm in groups[g]:
                    id_msp = _msp(arm.id[0], arm.id[1], name, lam)
                    for ds, (oraw, ovar) in arm.ood.items():
                        auroc[ds].append(float(AUROC(id_msp, _msp(oraw, ovar, name, lam))))
                sat = statistics.mean(
                    saturation_fraction(arm.id[0], arm.id[1], lam) for arm in groups[g]
                )
                row = [
                    name,
                    f"{lam:.4g}",
                    _fmt([m["acc"] for m in per_seed]),
                    _fmt([m["nll"] for m in per_seed]),
                    _fmt([m["brier"] for m in per_seed]),
                    _fmt([m["smece"] for m in per_seed]),
                    _fmt(auroc["cifar10"]),
                    _fmt(auroc["svhn"]),
                    f"{sat:.2f}",
                ]
                grid_rows.append(row)
                rows_csv.append(
                    {
                        "pass": "grid", "group": g, "predictive": name, "lambda": lam,
                        "temperature": 1.0, "offset": 0.0,
                        "acc": statistics.mean(m["acc"] for m in per_seed),
                        "nll": statistics.mean(m["nll"] for m in per_seed),
                        "brier": statistics.mean(m["brier"] for m in per_seed),
                        "smece": statistics.mean(m["smece"] for m in per_seed),
                        "auroc_cifar10": statistics.mean(auroc["cifar10"]) if auroc["cifar10"] else None,
                        "auroc_svhn": statistics.mean(auroc["svhn"]) if auroc["svhn"] else None,
                        "saturated_frac": sat,
                    }
                )
        lines += [
            f"### {groups[g][0].display}",
            "",
            *_table(
                grid_rows,
                ["predictive", "λ", "acc", "NLL", "Brier", "smECE", "MSP C-10", "MSP SVHN", "sat."],
            ),
            "",
        ]
    lines += [
        "`sat.` is the fraction of rows with two or more scaled logits above 6, where Φ is",
        "within 1e-9 of its ceiling and the normCDF link can no longer rank those classes",
        "apart — which is why its accuracy column moves at all. Softmax and the sigmoid link",
        "hold accuracy exactly.",
        "",
    ]

    # ---- (b) shift sensitivity ----
    lines += [
        "## (b) Shift sensitivity",
        "",
        f"A global logit offset added before the variance correction, at λ = {PRODUCTION_LAMBDA}.",
        "Cross-entropy never constrained this quantity — softmax is exactly invariant to it —",
        "so every column that moves here is a free parameter the link has made observable,",
        "not a property of the model.",
        "",
    ]
    for g in group_order:
        shift_rows: List[List[str]] = []
        for name in PREDICTIVE_NAMES:
            for offset in OFFSET_GRID:
                per_seed = [
                    _id_metrics(*arm.id, name, PRODUCTION_LAMBDA, offset=offset)
                    for arm in groups[g]
                ]
                shift_rows.append(
                    [
                        name,
                        f"{offset:+.1f}",
                        _fmt([m["acc"] for m in per_seed]),
                        _fmt([m["nll"] for m in per_seed]),
                        _fmt([m["smece"] for m in per_seed]),
                    ]
                )
                rows_csv.append(
                    {
                        "pass": "shift", "group": g, "predictive": name,
                        "lambda": PRODUCTION_LAMBDA, "temperature": 1.0, "offset": offset,
                        "acc": statistics.mean(m["acc"] for m in per_seed),
                        "nll": statistics.mean(m["nll"] for m in per_seed),
                        "brier": statistics.mean(m["brier"] for m in per_seed),
                        "smece": statistics.mean(m["smece"] for m in per_seed),
                        "auroc_cifar10": None, "auroc_svhn": None, "saturated_frac": None,
                    }
                )
        lines += [
            f"### {groups[g][0].display}",
            "",
            *_table(shift_rows, ["predictive", "offset", "acc", "NLL", "smECE"]),
            "",
        ]

    # ---- (c) best achievable ----
    lines += [
        "## (c) Best achievable, each predictive given its own best knobs",
        "",
        "`(λ, T, offset)` fit by NLL on a fixed random **half** of the test rows and reported",
        "on the other half, so no number here is self-graded. Softmax gets `(λ, T)`; the two",
        "normalized links also get `offset`, one free parameter more — an advantage they only",
        "have because they broke the shift-invariance that made it meaningless.",
        "",
    ]
    for g in group_order:
        best_rows: List[List[str]] = []
        for name in PREDICTIVE_NAMES:
            fits, metrics = [], []
            auroc: Dict[str, List[float]] = {ds: [] for ds in OOD_DATASETS}
            for arm in groups[g]:
                raw, var, targets = arm.id
                fit_idx, eval_idx = _halves(raw.shape[0])
                fit = _fit_best(raw[fit_idx], var[fit_idx], targets[fit_idx], name)
                lam, t, off, _ = fit
                fits.append(fit)
                metrics.append(
                    _id_metrics(
                        raw[eval_idx], var[eval_idx], targets[eval_idx], name, lam,
                        temperature=t, offset=off,
                    )
                )
                # OOD scored with the same fitted knobs, ID side restricted to the eval half
                # so the AUROC is not read off rows the knobs were chosen on.
                id_msp = _msp(raw[eval_idx], var[eval_idx], name, lam, temperature=t, offset=off)
                for ds, (oraw, ovar) in arm.ood.items():
                    auroc[ds].append(
                        float(AUROC(id_msp, _msp(oraw, ovar, name, lam, temperature=t, offset=off)))
                    )
            best_rows.append(
                [
                    name,
                    _fmt([f[0] for f in fits], 3),
                    _fmt([f[1] for f in fits], 3),
                    _fmt([f[2] for f in fits], 2),
                    _fmt([m["acc"] for m in metrics]),
                    _fmt([m["nll"] for m in metrics]),
                    _fmt([m["smece"] for m in metrics]),
                    _fmt(auroc["cifar10"]),
                    _fmt(auroc["svhn"]),
                ]
            )
            for arm, (lam, t, off, _), m in zip(groups[g], fits, metrics):
                rows_csv.append(
                    {
                        "pass": "best", "group": g, "predictive": name, "lambda": lam,
                        "temperature": t, "offset": off, "acc": m["acc"], "nll": m["nll"],
                        "brier": m["brier"], "smece": m["smece"],
                        "auroc_cifar10": statistics.mean(auroc["cifar10"]) if auroc["cifar10"] else None,
                        "auroc_svhn": statistics.mean(auroc["svhn"]) if auroc["svhn"] else None,
                        "saturated_frac": None,
                    }
                )
        lines += [
            f"### {groups[g][0].display}",
            "",
            *_table(
                best_rows,
                ["predictive", "λ*", "T*", "offset*", "acc", "NLL", "smECE", "MSP C-10", "MSP SVHN"],
            ),
            "",
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")

    if args.csv:
        import csv as _csv

        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
            writer.writeheader()
            writer.writerows(rows_csv)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
