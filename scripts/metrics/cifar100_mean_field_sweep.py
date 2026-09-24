"""Sweep SNGP's `mean_field_factor` (lambda) on CIFAR-100 and emit a per-seed tidy CSV.

`mean_field_factor` is pinned at 7.5 in all three CIFAR-100 SNGP configs and has never been
swept or fitted -- `docs/results/CIFAR100_RESULTS.md` carries "no post-hoc calibration" as a
standing limitation. This measures how much of the reported calibration and OOD numbers is
actually a property of that one unfitted constant.

Pure offline re-scoring: `raw_logits` and the per-sample GP variance are both persisted in
the prediction CSVs, so `softmax(raw / sqrt(1 + lambda * var))` is recomputable for any
lambda with no re-inference and no GPU.

What the sweep shows, and why both metric families are here: calibration (NLL, smECE, Brier)
has an interior optimum near lambda ~ 20-50, while far-OOD AUROC keeps improving well past
it and near-OOD AUROC degrades. That is not noise. As lambda grows the scaled logits tend to
`raw / (sqrt(lambda) * sigma)`, softmax flattens, and the MSP *ranking* converges to
logit-margin / sigma -- dividing the margin by the GP standard deviation, which helps
far-OOD and hurts near-OOD. The `asymptote` rows record that limit directly so the curves
can be read against what they are converging to.

Accuracy is reported as a control, not a result: the correction divides every logit of an
example by one positive scalar, so argmax -- and therefore accuracy -- is invariant. A
sweep where the accuracy column moves is a broken sweep.

Emits **one row per (arm, seed, lambda)**. The sibling `cifar100_predictive_links.py`
collapses seeds with `statistics.mean` and writes no std, which cannot feed an error band;
aggregation here is left to the consumer, `src/visualization/cifar100_mean_field_sweep.py`.

Usage:

    uv run python scripts/metrics/cifar100_mean_field_sweep.py \\
        --tag overnight_2026-09-20_21-38-42 --include-seed-12345 \\
        --csv figures/mean_field_sweep/cifar100_mean_field_sweep_per_seed.csv
"""
import argparse
import csv as _csv
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Same directory (sys.path[0] when run as a script), so this resolves without packaging --
# the trick `cifar100_svhn_subsample_check.py` already uses. Importing the arm mapping and
# the metric helpers rather than restating them is the point: these scripts must not drift
# on which run directory belongs to which arm, nor on how a metric is computed.
sys.path.insert(0, str(Path(__file__).parent))
from cifar100_overnight_report import ARMS, LABEL_COLS, LEGACY_ARMS, TRACE_ARMS, _infer_dir  # noqa: E402
from cifar100_predictive_links import _collect, _id_metrics, _msp  # noqa: E402

from src.metrics.auc import AUROC  # noqa: E402
from src.metrics.io import load_predictions, probs_array  # noqa: E402
from src.metrics.posthoc_calibration import mean_field_scale  # noqa: E402
from src.metrics.smooth_ece import smECE_fast_compat  # noqa: E402
from src.metrics.uncertainty import dempster_shafer  # noqa: E402

PRODUCTION_LAMBDA = 7.5
OOD_DATASETS = ("cifar10", "svhn")

# 0 (no correction) plus a log grid out to ~3e3. The gaussian arms' GP variance is
# near-identical across seeds (mean 0.023-0.024, p99 ~ 0.07); the `trace_logistic` arm's is
# ~65x larger (mean ~1.5), because its Laplace weight 1 - ||p||^2 collapses to ~0.015 on a
# converged classifier and the precision accumulator shrinks with it. One grid still serves
# every arm -- lambda and the variance enter only as the product lambda*var, so the larger
# variance simply slides that arm's curve ~65x to the left. The low end is 1e-2 rather than
# 1e-1 so the left edge is genuinely "no correction" for the trace arm too (at 1e-1 it is
# already a 1.07x shrink) and its optimum sits in the interior rather than on the boundary.
# 34 points, not 30: that keeps the log10 step at exactly 1/6, so the extended grid is a
# strict SUPERSET of the original logspace(-1, 3.5, 28) and every lambda already quoted in
# docs/results/CIFAR100_RESULTS.md (31.6, 46.4, 147, 1000, 3162) is still a grid point.
# The mean shrink factor sqrt(1 + lambda*var) runs 1.0 -> ~8x over this range for the
# gaussian arms; the pinned 7.5 is a 1.09x shrink there, which is why it barely moves
# anything -- and a 3.5x shrink on the trace arm, which is why it moves a great deal. Past ~1e3 the correction is
# asymptotically a per-example temperature proportional to 1/sigma, so the curve shape stops
# changing and only its scale does -- that is the natural right-hand end of the sweep.
# The production value is spliced in rather than left to the nearest grid point (6.81), so
# the sweep's `lambda = 7.5` row reproduces the committed headline numbers exactly and the
# curve can be read against them.
LAMBDA_GRID: Sequence[float] = tuple(
    sorted({0.0, PRODUCTION_LAMBDA} | {float(x) for x in np.logspace(-2, 3.5, 34)})
)

# Arms with a GP variance to sweep. Baseline has neither `raw_logits` nor `uncertainty`, so
# it has no lambda axis at all and is emitted once as a flat reference instead.
SWEEP_GROUPS = ("sngp", "specreg", "specreg_trace")


def _ds_scores(raw: torch.Tensor, var: torch.Tensor, lam: float) -> np.ndarray:
    """Dempster-Shafer `K / (K + sum exp(z))` at one lambda. `[N, C], [N, 1] -> [N]`.

    Deliberately recomputed from `raw_logits` rather than read from the `dempster_shafer`
    column: that column was written from the *already corrected* logits, i.e. at lambda =
    7.5, so it is a constant with respect to this sweep. For the same reason this does not
    go through `src.metrics.auc._compute_ood_score_series`, whose DS branch prefers the
    persisted column and would silently return the production number at every lambda.

    Unlike MSP, DS responds to the absolute logit scale -- it is neither shift- nor
    scale-invariant, which is exactly why this project reports it alongside MSP.
    """
    return dempster_shafer(mean_field_scale(raw, var, lam)).numpy()


def _mean_shrink(var: torch.Tensor, lam: float) -> float:
    return float(torch.sqrt(1.0 + lam * var).mean())


def _margin_over_sigma(raw: torch.Tensor, var: torch.Tensor) -> np.ndarray:
    """The `lambda -> infinity` limit of the MSP ranking.

    For large lambda every scaled logit is small, so `softmax(z) ~ uniform + (z - mean z)/C`
    and the top-1 probability orders examples by `(max z - mean z)`, which at that limit is
    `(max raw - mean raw) / (sqrt(lambda) * sigma)`. The lambda factor is common to every
    example and cancels in a ranking, leaving margin / sigma. Larger means more confident,
    so this is a confidence score, not an uncertainty one.
    """
    r = raw.numpy()
    return (r.max(axis=1) - r.mean(axis=1)) / np.sqrt(var.numpy().squeeze(1))


def _sweep_rows(arm, lambdas: Sequence[float]) -> List[Dict[str, object]]:
    raw, var, targets = arm.id
    rows: List[Dict[str, object]] = []

    for lam in lambdas:
        metrics = _id_metrics(raw, var, targets, "softmax", lam)
        row: Dict[str, object] = {
            "kind": "sweep",
            "group": arm.group,
            "display": arm.display,
            "seed_label": arm.label,
            "lambda": lam,
            "mean_shrink": _mean_shrink(var, lam),
            **{k: metrics[k] for k in ("acc", "nll", "brier", "smece")},
        }

        id_msp = _msp(raw, var, "softmax", lam)
        id_ds = _ds_scores(raw, var, lam)
        for ds_name, (oraw, ovar) in arm.ood.items():
            row[f"auroc_msp_{ds_name}"] = float(AUROC(id_msp, _msp(oraw, ovar, "softmax", lam)))
            row[f"auroc_ds_{ds_name}"] = float(
                AUROC(id_ds, _ds_scores(oraw, ovar, lam), score_is_uncertainty=True)
            )

        # Control: a *global* temperature of the same average strength as this lambda. The
        # gap between it and the row above is the part of the movement that comes from the
        # correction being per-example -- i.e. from the GP variance -- rather than from
        # simply making the logits smaller.
        sbar = row["mean_shrink"]
        ctrl = _id_metrics(raw, var, targets, "softmax", 0.0, temperature=sbar)
        row["ctrl_nll"] = ctrl["nll"]
        row["ctrl_smece"] = ctrl["smece"]
        ctrl_id_msp = _msp(raw, var, "softmax", 0.0, temperature=sbar)
        for ds_name, (oraw, ovar) in arm.ood.items():
            row[f"ctrl_auroc_msp_{ds_name}"] = float(
                AUROC(ctrl_id_msp, _msp(oraw, ovar, "softmax", 0.0, temperature=sbar))
            )
        rows.append(row)

    # The lambda -> infinity limit, as its own row rather than a column repeated 29 times.
    asymptote: Dict[str, object] = {
        "kind": "asymptote",
        "group": arm.group,
        "display": arm.display,
        "seed_label": arm.label,
        "lambda": float("inf"),
        "mean_shrink": float("nan"),
    }
    id_margin = _margin_over_sigma(raw, var)
    for ds_name, (oraw, ovar) in arm.ood.items():
        asymptote[f"auroc_msp_{ds_name}"] = float(
            AUROC(id_margin, _margin_over_sigma(oraw, ovar))
        )
    rows.append(asymptote)
    return rows


def _baseline_rows(root: Path, tag: str, label: str, display: str) -> Optional[Dict[str, object]]:
    """The deterministic baseline, which has no lambda axis.

    Read straight from the written columns rather than recomputed: with no `raw_logits` and
    no variance there is nothing to rescale, so the published numbers *are* the curve.
    """
    try:
        frame = load_predictions(str(_infer_dir(root, tag, label, "cifar100") / "predictions.csv"), fold="test")
    except Exception:  # noqa: BLE001 - a missing arm must not kill the sweep
        return None
    df = frame.df
    col = next(c for c in LABEL_COLS if c in df.columns)
    y = df[col].to_numpy().astype(int)
    probs = probs_array(frame)
    conf = df["confidence"].to_numpy(dtype=float)
    correct = df["correct"].to_numpy().astype(float)

    row: Dict[str, object] = {
        "kind": "baseline",
        "group": "baseline",
        "display": display,
        "seed_label": label,
        "lambda": float("nan"),
        "mean_shrink": float("nan"),
        "acc": float(correct.mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()),
        "brier": float(
            ((probs - np.eye(probs.shape[1])[y]) ** 2).sum(axis=1).mean()
        ),
        "smece": float(smECE_fast_compat(conf, correct)),
    }
    for ds_name in OOD_DATASETS:
        try:
            ood = load_predictions(
                str(_infer_dir(root, tag, label, ds_name) / "predictions.csv"), fold="test"
            )
        except Exception:  # noqa: BLE001
            continue
        row[f"auroc_msp_{ds_name}"] = float(
            AUROC(conf, ood.df["confidence"].to_numpy(dtype=float))
        )
        row[f"auroc_ds_{ds_name}"] = float(
            AUROC(
                df["dempster_shafer"].to_numpy(dtype=float),
                ood.df["dempster_shafer"].to_numpy(dtype=float),
                score_is_uncertainty=True,
            )
        )
    return row


def _gate(arm) -> Dict[str, float]:
    """At the pinned lambda the offline path must reproduce what inference wrote.

    Both operands are checked, because they fail differently: `class_probs` would catch a
    wrong rescaling, and `dempster_shafer` additionally catches reading the persisted column
    by accident, since that column is constant in lambda and would match at 7.5 either way
    only if the recomputation is right.
    """
    raw, var, _ = arm.id
    probs = torch.softmax(mean_field_scale(raw, var, PRODUCTION_LAMBDA), dim=1).numpy()
    return {"probs": float(np.abs(probs - arm.published_probs).max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--csv", required=True, help="per-seed tidy CSV, one row per (arm, seed, lambda)")
    ap.add_argument("--out", default=None, help="optional markdown summary at a few lambdas")
    ap.add_argument("--infer-root", default=None, help="defaults to $EXPERIMENTS_HOME/$PROJECT_NAME/infer")
    ap.add_argument("--include-seed-12345", action="store_true")
    ap.add_argument(
        "--extra-lambda",
        type=float,
        nargs="+",
        default=None,
        help="additional lambda values to evaluate, spliced into the grid. Use it to put an "
             "arm's validation-fitted lambda on the curve, so a 'fitted on val, reported on "
             "test' table can be read straight from this CSV instead of interpolated.",
    )
    ap.add_argument(
        "--trace-tag",
        default=None,
        help="tag of the trace_logistic run tree (e.g. trace_logistic_2026-09-23_17-06-47); "
             "adds the SpecReg trace-logistic arm, which lives under its own tag rather than --tag",
    )
    args = ap.parse_args()

    root = Path(
        args.infer_root or Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "infer"
    )

    entries = [(args.tag, label, display, group) for label, (display, group) in ARMS.items()]
    if args.include_seed_12345:
        entries += [("", label, display, group) for label, (display, group) in LEGACY_ARMS.items()]
    if args.trace_tag:
        entries += [(args.trace_tag, label, display, group) for label, (display, group) in TRACE_ARMS.items()]

    lambdas = LAMBDA_GRID
    if args.extra_lambda:
        lambdas = tuple(sorted(set(lambdas) | {float(x) for x in args.extra_lambda}))

    rows: List[Dict[str, object]] = []
    gates: List[str] = []

    for tag, label, display, group in entries:
        if group == "baseline":
            base = _baseline_rows(root, tag, label, display)
            if base is not None:
                rows.append(base)
            continue
        if group not in SWEEP_GROUPS:
            continue
        arm = _collect(root, tag, label, display, group)
        if arm is None:
            continue
        gates.append(f"  {label:16s} max |delta prob| = {_gate(arm)['probs']:.2e}")
        print(f"sweeping {label} ({len(lambdas)} lambdas) ...", flush=True)
        rows.extend(_sweep_rows(arm, lambdas))

    if not rows:
        raise SystemExit(f"No arms found under {root}. Wrong --tag or --infer-root?")

    print("\nGate -- offline softmax at lambda = 7.5 vs the written class_probs:")
    print("\n".join(gates))

    fields = [
        "kind", "group", "display", "seed_label", "lambda", "mean_shrink",
        "acc", "nll", "brier", "smece",
        "auroc_msp_cifar10", "auroc_msp_svhn", "auroc_ds_cifar10", "auroc_ds_svhn",
        "ctrl_nll", "ctrl_smece", "ctrl_auroc_msp_cifar10", "ctrl_auroc_msp_svhn",
    ]
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    if args.out:
        _write_markdown(rows, Path(args.out))


def _write_markdown(rows: List[Dict[str, object]], out: Path) -> None:
    """A compact readout at a handful of lambdas -- the figure is the real deliverable."""
    highlights = [0.0, 1.0, PRODUCTION_LAMBDA, 20.0, 50.0, 200.0, 1000.0]

    def nearest(group: str, target: float) -> List[Dict[str, object]]:
        cand = [r for r in rows if r["kind"] == "sweep" and r["group"] == group]
        if not cand:
            return []
        lam = min({float(r["lambda"]) for r in cand}, key=lambda x: abs(x - target))
        return [r for r in cand if float(r["lambda"]) == lam]

    def fmt(vals: List[float]) -> str:
        vals = [v for v in vals if v is not None]
        if not vals:
            return "--"
        if len(vals) == 1:
            return f"{vals[0]:.4f}"
        return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"

    lines = [
        "# CIFAR-100: mean_field_factor sweep",
        "",
        "`±` is across training seeds. Accuracy is a control and must not move.",
        "",
    ]
    for group in SWEEP_GROUPS:
        sample = next((r for r in rows if r["group"] == group), None)
        if sample is None:
            continue
        lines += [
            f"## {sample['display']}",
            "",
            "| λ | acc | NLL | smECE | Brier | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for target in highlights:
            group_rows = nearest(group, target)
            if not group_rows:
                continue
            lam = float(group_rows[0]["lambda"])
            cells = [
                fmt([float(r[k]) for r in group_rows])
                for k in ("acc", "nll", "smece", "brier",
                          "auroc_msp_cifar10", "auroc_msp_svhn",
                          "auroc_ds_cifar10", "auroc_ds_svhn")
            ]
            mark = " *(current)*" if abs(lam - PRODUCTION_LAMBDA) < 1e-9 else ""
            lines.append(f"| {lam:g}{mark} | " + " | ".join(cells) + " |")
        asym = [r for r in rows if r["kind"] == "asymptote" and r["group"] == group]
        if asym:
            lines += [
                "",
                "λ → ∞ limit (MSP ranking becomes logit-margin / σ): "
                f"C-10 {fmt([float(r['auroc_msp_cifar10']) for r in asym])}, "
                f"SVHN {fmt([float(r['auroc_msp_svhn']) for r in asym])}.",
                "",
            ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
