"""Assemble the overnight CIFAR-100 follow-up into one markdown summary.

Driven by scripts/tmux/cifar100_overnight.sh. Reads the prediction CSVs that run wrote
and reports, per arm: in-distribution accuracy / NLL / top-label smECE, and
Dempster-Shafer OOD AUROC against CIFAR-10 and SVHN. Where an arm has several seeds it
reports mean ± std across seeds -- the thing the first run could not provide.

Deliberately failure-tolerant: a missing arm becomes a "--" row, never an exception, so
one bad run cannot cost the whole night's summary.
"""
import argparse
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.auc import AUROC, _compute_ood_score_series  # noqa: E402
from src.metrics.io import load_predictions, probs_array  # noqa: E402
from src.metrics.smooth_ece import smECE_fast_compat  # noqa: E402

LABEL_COLS = ("label", "target", "ground_truth", "true_label")
SEEDS = {"s1": 1, "s2": 2}
# label -> (display name, group key for seed aggregation)
ARMS = {
    "s1_baseline": ("Baseline", "baseline"),
    "s2_baseline": ("Baseline", "baseline"),
    "s1_sngp": ("SNGP (c=6.0)", "sngp"),
    "s2_sngp": ("SNGP (c=6.0)", "sngp"),
    "s1_specreg": ("SpecReg (matched)", "specreg"),
    "s2_specreg": ("SpecReg (matched)", "specreg"),
    "c41_sngp": ("SNGP (c=4.1)", "sngp_c41"),
    "literal_rerun": ("SpecReg (literal)", "literal"),
}


def _infer_dir(root: Path, tag: str, label: str, dataset: str) -> Path:
    return root / f"{tag}_{label}__{dataset}"


def _id_metrics(path: Path) -> Optional[Dict[str, float]]:
    try:
        frame = load_predictions(str(path / "predictions.csv"), fold="test")
    except Exception:  # noqa: BLE001 - a missing arm must not kill the summary
        return None
    df = frame.df
    probs = probs_array(frame)
    col = next(c for c in LABEL_COLS if c in df.columns)
    y = df[col].to_numpy().astype(int)
    return {
        "acc": float(df["correct"].mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()),
        "smece": float(
            smECE_fast_compat(
                df["confidence"].to_numpy(dtype=float), df["correct"].to_numpy().astype(float)
            )
        ),
    }


def _ood_auroc(id_path: Path, ood_path: Path) -> Optional[float]:
    try:
        id_df = pd.read_csv(id_path / "predictions.csv")
        ood_df = pd.read_csv(ood_path / "predictions.csv")
        id_s = _compute_ood_score_series(id_df, "dempster_shafer")
        ood_s = _compute_ood_score_series(ood_df, "dempster_shafer")
    except Exception:  # noqa: BLE001
        return None
    n = min(1000, len(id_s), len(ood_s))
    if n == 0:
        return None
    vals = [
        AUROC(id_s.sample(n, random_state=s), ood_s.sample(n, random_state=s),
              score_is_uncertainty=True)
        for s in (42, 1337, 12345, 8675309, 314159)
    ]
    return float(np.mean(vals))


def _fmt(values: List[float], places: int = 4) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{places}f}"
    return f"{statistics.mean(vals):.{places}f} ± {statistics.stdev(vals):.{places}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--infer-root",
        default=None,
        help="defaults to $EXPERIMENTS_HOME/$PROJECT_NAME/infer",
    )
    args = ap.parse_args()

    root = Path(
        args.infer_root
        or Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "infer"
    )

    per_arm: Dict[str, Dict[str, List[float]]] = {}
    names: Dict[str, str] = {}
    present: Dict[str, List[str]] = {}
    for label, (display, group) in ARMS.items():
        idm = _id_metrics(_infer_dir(root, args.tag, label, "cifar100"))
        if idm is None:
            continue
        bucket = per_arm.setdefault(group, {"acc": [], "nll": [], "smece": [], "c10": [], "svhn": []})
        names[group] = display
        present.setdefault(group, []).append(label)
        bucket["acc"].append(idm["acc"])
        bucket["nll"].append(idm["nll"])
        bucket["smece"].append(idm["smece"])
        for key, ds in (("c10", "cifar10"), ("svhn", "svhn")):
            val = _ood_auroc(
                _infer_dir(root, args.tag, label, "cifar100"),
                _infer_dir(root, args.tag, label, ds),
            )
            if val is not None:
                bucket[key].append(val)

    lines = [
        f"# CIFAR-100 overnight follow-up — `{args.tag}`",
        "",
        "All rows are `last.ckpt` (epoch 249), the reference's reporting point. Where an",
        "arm ran more than one seed the cell is **mean ± std across seeds** — that spread",
        "is the thing the first run could not provide, so read it before any between-arm",
        "gap. smECE is top-label; OOD AUROC is Dempster-Shafer, mean over 5 resamples.",
        "",
        "| Arm | seeds | Accuracy | NLL | smECE | AUROC vs CIFAR-10 | AUROC vs SVHN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("baseline", "sngp", "sngp_c41", "specreg", "literal"):
        if group not in per_arm:
            lines.append(f"| {names.get(group, group)} | 0 | -- | -- | -- | -- | -- |")
            continue
        b = per_arm[group]
        lines.append(
            f"| {names[group]} | {len(b['acc'])} | {_fmt(b['acc'])} | {_fmt(b['nll'])} | "
            f"{_fmt(b['smece'])} | {_fmt(b['c10'])} | {_fmt(b['svhn'])} |"
        )

    lines += [
        "",
        "Seed-12345 rows from the first run are in",
        "`docs/results/CIFAR100_RESULTS.md`; pooling them with these gives 3 seeds per arm.",
        "",
        "## What to check",
        "",
        "- **Do the seed spreads overlap?** If the std on NLL / smECE / OOD AUROC is the",
        "  size of the gap between SpecReg and SNGP, the first run's apparent win is not",
        "  yet established.",
        "- **`SNGP (c=4.1)`** is the operator-norm-equivalent control for the 1.46x",
        "  estimator mismatch. If it beats `c=6.0` on OOD, the original SNGP arm was",
        "  under-constrained and understated.",
        "- **`SpecReg (literal)`** now has its own run directory, so its `last.ckpt`",
        "  exists this time.",
        "",
        f"Arms found: " + ", ".join(f"{g} ({'+'.join(present[g])})" for g in sorted(present)),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
