"""Assemble the overnight CIFAR-100 follow-up into one markdown summary.

Driven by scripts/tmux/cifar100_overnight.sh. Reads the prediction CSVs that run wrote
and reports, per arm: in-distribution accuracy / NLL / top-label smECE, and OOD AUROC
against CIFAR-10 (near) and SVHN (far) under both MSP and Dempster-Shafer. Where an arm
has several seeds it reports mean ± std across seeds -- the thing the first run could
not provide, and now the *only* thing ± ever means here.

OOD AUROC follows the SNGP paper's protocol (arXiv 2205.00403 §6.2.1, appendix C.1):
the full CIFAR-100 test set against the full OOD test set, no subsampling, one
deterministic number per run. MSP is the paper's own score; DS is the reference
implementation's, kept as a companion table.

Deliberately failure-tolerant: a missing arm becomes a "--" row, never an exception, so
one bad run cannot cost the whole night's summary.
"""
import argparse
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.auc import AUROC_across_dataset_full_population  # noqa: E402
from src.metrics.io import load_predictions, probs_array  # noqa: E402
from src.metrics.smooth_ece import smECE_fast_compat  # noqa: E402

LABEL_COLS = ("label", "target", "ground_truth", "true_label")
OOD_DATASETS = (("c10", "cifar10"), ("svhn", "svhn"))
# (score_mode, bucket suffix, column header prefix)
SCORE_MODES = (("msp", "msp", "MSP"), ("dempster_shafer", "ds", "DS"))
GROUP_ORDER = ("baseline", "sngp", "sngp_c41", "specreg", "literal")

# label -> (display name, group key for seed aggregation). These live under the
# overnight tag prefix: <root>/<tag>_<label>__<dataset>.
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

# Seed 12345, from the first run -- a different directory convention with no tag
# prefix (<root>/<label>__<dataset>, see scripts/metrics/stage_cifar100_ood.sh).
# Opt-in via --include-seed-12345; pooling these with ARMS is what makes the headline
# table three seeds per arm rather than two.
LEGACY_ARMS = {
    "cifar100last_baseline": ("Baseline", "baseline"),
    "cifar100last_sngp": ("SNGP (c=6.0)", "sngp"),
    "cifar100last_specreg": ("SpecReg (matched)", "specreg"),
}


def _infer_dirname(tag: str, label: str, dataset: str) -> str:
    """`<tag>_<label>__<dataset>`, or `<label>__<dataset>` when `tag` is empty (the
    first run's convention)."""
    return f"{tag}_{label}__{dataset}" if tag else f"{label}__{dataset}"


def _infer_dir(root: Path, tag: str, label: str, dataset: str) -> Path:
    return root / _infer_dirname(tag, label, dataset)


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


def _ood_auroc(root: Path, tag: str, label: str, score_mode: str) -> Dict[str, float]:
    """Full-population AUROC for one arm against both OOD sets, in one call.

    No resampling and no seed loop: the SNGP protocol's number is a single deterministic
    AUROC over the whole test sets, and the only dispersion this report shows is across
    training seeds (`_fmt` over an ARMS group).

    `score_is_uncertainty` is derived from `score_mode` inside
    `AUROC_across_dataset_full_population` rather than hardcoded here -- it differs
    between MSP and DS, and getting it wrong silently returns `1 - AUROC`.
    """
    names = {
        ds: f"{_infer_dirname(tag, label, ds)}/predictions.csv"
        for ds in ("cifar100", "cifar10", "svhn")
    }
    try:
        return AUROC_across_dataset_full_population(
            str(root), names, "cifar100", [ds for _, ds in OOD_DATASETS],
            score_mode=score_mode,
        )
    except Exception:  # noqa: BLE001 - a missing arm must not kill the summary
        return {}


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
    ap.add_argument(
        "--include-seed-12345",
        action="store_true",
        help="also pool the first run's seed-12345 arms (a different, untagged directory "
             "convention), making the table three seeds per arm instead of two",
    )
    args = ap.parse_args()

    root = Path(
        args.infer_root
        or Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "infer"
    )

    # (tag, label, display, group) -- the legacy arms carry an empty tag.
    entries: List[Tuple[str, str, str, str]] = [
        (args.tag, label, display, group) for label, (display, group) in ARMS.items()
    ]
    if args.include_seed_12345:
        entries += [("", label, display, group) for label, (display, group) in LEGACY_ARMS.items()]

    bucket_keys = ["acc", "nll", "smece"] + [
        f"{key}_{suffix}" for key, _ in OOD_DATASETS for _, suffix, _ in SCORE_MODES
    ]
    per_arm: Dict[str, Dict[str, List[float]]] = {}
    names: Dict[str, str] = {}
    present: Dict[str, List[str]] = {}
    for tag, label, display, group in entries:
        idm = _id_metrics(_infer_dir(root, tag, label, "cifar100"))
        if idm is None:
            continue
        bucket = per_arm.setdefault(group, {k: [] for k in bucket_keys})
        names[group] = display
        present.setdefault(group, []).append(label)
        for stat in ("acc", "nll", "smece"):
            bucket[stat].append(idm[stat])
        for score_mode, suffix, _ in SCORE_MODES:
            res = _ood_auroc(root, tag, label, score_mode)
            for key, ds in OOD_DATASETS:
                if ds in res:
                    bucket[f"{key}_{suffix}"].append(res[ds])

    def _table(header: List[str], keys: List[str]) -> List[str]:
        rows = [
            "| " + " | ".join(header) + " |",
            "|---" + "|---:" * (len(header) - 1) + "|",
        ]
        for group in GROUP_ORDER:
            if group not in per_arm:
                rows.append(f"| {names.get(group, group)} | 0 |" + " -- |" * (len(header) - 2))
                continue
            b = per_arm[group]
            cells = " | ".join(_fmt(b[k]) for k in keys)
            rows.append(f"| {names[group]} | {len(b['acc'])} | {cells} |")
        return rows

    lines = [
        f"# CIFAR-100 overnight follow-up — `{args.tag}`",
        "",
        "All rows are `last.ckpt` (epoch 249), the reference's reporting point. Where an",
        "arm ran more than one seed the cell is **mean ± std across seeds** — that is the",
        "only thing ± means here, so read it before any between-arm gap. smECE is",
        "top-label.",
        "",
        "OOD AUROC follows the SNGP paper's protocol (arXiv 2205.00403 §6.2.1, appendix",
        "C.1): the **full** CIFAR-100 test set (10,000) against the **full** OOD test set",
        "(CIFAR-10 10,000, SVHN 26,032), no subsampling, one deterministic number per run.",
        "Unequal group sizes are fine — AUROC is a rank statistic. MSP is the paper's own",
        "score and the table to compare against its Table 3; DS is the reference",
        "*implementation*'s score (Table 11), below.",
        "",
        "Not comparable to numbers from `AUROC_across_dataset`, which subsamples 1000 rows",
        "per frame over 10 fixed seeds and reports a population std (`np.std`); the ± here",
        "is a sample std across seeds (`statistics.stdev`, ddof=1).",
        "",
        "## Headline — MSP (the paper's protocol)",
        "",
    ]
    lines += _table(
        ["Arm", "seeds", "Accuracy", "NLL", "smECE", "MSP AUROC vs CIFAR-10", "MSP AUROC vs SVHN"],
        ["acc", "nll", "smece", "c10_msp", "svhn_msp"],
    )
    lines += [
        "",
        "## Dempster-Shafer companion",
        "",
        "`K / (K + Σ exp(logit))` — total evidence mass, which softmax discards: it is",
        "shift-invariant, so MSP is blind to exactly the logit magnitude an SNGP head is",
        "trained to modulate. Where MSP and DS disagree on the ordering of the arms, that",
        "disagreement is itself the finding.",
        "",
    ]
    lines += _table(
        ["Arm", "seeds", "DS AUROC vs CIFAR-10", "DS AUROC vs SVHN"],
        ["c10_ds", "svhn_ds"],
    )

    lines += [
        "",
        "## What to check",
        "",
        "- **Do the seed spreads overlap?** If the std on NLL / smECE / OOD AUROC is the",
        "  size of the gap between SpecReg and SNGP, the first run's apparent win is not",
        "  yet established.",
        "- **Do MSP and DS agree?** The published CIFAR-100 rows are MSP 0.795 (DNN) /",
        "  0.798 (SNGP) near-OOD and 0.799 / 0.846 far-OOD; DS 0.804 / 0.784 and 0.841 /",
        "  0.894. DS flatters SNGP on SVHN and *hurts* it on CIFAR-10 in the paper too.",
        "- **`SNGP (c=4.1)`** is the operator-norm-equivalent control for the 1.46x",
        "  estimator mismatch. If it beats `c=6.0` on OOD, the original SNGP arm was",
        "  under-constrained and understated.",
        "- **`SpecReg (literal)`** now has its own run directory, so its `last.ckpt`",
        "  exists this time.",
        "",
        "Arms found: " + ", ".join(f"{g} ({'+'.join(present[g])})" for g in sorted(present)),
    ]
    if not args.include_seed_12345:
        lines += [
            "",
            "Seed 12345 is **not** pooled in — pass `--include-seed-12345` for three seeds",
            "per arm.",
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
