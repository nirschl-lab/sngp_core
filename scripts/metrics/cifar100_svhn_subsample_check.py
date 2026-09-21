"""Does capping SVHN at 10,000 rows change the CIFAR-100 far-OOD AUROC? No.

The results page reports SVHN AUROC over the full 26,032-row test split, while the SNGP
paper's CIFAR splits are 10,000 -- which leaves open whether our 7-10 pt shortfall against
the published far-OOD numbers is just the sample count. This script settles that: it
computes each run's SVHN AUROC over the full split and over ten independent 10,000-row
draws, and prints the comparison.

AUROC estimates `P(score_OOD > score_ID)`, a rank statistic, so dropping 60% of the OOD
rows at random should cost precision and not position -- and measurably it does: the
3-seed means move by <= 0.0004. Recorded in docs/results/CIFAR100_RESULTS.md.

A one-off check, deliberately standalone: nothing imports it, it is not wired into
scripts/tmux/cifar100_overnight.sh, and it adds no option to src/metrics/. There is no
test for it either -- it takes ~40s and reads a cluster path CI does not have.

    uv run python scripts/metrics/cifar100_svhn_subsample_check.py \
        --tag overnight_2026-09-20_21-38-42
"""
import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.auc import (  # noqa: E402
    AUROC,
    UNCERTAINTY_SCORE_MODES,
    _compute_ood_score_series,
    seeds,
)

# Same directory (sys.path[0] when run as a script), so this resolves without packaging.
# Importing the mapping rather than restating it is the point: the two scripts must not
# drift on which run directory belongs to which arm.
sys.path.insert(0, str(Path(__file__).parent))
from cifar100_overnight_report import ARMS, LEGACY_ARMS, _infer_dirname  # noqa: E402

SUBSAMPLE_N = 10_000
SCORE_MODES = (("msp", "MSP"), ("dempster_shafer", "DS"))
# Only the arms that ran more than one seed -- a single-seed arm has no 3-seed mean to
# compare, and the question here is about the aggregate.
GROUPS = {"baseline": "Baseline", "sngp": "SNGP (c = 6.0)", "specreg": "SpecReg (matched)"}


def _runs(tag: str) -> Dict[str, List[Tuple[str, str]]]:
    """group -> [(tag, label)], pooling the overnight arms with the first run's seed 12345."""
    out: Dict[str, List[Tuple[str, str]]] = {g: [] for g in GROUPS}
    for label, (_, group) in LEGACY_ARMS.items():
        if group in out:
            out[group].append(("", label))
    for label, (_, group) in ARMS.items():
        if group in out:
            out[group].append((tag, label))
    return out


def _compare(root: Path, tag: str, label: str, score_mode: str) -> Tuple[float, List[float]]:
    """(full-population AUROC, one AUROC per 10k draw) for this run against SVHN."""
    id_df = pd.read_csv(root / _infer_dirname(tag, label, "cifar100") / "predictions.csv")
    ood_df = pd.read_csv(root / _infer_dirname(tag, label, "svhn") / "predictions.csv")
    id_s = _compute_ood_score_series(id_df, score_mode)
    ood_s = _compute_ood_score_series(ood_df, score_mode)
    uncertain = score_mode in UNCERTAINTY_SCORE_MODES

    full = float(AUROC(id_s, ood_s, score_is_uncertainty=uncertain))
    draws = [
        float(AUROC(id_s, ood_s.sample(SUBSAMPLE_N, random_state=s), score_is_uncertainty=uncertain))
        for s in seeds
    ]
    return full, draws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--infer-root", default=None, help="defaults to $EXPERIMENTS_HOME/$PROJECT_NAME/infer"
    )
    args = ap.parse_args()

    root = Path(
        args.infer_root
        or Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "infer"
    )
    runs = _runs(args.tag)

    print(f"SVHN: full test split (26,032) vs {SUBSAMPLE_N:,}-row draws, {len(seeds)} draws each\n")
    print("| Arm | Score | SVHN full (26,032) | 10K draws (mean) | Δ |")
    print("|---|---|---:|---:|---:|")

    spreads = []
    for score_mode, score_label in SCORE_MODES:
        for group, display in GROUPS.items():
            fulls, draw_means = [], []
            for tag, label in runs[group]:
                try:
                    full, draws = _compare(root, tag, label, score_mode)
                except Exception as exc:  # noqa: BLE001 - report the gap, don't die on it
                    print(f"<!-- skipped {group}/{label}: {exc} -->", file=sys.stderr)
                    continue
                fulls.append(full)
                draw_means.append(statistics.mean(draws))
                spreads.append(max(draws) - min(draws))
            if not fulls:
                print(f"| {display} | {score_label} | -- | -- | -- |")
                continue
            f_m, d_m = statistics.mean(fulls), statistics.mean(draw_means)
            f_sd = statistics.stdev(fulls) if len(fulls) > 1 else 0.0
            d_sd = statistics.stdev(draw_means) if len(draw_means) > 1 else 0.0
            print(
                f"| {display} | {score_label} | {f_m:.4f} ± {f_sd:.4f} | "
                f"{d_m:.4f} ± {d_sd:.4f} | {d_m - f_m:+.4f} |"
            )

    if spreads:
        print(
            f"\nPer-run spread across the {len(seeds)} draws (max − min): "
            f"{min(spreads):.4f}–{max(spreads):.4f}."
        )


if __name__ == "__main__":
    main()
