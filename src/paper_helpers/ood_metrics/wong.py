"""Wong (in-distribution) vs. every other dataset (OOD) AUROC comparison across
Baseline/MC-Dropout/SNGP, using entropy-based uncertainty (ISBI 2026 run). See
src/paper_helpers/ood_metrics/runner.py.
"""
from pathlib import Path

from src.paper_helpers.ood_metrics.runner import run_ood_comparison

run_ood_comparison(
    dataset="wong",
    methods={
        "Baseline": Path("csv/isbi_test_files/wong_baseline"),
        "MC Dropout": Path("csv/isbi_test_files/wong_mc"),
        "SNGP": Path("csv/isbi_test_files/wong_sngp"),
    },
    ood_datasets=["tang", "kather2018", "kather2016", "jung", "nirschl", "acevedo"],
    out_dir=Path("csv/isbi_test_files"),
    score_mode="entropy",
    out_filename="wong_results_entropy.csv",
)
