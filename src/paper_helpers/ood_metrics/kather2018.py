"""Kather2018 (in-distribution) vs. every other dataset (OOD) AUROC comparison across
Baseline/MC-Dropout/SNGP. See src/paper_helpers/ood_metrics/runner.py.
"""
from pathlib import Path

from src.paper_helpers.ood_metrics.runner import run_ood_comparison

run_ood_comparison(
    dataset="kather2018",
    methods={
        "Baseline": Path("csv/final/kather2018_baseline"),
        "MC Dropout": Path("csv/final/kather2018_mc"),
        "SNGP": Path("csv/final/kather2018_sngp"),
    },
    ood_datasets=["kather2016", "nirschl", "wong", "tang", "jung", "acevedo"],
    out_dir=Path("csv/final"),
    score_mode="msp",
)
