"""Acevedo (in-distribution) vs. every other dataset (OOD) AUROC comparison across
Baseline/MC-Dropout/SNGP. See src/paper_helpers/ood_metrics/runner.py -- this file is
now just the config for one comparison, not a copy of the comparison loop itself.
"""
from pathlib import Path

from src.paper_helpers.ood_metrics.runner import run_ood_comparison

run_ood_comparison(
    dataset="acevedo",
    methods={
        "Baseline": Path("csv/final/acevedo_baseline"),
        "MC Dropout": Path("csv/final/acevedo_mc"),
        "SNGP": Path("csv/final/acevedo_sngp"),
    },
    ood_datasets=["jung", "wong", "kather2016", "kather2018", "nirschl", "tang"],
    out_dir=Path("csv/final"),
    score_mode="msp",
    fid_scores={
        "jung": 306.65,
        "wong": 351.25,
        "kather2016": 375.57,
        "kather2018": 377.93,
        "nirschl": 379.89,
        "tang": 388.94,
    },
)
