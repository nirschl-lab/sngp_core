"""Generalized cross-dataset OOD AUROC comparison.

Replaces the copy-pasted per-ID-dataset scripts that used to live here (acevedo.py,
kather2018.py, wong.py -- ~55-75 lines each, differing only in dataset name, CSV
paths, and score mode) with one parametrized function, modeled on the
already-well-designed `src/metrics/artifact_quantification.py`. Adding a new
in-distribution dataset is now a config-only call site (see acevedo.py/kather2018.py/
wong.py for examples), never a new copy of this loop.
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.metrics.auc import AUROC_across_dataset

# Every dataset's prediction CSVs use this filename regardless of which dataset's
# checkpoint produced them -- only the containing directory (one per method) changes
# per comparison. See src/inference/infer.py / the `inference` skill for how these
# CSVs get produced.
DATASET_CSV_FILENAMES = {
    "acevedo": "acevedo_et_al_2020.csv",
    "jung": "jung_et_al_2022.csv",
    "kather2016": "kather_et_al_2016.csv",
    "kather2018": "kather_et_al_2018.csv",
    "nirschl": "nirschl_et_al_2018.csv",
    "tang": "tang_et_al_2019.csv",
    "wong": "wong_et_al_2022.csv",
}


def run_ood_comparison(
    dataset: str,
    methods: Dict[str, Path],
    ood_datasets: List[str],
    out_dir: Path,
    score_mode: str = "msp",
    fid_scores: Optional[Dict[str, float]] = None,
    csv_filenames: Optional[Dict[str, str]] = None,
    out_filename: Optional[str] = None,
) -> pd.DataFrame:
    """Cross-dataset OOD AUROC comparison for one in-distribution dataset across
    multiple trained methods (e.g. Baseline / MC-Dropout / SNGP).

    Args:
        dataset: in-distribution dataset short name (a key of `csv_filenames`).
        methods: `{method_name: csv_dir}` -- `csv_dir` holds one predictions CSV per
            dataset (see `DATASET_CSV_FILENAMES`).
        ood_datasets: short names of datasets to treat as OOD.
        out_dir: directory to write the results CSV to.
        score_mode: `"msp"` (max softmax probability) or `"entropy"` -- see
            `src/metrics/auc.py::AUROC_across_dataset`.
        fid_scores: optional `{ood_dataset: fid_score}` row appended to the output.
        csv_filenames: override `DATASET_CSV_FILENAMES`.
        out_filename: override the default `"<dataset>_results.csv"` output name.

    Returns:
        The results DataFrame (one row per method, one column per OOD dataset).
    """
    csv_filenames = csv_filenames or DATASET_CSV_FILENAMES

    rows = []
    for method_name, csv_dir in methods.items():
        res = AUROC_across_dataset(str(csv_dir), csv_filenames, [dataset], ood_datasets, score_mode=score_mode)
        rows.append({"Method": method_name, **res})

    if fid_scores:
        rows.append({
            "Method": "FID Score",
            **{name: f"{fid_scores[name]:.2f}" for name in ood_datasets if name in fid_scores},
        })

    df = pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_filename or f"{dataset}_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    print(df)

    return df
