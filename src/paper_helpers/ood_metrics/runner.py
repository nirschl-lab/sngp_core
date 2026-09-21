"""Generalized cross-dataset OOD AUROC comparison.

Replaces the copy-pasted per-ID-dataset scripts that used to live here (acevedo.py,
kather2018.py, wong.py -- ~55-75 lines each, differing only in dataset name, CSV
paths, and score mode) with one parametrized function, modeled on the
already-well-designed `src/metrics/artifact_quantification.py`. Adding a new
in-distribution dataset is now a config-only call site (see acevedo.py/kather2018.py/
wong.py for examples), never a new copy of this loop.
"""
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd

from src.metrics.auc import AUROC_across_dataset, AUROC_across_dataset_full_population
from src.metrics.io import LEGACY_ISBI_FOLD_POLICY, SYMMETRIC_FOLD_POLICY, OodFoldPolicy

# Each estimator's own default fold policy, resolved explicitly rather than by omitting
# the kwarg: the frozen subsample path keeps the published ISBI asymmetry, the
# full-population path filters both frames to test (see src/metrics/auc.py).
_DEFAULT_FOLD_POLICY = {
    "bootstrap": LEGACY_ISBI_FOLD_POLICY,
    "full_population": SYMMETRIC_FOLD_POLICY,
}

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
    *,
    estimator: Literal["bootstrap", "full_population"] = "bootstrap",
    fold_policy: Optional[OodFoldPolicy] = None,
) -> pd.DataFrame:
    """Cross-dataset OOD AUROC comparison for one in-distribution dataset across
    multiple trained methods (e.g. Baseline / MC-Dropout / SNGP).

    Args:
        dataset: in-distribution dataset short name (a key of `csv_filenames`).
        methods: `{method_name: csv_dir}` -- `csv_dir` holds one predictions CSV per
            dataset (see `DATASET_CSV_FILENAMES`).
        ood_datasets: short names of datasets to treat as OOD.
        out_dir: directory to write the results CSV to.
        score_mode: `"msp"` (max softmax probability), `"entropy"`, or
            `"dempster_shafer"` -- see `src/metrics/auc.py::_compute_ood_score_series`.
        fid_scores: optional `{ood_dataset: fid_score}` row appended to the output.
        csv_filenames: override `DATASET_CSV_FILENAMES`.
        out_filename: override the default `"<dataset>_results.csv"` output name.
        estimator: `"bootstrap"` (default) gives `"mean ± std"` strings over 10
            fixed-seed subsamples -- frozen, reproduces the published ISBI numbers.
            `"full_population"` gives one deterministic float over every row of both
            test sets, the SNGP paper's protocol. The two do not agree to 4 decimals.
        fold_policy: override the estimator's default (`_DEFAULT_FOLD_POLICY`).

    Returns:
        The results DataFrame (one row per method, one column per OOD dataset). Cells
        are strings under `"bootstrap"` and floats under `"full_population"`.
    """
    if estimator not in _DEFAULT_FOLD_POLICY:
        raise ValueError(
            f"Unknown estimator {estimator!r}. Use 'bootstrap' (frozen, published ISBI "
            "numbers) or 'full_population' (the SNGP paper's protocol)."
        )
    csv_filenames = csv_filenames or DATASET_CSV_FILENAMES
    fold_policy = fold_policy or _DEFAULT_FOLD_POLICY[estimator]

    rows = []
    for method_name, csv_dir in methods.items():
        if estimator == "full_population":
            res = AUROC_across_dataset_full_population(
                str(csv_dir), csv_filenames, dataset, ood_datasets,
                score_mode=score_mode, fold_policy=fold_policy,
            )
        else:
            res = AUROC_across_dataset(
                str(csv_dir), csv_filenames, [dataset], ood_datasets,
                score_mode=score_mode, fold_policy=fold_policy,
            )
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
    # float_format keeps the full_population path's cells at 4 dp rather than
    # 0.81290000000001; a no-op for the bootstrap path, whose cells are strings.
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Results saved to {out_path}")
    print(df)

    return df
