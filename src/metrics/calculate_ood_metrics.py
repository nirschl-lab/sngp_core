"""calculate_ood_metrics.py in src/metrics.

Cross-dataset OOD AUROC for a single checkpoint's inference run -- compares one
in-distribution dataset against one or more out-of-distribution datasets, using both
max-softmax-probability (`confidence`) and normalized-entropy (`class_probs`) scores.

Usage:
    uv run src/metrics/calculate_ood_metrics.py \\
        --run-dir /data1/maheswararao/experiments/uncertaity-aware-ml/infer/baseline_classifier_acevedo/2026-08-25_14-31-36 \\
        --indist acevedo \\
        --outdist jung kather2016 kather2018 nirschl2018 tang wong \\
        --out-dir csv/ood_metrics \\
        --name baseline_acevedo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.auc import AUROC, _normalized_entropy, _parse_class_probs, seeds  # noqa: E402
from src.metrics.auc import sample_rate as DEFAULT_SAMPLE_RATE  # noqa: E402
from src.metrics.io import filter_fold  # noqa: E402


def load_ood_scores(predictions_csv: Path, fold: str | None = None) -> tuple[pd.Series, pd.Series]:
    """Load a predictions.csv and return (msp_scores, entropy_scores)."""
    if not predictions_csv.exists():
        raise FileNotFoundError(f"No predictions.csv at {predictions_csv}")

    df = pd.read_csv(predictions_csv)
    df = filter_fold(df, fold, source=predictions_csv)

    if "confidence" not in df.columns:
        raise KeyError(f"{predictions_csv} is missing the 'confidence' column needed for msp AUROC.")
    if "class_probs" not in df.columns:
        raise KeyError(f"{predictions_csv} is missing the 'class_probs' column needed for entropy AUROC.")

    msp_scores = pd.to_numeric(df["confidence"], errors="coerce").dropna()

    class_probs = df["class_probs"].map(_parse_class_probs)
    entropy_scores = class_probs.map(lambda p: _normalized_entropy(p) if p is not None else np.nan).dropna()

    return msp_scores, entropy_scores


def _auroc_mean_std_parts(
    id_scores: pd.Series, ood_scores: pd.Series, score_is_uncertainty: bool, sample_rate: int
) -> tuple[float, float]:
    """Mean and std of AUROC across `seeds`, as floats -- the core `_auroc_mean_std`
    (below) formats into `"mean ± std"` for, and what a metric-registry adapter wants
    directly rather than re-parsing that string."""
    n_samples = min(sample_rate, len(id_scores), len(ood_scores))
    if n_samples == 0:
        raise ValueError("No valid rows available to compute AUROC.")

    auroc_list = [
        AUROC(
            id_scores.sample(n_samples, random_state=seed),
            ood_scores.sample(n_samples, random_state=seed),
            score_is_uncertainty=score_is_uncertainty,
        )
        for seed in seeds
    ]
    return float(np.mean(auroc_list)), float(np.std(auroc_list))


def _auroc_mean_std(id_scores: pd.Series, ood_scores: pd.Series, score_is_uncertainty: bool, sample_rate: int) -> str:
    mean, std = _auroc_mean_std_parts(id_scores, ood_scores, score_is_uncertainty, sample_rate)
    return f"{mean:.4f} ± {std:.4f}"


def compute_ood_auroc(
    run_dir: Path,
    indist: str,
    ood_datasets: list[str],
    fold: str | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> pd.DataFrame:
    """Cross-dataset OOD AUROC (msp + entropy) for one in-distribution dataset against
    each of `ood_datasets`, reading `<run_dir>/<name>/predictions.csv` for each name.
    """
    id_msp, id_entropy = load_ood_scores(run_dir / indist / "predictions.csv", fold=fold)

    rows = []
    for name in ood_datasets:
        ood_msp, ood_entropy = load_ood_scores(run_dir / name / "predictions.csv", fold=fold)
        try:
            msp_auroc = _auroc_mean_std(id_msp, ood_msp, score_is_uncertainty=False, sample_rate=sample_rate)
            entropy_auroc = _auroc_mean_std(
                id_entropy, ood_entropy, score_is_uncertainty=True, sample_rate=sample_rate
            )
        except ValueError as e:
            raise ValueError(f"{e} (indist={indist!r}, outdist={name!r})") from e
        rows.append({"dataset": name, "msp_auroc": msp_auroc, "entropy_auroc": entropy_auroc})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset OOD AUROC for a single checkpoint's inference run.")
    parser.add_argument(
        "--run-dir", required=True, type=Path, help="Directory whose immediate subfolders hold predictions.csv."
    )
    parser.add_argument("--indist", required=True, help="Folder name under --run-dir treated as in-distribution.")
    parser.add_argument(
        "--outdist", required=True, nargs="+", help="Folder name(s) under --run-dir treated as out-of-distribution."
    )
    parser.add_argument("--out-dir", default=str(ROOT / "csv" / "ood_metrics"))
    parser.add_argument("--name", default=None, help="Output filename stem (default: '<indist>_ood_metrics').")
    parser.add_argument("--fold", default=None, help="Filter rows to this fold before scoring (default: none).")
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Per-seed sample size for AUROC."
    )
    args = parser.parse_args()

    if args.indist in args.outdist:
        raise ValueError(f"--indist {args.indist!r} cannot also appear in --outdist.")
    if len(args.outdist) != len(set(args.outdist)):
        raise ValueError(f"Duplicate dataset name(s) in --outdist: {args.outdist}")

    missing = [
        name
        for name in [args.indist, *args.outdist]
        if not (args.run_dir / name / "predictions.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(f"No predictions.csv found for: {missing} under {args.run_dir}")

    df = compute_ood_auroc(
        args.run_dir, args.indist, args.outdist, fold=args.fold, sample_rate=args.sample_rate
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{args.indist}_ood_metrics"
    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    print(df)


if __name__ == "__main__":
    main()
