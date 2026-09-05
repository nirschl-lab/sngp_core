"""Batch metric driver: run every registered metric (or a chosen subset) across every
manifest-tracked inference run (or a chosen subset), writing one long-format CSV.

Before this, adding a metric to the picture across all 76 tracked prediction CSVs
meant hand-driving a per-run script 15+ times. This is the loop `.claude/skills/
metrics/SKILL.md` already forbids reimplementing per-dataset -- reusing
`src/metrics/manifest.py` for run identity and `src/metrics/registry.py` for what to
compute, over `src/metrics/io.py`'s normalized `PredictionFrame`.

Usage:
    uv run src/metrics/run_manifest_metrics.py
    uv run src/metrics/run_manifest_metrics.py --runs baseline_acevedo sngp_acevedo
    uv run src/metrics/run_manifest_metrics.py --metrics basic_stats dempster_shafer
    uv run src/metrics/run_manifest_metrics.py --validate-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import rootutils
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.calculate_ood_metrics import DEFAULT_SAMPLE_RATE  # noqa: E402
from src.metrics.io import MissingPredictionData, load_predictions  # noqa: E402
from src.metrics.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    Manifest,
    RunEntry,
    check_master_docs_are_current,
    load_manifest,
    resolve_default_infer_root,
    validate_manifest,
)
from src.metrics.registry import METRIC_REGISTRY, MetricContext, get_metric  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "csv" / "run_metrics"


def _row(
    run: RunEntry,
    metric: str,
    scope: str,
    *,
    status: str,
    value: Optional[float] = None,
    std: Optional[float] = None,
    value_str: Optional[str] = None,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "label": run.label,
        "method": run.method,
        "train_dataset": run.train_dataset,
        "id_dataset": run.id_dataset,
        "scope": scope,
        "metric": metric,
        "value": value,
        "std": std,
        "value_str": value_str,
        "status": status,
        "note": note,
    }


def run_metrics_for_entry(
    run: RunEntry,
    manifest: Manifest,
    *,
    metric_names: Sequence[str],
    fold: Optional[str],
    sample_rate: int,
) -> List[Dict[str, Any]]:
    """Run every named metric against one manifest entry, returning long-format rows.

    Never raises for an expected degraded state -- a manifest entry with a single
    eval dataset (`needs_ood`), a run predating a needed column (`requires`), or one
    OOD dataset's predictions.csv failing to load -- all become `status` values on a
    row instead. `--fold` (default `None`, matching
    `calculate_ood_metrics.py --fold`'s own default) is applied identically to the ID
    frame and every OOD frame, since that -- not `AUROC_across_dataset`'s asymmetric
    `OodFoldPolicy`, a different function for the older `csv/final/` layout -- is
    what actually produced the published `csv/ood_metrics/*.csv` numbers.
    """
    id_path = manifest.infer_root / run.run_dir / run.id_dataset / "predictions.csv"
    try:
        frame = load_predictions(id_path, fold=fold)
    except (FileNotFoundError, MissingPredictionData) as e:
        logger.warning(f"{run.label}: could not load ID frame at {id_path}: {e}")
        return [_row(run, name, "run", status="error", note=str(e)) for name in metric_names]

    ood_frames = {}
    for dataset in run.eval_datasets:
        if dataset == run.id_dataset:
            continue
        ood_path = manifest.infer_root / run.run_dir / dataset / "predictions.csv"
        try:
            ood_frames[dataset] = load_predictions(ood_path, fold=fold)
        except (FileNotFoundError, MissingPredictionData) as e:
            logger.warning(f"{run.label}: could not load OOD frame {dataset!r} at {ood_path}: {e}")

    ctx = MetricContext(
        frame=frame,
        ood_frames=ood_frames,
        run=run.__dict__,
        options={"sample_rate": sample_rate},
    )

    rows: List[Dict[str, Any]] = []
    for name in metric_names:
        spec = get_metric(name)

        if spec.needs_ood and not ood_frames:
            rows.append(_row(run, name, "run", status="skipped", note="no OOD datasets in manifest entry"))
            continue

        missing = spec.requires - frame.capabilities
        if missing:
            rows.append(
                _row(run, name, "run", status="skipped", note=f"missing capabilities: {sorted(missing)}")
            )
            continue

        for metric_row in spec.fn(ctx):
            rows.append(
                _row(
                    run,
                    metric_row.metric,
                    metric_row.scope,
                    status="ok",
                    value=metric_row.value,
                    std=metric_row.std,
                    value_str=metric_row.value_str,
                )
            )

    return rows


def run_manifest_metrics(
    manifest: Manifest,
    *,
    runs: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    fold: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> pd.DataFrame:
    """Run `metrics` (default: every registered metric) across `runs` (default: every
    manifest entry), returning the long-format result as a DataFrame."""
    selected_runs = [manifest.get(label) for label in runs] if runs else list(manifest.runs)
    metric_names = list(metrics) if metrics else sorted(METRIC_REGISTRY)

    all_rows: List[Dict[str, Any]] = []
    for run in selected_runs:
        all_rows.extend(
            run_metrics_for_entry(run, manifest, metric_names=metric_names, fold=fold, sample_rate=sample_rate)
        )
    return pd.DataFrame(all_rows)


def main() -> None:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), type=Path)
    parser.add_argument("--infer-root", default=None, help="Override infer_root instead of resolving it from env.")
    parser.add_argument("--runs", nargs="+", default=None, help="Manifest labels to include (default: all).")
    parser.add_argument("--metrics", nargs="+", default=None, help="Registered metric names to run (default: all).")
    parser.add_argument(
        "--fold", default=None, help="Filter loaded predictions to this fold (default: none, i.e. unfiltered)."
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate the manifest and exit -- no metrics are run."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run without loading any CSVs or writing output."
    )
    args = parser.parse_args()

    infer_root = args.infer_root or resolve_default_infer_root()
    manifest = load_manifest(args.manifest, infer_root=infer_root)

    if args.validate_only:
        problems = validate_manifest(manifest, check_disk=True) + check_master_docs_are_current(manifest)
        if problems:
            for problem in problems:
                print(f"PROBLEM: {problem}")
            raise SystemExit(f"{len(problems)} problem(s) found.")
        print(f"OK -- {len(manifest.runs)} manifest entries validated.")
        return

    selected_runs = [manifest.get(label) for label in args.runs] if args.runs else list(manifest.runs)
    metric_names = args.metrics or sorted(METRIC_REGISTRY)

    if args.dry_run:
        print(f"Would run {metric_names} across {len(selected_runs)} run(s):")
        for run in selected_runs:
            print(f"  {run.label} ({run.method}, id_dataset={run.id_dataset}, eval={list(run.eval_datasets)})")
        return

    df = run_manifest_metrics(
        manifest, runs=args.runs, metrics=metric_names, fold=args.fold, sample_rate=args.sample_rate
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_long.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows across {len(selected_runs)} run(s))")


if __name__ == "__main__":
    main()
