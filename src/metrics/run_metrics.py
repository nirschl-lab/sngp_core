"""Batch metric driver: run every registered metric (or a chosen subset) across a set
of prediction CSVs discovered on disk, writing one long-format CSV.

Point it at explicit predictions.csv paths and/or a glob; each run's identity and OOD
candidates are inferred from the on-disk layout (`<run_dir>/<dataset_name>/
predictions.csv`, see docs/OUTPUT_LAYOUT.md) -- every sibling dataset directory under
the same run_dir becomes an OOD frame for `needs_ood` metrics automatically. Re-running
after registering a new metric in src/metrics/registered.py upserts only the rows for
that metric into the existing output CSV -- everything else already there is left
alone (see docs/METRICS_GUIDE.md's "Adding a new metric to already-computed runs").

Usage:
    uv run src/metrics/run_metrics.py --predictions <path/to/predictions.csv> [<path> ...]
    uv run src/metrics/run_metrics.py --glob '/data1/.../infer/**/predictions.csv'
    uv run src/metrics/run_metrics.py --glob '...' --metrics basic_stats dempster_shafer
    uv run src/metrics/run_metrics.py --glob '...' --root /data1/.../infer
"""

from __future__ import annotations

import argparse
import glob as glob_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import rootutils
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.calculate_ood_metrics import DEFAULT_SAMPLE_RATE  # noqa: E402
from src.metrics.io import MissingPredictionData, load_predictions  # noqa: E402
from src.metrics.registry import METRIC_REGISTRY, MetricContext, get_metric  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "csv" / "run_metrics"
OUTPUT_COLUMNS = ["label", "run_dir", "dataset", "scope", "metric", "value", "std", "value_str", "status", "note"]
UPSERT_KEY = ["label", "metric", "scope"]


def discover_ood_siblings(predictions_path: Path) -> Dict[str, Path]:
    """Every sibling dataset directory under the same run_dir that has its own
    predictions.csv, keyed by dataset name -- excludes `images/` and the dataset
    `predictions_path` itself belongs to."""
    dataset = predictions_path.parent.name
    run_dir = predictions_path.parent.parent
    siblings: Dict[str, Path] = {}
    if not run_dir.is_dir():
        return siblings
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or child.name in (dataset, "images"):
            continue
        candidate = child / "predictions.csv"
        if candidate.exists():
            siblings[child.name] = candidate
    return siblings


def label_for(path: Path, root: Optional[Path]) -> str:
    """A human-readable run identifier: the path relative to `root` if given (minus
    the trailing predictions.csv), else the last two path components
    (`<run_dir_name>/<dataset>`)."""
    if root is not None:
        try:
            return str(path.relative_to(root).parent)
        except ValueError:
            pass
    return f"{path.parent.parent.name}/{path.parent.name}"


def _row(
    label: str,
    run_dir: str,
    dataset: str,
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
        "label": label,
        "run_dir": run_dir,
        "dataset": dataset,
        "scope": scope,
        "metric": metric,
        "value": value,
        "std": std,
        "value_str": value_str,
        "status": status,
        "note": note,
    }


def run_metrics_for_path(
    predictions_path: Path,
    *,
    metric_names: Sequence[str],
    fold: Optional[str],
    sample_rate: int,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run every named metric against one predictions.csv, returning long-format rows.

    Never raises for an expected degraded state -- no sibling OOD directories on disk
    (`needs_ood`), a run predating a needed column (`requires`), or one OOD sibling's
    predictions.csv failing to load -- all become `status` values on a row instead.
    """
    label = label_for(predictions_path, root)
    dataset = predictions_path.parent.name
    run_dir = str(predictions_path.parent.parent)

    try:
        frame = load_predictions(predictions_path, fold=fold)
    except (FileNotFoundError, MissingPredictionData) as e:
        logger.warning(f"{label}: could not load {predictions_path}: {e}")
        return [_row(label, run_dir, dataset, name, "run", status="error", note=str(e)) for name in metric_names]

    ood_frames = {}
    for name, sibling_path in discover_ood_siblings(predictions_path).items():
        try:
            ood_frames[name] = load_predictions(sibling_path, fold=fold)
        except (FileNotFoundError, MissingPredictionData) as e:
            logger.warning(f"{label}: could not load OOD sibling {name!r} at {sibling_path}: {e}")

    ctx = MetricContext(
        frame=frame,
        ood_frames=ood_frames,
        run={"label": label, "run_dir": run_dir, "dataset": dataset},
        options={"sample_rate": sample_rate},
    )

    rows: List[Dict[str, Any]] = []
    for name in metric_names:
        spec = get_metric(name)

        if spec.needs_ood and not ood_frames:
            rows.append(
                _row(label, run_dir, dataset, name, "run", status="skipped", note="no sibling OOD datasets on disk")
            )
            continue

        missing = spec.requires - frame.capabilities
        if missing:
            rows.append(
                _row(
                    label,
                    run_dir,
                    dataset,
                    name,
                    "run",
                    status="skipped",
                    note=f"missing capabilities: {sorted(missing)}",
                )
            )
            continue

        for metric_row in spec.fn(ctx):
            rows.append(
                _row(
                    label,
                    run_dir,
                    dataset,
                    metric_row.metric,
                    metric_row.scope,
                    status="ok",
                    value=metric_row.value,
                    std=metric_row.std,
                    value_str=metric_row.value_str,
                )
            )

    return rows


def run_metrics(
    predictions_paths: Sequence[Path],
    *,
    metrics: Optional[Sequence[str]] = None,
    fold: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Run `metrics` (default: every registered metric) across `predictions_paths`,
    returning the long-format result as a DataFrame."""
    metric_names = list(metrics) if metrics else sorted(METRIC_REGISTRY)

    all_rows: List[Dict[str, Any]] = []
    for path in predictions_paths:
        all_rows.extend(
            run_metrics_for_path(
                Path(path), metric_names=metric_names, fold=fold, sample_rate=sample_rate, root=root
            )
        )
    return pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)


def upsert(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Replace rows in `existing` that share a `(label, metric, scope)` key with the
    matching row in `new`; every other existing row is kept as-is."""
    if existing.empty:
        return new
    new_keys = set(map(tuple, new[UPSERT_KEY].itertuples(index=False, name=None)))
    existing_keys = existing[UPSERT_KEY].apply(tuple, axis=1)
    kept = existing[~existing_keys.isin(new_keys)]
    return pd.concat([kept, new], ignore_index=True)


def main() -> None:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", default=None, type=Path, help="Explicit predictions.csv path(s).")
    parser.add_argument("--glob", default=None, help="Recursive glob pattern for predictions.csv files.")
    parser.add_argument("--root", default=None, type=Path, help="Base dir for tidier labels (path made relative to this).")
    parser.add_argument("--metrics", nargs="+", default=None, help="Registered metric names to run (default: all).")
    parser.add_argument(
        "--fold", default=None, help="Filter loaded predictions to this fold (default: none, i.e. unfiltered)."
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-name", default="metrics_long.csv")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run without loading any CSVs or writing output."
    )
    args = parser.parse_args()

    paths: List[Path] = list(args.predictions or [])
    if args.glob:
        paths.extend(Path(p) for p in sorted(glob_module.glob(args.glob, recursive=True)))
    if not paths:
        raise SystemExit("No predictions.csv paths given -- pass --predictions and/or --glob.")

    metric_names = args.metrics or sorted(METRIC_REGISTRY)

    if args.dry_run:
        print(f"Would run {metric_names} across {len(paths)} file(s):")
        for path in paths:
            print(f"  {label_for(path, args.root)} ({path})")
        return

    df = run_metrics(paths, metrics=metric_names, fold=args.fold, sample_rate=args.sample_rate, root=args.root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    if out_path.exists():
        df = upsert(pd.read_csv(out_path), df)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows across {len(paths)} file(s))")


if __name__ == "__main__":
    main()
