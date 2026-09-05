"""Registers this project's metric functions into `src.metrics.registry.METRIC_REGISTRY`.

Importing this module is enough to populate the registry -- `src/metrics/__init__.py`
does exactly that, so `from src.metrics.registry import METRIC_REGISTRY` (which
imports the `src.metrics` package first, running its `__init__.py`) sees every metric
below regardless of which module a test, script, or the batch driver happens to import
first. `pyproject.toml` runs tests via `pytest-randomly` (randomized order), so this
indirection is what keeps registration order-independent -- nothing here may depend on
having been imported before or after any other module.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.metrics.calculate_ood_metrics import DEFAULT_SAMPLE_RATE, _auroc_mean_std_parts
from src.metrics.dempster_shafer_uncertainity import DempsterShaferUncertainty
from src.metrics.io import logits_array
from src.metrics.registry import MetricContext, MetricRow, register_metric


def _ood_auroc_rows(
    ctx: MetricContext,
    *,
    metric_name: str,
    score_column: str,
    score_is_uncertainty: bool,
) -> List[MetricRow]:
    """Shared body for `ood_auroc_msp`/`ood_auroc_entropy`: one row per OOD dataset,
    reusing `calculate_ood_metrics._auroc_mean_std_parts` (the float-returning core
    behind `csv/ood_metrics/*.csv`'s published `"mean ± std"` strings) so this
    reproduces those numbers exactly rather than recomputing AUROC a third way.
    Both scores are already columns on `PredictionFrame.df`
    (`confidence`/`entropy_norm`), so no re-parsing of `class_probs` is needed here.
    """
    sample_rate = int(ctx.options.get("sample_rate", DEFAULT_SAMPLE_RATE))
    id_scores = ctx.frame.df[score_column].dropna()

    rows: List[MetricRow] = []
    for dataset_name, ood_frame in ctx.ood_frames.items():
        ood_scores = ood_frame.df[score_column].dropna()
        mean, std = _auroc_mean_std_parts(id_scores, ood_scores, score_is_uncertainty, sample_rate)
        rows.append(
            MetricRow(
                metric=metric_name,
                value=mean,
                std=std,
                value_str=f"{mean:.4f} ± {std:.4f}",
                scope=dataset_name,
            )
        )
    return rows


@register_metric(
    "ood_auroc_msp",
    needs_ood=True,
    description="Max-softmax-probability OOD AUROC, ID vs each OOD dataset, bootstrapped over 10 seeds.",
)
def _ood_auroc_msp(ctx: MetricContext) -> List[MetricRow]:
    return _ood_auroc_rows(ctx, metric_name="ood_auroc_msp", score_column="confidence", score_is_uncertainty=False)


@register_metric(
    "ood_auroc_entropy",
    needs_ood=True,
    description="Normalized-entropy OOD AUROC, ID vs each OOD dataset, bootstrapped over 10 seeds.",
)
def _ood_auroc_entropy(ctx: MetricContext) -> List[MetricRow]:
    return _ood_auroc_rows(
        ctx, metric_name="ood_auroc_entropy", score_column="entropy_norm", score_is_uncertainty=True
    )


@register_metric(
    "dempster_shafer",
    requires=frozenset({"logits"}),
    description="Mean Dempster-Shafer uncertainty (K / (K + sum(exp(logits)))) from class_logits.",
)
def _dempster_shafer(ctx: MetricContext) -> List[MetricRow]:
    """Skips (via `requires={"logits"}`) on any run written before
    `predictions_csv_schema: 2` -- which, until the backfill in
    `scripts/inference/rerun_with_logits.sh` runs, is every run on disk today."""
    logits = logits_array(ctx.frame)
    uncertainty = DempsterShaferUncertainty(logits)
    return [MetricRow(metric="dempster_shafer", value=float(np.mean(uncertainty)), scope="run")]


@register_metric(
    "basic_stats",
    description="Accuracy, mean confidence, and mean normalized entropy from a single run's predictions.",
)
def _basic_stats(ctx: MetricContext) -> List[MetricRow]:
    df = ctx.frame.df
    return [
        MetricRow(metric="accuracy", value=float(df["correct"].mean()), scope="run"),
        MetricRow(metric="mean_confidence", value=float(df["confidence"].mean()), scope="run"),
        MetricRow(metric="mean_entropy", value=float(df["entropy_norm"].mean()), scope="run"),
    ]
