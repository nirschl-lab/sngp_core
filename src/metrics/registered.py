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

from dataclasses import replace
from typing import List

import numpy as np
import pandas as pd
import torch

from src.metrics.calculate_ood_metrics import DEFAULT_SAMPLE_RATE, _auroc_mean_std_parts
from src.metrics.dempster_shafer_uncertainity import DempsterShaferUncertainty
from src.metrics.dispersion import mean_std_sem
from src.metrics.io import logits_array, uncertainty_kind
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
    `predictions_csv_schema: 2` -- which, until that checkpoint/dataset is manually
    re-run through `infer.py`, is every run on disk today."""
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


def _mean_row(metric: str, values: pd.Series) -> MetricRow:
    """A `MetricRow` carrying both the mean and the spread of a per-sample column.

    `MetricRow.std` has been part of the contract since the registry was written but
    only `ood_auroc_*` ever filled it; these per-sample columns are the other natural
    source, since their spread is exact rather than resampled
    (`src/metrics/dispersion.py`).
    """
    summary = mean_std_sem(torch.tensor(values.to_numpy(dtype=float)))
    return MetricRow(
        metric=metric,
        value=summary.mean,
        std=summary.std,
        value_str=f"{summary.mean:.4f} ± {summary.std:.4f}",
        scope="run",
        extra={"sem": summary.sem, "n": summary.n},
    )


@register_metric(
    "predictive_uncertainty",
    requires=frozenset({"comparable_uncertainty"}),
    description=(
        "Mean predictive entropy (nats), confidence margin, and Dempster-Shafer uncertainty -- the "
        "cross-family comparable per-sample uncertainty columns, each with its across-sample spread."
    ),
)
def _predictive_uncertainty(ctx: MetricContext) -> List[MetricRow]:
    """Unlike `dempster_shafer` (which recomputes from `class_logits`), this reads the
    columns the write path already produced, so it also covers the callback schema and
    costs no re-derivation. The two agree by construction -- `build_records` writes
    `src.metrics.uncertainty.dempster_shafer`, which reproduces the numpy original."""
    df = ctx.frame.df
    return [
        _mean_row("mean_predictive_entropy", df["predictive_entropy"]),
        _mean_row("mean_confidence_margin", df["confidence_margin"]),
        _mean_row("mean_dempster_shafer", df["dempster_shafer"]),
    ]


@register_metric(
    "uncertainty_decomposition",
    requires=frozenset({"uncertainty_decomposition"}),
    description=(
        "Mean total/aleatoric/epistemic predictive entropy (nats) for runs with a member stack "
        "(Deep Ensemble or MC-Dropout), each with its across-sample spread."
    ),
)
def _uncertainty_decomposition(ctx: MetricContext) -> List[MetricRow]:
    """Skips on SNGP and plain-Baseline runs: a single latent variance is not a sample
    over models, so there is nothing to decompose (`requires` handles the skip, giving
    a `status=skipped` row with a reason rather than an error)."""
    df = ctx.frame.df
    return [
        _mean_row("mean_total_entropy", df["total_entropy"]),
        _mean_row("mean_aleatoric_entropy", df["aleatoric_entropy"]),
        _mean_row("mean_mutual_information", df["mutual_information"]),
    ]


@register_metric(
    "model_uncertainty",
    requires=frozenset({"uncertainty"}),
    description=(
        "Mean model-side uncertainty (SNGP GP variance / ensemble logit variance / MC-Dropout logit "
        "std). Units differ per family -- the scope names which, via uncertainty_kind."
    ),
)
def _model_uncertainty(ctx: MetricContext) -> List[MetricRow]:
    """The `uncertainty` column's unit is family-dependent, so this row is *not*
    comparable across families -- `use predictive_uncertainty` for that. The kind is
    carried in `extra` so a reader that aggregates these rows can tell that two of them
    are in different units instead of averaging them together."""
    row = _mean_row("mean_model_uncertainty", ctx.frame.df["uncertainty"])
    kind = uncertainty_kind(ctx.frame)
    return [replace(row, extra={**row.extra, "uncertainty_kind": kind or "unknown"})]
