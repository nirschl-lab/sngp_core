"""Tests for src/metrics/registry.py -- the metric registry mechanics -- and
src/metrics/registered.py's four initial metrics.
"""

import pytest

from src.metrics.io import load_predictions
from src.metrics.registry import (
    METRIC_REGISTRY,
    MetricContext,
    get_metric,
    register_metric,
)
from tests.helpers.predictions import write_predictions_csv


# --- registry mechanics --------------------------------------------------------------


def test_registered_metrics_are_populated_regardless_of_import_order():
    """src/metrics/__init__.py imports registered.py as a side effect, so importing
    just `src.metrics.registry` (as this test does, before anything else in this file
    touches `src.metrics.registered`) must already see every metric."""
    assert {"ood_auroc_msp", "ood_auroc_entropy", "dempster_shafer", "basic_stats"}.issubset(METRIC_REGISTRY)


def test_get_metric_returns_the_registered_spec():
    spec = get_metric("basic_stats")
    assert spec.name == "basic_stats"
    assert callable(spec.fn)


def test_get_metric_raises_key_error_listing_registered_names():
    with pytest.raises(KeyError, match="basic_stats"):
        get_metric("not_a_real_metric")


def test_register_metric_rejects_a_duplicate_name_bound_to_a_different_function():
    @register_metric("test_registry_dup")
    def _first(ctx):
        return []

    with pytest.raises(ValueError, match="already registered"):

        @register_metric("test_registry_dup")
        def _second(ctx):
            return []


def test_register_metric_allows_re_registering_the_same_function():
    """Re-importing a module (e.g. under pytest-randomly re-collection) re-runs its
    decorators -- registering the identical function object a second time must not
    raise."""

    def _fn(ctx):
        return []

    register_metric("test_registry_idempotent")(_fn)
    register_metric("test_registry_idempotent")(_fn)  # should not raise


def test_register_metric_returns_the_function_unchanged():
    def _fn(ctx):
        return []

    returned = register_metric("test_registry_returns_fn")(_fn)
    assert returned is _fn


def test_requires_and_needs_ood_are_stored_on_the_spec():
    @register_metric("test_registry_requires", requires={"logits"}, needs_ood=True)
    def _fn(ctx):
        return []

    spec = get_metric("test_registry_requires")
    assert spec.requires == frozenset({"logits"})
    assert spec.needs_ood is True


# --- fixtures for the four registered metrics -----------------------------------------


def _write_inference_csv(path, *, n, num_classes, confidences, seed, with_logits, all_correct=False):
    write_predictions_csv(
        path,
        n=n,
        num_classes=num_classes,
        confidences=confidences,
        seed=seed,
        with_logits=with_logits,
        all_correct=all_correct,
    )


def test_basic_stats_reports_accuracy_confidence_and_entropy(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(
        path, n=4, num_classes=2, confidences=[0.9, 0.9, 0.9, 0.9], seed=0, with_logits=False, all_correct=True
    )
    frame = load_predictions(path)

    rows = get_metric("basic_stats").fn(MetricContext(frame=frame))

    by_name = {r.metric: r for r in rows}
    assert set(by_name) == {"accuracy", "mean_confidence", "mean_entropy"}
    assert by_name["accuracy"].value == pytest.approx(1.0)
    assert by_name["mean_confidence"].value == pytest.approx(0.9)
    assert all(r.scope == "run" for r in rows)


def test_dempster_shafer_skips_without_logits_via_requires(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=3, num_classes=2, confidences=[0.9, 0.8, 0.7], seed=0, with_logits=False)
    frame = load_predictions(path)

    spec = get_metric("dempster_shafer")
    missing = spec.requires - frame.capabilities
    assert missing == {"logits"}  # what a driver checks before calling spec.fn at all


def test_dempster_shafer_runs_when_logits_are_present(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=3, num_classes=2, confidences=[0.9, 0.8, 0.7], seed=0, with_logits=True)
    frame = load_predictions(path)

    assert "logits" in frame.capabilities
    rows = get_metric("dempster_shafer").fn(MetricContext(frame=frame))
    assert len(rows) == 1
    assert rows[0].metric == "dempster_shafer"
    assert 0.0 <= rows[0].value <= 1.0


def test_ood_auroc_msp_and_entropy_produce_one_row_per_ood_dataset(tmp_path):
    id_path = tmp_path / "id" / "predictions.csv"
    ood_a_path = tmp_path / "ood_a" / "predictions.csv"
    ood_b_path = tmp_path / "ood_b" / "predictions.csv"
    _write_inference_csv(id_path, n=20, num_classes=2, confidences=[0.95] * 20, seed=0, with_logits=False)
    _write_inference_csv(ood_a_path, n=20, num_classes=2, confidences=[0.55] * 20, seed=1, with_logits=False)
    _write_inference_csv(ood_b_path, n=20, num_classes=2, confidences=[0.5] * 20, seed=2, with_logits=False)

    ctx = MetricContext(
        frame=load_predictions(id_path),
        ood_frames={"ood_a": load_predictions(ood_a_path), "ood_b": load_predictions(ood_b_path)},
        options={"sample_rate": 10},
    )

    msp_rows = get_metric("ood_auroc_msp").fn(ctx)
    entropy_rows = get_metric("ood_auroc_entropy").fn(ctx)

    assert {r.scope for r in msp_rows} == {"ood_a", "ood_b"}
    assert {r.scope for r in entropy_rows} == {"ood_a", "ood_b"}
    for row in msp_rows + entropy_rows:
        assert row.value is not None
        assert row.std is not None
        assert "±" in row.value_str


def test_ood_auroc_matches_calculate_ood_metrics_exactly(tmp_path):
    """The registry adapter must reproduce calculate_ood_metrics.py's published
    numbers, not just be structurally similar to them -- same seeds, same sampling."""
    from src.metrics.calculate_ood_metrics import compute_ood_auroc

    run_dir = tmp_path / "run"
    _write_inference_csv(
        run_dir / "acevedo" / "predictions.csv", n=30, num_classes=2, confidences=[0.9] * 30, seed=0, with_logits=False
    )
    _write_inference_csv(
        run_dir / "wong" / "predictions.csv", n=30, num_classes=2, confidences=[0.5] * 30, seed=1, with_logits=False
    )

    reference = compute_ood_auroc(run_dir, indist="acevedo", ood_datasets=["wong"], sample_rate=15)
    reference_row = reference.iloc[0]

    ctx = MetricContext(
        frame=load_predictions(run_dir / "acevedo" / "predictions.csv"),
        ood_frames={"wong": load_predictions(run_dir / "wong" / "predictions.csv")},
        options={"sample_rate": 15},
    )
    msp_row = get_metric("ood_auroc_msp").fn(ctx)[0]
    entropy_row = get_metric("ood_auroc_entropy").fn(ctx)[0]

    assert msp_row.value_str == reference_row["msp_auroc"]
    assert entropy_row.value_str == reference_row["entropy_auroc"]


def test_ood_auroc_empty_ood_frames_yields_no_rows(tmp_path):
    """A run with no sibling OOD dataset on disk (e.g. wong_ucdavis) has no OOD frames
    at all -- the metric itself degrades to zero rows; a driver is responsible for
    turning that into an explicit skip status, not this function."""
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=3, num_classes=2, confidences=[0.9, 0.8, 0.7], seed=0, with_logits=False)

    ctx = MetricContext(frame=load_predictions(path), ood_frames={})
    assert get_metric("ood_auroc_msp").fn(ctx) == []
    assert get_metric("ood_auroc_entropy").fn(ctx) == []
