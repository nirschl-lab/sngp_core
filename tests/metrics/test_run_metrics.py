"""Tests for src/metrics/run_metrics.py -- the manifest-free batch driver.

A synthetic two-run `infer/`-shaped tmp_path tree covers the mechanics (expected
rows/statuses, the no-sibling-OOD skip, a missing-logits skip, a missing OOD sibling
being dropped rather than blowing up the whole run, and the upsert behavior against a
pre-existing output CSV).
"""

from pathlib import Path

import pandas as pd

from src.metrics.run_metrics import label_for, run_metrics, upsert
from tests.helpers.predictions import write_predictions_csv


def _populate_two_run_tree(infer_root: Path, *, with_logits_on_first: bool = False):
    """`multi/a` has a sibling OOD dataset (`multi/b`), so `needs_ood` metrics run;
    `single/c` has no siblings, so they skip. Matches the shape of a real
    acevedo-style run vs a wong_ucdavis-style leaf."""
    multi_a = infer_root / "multi" / "a" / "predictions.csv"
    multi_b = infer_root / "multi" / "b" / "predictions.csv"
    single_c = infer_root / "single" / "c" / "predictions.csv"

    write_predictions_csv(multi_a, n=20, num_classes=2, confidences=[0.95] * 20, seed=0, with_logits=with_logits_on_first)
    write_predictions_csv(multi_b, n=20, num_classes=2, confidences=[0.5] * 20, seed=1)
    write_predictions_csv(single_c, n=10, num_classes=2, confidences=[0.9] * 10, seed=2)

    return multi_a, multi_b, single_c


# --- run_metrics: mechanics ----------------------------------------------------------


def test_run_metrics_produces_a_row_per_metric_and_scope(tmp_path):
    multi_a, _, _ = _populate_two_run_tree(tmp_path / "infer")

    df = run_metrics([multi_a], metrics=["basic_stats"])

    assert set(df["metric"]) == {"accuracy", "mean_confidence", "mean_entropy"}
    assert (df["status"] == "ok").all()
    assert (df["scope"] == "run").all()
    assert (df["dataset"] == "a").all()


def test_needs_ood_metric_skips_when_no_sibling_dataset_on_disk(tmp_path):
    _, _, single_c = _populate_two_run_tree(tmp_path / "infer")

    df = run_metrics([single_c], metrics=["ood_auroc_msp"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "skipped"
    assert "no sibling OOD" in row["note"]


def test_needs_ood_metric_runs_when_a_sibling_dataset_exists(tmp_path):
    multi_a, _, _ = _populate_two_run_tree(tmp_path / "infer")

    df = run_metrics([multi_a], metrics=["ood_auroc_msp"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "ok"
    assert row["scope"] == "b"
    assert "±" in row["value_str"]


def test_requires_metric_skips_when_capability_is_missing(tmp_path):
    multi_a, _, _ = _populate_two_run_tree(tmp_path / "infer", with_logits_on_first=False)

    df = run_metrics([multi_a], metrics=["dempster_shafer"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "skipped"
    assert "logits" in row["note"]


def test_requires_metric_runs_when_capability_is_present(tmp_path):
    multi_a, _, _ = _populate_two_run_tree(tmp_path / "infer", with_logits_on_first=True)

    df = run_metrics([multi_a], metrics=["dempster_shafer"])

    assert len(df) == 1
    assert df.iloc[0]["status"] == "ok"


def test_missing_predictions_file_yields_an_error_row_per_requested_metric(tmp_path):
    missing_path = tmp_path / "infer" / "no_data" / "a" / "predictions.csv"
    # Deliberately don't write anything at missing_path.

    df = run_metrics([missing_path], metrics=["basic_stats", "dempster_shafer"])

    assert len(df) == 2
    assert (df["status"] == "error").all()
    assert set(df["metric"]) == {"basic_stats", "dempster_shafer"}


def test_a_missing_ood_sibling_is_dropped_not_fatal(tmp_path):
    """One bad sibling directory shouldn't block metrics for the other OOD siblings or
    for non-OOD metrics on the same run."""
    infer_root = tmp_path / "infer"
    run_a = infer_root / "partial" / "a" / "predictions.csv"
    run_b = infer_root / "partial" / "b" / "predictions.csv"
    write_predictions_csv(run_a, n=10, num_classes=2, seed=0)
    write_predictions_csv(run_b, n=10, num_classes=2, seed=1)
    # "missing" sibling directory doesn't exist at all -- discover_ood_siblings only
    # finds directories that actually have a predictions.csv, so it's simply absent.
    (infer_root / "partial" / "missing").mkdir(parents=True)

    df = run_metrics([run_a], metrics=["ood_auroc_msp", "basic_stats"])

    ood_rows = df[df["metric"] == "ood_auroc_msp"]
    assert set(ood_rows["scope"]) == {"b"}
    assert (df[df["metric"] == "basic_stats"]["status"] == "ok").all()


def test_default_metrics_is_every_registered_metric(tmp_path):
    multi_a, _, _ = _populate_two_run_tree(tmp_path / "infer")
    df = run_metrics([multi_a])
    assert {"accuracy", "mean_confidence", "mean_entropy", "dempster_shafer", "ood_auroc_msp", "ood_auroc_entropy"} <= set(
        df["metric"]
    )


# --- label_for -------------------------------------------------------------------


def test_label_for_uses_root_relative_path_when_given(tmp_path):
    root = tmp_path / "infer"
    path = root / "baseline_acevedo" / "wong" / "predictions.csv"
    assert label_for(path, root) == "baseline_acevedo/wong"


def test_label_for_falls_back_to_last_two_components_without_root(tmp_path):
    path = tmp_path / "some" / "deep" / "baseline_acevedo" / "wong" / "predictions.csv"
    assert label_for(path, None) == "baseline_acevedo/wong"


# --- upsert ------------------------------------------------------------------------


def test_upsert_replaces_matching_keys_and_keeps_the_rest():
    existing = pd.DataFrame(
        [
            {"label": "run1", "run_dir": "d1", "dataset": "a", "scope": "run", "metric": "basic_stats", "value": 1.0, "std": None, "value_str": None, "status": "ok", "note": ""},
            {"label": "run1", "run_dir": "d1", "dataset": "a", "scope": "run", "metric": "old_metric", "value": 2.0, "std": None, "value_str": None, "status": "ok", "note": ""},
        ]
    )
    new = pd.DataFrame(
        [
            {"label": "run1", "run_dir": "d1", "dataset": "a", "scope": "run", "metric": "basic_stats", "value": 99.0, "std": None, "value_str": None, "status": "ok", "note": ""},
        ]
    )

    result = upsert(existing, new)

    assert len(result) == 2
    updated = result[(result["label"] == "run1") & (result["metric"] == "basic_stats")]
    assert updated.iloc[0]["value"] == 99.0
    assert "old_metric" in set(result["metric"])


def test_upsert_appends_brand_new_keys():
    existing = pd.DataFrame(
        [
            {"label": "run1", "run_dir": "d1", "dataset": "a", "scope": "run", "metric": "basic_stats", "value": 1.0, "std": None, "value_str": None, "status": "ok", "note": ""},
        ]
    )
    new = pd.DataFrame(
        [
            {"label": "run1", "run_dir": "d1", "dataset": "a", "scope": "run", "metric": "dempster_shafer", "value": 0.5, "std": None, "value_str": None, "status": "ok", "note": ""},
        ]
    )

    result = upsert(existing, new)

    assert len(result) == 2
    assert set(result["metric"]) == {"basic_stats", "dempster_shafer"}
