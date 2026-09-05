"""Tests for src/metrics/run_manifest_metrics.py -- the manifest-driven batch driver.

A synthetic 2-run manifest over a tmp_path fake `infer/` tree covers the mechanics
(expected rows/statuses, the single-eval-dataset skip, a missing-logits skip, a
missing OOD file being dropped rather than blowing up the whole run). A separate,
env-gated test reproduces the real csv/ood_metrics/*.csv numbers exactly off the real
checked-in manifest and the real tree -- the strongest, still GPU-free, signal that
the whole pipeline (manifest -> io.py -> registry) agrees with what's already
published.
"""

import os

import pandas as pd
import pytest

from src.metrics.manifest import Manifest, RunEntry
from src.metrics.run_manifest_metrics import run_manifest_metrics, run_metrics_for_entry
from tests.helpers.predictions import write_predictions_csv

_HAS_REAL_ENV = bool(os.environ.get("EXPERIMENTS_HOME")) and bool(os.environ.get("PROJECT_NAME"))


def _make_run(
    label,
    *,
    id_dataset="a",
    eval_datasets=("a", "b"),
    method="baseline",
    train_dataset="a",
    is_ensemble=False,
    uses_mc_dropout=False,
    run_dir=None,
    paired_streams=(),
) -> RunEntry:
    return RunEntry(
        label=label,
        method=method,
        train_dataset=train_dataset,
        is_ensemble=is_ensemble,
        uses_mc_dropout=uses_mc_dropout,
        run_dir=run_dir or label,
        ckpt=f"train/{label}/runs/2026-01-01_00-00-00/checkpoints/best.ckpt",
        id_dataset=id_dataset,
        eval_datasets=tuple(eval_datasets),
        paired_streams=tuple(paired_streams),
    )


def _populate_two_run_tree(infer_root, *, with_logits_on_first=False):
    """A two-entry manifest: `multi_dataset` has an ID + one OOD dataset (so
    `needs_ood` metrics run); `single_dataset` has only its own ID dataset (so they
    skip). Matches the shape of a real acevedo-style entry vs a wong_ucdavis one."""
    multi = _make_run("multi_dataset", id_dataset="a", eval_datasets=("a", "b"))
    single = _make_run("single_dataset", id_dataset="c", eval_datasets=("c",))
    manifest = Manifest(schema_version=1, infer_root=infer_root, runs=(multi, single))

    write_predictions_csv(
        infer_root / multi.run_dir / "a" / "predictions.csv",
        n=20,
        num_classes=2,
        confidences=[0.95] * 20,
        seed=0,
        with_logits=with_logits_on_first,
    )
    write_predictions_csv(
        infer_root / multi.run_dir / "b" / "predictions.csv", n=20, num_classes=2, confidences=[0.5] * 20, seed=1
    )
    write_predictions_csv(
        infer_root / single.run_dir / "c" / "predictions.csv", n=10, num_classes=2, confidences=[0.9] * 10, seed=2
    )
    return manifest, multi, single


# --- run_metrics_for_entry / run_manifest_metrics: mechanics ------------------------


def test_run_manifest_metrics_produces_a_row_per_metric_and_scope(tmp_path):
    manifest, multi, single = _populate_two_run_tree(tmp_path / "infer")

    df = run_manifest_metrics(manifest, metrics=["basic_stats"])

    multi_rows = df[df["label"] == "multi_dataset"]
    assert set(multi_rows["metric"]) == {"accuracy", "mean_confidence", "mean_entropy"}
    assert (multi_rows["status"] == "ok").all()
    assert (multi_rows["scope"] == "run").all()


def test_needs_ood_metric_skips_for_a_single_eval_dataset_run(tmp_path):
    manifest, multi, single = _populate_two_run_tree(tmp_path / "infer")

    df = run_manifest_metrics(manifest, runs=["single_dataset"], metrics=["ood_auroc_msp"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "skipped"
    assert "no OOD datasets" in row["note"]


def test_needs_ood_metric_runs_for_a_multi_eval_dataset_run(tmp_path):
    manifest, multi, single = _populate_two_run_tree(tmp_path / "infer")

    df = run_manifest_metrics(manifest, runs=["multi_dataset"], metrics=["ood_auroc_msp"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "ok"
    assert row["scope"] == "b"
    assert "±" in row["value_str"]


def test_requires_metric_skips_when_capability_is_missing(tmp_path):
    manifest, multi, single = _populate_two_run_tree(tmp_path / "infer", with_logits_on_first=False)

    df = run_manifest_metrics(manifest, runs=["multi_dataset"], metrics=["dempster_shafer"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "skipped"
    assert "logits" in row["note"]


def test_requires_metric_runs_when_capability_is_present(tmp_path):
    manifest, multi, single = _populate_two_run_tree(tmp_path / "infer", with_logits_on_first=True)

    df = run_manifest_metrics(manifest, runs=["multi_dataset"], metrics=["dempster_shafer"])

    assert len(df) == 1
    assert df.iloc[0]["status"] == "ok"


def test_missing_id_frame_yields_an_error_row_per_requested_metric(tmp_path):
    infer_root = tmp_path / "infer"
    run = _make_run("no_data", id_dataset="a", eval_datasets=("a",))
    manifest = Manifest(schema_version=1, infer_root=infer_root, runs=(run,))
    # Deliberately don't write anything under infer_root.

    df = run_manifest_metrics(manifest, metrics=["basic_stats", "dempster_shafer"])

    assert len(df) == 2
    assert (df["status"] == "error").all()
    assert (df["metric"].isin(["basic_stats", "dempster_shafer"])).all()


def test_a_missing_ood_dataset_is_dropped_not_fatal(tmp_path, caplog):
    """One bad OOD file shouldn't block metrics for the other OOD datasets or for
    non-OOD metrics on the same run."""
    infer_root = tmp_path / "infer"
    run = _make_run("partial", id_dataset="a", eval_datasets=("a", "b", "missing"))
    manifest = Manifest(schema_version=1, infer_root=infer_root, runs=(run,))

    write_predictions_csv(infer_root / run.run_dir / "a" / "predictions.csv", n=10, num_classes=2, seed=0)
    write_predictions_csv(infer_root / run.run_dir / "b" / "predictions.csv", n=10, num_classes=2, seed=1)
    # No predictions.csv written for "missing".

    df = run_manifest_metrics(manifest, metrics=["ood_auroc_msp", "basic_stats"])

    ood_rows = df[df["metric"] == "ood_auroc_msp"]
    assert set(ood_rows["scope"]) == {"b"}  # "missing" silently excluded
    assert (df[df["metric"] == "basic_stats"]["status"] == "ok").all()


def test_runs_filter_selects_only_named_labels(tmp_path):
    manifest, multi, single = _populate_two_run_tree(tmp_path / "infer")
    df = run_manifest_metrics(manifest, runs=["single_dataset"], metrics=["basic_stats"])
    assert set(df["label"]) == {"single_dataset"}


def test_default_metrics_is_every_registered_metric(tmp_path):
    manifest, multi, single = _populate_two_run_tree(tmp_path / "infer")
    df = run_manifest_metrics(manifest, runs=["multi_dataset"])
    assert {"accuracy", "mean_confidence", "mean_entropy", "dempster_shafer", "ood_auroc_msp", "ood_auroc_entropy"} <= set(
        df["metric"]
    )


# --- the real, checked-in manifest against the real tree ----------------------------


@pytest.mark.skipif(not _HAS_REAL_ENV, reason="requires EXPERIMENTS_HOME/PROJECT_NAME and the real infer/ tree")
def test_reproduces_published_ood_metrics_csvs_exactly():
    """The strongest available regression signal: every value_str this driver
    produces for a label with an existing csv/ood_metrics/<label>.csv must equal the
    published string exactly -- proving load_predictions + the registry adapters
    reproduce calculate_ood_metrics.py's own numbers, not just something similar."""
    from pathlib import Path

    from src.metrics.manifest import DEFAULT_MANIFEST_PATH, load_manifest, resolve_default_infer_root

    published_dir = Path(__file__).resolve().parents[2] / "csv" / "ood_metrics"
    published_labels = sorted(p.stem for p in published_dir.glob("*.csv"))
    assert published_labels, "expected at least one published csv/ood_metrics/*.csv to compare against"

    manifest = load_manifest(DEFAULT_MANIFEST_PATH, infer_root=resolve_default_infer_root())
    df = run_manifest_metrics(
        manifest, runs=published_labels, metrics=["ood_auroc_msp", "ood_auroc_entropy"], sample_rate=1000
    )

    for label in published_labels:
        published = pd.read_csv(published_dir / f"{label}.csv")
        for _, prow in published.iterrows():
            for metric, col in (("ood_auroc_msp", "msp_auroc"), ("ood_auroc_entropy", "entropy_auroc")):
                match = df[(df["label"] == label) & (df["metric"] == metric) & (df["scope"] == prow["dataset"])]
                assert not match.empty, f"missing {label}/{metric}/{prow['dataset']}"
                assert match.iloc[0]["value_str"] == prow[col], f"mismatch at {label}/{metric}/{prow['dataset']}"
