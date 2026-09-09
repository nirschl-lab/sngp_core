"""Smoke test for src/metrics/artifact_quantification.py: import + a couple of calls on
tiny synthetic data, not full behavioral coverage.
"""
import json

import pandas as pd
import pytest

from src.metrics.artifact_quantification import quantify_artifact_impact
from tests.helpers.predictions import write_predictions_csv


def _write_paired_csv(path, n=20, num_classes=3, seed=0):
    real_df = write_predictions_csv(path.parent / f"{path.stem}_real_tmp.csv", n=n, num_classes=num_classes, seed=seed, stream="real")
    # High, mostly-correct confidences on the artifact stream so risk stays low enough at some
    # coverage level for `CovAt5Risk` to be well-defined (it's legitimately NaN, per the
    # library's own docs, when risk never drops to <=5% at any coverage -- not something to
    # special-case with an "all_correct" fixture that would make every new metric trivially 0).
    artifact_df = write_predictions_csv(
        path.parent / f"{path.stem}_artifact_tmp.csv",
        n=n,
        num_classes=num_classes,
        seed=seed + 1,
        stream="artifact",
        confidences=[0.9] * (n * 4 // 5) + [0.4] * (n - n * 4 // 5),
        all_correct=True,
    )
    combined = pd.concat([real_df, artifact_df], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined


NEW_COLUMNS = [
    "artifact_ece_plus",
    "artifact_ece_minus",
    "artifact_mce",
    "artifact_aece",
    "artifact_smece",
    "artifact_aurc",
    "artifact_augrc",
    "artifact_cov_5risk",
    "artifact_risk_80cov",
    "artifact_nll",
    "artifact_brier",
]


def test_quantify_artifact_impact_new_columns_without_metrics_json(tmp_path):
    csv_path = tmp_path / "baseline.csv"
    _write_paired_csv(csv_path, n=30, num_classes=3, seed=1)

    out_dir = tmp_path / "out"
    summary_df, _ = quantify_artifact_impact(
        model_csv_map={"Baseline": csv_path},
        output_dir=out_dir,
    )

    assert list(summary_df["model"]) == ["Baseline"]
    for col in NEW_COLUMNS:
        assert col in summary_df.columns, f"missing column {col}"
        value = summary_df.iloc[0][col]
        assert value == value  # not NaN

    assert 0.0 <= summary_df.iloc[0]["artifact_aurc"] <= 1.0
    assert 0.0 <= summary_df.iloc[0]["artifact_augrc"] <= 0.6
    assert summary_df.iloc[0]["artifact_nll"] >= 0.0
    assert summary_df.iloc[0]["artifact_brier"] >= 0.0


def test_quantify_artifact_impact_prefers_metrics_json_nll_brier(tmp_path):
    csv_path = tmp_path / "sngp.csv"
    _write_paired_csv(csv_path, n=20, num_classes=2, seed=2)

    metrics_json_path = tmp_path / "metrics.json"
    metrics_json_path.write_text(json.dumps({"artifact.nll": 1.2345, "artifact.brier": 0.6789}))

    out_dir = tmp_path / "out"
    summary_df, _ = quantify_artifact_impact(
        model_csv_map={"SNGP": csv_path},
        output_dir=out_dir,
        metrics_json_map={"SNGP": metrics_json_path},
    )

    assert summary_df.iloc[0]["artifact_nll"] == pytest.approx(1.2345)
    assert summary_df.iloc[0]["artifact_brier"] == pytest.approx(0.6789)
