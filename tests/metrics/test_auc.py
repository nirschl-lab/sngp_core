"""Tests for src/metrics/auc.py's fold-filtering behavior in AUROC_across_dataset.

Not a full behavioral suite for AUROC itself (untested before this change, and out of
scope here) -- these pin the one thing this change touches: the ID/OOD fold-filtering
asymmetry, now an explicit `fold_policy` parameter instead of a hardcoded line.
"""

import pandas as pd

from src.metrics import auc as auc_module
from src.metrics.auc import AUROC_across_dataset
from src.metrics.io import SYMMETRIC_FOLD_POLICY


def _write_legacy_csv(path, prob_scores, folds):
    pd.DataFrame(
        {
            "prediction_prob_score": prob_scores,
            "class_probs": [str([p, 1 - p]) for p in prob_scores],
            "fold": folds,
        }
    ).to_csv(path, index=False)


def _spy_on_compute_ood_score_series(monkeypatch):
    seen_lengths = []
    real_compute = auc_module._compute_ood_score_series

    def _spy(df, score_mode):
        seen_lengths.append(len(df))
        return real_compute(df, score_mode)

    monkeypatch.setattr(auc_module, "_compute_ood_score_series", _spy)
    return seen_lengths


def test_auroc_across_dataset_default_policy_filters_id_only_not_ood(tmp_path, monkeypatch):
    """Regression pin for the historical (previously undocumented) asymmetry:
    LEGACY_ISBI_FOLD_POLICY filters the ID frame to fold=='test' but leaves the OOD
    frame untouched -- load-bearing for every already-published csv/final/ and
    csv/isbi_test_files/ number, so this must stay the default."""
    _write_legacy_csv(tmp_path / "id.csv", [0.9] * 4, ["test"] * 4)
    _write_legacy_csv(tmp_path / "ood.csv", [0.5] * 3 + [0.1] * 3, ["test"] * 3 + ["train"] * 3)

    seen_lengths = _spy_on_compute_ood_score_series(monkeypatch)

    AUROC_across_dataset(str(tmp_path), {"id": "id.csv", "ood": "ood.csv"}, ["id"], ["ood"], score_mode="msp")

    # One call per frame: id filtered down to its 4 test rows, ood left at all 6.
    assert seen_lengths == [4, 6]


def test_auroc_across_dataset_symmetric_policy_also_filters_ood(tmp_path, monkeypatch):
    _write_legacy_csv(tmp_path / "id.csv", [0.9] * 4, ["test"] * 4)
    _write_legacy_csv(tmp_path / "ood.csv", [0.5] * 3 + [0.1] * 3, ["test"] * 3 + ["train"] * 3)

    seen_lengths = _spy_on_compute_ood_score_series(monkeypatch)

    AUROC_across_dataset(
        str(tmp_path),
        {"id": "id.csv", "ood": "ood.csv"},
        ["id"],
        ["ood"],
        score_mode="msp",
        fold_policy=SYMMETRIC_FOLD_POLICY,
    )

    # Both frames filtered to their 'test' rows.
    assert seen_lengths == [4, 3]
