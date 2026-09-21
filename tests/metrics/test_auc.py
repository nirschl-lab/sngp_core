"""Tests for src/metrics/auc.py's fold-filtering behavior in AUROC_across_dataset.

Not a full behavioral suite for AUROC itself (untested before this change, and out of
scope here) -- these pin the one thing this change touches: the ID/OOD fold-filtering
asymmetry, now an explicit `fold_policy` parameter instead of a hardcoded line.
"""

import numpy as np
import pandas as pd
import pytest

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


class TestDempsterShaferScoreMode:
    """`score_mode="dempster_shafer"` -- added for the CIFAR-100 SNGP benchmark.

    `K / (K + sum_c exp(logit_c))` measures total evidence, which softmax destroys: it is
    shift-invariant, so MSP and entropy are blind to exactly the logit magnitude an SNGP
    head is trained to modulate. It is also the score the reference uses for CIFAR OOD
    (`dempster_shafer_ood` in google/uncertainty-baselines `baselines/cifar/sngp.py`).
    """

    def test_prefers_the_persisted_column(self):
        df = pd.DataFrame({"dempster_shafer": [0.1, 0.4, 0.7]})
        scores = auc_module._compute_ood_score_series(df, "dempster_shafer")
        assert list(scores) == [0.1, 0.4, 0.7]

    def test_falls_back_to_recomputing_from_logits(self):
        """Older CSVs predate the column but carry `class_logits`."""
        logits = [[2.0, 1.0, 0.5], [0.1, 0.1, 0.1]]
        df = pd.DataFrame({"class_logits": [str(row) for row in logits]})
        scores = auc_module._compute_ood_score_series(df, "dempster_shafer")

        from src.metrics.dempster_shafer_uncertainity import DempsterShaferUncertainty

        expected = DempsterShaferUncertainty(np.asarray(logits))
        assert np.allclose(scores.to_numpy(), expected)

    def test_probabilities_alone_are_rejected(self):
        """Refusing is the point: softmax has already discarded the evidence mass, so
        silently scoring off `class_probs` would return a confidently wrong number."""
        df = pd.DataFrame({"class_probs": ["[0.7, 0.3]"]})
        with pytest.raises(KeyError, match="dempster_shafer"):
            auc_module._compute_ood_score_series(df, "dempster_shafer")

    def test_counts_as_an_uncertainty_score(self):
        """Larger = more uncertain, so AUROC must not invert it the way it does MSP."""
        assert "dempster_shafer" in auc_module.UNCERTAINTY_SCORE_MODES
        assert "msp" not in auc_module.UNCERTAINTY_SCORE_MODES

    def test_end_to_end_auroc_separates_confident_id_from_diffuse_ood(self, tmp_path):
        """High-evidence ID logits vs low-evidence OOD logits must give AUROC ~1, not ~0
        -- the direction check that a sign error in `score_is_uncertainty` would flip."""
        rng = np.random.default_rng(0)
        for name, scale in (("id", 8.0), ("ood", 0.05)):
            logits = rng.normal(scale=scale, size=(1200, 5))
            pd.DataFrame({
                # `.tolist()`, not `list()`: numpy 2 renders np.float64 as
                # "np.float64(1.23)", which the CSV parser cannot read back.
                "class_logits": [str(row.tolist()) for row in logits],
                "fold": ["test"] * len(logits),
            }).to_csv(tmp_path / f"{name}.csv", index=False)

        res = AUROC_across_dataset(
            str(tmp_path), {"id": "id.csv", "ood": "ood.csv"}, ["id"], ["ood"],
            score_mode="dempster_shafer",
        )
        # Threshold is loose on purpose: the property under test is the *direction*
        # (a sign error in `score_is_uncertainty` lands near 0.06, not near 1), and
        # some Gaussian ID rows come out all-negative and so genuinely low-evidence.
        assert float(res["ood"].split("±")[0]) > 0.9
