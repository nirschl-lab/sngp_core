"""Tests for src/metrics/auc.py's fold-filtering behavior in AUROC_across_dataset.

Not a full behavioral suite for AUROC itself (untested before this change, and out of
scope here) -- these pin the one thing this change touches: the ID/OOD fold-filtering
asymmetry, now an explicit `fold_policy` parameter instead of a hardcoded line.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.metrics import auc as auc_module
from src.metrics.auc import AUROC_across_dataset, AUROC_across_dataset_full_population
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


class TestMspScoreColumns:
    """MSP lives under two names -- `prediction_prob_score` (callback schema) and
    `confidence` (inference schema) -- for the same quantity. Before the fallback, the
    inference schema simply raised, so every CIFAR-100 prediction CSV was unscoreable
    under `score_mode="msp"`."""

    def test_prefers_prediction_prob_score_over_confidence(self):
        """The one that protects the frozen path: a published call site must keep
        resolving to the column it always resolved to, whatever else the frame carries."""
        df = pd.DataFrame({"prediction_prob_score": [0.9, 0.8], "confidence": [0.1, 0.2]})
        scores = auc_module._compute_ood_score_series(df, "msp")
        assert list(scores) == [0.9, 0.8]

    def test_falls_back_to_confidence_for_the_inference_schema(self):
        df = pd.DataFrame({"confidence": [0.9, 0.8]})
        scores = auc_module._compute_ood_score_series(df, "msp")
        assert list(scores) == [0.9, 0.8]

    def test_raises_naming_both_columns_when_neither_is_present(self):
        df = pd.DataFrame({"class_probs": ["[0.7, 0.3]"]})
        with pytest.raises(KeyError, match="confidence"):
            auc_module._compute_ood_score_series(df, "msp")

    def test_drops_non_numeric_rows(self):
        """Parity with the pre-fallback `to_numeric(errors='coerce').dropna()`."""
        df = pd.DataFrame({"confidence": ["0.9", "not-a-number", "0.5"]})
        scores = auc_module._compute_ood_score_series(df, "msp")
        assert list(scores) == [0.9, 0.5]


class TestFullPopulationAuroc:
    """`AUROC_across_dataset_full_population` -- the SNGP paper's protocol (arXiv
    2205.00403 section 6.2.1 / appendix C.1): the whole ID test set against the whole
    OOD test set, no subsampling, one deterministic number."""

    @staticmethod
    def _write(tmp_path, name, confidences, folds=None):
        folds = folds if folds is not None else ["test"] * len(confidences)
        pd.DataFrame({"confidence": confidences, "fold": folds}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )

    @staticmethod
    def _files():
        return {"id": "id.csv", "ood": "ood.csv"}

    def _fixture(self, tmp_path, n_id=3000, n_ood=2000):
        """Deliberately n_id != n_ood, and both above `sample_rate`: unequal groups are
        the real CIFAR-100 (10,000) vs SVHN (26,032) case, and staying above 1000 is
        what makes the no-subsampling assertions meaningful."""
        rng = np.random.default_rng(0)
        id_conf = rng.uniform(0.5, 1.0, size=n_id)
        ood_conf = rng.uniform(0.0, 0.6, size=n_ood)
        self._write(tmp_path, "id", id_conf)
        self._write(tmp_path, "ood", ood_conf)
        return id_conf, ood_conf

    def test_matches_sklearn_over_every_row(self, tmp_path):
        """The actual specification: one `roc_auc_score` over both full arrays."""
        id_conf, ood_conf = self._fixture(tmp_path)
        res = AUROC_across_dataset_full_population(
            str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
        )
        expected = roc_auc_score(
            np.concatenate([np.zeros(len(id_conf)), np.ones(len(ood_conf))]),
            np.concatenate([1 - id_conf, 1 - ood_conf]),
        )
        assert res["ood"] == pytest.approx(expected)

    def test_ignores_the_module_sample_rate(self, tmp_path, monkeypatch):
        """Pins "no subsampling" in a way a refactor cannot fake: drop `sample_rate` to
        5 and the answer must not move by even a float ulp."""
        self._fixture(tmp_path)
        before = AUROC_across_dataset_full_population(
            str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
        )
        monkeypatch.setattr(auc_module, "sample_rate", 5)
        after = AUROC_across_dataset_full_population(
            str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
        )
        assert before["ood"] == after["ood"]

    def test_is_deterministic_and_returns_floats(self, tmp_path):
        self._fixture(tmp_path)
        calls = [
            AUROC_across_dataset_full_population(
                str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
            )["ood"]
            for _ in range(2)
        ]
        assert calls[0] == calls[1]
        assert isinstance(calls[0], float)

    def test_msp_direction(self, tmp_path):
        """Confident ID vs diffuse OOD must give AUROC near 1, not near 0. The existing
        end-to-end test only covers the uncertainty branch, so a sign flip on the
        *non*-uncertainty branch had nothing catching it."""
        self._write(tmp_path, "id", [0.99] * 500)
        self._write(tmp_path, "ood", [0.30] * 500)
        res = AUROC_across_dataset_full_population(
            str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
        )
        assert res["ood"] > 0.9

    def test_dempster_shafer_direction(self, tmp_path):
        rng = np.random.default_rng(0)
        for name, scale in (("id", 8.0), ("ood", 0.05)):
            logits = rng.normal(scale=scale, size=(1200, 5))
            pd.DataFrame({
                "class_logits": [str(row.tolist()) for row in logits],
                "fold": ["test"] * len(logits),
            }).to_csv(tmp_path / f"{name}.csv", index=False)
        res = AUROC_across_dataset_full_population(
            str(tmp_path), self._files(), "id", ["ood"], score_mode="dempster_shafer"
        )
        assert res["ood"] > 0.9

    def test_default_fold_policy_is_symmetric(self, tmp_path, monkeypatch):
        """Deliberately the opposite default from `AUROC_across_dataset` (see the
        `..._filters_id_only_not_ood` test above): test-vs-test is what the protocol
        says, and no published number rides on this function."""
        self._write(tmp_path, "id", [0.9] * 4, ["test"] * 4)
        self._write(tmp_path, "ood", [0.5] * 3 + [0.1] * 3, ["test"] * 3 + ["train"] * 3)

        seen_lengths = _spy_on_compute_ood_score_series(monkeypatch)
        AUROC_across_dataset_full_population(
            str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
        )
        assert seen_lengths == [4, 3]

    def test_reads_the_id_frame_once_per_call(self, tmp_path, monkeypatch):
        self._write(tmp_path, "id", [0.9] * 4)
        self._write(tmp_path, "ood", [0.5] * 4)
        self._write(tmp_path, "ood2", [0.4] * 4)

        seen_lengths = _spy_on_compute_ood_score_series(monkeypatch)
        AUROC_across_dataset_full_population(
            str(tmp_path),
            {"id": "id.csv", "ood": "ood.csv", "ood2": "ood2.csv"},
            "id",
            ["ood", "ood2"],
            score_mode="msp",
        )
        # 1 ID read + 1 per OOD dataset, not 1 ID read per OOD dataset.
        assert len(seen_lengths) == 3

    def test_raises_naming_both_datasets_when_a_frame_is_empty(self, tmp_path):
        self._write(tmp_path, "id", [0.9] * 4, ["train"] * 4)
        self._write(tmp_path, "ood", [0.5] * 4)
        with pytest.raises(ValueError, match="ID='id'.*OOD='ood'"):
            AUROC_across_dataset_full_population(
                str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
            )

    def test_the_frozen_path_still_returns_mean_plus_minus_std(self, tmp_path):
        """Regression pin for the `_read_ood_score_series` extraction: the two
        estimators must stay distinguishable by return type, and are not expected to
        agree on a value."""
        self._fixture(tmp_path)
        legacy = AUROC_across_dataset(
            str(tmp_path), self._files(), ["id"], ["ood"], score_mode="msp"
        )
        full = AUROC_across_dataset_full_population(
            str(tmp_path), self._files(), "id", ["ood"], score_mode="msp"
        )
        assert isinstance(legacy["ood"], str) and "±" in legacy["ood"]
        assert isinstance(full["ood"], float)
