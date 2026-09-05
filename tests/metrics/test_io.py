"""Tests for src/metrics/io.py -- the shared prediction-CSV loader.

Schema fixtures follow the real writers' conventions exactly: the inference schema
(`src/inference/records.py`) JSON-encodes list columns; the callback schema
(`src/callbacks/test_artifacts_callback.py`) writes them as Python-list `repr` (which
is what pandas produces when a DataFrame column holds real list objects and gets
`to_csv`'d), exercising the `ast.literal_eval` fallback.
"""

import json

import pandas as pd
import pytest

from src.metrics.io import (
    LEGACY_ISBI_FOLD_POLICY,
    SYMMETRIC_FOLD_POLICY,
    MissingPredictionData,
    canonicalize_stream,
    filter_fold,
    load_predictions,
    logits_array,
    member_logits_array,
    normalized_entropy,
    parse_float_list,
    parse_nested_float_list,
    probs_array,
    shannon_entropy_nats,
)
from tests.helpers.predictions import write_predictions_csv


def _write_inference_csv(path, *, n=4, num_classes=2, with_logits=False, with_member_logits=False, members=3):
    write_predictions_csv(
        path,
        n=n,
        num_classes=num_classes,
        with_logits=with_logits,
        with_member_logits=with_member_logits,
        members=members,
    )


def _write_callback_csv(path, *, n=4, num_classes=2):
    write_predictions_csv(path, n=n, num_classes=num_classes, schema="callback", seed=1)


# --- schema detection ---------------------------------------------------------------


def test_load_predictions_detects_inference_schema(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path)
    frame = load_predictions(path)
    assert frame.schema == "inference"


def test_load_predictions_detects_callback_schema(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_callback_csv(path)
    frame = load_predictions(path)
    assert frame.schema == "callback"


def test_load_predictions_rejects_a_file_with_both_confidence_columns(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "confidence": [0.9],
            "prediction_prob_score": [0.9],
            "class_probs": ["[0.9, 0.1]"],
            "target": [0],
            "prediction": [0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="both 'confidence' and 'prediction_prob_score'"):
        load_predictions(path)


def test_load_predictions_raises_missing_prediction_data_for_neither_schema(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"class_probs": ["[0.9, 0.1]"], "target": [0], "prediction": [0]}).to_csv(path, index=False)

    with pytest.raises(MissingPredictionData):
        load_predictions(path)


def test_missing_prediction_data_is_a_key_error_subclass():
    """So existing `except KeyError:` call sites keep catching it."""
    assert issubclass(MissingPredictionData, KeyError)


# --- derived columns ------------------------------------------------------------------


def test_load_predictions_derives_confidence_and_correct_for_inference_schema(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=6)
    frame = load_predictions(path)

    assert (frame.df["confidence"] == pd.read_csv(path)["confidence"]).all()
    assert (frame.df["correct"] == (frame.df["prediction"] == frame.df["target"])).all()


def test_load_predictions_derives_confidence_and_correct_for_callback_schema(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_callback_csv(path, n=6)
    frame = load_predictions(path)

    raw = pd.read_csv(path)
    assert (frame.df["confidence"] == raw["prediction_prob_score"]).all()
    assert (frame.df["correct"] == raw["true_bin_label"].astype(bool)).all()


def test_load_predictions_probs_and_entropy_columns(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=3, num_classes=3)
    frame = load_predictions(path)

    for probs, entropy_norm, entropy_nats in zip(
        frame.df["probs"], frame.df["entropy_norm"], frame.df["entropy_nats"]
    ):
        assert entropy_norm == pytest.approx(normalized_entropy(probs))
        assert entropy_nats == pytest.approx(shannon_entropy_nats(probs))


def test_load_predictions_defaults_stream_canonical_when_no_stream_column(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_callback_csv(path)  # callback schema has no `stream` column at all
    frame = load_predictions(path)
    assert set(frame.df["stream_canonical"]) == {"default"}
    assert frame.streams == ("default",)


def test_load_predictions_num_classes_matches_class_probs_length(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, num_classes=5)
    frame = load_predictions(path)
    assert frame.num_classes == 5


# --- capabilities ----------------------------------------------------------------------


def test_capabilities_include_logits_only_when_class_logits_present(tmp_path):
    without = tmp_path / "without" / "predictions.csv"
    with_logits = tmp_path / "with" / "predictions.csv"
    _write_inference_csv(without, with_logits=False)
    _write_inference_csv(with_logits, with_logits=True)

    assert "logits" not in load_predictions(without).capabilities
    assert "logits" in load_predictions(with_logits).capabilities


def test_capabilities_include_member_logits_only_when_present(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, with_member_logits=True, members=4)
    frame = load_predictions(path)
    assert "member_logits" in frame.capabilities
    assert member_logits_array(frame).shape == (4, 4, 2)  # n=4 rows, M=4, C=2


def test_capabilities_always_include_probs_and_targets(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path)
    frame = load_predictions(path)
    assert {"probs", "targets"}.issubset(frame.capabilities)


def test_capabilities_include_paired_streams_only_when_both_real_and_artifact_present(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=4)
    df = pd.read_csv(path)
    df["stream"] = ["real", "artifact", "real", "artifact"]
    df.to_csv(path, index=False)

    frame = load_predictions(path)
    assert "paired_streams" in frame.capabilities
    assert set(frame.streams) == {"real", "artifact"}


def test_require_raises_missing_prediction_data_naming_the_missing_capability(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, with_logits=False)

    with pytest.raises(MissingPredictionData, match="logits"):
        load_predictions(path, require=["logits"])


# --- array accessors ---------------------------------------------------------------


def test_probs_array_shape(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=5, num_classes=3)
    frame = load_predictions(path)
    assert probs_array(frame).shape == (5, 3)


def test_logits_array_raises_a_specific_error_when_absent(tmp_path):
    """Not a bare KeyError -- the 76 real runs on disk predate class_logits, and the
    error should say why (softmax is shift-invariant) and what to do (re-run)."""
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, with_logits=False)
    frame = load_predictions(path)

    with pytest.raises(MissingPredictionData, match="shift-invariant"):
        logits_array(frame)


def test_logits_array_returns_values_when_present(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=3, num_classes=2, with_logits=True)
    frame = load_predictions(path)
    assert logits_array(frame).shape == (3, 2)


def test_member_logits_array_raises_when_absent(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, with_member_logits=False)
    frame = load_predictions(path)

    with pytest.raises(MissingPredictionData, match="save_member_logits"):
        member_logits_array(frame)


# --- fold filtering ------------------------------------------------------------------


def test_load_predictions_fold_argument_filters_rows(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_inference_csv(path, n=4)
    df = pd.read_csv(path)
    df.loc[0, "fold"] = "train"
    df.to_csv(path, index=False)

    frame = load_predictions(path, fold="test")
    assert len(frame.df) == 3


def test_filter_fold_passthrough_when_fold_is_none():
    df = pd.DataFrame({"fold": ["test", "train"]})
    assert filter_fold(df, None, source="x").equals(df)


def test_filter_fold_raises_key_error_when_no_fold_column():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError, match="no 'fold' column"):
        filter_fold(df, "test", source="some/path.csv")


def test_filter_fold_raises_value_error_when_filter_empties_the_frame():
    df = pd.DataFrame({"fold": ["train", "train"]})
    with pytest.raises(ValueError, match="No rows with fold='test'"):
        filter_fold(df, "test", source="some/path.csv")


# --- parsing helpers -----------------------------------------------------------------


def test_parse_float_list_handles_json_and_repr_and_none():
    assert parse_float_list("[0.1, 0.2]") == [0.1, 0.2]
    assert parse_float_list([0.1, 0.2]) == [0.1, 0.2]
    assert parse_float_list("") is None
    assert parse_float_list(3.14) is None


def test_parse_nested_float_list_handles_json_and_none():
    assert parse_nested_float_list(json.dumps([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]
    assert parse_nested_float_list("not json") is None
    assert parse_nested_float_list(None) is None


def test_canonicalize_stream_maps_synonyms():
    assert canonicalize_stream("Real") == "real"
    assert canonicalize_stream("simulated") == "artifact"
    assert canonicalize_stream("something_else") == "something_else"


def test_fold_policy_constants_are_distinct():
    assert LEGACY_ISBI_FOLD_POLICY.ood_fold is None
    assert SYMMETRIC_FOLD_POLICY.ood_fold == SYMMETRIC_FOLD_POLICY.id_fold == "test"
