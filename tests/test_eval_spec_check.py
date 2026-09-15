"""`src/eval.py::_assert_model_cfg_matches_checkpoint` -- config-authoritative eval must
still accept checkpoints written before an *additive* spec key existed (`temperature`,
`spectral_norm_bound`), while refusing real architecture mismatches."""
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import src.eval as eval_module
from src.checkpointing.spec import FORMAT_VERSION, CheckpointMeta

_BASE_SPEC = {"name": "baseline_classifier", "arch": "resnet18", "num_classes": 4, "dropout_p": 0.2, "pretrained": False}


def _meta(net_spec: dict) -> CheckpointMeta:
    return CheckpointMeta(format_version=FORMAT_VERSION, lit_module="x.Y", net_spec=net_spec, num_classes=4)


def _model(spec: dict):
    return SimpleNamespace(net=SimpleNamespace(spec=spec))


def _cfg():
    return OmegaConf.create({"ckpt_path": "/nonexistent/best.ckpt"})


def _patch(monkeypatch, ckpt_spec: dict) -> None:
    monkeypatch.setattr(eval_module, "read_meta", lambda _path: _meta(ckpt_spec))


def test_additive_key_missing_from_checkpoint_is_tolerated(monkeypatch):
    _patch(monkeypatch, dict(_BASE_SPEC))
    eval_module._assert_model_cfg_matches_checkpoint(_cfg(), _model({**_BASE_SPEC, "temperature": 1.0}))


def test_pretrained_disagreement_is_ignored(monkeypatch):
    _patch(monkeypatch, dict(_BASE_SPEC))
    eval_module._assert_model_cfg_matches_checkpoint(_cfg(), _model({**_BASE_SPEC, "pretrained": True}))


def test_real_mismatch_still_raises(monkeypatch):
    _patch(monkeypatch, {**_BASE_SPEC, "dropout_p": 0.5})
    with pytest.raises(ValueError, match="dropout_p"):
        eval_module._assert_model_cfg_matches_checkpoint(_cfg(), _model(dict(_BASE_SPEC)))


def test_key_recorded_in_checkpoint_but_unknown_to_config_raises(monkeypatch):
    _patch(monkeypatch, {**_BASE_SPEC, "temperature": 1.7})
    with pytest.raises(ValueError, match="temperature") as excinfo:
        eval_module._assert_model_cfg_matches_checkpoint(_cfg(), _model(dict(_BASE_SPEC)))
    assert "calibrated" in str(excinfo.value)


def test_calibrated_checkpoint_value_mismatch_mentions_the_knob(monkeypatch):
    _patch(monkeypatch, {**_BASE_SPEC, "temperature": 1.7})
    with pytest.raises(ValueError, match="model.net.temperature"):
        eval_module._assert_model_cfg_matches_checkpoint(_cfg(), _model({**_BASE_SPEC, "temperature": 1.0}))


def test_matching_calibrated_value_passes(monkeypatch):
    _patch(monkeypatch, {**_BASE_SPEC, "temperature": 1.7})
    eval_module._assert_model_cfg_matches_checkpoint(_cfg(), _model({**_BASE_SPEC, "temperature": 1.7}))
