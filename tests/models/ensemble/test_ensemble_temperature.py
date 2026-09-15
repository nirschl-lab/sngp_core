"""`DeepEnsemble.temperature`: an ensemble-level post-hoc knob on the pooled logits.
Members keep their own (default 1.0) temperature; `member_logits`/`variance` stay in raw
member-logit units."""
import json

import pytest
import torch

from src.models.ensemble.deep_ensemble_model import DeepEnsemble
from src.models.registry import build_net

NUM_CLASSES = 3


def _spec() -> dict:
    # dropout_p=0 so train-mode member forwards are deterministic for the same weights.
    return {"name": "baseline_classifier", "arch": "resnet18", "num_classes": NUM_CLASSES, "pretrained": False, "dropout_p": 0.0}


def _pair(temperature: float):
    torch.manual_seed(0)
    reference = DeepEnsemble(_spec(), num_estimators=2, temperature=1.0)
    torch.manual_seed(0)
    scaled = DeepEnsemble(_spec(), num_estimators=2, temperature=temperature)
    scaled.load_state_dict(reference.state_dict(), strict=True)
    return reference, scaled


class TestDeepEnsembleTemperature:
    def test_default_is_one_and_lives_at_ensemble_level(self):
        ensemble = DeepEnsemble(_spec(), num_estimators=2)
        assert ensemble.temperature == 1.0
        assert ensemble.spec["temperature"] == 1.0
        assert "temperature" not in ensemble.spec["base_model_spec"]
        assert all(m.temperature == 1.0 for m in ensemble.ensemble_members)

    def test_eval_pooled_logits_scaled_members_unscaled(self):
        reference, scaled = _pair(2.0)
        reference.eval()
        scaled.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            ref_out, scaled_out = reference(x), scaled(x)
        assert torch.allclose(scaled_out.logits, ref_out.logits / 2.0, atol=1e-6)
        assert torch.equal(scaled_out.member_logits, ref_out.member_logits)
        assert torch.equal(scaled_out.variance, ref_out.variance)
        assert torch.equal(scaled_out.logits.argmax(1), ref_out.logits.argmax(1))

    def test_train_branch_scaled_too(self):
        reference, scaled = _pair(2.0)
        for net in (reference, scaled):
            net.train()
            net.set_active_member(0)
        x = torch.randn(2, 3, 224, 224)
        assert torch.allclose(scaled(x).logits, reference(x).logits / 2.0, atol=1e-5)

    @pytest.mark.parametrize("bad", [0.0, -2.0])
    def test_invalid_temperature_raises(self, bad):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            DeepEnsemble(_spec(), num_estimators=2, temperature=bad)

    def test_spec_round_trips_and_is_json(self):
        ensemble = DeepEnsemble(_spec(), num_estimators=2, temperature=1.5)
        rebuilt = build_net(ensemble.spec)
        assert isinstance(rebuilt, DeepEnsemble)
        assert rebuilt.temperature == 1.5
        assert rebuilt.spec == ensemble.spec
        assert json.loads(json.dumps(ensemble.spec)) == ensemble.spec

    def test_spec_without_temperature_key_rebuilds_at_one(self):
        spec = DeepEnsemble(_spec(), num_estimators=2).spec
        del spec["temperature"]
        assert build_net(spec).temperature == 1.0

    def test_temperature_is_not_a_buffer(self):
        assert not any("temperature" in k for k in DeepEnsemble(_spec(), num_estimators=2, temperature=2.0).state_dict())
