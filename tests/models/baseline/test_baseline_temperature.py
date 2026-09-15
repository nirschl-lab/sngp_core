"""`BaselineClassifier.temperature`: the family's single post-hoc calibration knob.
Never trained -- fit afterwards on validation NLL and written into the checkpoint's
`net_spec` -- so the contract here is purely about how it is applied and persisted."""
import json

import pytest
import torch

from src.models.baseline.baseline_models import BaselineClassifier
from src.models.registry import build_net

NUM_CLASSES = 4


def _net(temperature: float = 1.0) -> BaselineClassifier:
    torch.manual_seed(0)
    return BaselineClassifier(arch="resnet18", num_classes=NUM_CLASSES, pretrained=False, dropout_p=0.2, temperature=temperature)


def _pair(temperature: float):
    """Two nets with identical weights, differing only in temperature."""
    reference = _net(1.0)
    scaled = _net(temperature)
    scaled.load_state_dict(reference.state_dict(), strict=True)
    return reference, scaled


class TestBaselineTemperature:
    def test_default_is_one_and_in_spec(self):
        net = _net()
        assert net.temperature == 1.0
        assert net.spec["temperature"] == 1.0

    def test_identity_at_default_is_bit_exact(self):
        net = _net()
        net.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = net(x).logits
            expected = net.classifier(net.feature_extractor(x))
        assert torch.equal(logits, expected)

    def test_eval_logits_scale_by_temperature(self):
        reference, scaled = _pair(2.0)
        reference.eval()
        scaled.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            assert torch.allclose(scaled(x).logits, reference(x).logits / 2.0, atol=1e-6)

    def test_argmax_is_invariant(self):
        reference, scaled = _pair(3.0)
        reference.eval()
        scaled.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            assert torch.equal(scaled(x).logits.argmax(1), reference(x).logits.argmax(1))

    def test_mc_dropout_passes_inherit_temperature(self):
        """`mc_forward_samples` runs `forward` in train mode with dropout on; the same seed
        gives the same masks, so the only difference must be the temperature."""
        reference, scaled = _pair(2.0)
        x = torch.randn(2, 3, 224, 224)
        torch.manual_seed(7)
        ref_stack = reference.mc_forward_samples(x, T=3)
        torch.manual_seed(7)
        scaled_stack = scaled.mc_forward_samples(x, T=3)
        assert ref_stack.shape == (3, 2, NUM_CLASSES)
        assert torch.allclose(scaled_stack, ref_stack / 2.0, atol=1e-6)

    def test_raw_logits_stays_none(self):
        """`raw_logits` is the SNGP marker downstream (test_artifacts_callback, records.py);
        the baseline must not start reporting it."""
        net = _net(2.0)
        net.eval()
        assert net(torch.randn(1, 3, 224, 224)).raw_logits is None

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_invalid_temperature_raises(self, bad):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            BaselineClassifier(arch="resnet18", num_classes=NUM_CLASSES, pretrained=False, temperature=bad)

    def test_spec_round_trips_and_is_json(self):
        net = _net(1.7)
        rebuilt = build_net(net.spec)
        assert isinstance(rebuilt, BaselineClassifier)
        assert rebuilt.temperature == 1.7
        assert rebuilt.spec == net.spec
        assert json.loads(json.dumps(net.spec)) == net.spec

    def test_spec_without_temperature_key_rebuilds_at_one(self):
        """Back-compat: checkpoints written before the key existed."""
        spec = _net().spec
        del spec["temperature"]
        assert build_net(spec).temperature == 1.0

    def test_temperature_is_not_a_buffer(self):
        """A buffer would break strict=True loading of every existing checkpoint."""
        assert not any("temperature" in k for k in _net(2.0).state_dict())
