import pytest
import torch

from src.models.outputs import ModelOutput
from src.models.sngp.sngp_classifier import SNGPClassifier


class TestSNGPClassifier:
    @pytest.mark.parametrize("arch", ["resnet18", "resnet34", "resnet50"])
    def test_resnet_architectures(self, arch):
        model = SNGPClassifier(num_classes=10, arch=arch, pretrained=False, rff_dim=128)
        model.eval()
        output = model(torch.randn(2, 3, 224, 224))
        assert isinstance(output, ModelOutput)
        assert output.logits.shape == (2, 10)
        assert output.raw_logits.shape == (2, 10)
        assert output.variance.shape == (2, 1)

    @pytest.mark.parametrize("arch", ["vit_b_16", "vit_b_32", "vit_l_16"])
    def test_vit_architectures_unsupported(self, arch):
        """SNGP's spectral-norm wrapping is not supported for ViT backbones."""
        with pytest.raises(ValueError, match="not supported"):
            SNGPClassifier(num_classes=10, arch=arch, pretrained=False, rff_dim=128)

    def test_unsupported_architecture(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            SNGPClassifier(num_classes=10, arch="unsupported_arch")

    def test_pretrained_model(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", pretrained=True, rff_dim=128)
        assert model is not None

    def test_spectral_norm_applied(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", pretrained=False, rff_dim=128)
        assert any(hasattr(m, "weight_u") for m in model.backbone.modules())

    def test_different_input_sizes(self):
        model = SNGPClassifier(num_classes=5, arch="resnet18", rff_dim=64)
        model.eval()
        for batch_size in [1, 4, 8]:
            output = model(torch.randn(batch_size, 3, 224, 224))
            assert output.logits.shape == (batch_size, 5)
            assert output.raw_logits.shape == (batch_size, 5)
            assert output.variance.shape == (batch_size, 1)

    @pytest.mark.parametrize("num_classes", [1, 5, 100, 1000])
    def test_different_num_classes(self, num_classes):
        model = SNGPClassifier(num_classes=num_classes, arch="resnet18", rff_dim=64)
        model.eval()
        output = model(torch.randn(2, 3, 224, 224))
        assert output.logits.shape == (2, num_classes)
        assert output.raw_logits.shape == (2, num_classes)
        assert output.variance.shape == (2, 1)

    def test_train_mode_returns_raw_logits_and_no_variance(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        model.train()
        output = model(torch.randn(2, 3, 224, 224))
        assert output.variance is None
        assert torch.equal(output.logits, output.raw_logits)

    def test_update_precision_parameter(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        x = torch.randn(2, 3, 224, 224)
        model.train()

        model(x, update_precision=True)
        after_update = model.gp_head.precision_accum.clone()
        assert torch.count_nonzero(after_update) > 0

        model(x, update_precision=False)
        assert torch.equal(model.gp_head.precision_accum, after_update)

    def test_eval_mode_freezes_precision(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        model.eval()
        before = model.gp_head.precision_accum.clone()
        model(torch.randn(2, 3, 224, 224))
        assert torch.equal(model.gp_head.precision_accum, before)

    def test_reset_precision_delegates_to_head(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        model.train()
        model(torch.randn(2, 3, 224, 224))
        assert torch.count_nonzero(model.gp_head.precision_accum) > 0
        model.reset_precision()
        assert torch.count_nonzero(model.gp_head.precision_accum) == 0

    def test_grad_flow(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        model(x).logits.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_spec_round_trips_through_registry(self):
        """Every ctor kwarg must survive the round trip -- the old version of this test
        checked only `arch` and `num_classes`, so a dropped spec key went unnoticed."""
        from src.models.registry import build_net

        model = SNGPClassifier(
            num_classes=6,
            arch="resnet18",
            pretrained=False,
            rff_dim=64,
            length_scale=2.5,
            ridge_penalty=1e-2,
            cov_momentum=0.95,
            mean_field=False,
            n_power_iterations_sn=3,
            mean_field_factor=0.5,
            normalize_input=False,
            likelihood="binary_logistic",
            output_bias=True,
        )
        rebuilt = build_net(model.spec)
        assert isinstance(rebuilt, SNGPClassifier)
        assert rebuilt.spec == model.spec

    def test_spec_is_json_serializable(self):
        import json

        model = SNGPClassifier(num_classes=6, arch="resnet18", pretrained=False, rff_dim=64)
        assert json.loads(json.dumps(model.spec)) == model.spec

    def test_output_bias_default_matches_reference(self):
        """The reference uses a fixed zero GP output bias, i.e. no trainable one."""
        model = SNGPClassifier(num_classes=6, arch="resnet18", rff_dim=64)
        assert model.gp_head.classifier.bias is None
        with_bias = SNGPClassifier(num_classes=6, arch="resnet18", rff_dim=64, output_bias=True)
        assert with_bias.gp_head.classifier.bias is not None
