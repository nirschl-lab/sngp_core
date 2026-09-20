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
            spectral_norm_bound=2.0,
        )
        rebuilt = build_net(model.spec)
        assert isinstance(rebuilt, SNGPClassifier)
        assert rebuilt.spec == model.spec
        assert rebuilt.spec["spectral_norm_bound"] == 2.0

    def test_ctor_defaults_are_the_paper_constants(self):
        """A bare SNGPClassifier is the Liu et al. 2022 Table 9 model -- except the
        spectral-norm bound, whose ctor default stays None (stock hard normalization) so
        checkpoints written before the key existed rebuild identically."""
        model = SNGPClassifier(num_classes=4, rff_dim=64)
        assert model.rff_dim == 64  # explicit here; the default is 1024
        assert SNGPClassifier.__init__.__defaults__ is not None
        assert model.length_scale == pytest.approx(1.4142)
        assert model.cov_momentum == -1.0
        assert model.random_feature_type == "orf"
        assert model.n_power_iterations_sn == 1
        assert model.spectral_norm_bound is None
        assert model.use_spectral_norm is True

    def test_spec_without_bound_key_rebuilds_with_stock_normalization(self):
        """Back-compat: a pre-bound checkpoint's spec lacks `spectral_norm_bound`."""
        from torch.nn.utils.spectral_norm import SpectralNorm

        from src.models.registry import build_net

        spec = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64).spec
        del spec["spectral_norm_bound"]
        rebuilt = build_net(spec)
        assert rebuilt.spectral_norm_bound is None
        hooks = [h for m in rebuilt.backbone.modules() for h in m._forward_pre_hooks.values()]
        assert hooks and all(type(h) is SpectralNorm for h in hooks)

    def test_bounded_classifier_layers_respect_bound(self):
        from src.models.components.spectral_norm import BoundedSpectralNorm

        torch.manual_seed(0)
        bound = 1.0
        model = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64, spectral_norm_bound=bound)
        wrapped = [m for m in model.backbone.modules() if hasattr(m, "weight_u")]
        assert wrapped
        for m in wrapped:
            hook = next(iter(m._forward_pre_hooks.values()))
            assert isinstance(hook, BoundedSpectralNorm) and hook.bound == bound
            sigma = torch.linalg.matrix_norm(m.weight.reshape(m.weight.shape[0], -1), ord=2).item()
            assert sigma <= bound + 0.05, f"{m}: sigma={sigma}"
        # An integer override (as a W&B sweep may emit) is stored as a float in the spec.
        assert isinstance(SNGPClassifier(num_classes=4, rff_dim=64, spectral_norm_bound=2).spec["spectral_norm_bound"], float)

    def test_bounded_model_forward_is_finite(self):
        model = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64, spectral_norm_bound=2.0)
        model.eval()
        out = model(torch.randn(2, 3, 224, 224))
        assert torch.isfinite(out.logits).all()
        assert torch.isfinite(out.variance).all()

    def test_spec_is_json_serializable(self):
        import json

        model = SNGPClassifier(num_classes=6, arch="resnet18", pretrained=False, rff_dim=64)
        assert json.loads(json.dumps(model.spec)) == model.spec

    def test_use_spectral_norm_false_builds_a_plain_backbone(self):
        """The spectral-regularization variant: same backbone + GP head, no hooks, no
        `weight_orig`/`weight_u`/`weight_v` -- plain `weight` keys in the state dict."""
        from src.models.registry import build_net

        model = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64, use_spectral_norm=False)
        assert model.use_spectral_norm is False
        assert not any(hasattr(m, "weight_u") for m in model.backbone.modules())
        assert not any(m._forward_pre_hooks for m in model.backbone.modules())
        assert not any(k.endswith(("weight_orig", "weight_u", "weight_v")) for k in model.state_dict())
        assert any(k.endswith("conv1.weight") for k in model.state_dict())

        model.eval()
        out = model(torch.randn(2, 3, 224, 224))
        assert torch.isfinite(out.logits).all() and torch.isfinite(out.variance).all()

        assert model.spec["use_spectral_norm"] is False
        rebuilt = build_net(model.spec)
        assert rebuilt.spec == model.spec
        assert not any(hasattr(m, "weight_u") for m in rebuilt.backbone.modules())

    def test_spec_without_use_spectral_norm_key_rebuilds_with_spectral_norm(self):
        """Back-compat: every spec written before the key existed is a spectral-normed SNGP."""
        from src.models.registry import build_net

        spec = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64).spec
        del spec["use_spectral_norm"]
        rebuilt = build_net(spec)
        assert rebuilt.use_spectral_norm is True
        assert any(hasattr(m, "weight_u") for m in rebuilt.backbone.modules())

    def test_use_spectral_norm_false_lifts_the_vit_guard(self):
        """The ViT rejection exists because *spectral normalization* is unvalidated on ViT
        internals; with no normalization applied there is nothing to guard."""
        model = SNGPClassifier(num_classes=4, arch="vit_b_32", pretrained=False, rff_dim=64, use_spectral_norm=False)
        assert model.arch == "vit_b_32"
        assert not any(hasattr(m, "weight_u") for m in model.backbone.modules())

    def test_output_bias_default_matches_reference(self):
        """The reference uses a fixed zero GP output bias, i.e. no trainable one."""
        model = SNGPClassifier(num_classes=6, arch="resnet18", rff_dim=64)
        assert model.gp_head.classifier.bias is None
        with_bias = SNGPClassifier(num_classes=6, arch="resnet18", rff_dim=64, output_bias=True)
        assert with_bias.gp_head.classifier.bias is not None


def test_spec_without_scale_random_features_rebuilds_unchanged():
    """`scale_random_features` was added after checkpoints existed. A `net_spec` written
    before it must still rebuild bit-identically, so the key has to default to True and
    `build_net` must tolerate its absence."""
    from src.models.registry import build_net

    net = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64)
    assert net.spec["scale_random_features"] is True

    legacy_spec = {k: v for k, v in net.spec.items() if k != "scale_random_features"}
    rebuilt = build_net(legacy_spec)
    assert rebuilt.gp_head.scale_random_features is True
    assert rebuilt.gp_head.rff_scale == net.gp_head.rff_scale
