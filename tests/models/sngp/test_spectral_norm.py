import pytest
import torch
import torch.nn as nn
from src.models.components.spectral_norm import apply_spectral_norm, assert_spectral_norm_compatible

class TestApplySpectralNorm:
    def test_apply_spectral_norm_conv2d(self):
        # Test with a container module
        container = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3))
        apply_spectral_norm(container)
        assert hasattr(container[0], 'weight_u')

    def test_apply_spectral_norm_linear(self):
        # Test with a container module
        container = nn.Sequential(nn.Linear(512, 10))
        apply_spectral_norm(container)
        assert hasattr(container[0], 'weight_u')

    def test_skip_batchnorm(self):
        container = nn.Sequential(nn.BatchNorm2d(64))
        apply_spectral_norm(container)
        assert not hasattr(container[0], 'weight_u')

    def test_avoid_double_wrapping(self):
        container = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3))
        apply_spectral_norm(container)
        apply_spectral_norm(container)  # Should not wrap twice
        assert hasattr(container[0], 'weight_u')

    def test_nested_modules(self):
        # Test with nested structure
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Sequential(
                nn.Conv2d(32, 64, 3),
                nn.BatchNorm2d(64)
            ),
            nn.Linear(64, 10)
        )
        apply_spectral_norm(model)

        # Check conv layers have spectral norm
        assert hasattr(model[0], 'weight_u')  # First conv
        assert hasattr(model[2][0], 'weight_u')  # Nested conv
        assert hasattr(model[3], 'weight_u')  # Linear layer

        # Check BatchNorm doesn't have spectral norm
        assert not hasattr(model[2][1], 'weight_u')


class TestSpectralNormCompatibility:
    @pytest.mark.parametrize("arch", ["resnet18", "resnet34", "resnet50"])
    def test_resnet_is_compatible(self, arch):
        assert_spectral_norm_compatible(arch)  # should not raise

    @pytest.mark.parametrize("arch", ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"])
    def test_vit_is_incompatible(self, arch):
        with pytest.raises(ValueError, match="not supported"):
            assert_spectral_norm_compatible(arch)

class TestPowerIterationWarmup:
    """`spectral_norm`'s power iteration only advances on train-mode forwards, so a
    freshly built net would otherwise normalize by `u^T W v` for random unit u, v. The
    error compounds across layers: an unwarmed spectral-normed resnet18 emits ~1e30 in
    eval mode. `cos()` in the GP head used to hide this; LayerNorm on the GP input turns
    it into NaN, and it corrupts Lightning's pre-training sanity-check validation either
    way.
    """

    def test_warmup_keeps_untrained_resnet_activations_sane(self):
        from src.models.backbones import build_backbone

        torch.manual_seed(0)
        backbone, _ = build_backbone("resnet18", False)
        apply_spectral_norm(backbone)
        backbone.eval()
        out = backbone(torch.randn(2, 3, 224, 224))
        assert torch.isfinite(out).all()
        assert out.abs().max() < 100

    def test_without_warmup_activations_explode(self):
        """Documents the failure mode the warmup exists to prevent."""
        from src.models.backbones import build_backbone

        torch.manual_seed(0)
        backbone, _ = build_backbone("resnet18", False)
        apply_spectral_norm(backbone, warmup_iterations=0)
        backbone.eval()
        assert backbone(torch.randn(2, 3, 224, 224)).abs().max() > 1e6

    def test_warmup_drives_the_normalized_weight_to_unit_spectral_norm(self):
        from src.models.components.spectral_norm import warm_up_spectral_norm

        def sigma_of(module):
            return torch.linalg.matrix_norm(module.weight.reshape(module.weight.shape[0], -1), ord=2).item()

        torch.manual_seed(0)
        wrapped = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(wrapped, warmup_iterations=0)
        unwarmed = sigma_of(wrapped[0])

        warm_up_spectral_norm(wrapped)
        warmed = sigma_of(wrapped[0])

        # Spectral norm targets sigma == 1; random u/v under-shoots badly.
        assert abs(warmed - 1.0) < abs(unwarmed - 1.0)
        assert warmed == pytest.approx(1.0, abs=0.05)

    def test_further_warmup_does_not_destabilize_sigma(self):
        """Power iteration keeps refining (and u may converge to -v), so u itself is not
        stable to compare -- what must hold is that sigma stays pinned near 1."""
        from src.models.components.spectral_norm import warm_up_spectral_norm

        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(model)
        for _ in range(3):
            warm_up_spectral_norm(model)
            sigma = torch.linalg.matrix_norm(model[0].weight.reshape(16, -1), ord=2).item()
            assert sigma == pytest.approx(1.0, abs=0.05)
