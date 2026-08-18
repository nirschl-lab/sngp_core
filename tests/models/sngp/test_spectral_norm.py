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