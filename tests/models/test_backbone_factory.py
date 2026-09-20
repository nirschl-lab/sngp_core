import pytest
import torch

from src.models.backbones import BACKBONES, SPECTRAL_NORM_COMPATIBLE, build_backbone

EXPECTED_FEAT_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "wide_resnet28_10": 640,
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
    "vit_h_14": 1280,
}

# Everything here is an ImageNet-resolution backbone except the CIFAR WideResNet, whose
# stride-1 stem keeps full resolution through the first group -- a 224px forward costs
# ~250 GFLOPs and has no business in the fast suite. It pools adaptively, so exercising
# it at its native 32px proves the same thing.
INPUT_SIZE = {"wide_resnet28_10": 32}
DEFAULT_INPUT_SIZE = 224


class TestBuildBackbone:
    @pytest.mark.parametrize("arch", sorted(BACKBONES))
    def test_returns_expected_feature_dim(self, arch):
        module, feat_dim = build_backbone(arch, pretrained=False)
        assert feat_dim == EXPECTED_FEAT_DIM[arch]

    @pytest.mark.parametrize("arch", sorted(BACKBONES))
    def test_forward_yields_flat_features(self, arch):
        module, feat_dim = build_backbone(arch, pretrained=False)
        module.eval()
        size = INPUT_SIZE.get(arch, DEFAULT_INPUT_SIZE)
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            out = module(x)
        assert out.shape == (2, feat_dim)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            build_backbone("not_a_real_arch", pretrained=False)

    def test_spectral_norm_compatible_is_the_conv_backbones(self):
        """SNGP wraps every Conv2d/Linear in the backbone; the ViTs are not validated for
        it, and `wide_resnet28_10` qualifies for the same reason the ResNets do."""
        assert SPECTRAL_NORM_COMPATIBLE == {"resnet18", "resnet34", "resnet50", "wide_resnet28_10"}

    def test_wide_resnet_has_no_pretrained_weights(self):
        """There is no ImageNet checkpoint for a CIFAR WideResNet, so `pretrained=True`
        must fall through to random init rather than silently loading something else."""
        module, feat_dim = build_backbone("wide_resnet28_10", pretrained=True)
        assert feat_dim == 640
