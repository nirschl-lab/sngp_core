import pytest
import torch

from src.models.backbones import BACKBONES, SPECTRAL_NORM_COMPATIBLE, build_backbone

EXPECTED_FEAT_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
    "vit_h_14": 1280,
}


class TestBuildBackbone:
    @pytest.mark.parametrize("arch", sorted(BACKBONES))
    def test_returns_expected_feature_dim(self, arch):
        module, feat_dim = build_backbone(arch, pretrained=False)
        assert feat_dim == EXPECTED_FEAT_DIM[arch]

    @pytest.mark.parametrize("arch", sorted(BACKBONES))
    def test_forward_yields_flat_features(self, arch):
        module, feat_dim = build_backbone(arch, pretrained=False)
        module.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = module(x)
        assert out.shape == (2, feat_dim)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            build_backbone("not_a_real_arch", pretrained=False)

    def test_spectral_norm_compatible_is_resnet_only(self):
        assert SPECTRAL_NORM_COMPATIBLE == {"resnet18", "resnet34", "resnet50"}
