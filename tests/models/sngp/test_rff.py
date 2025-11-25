import pytest
import torch
import torch.nn as nn
from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess

class TestRandomFeatureGaussianProcess:
    @pytest.fixture
    def rff_gp(self):
        return RandomFeatureGaussianProcess(
            in_dim=512,
            num_classes=10,
            rff_dim=128,
            length_scale=1.0,
            ridge_penalty=1e-3,
            cov_momentum=0.999,
            mean_field=True
        )
    
    def test_features_shape(self, rff_gp):
        x = torch.randn(4, 512)
        phi = rff_gp._features(x)
        assert phi.shape == (4, 128)
    
    def test_forward_shapes(self, rff_gp):
        x = torch.randn(4, 512)
        mean_field_logits, raw_logits, pred_var = rff_gp(x)
        assert mean_field_logits.shape == (4, 10)
        assert raw_logits.shape == (4, 10)
        assert pred_var.shape == (4, 1)
    
    def test_covariance_update(self, rff_gp):
        x = torch.randn(4, 512)
        rff_gp.train()
        initial_updates = rff_gp.num_updates.item()
        rff_gp(x, update_cov=True)
        assert rff_gp.num_updates.item() == initial_updates + 1
    
    def test_no_covariance_update_when_eval(self, rff_gp):
        x = torch.randn(4, 512)
        rff_gp.eval()
        initial_updates = rff_gp.num_updates.item()
        rff_gp(x, update_cov=False)
        assert rff_gp.num_updates.item() == initial_updates
    
    def test_mean_field_vs_raw_logits(self, rff_gp):
        x = torch.randn(4, 512)
        mean_field_logits, raw_logits, pred_var = rff_gp(x)
        # Mean field logits should be scaled down by uncertainty
        assert not torch.allclose(mean_field_logits, raw_logits)
    
    def test_without_mean_field(self):
        rff_gp = RandomFeatureGaussianProcess(
            in_dim=512, num_classes=10, rff_dim=128, mean_field=False
        )
        x = torch.randn(4, 512)
        mean_field_logits, raw_logits, pred_var = rff_gp(x)
        assert torch.allclose(mean_field_logits, raw_logits)
