import pytest
import torch
import torch.nn as nn
from src.models.sngp.sngp_classifier import SNGPClassifier 

class TestSNGPClassifier:
    @pytest.mark.parametrize("arch", ["resnet18", "resnet34", "resnet50"])
    def test_resnet_architectures(self, arch):
        model = SNGPClassifier(
            num_classes=10,
            arch=arch,
            pretrained=False,
            rff_dim=128
        )
        x = torch.randn(2, 3, 224, 224)
        mean_field_logits, raw_logits, pred_var = model(x)
        assert mean_field_logits.shape == (2, 10)
        assert raw_logits.shape == (2, 10)
        assert pred_var.shape == (2, 1)
    
    @pytest.mark.parametrize("arch", ["vit_b_16", "vit_b_32", "vit_l_16"])
    def test_vit_architectures(self, arch):
        model = SNGPClassifier(
            num_classes=10,
            arch=arch,
            pretrained=False,
            rff_dim=128
        )
        x = torch.randn(2, 3, 224, 224)
        mean_field_logits, raw_logits, pred_var = model(x)
        assert mean_field_logits.shape == (2, 10)
        assert raw_logits.shape == (2, 10)
        assert pred_var.shape == (2, 1)
    
    def test_unsupported_architecture(self):
        with pytest.raises(ValueError, match="Unsupported arch"):
            SNGPClassifier(num_classes=10, arch="unsupported_arch")
    
    def test_pretrained_model(self):
        model = SNGPClassifier(
            num_classes=10,
            arch="resnet18",
            pretrained=True,
            rff_dim=128
        )
        assert model is not None
    
    def test_spectral_norm_applied(self):
        model = SNGPClassifier(
            num_classes=10,
            arch="resnet18",
            pretrained=False,
            rff_dim=128
        )
        # Check that spectral norm was applied to backbone
        has_spectral_norm = False
        for module in model.backbone.modules():
            if hasattr(module, 'weight_u'):
                has_spectral_norm = True
                break
        assert has_spectral_norm
    
    def test_different_input_sizes(self):
        model = SNGPClassifier(num_classes=5, arch="resnet18", rff_dim=64)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            mean_field_logits, raw_logits, pred_var = model(x)
            assert mean_field_logits.shape == (batch_size, 5)
            assert raw_logits.shape == (batch_size, 5)
            assert pred_var.shape == (batch_size, 1)
    
    def test_update_cov_parameter(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        x = torch.randn(2, 3, 224, 224)
        
        model.train()
        initial_updates = model.gp_head.num_updates.item()
        
        # With update_cov=True (default)
        model(x, update_cov=True)
        assert model.gp_head.num_updates.item() == initial_updates + 1
        
        # With update_cov=False
        model(x, update_cov=False)
        assert model.gp_head.num_updates.item() == initial_updates + 1
    
    def test_eval_mode(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        x = torch.randn(2, 3, 224, 224)
        
        model.eval()
        initial_updates = model.gp_head.num_updates.item()
        model(x)
        # In eval mode, covariance should not update
        assert model.gp_head.num_updates.item() == initial_updates
    
    def test_grad_flow(self):
        model = SNGPClassifier(num_classes=10, arch="resnet18", rff_dim=64)
        x = torch.randn(2, 3, 224, 224)
        x.requires_grad_(True)
        
        mean_field_logits, _, _ = model(x)
        loss = mean_field_logits.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    @pytest.mark.parametrize("num_classes", [1, 5, 100, 1000])
    def test_different_num_classes(self, num_classes):
        model = SNGPClassifier(
            num_classes=num_classes,
            arch="resnet18",
            rff_dim=64
        )
        x = torch.randn(2, 3, 224, 224)
        mean_field_logits, raw_logits, pred_var = model(x)
        assert mean_field_logits.shape == (2, num_classes)
        assert raw_logits.shape == (2, num_classes)
        assert pred_var.shape == (2, 1)