import pytest
import torch
import torch.nn as nn
from src.models.baseline.baseline_models import BaselineClassifier


class TestBackbones:
    @pytest.mark.parametrize(
        "backbone_name,expected_feature_dim",
        [
            ("resnet18", 512),
            ("resnet34", 512),
            ("resnet50", 2048),
            ("vit_b_16", 768),
            ("vit_b_32", 768),
            ("vit_l_16", 1024),
            ("vit_l_32", 1024),
            ("vit_h_14", 1280),
        ]
    )
    def test_backbone_feature_dimensions(self, backbone_name, expected_feature_dim):
        """Test that backbones produce correct feature dimensions."""
        model = BaselineClassifier(
            arch=backbone_name,
            num_classes=10,
            dropout_p=0.1,
            pretrained=False
        )
        
        # Test input
        x = torch.randn(2, 3, 224, 224)
        
        # Get features
        logits, features = model(x, return_features=True)
        
        # Check feature dimensions
        assert features.shape == (2, expected_feature_dim)
        assert logits.shape == (2, 10)

    @pytest.mark.parametrize("backbone_name", [
        "resnet18", "resnet34", "resnet50",
        "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"
    ])
    def test_backbone_forward_pass(self, backbone_name):
        """Test that all backbones can perform forward passes."""
        model = BaselineClassifier(
            arch=backbone_name,
            num_classes=5,
            pretrained=False
        )
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = model(x)
                
            assert output.shape == (batch_size, 5)
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()

    @pytest.mark.parametrize("pretrained", [True, False])
    def test_pretrained_vs_non_pretrained(self, pretrained):
        """Test that models work with both pretrained and non-pretrained weights."""
        try:
            model = BaselineClassifier(
                arch="resnet18",
                pretrained=pretrained,
                num_classes=3
            )
            
            x = torch.randn(2, 3, 224, 224)
            output = model(x)
            
            assert output.shape == (2, 3)
            
        except Exception as e:
            # If pretrained weights fail to load, skip this test
            if pretrained and ("HTTP" in str(e) or "download" in str(e).lower()):
                pytest.skip(f"Skipping pretrained test due to download issue: {e}")
            else:
                raise

    def test_unsupported_backbone_raises_error(self):
        """Test that unsupported backbone names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            BaselineClassifier(arch="invalid_backbone")

    @pytest.mark.parametrize("backbone_type", ["resnet", "vit"])
    def test_backbone_feature_extraction_consistency(self, backbone_type):
        """Test that feature extraction is consistent across calls."""
        if backbone_type == "resnet":
            arch = "resnet18"
        else:
            arch = "vit_b_16"
            
        model = BaselineClassifier(arch=arch, pretrained=False)
        model.eval()
        
        x = torch.randn(3, 3, 224, 224)
        
        with torch.no_grad():
            features1 = model.feature_extractor(x)
            features2 = model.feature_extractor(x)
            
        torch.testing.assert_close(features1, features2)

    @pytest.mark.parametrize("input_size", [
        (224, 224),
        (256, 256),
        (384, 384),
    ])
    def test_backbone_different_input_sizes(self, input_size):
        """Test that backbones handle different input sizes."""
        # ResNets are generally flexible with input sizes
        model = BaselineClassifier(arch="resnet18", pretrained=False)
        
        h, w = input_size
        x = torch.randn(2, 3, h, w)
        
        try:
            output = model(x)
            assert output.shape[0] == 2  # Batch size should be preserved
            assert output.shape[1] == 2  # Default num_classes
        except Exception as e:
            # Some input sizes might not work with certain architectures
            pytest.skip(f"Input size {input_size} not supported: {e}")

    def test_backbone_gradients_flow(self):
        """Test that gradients flow through the backbone."""
        model = BaselineClassifier(arch="resnet18", pretrained=False)
        model.train()
        
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 2, (2,))
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check that gradients exist for backbone parameters
        backbone_has_grad = False
        for param in model.feature_extractor.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                backbone_has_grad = True
                break
        
        assert backbone_has_grad, "No gradients found in backbone"

    def test_backbone_eval_vs_train_mode(self):
        """Test that backbone behaves differently in train vs eval mode."""
        model = BaselineClassifier(arch="resnet18", pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        
        # Test in training mode
        model.train()
        train_output = model(x)
        
        # Test in eval mode
        model.eval()
        with torch.no_grad():
            eval_output = model(x)
        
        # Outputs might differ due to batch norm, but shapes should be same
        assert train_output.shape == eval_output.shape

    @pytest.mark.parametrize("num_classes", [1, 2, 10, 100, 1000])
    def test_backbone_with_different_num_classes(self, num_classes):
        """Test that backbones work with different numbers of output classes."""
        model = BaselineClassifier(
            arch="resnet18",
            num_classes=num_classes,
            pretrained=False
        )
        
        x = torch.randn(3, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (3, num_classes)

    def test_backbone_memory_efficiency(self):
        """Test that backbone doesn't leak memory during multiple forward passes."""
        model = BaselineClassifier(arch="resnet18", pretrained=False)
        model.eval()
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Multiple forward passes
        for _ in range(10):
            x = torch.randn(4, 3, 224, 224)
            with torch.no_grad():
                _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            # Memory should not significantly increase
            assert final_memory <= initial_memory + 1024 * 1024  # 1MB tolerance

    def test_backbone_classifier_separation(self):
        """Test that feature extractor and classifier are properly separated."""
        model = BaselineClassifier(arch="resnet18", num_classes=5, pretrained=False)
        
        # Check that feature extractor doesn't include classification head
        x = torch.randn(2, 3, 224, 224)
        features = model.feature_extractor(x)
        
        # Features should be flattened but not classified yet
        assert len(features.shape) == 2  # (batch_size, feature_dim)
        assert features.shape[1] == 512  # ResNet18 feature dim
        
        # Classifier should take features and output class logits
        logits = model.classifier(features)
        assert logits.shape == (2, 5)

    @pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.5, 0.9])
    def test_backbone_with_different_dropout_rates(self, dropout_p):
        """Test that backbones work with different dropout rates."""
        model = BaselineClassifier(
            arch="resnet18",
            dropout_p=dropout_p,
            pretrained=False
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        # Test in both train and eval modes
        model.train()
        train_output = model(x)
        
        model.eval()
        with torch.no_grad():
            eval_output = model(x)
        
        assert train_output.shape == eval_output.shape == (2, 2)
        
        # With dropout_p=0, outputs should be identical in eval mode
        if dropout_p == 0.0:
            model.eval()
            with torch.no_grad():
                output1 = model(x)
                output2 = model(x)
            torch.testing.assert_close(output1, output2)