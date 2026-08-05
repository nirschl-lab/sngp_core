"""
Simple test to verify Deep Ensemble implementation.

Run this to ensure everything is working before full training:
    python scripts/test_ensemble.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.baseline.baseline_models import BaselineClassifier
from src.models.ensemble import DeepEnsemble


def test_ensemble_creation():
    """Test that ensemble can be created correctly."""
    print("Testing ensemble creation...")
    
    ensemble = DeepEnsemble(
        base_model_class=BaselineClassifier,
        base_model_kwargs={
            'arch': 'resnet18',
            'num_classes': 8,
            'dropout_p': 0.2,
            'pretrained': False  # Faster for testing
        },
        num_estimators=3,
        task='classification'
    )
    
    print(f"✓ Created ensemble with {ensemble.num_estimators} members")
    print(f"✓ Model: {ensemble}")
    
    return ensemble


def test_training_mode(ensemble):
    """Test training mode with active member."""
    print("\nTesting training mode...")
    
    ensemble.train()
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test each member
    for i in range(ensemble.num_estimators):
        ensemble.set_active_member(i)
        output = ensemble(x)
        
        assert output.shape == (batch_size, 8), f"Expected shape (4, 8), got {output.shape}"
        print(f"✓ Member {i+1} forward pass: output shape {output.shape}")
    
    print("✓ All members work in training mode")


def test_inference_mode(ensemble):
    """Test inference mode with ensemble averaging."""
    print("\nTesting inference mode...")
    
    ensemble.eval()
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        # Test ensemble prediction
        mean_output = ensemble(x)
        assert mean_output.shape == (batch_size, 8), f"Expected shape (4, 8), got {mean_output.shape}"
        print(f"✓ Ensemble forward pass: output shape {mean_output.shape}")
        
        # Test individual predictions
        mean_output2, individual_outputs = ensemble.ensemble_predict(x, return_individual=True)
        assert individual_outputs.shape == (3, batch_size, 8), \
            f"Expected shape (3, 4, 8), got {individual_outputs.shape}"
        print(f"✓ Individual predictions: shape {individual_outputs.shape}")
        
        # Verify averaging
        manual_mean = individual_outputs.mean(dim=0)
        assert torch.allclose(mean_output, manual_mean, atol=1e-5), "Mean output doesn't match manual average"
        print("✓ Ensemble averaging verified")


def test_uncertainty_estimation(ensemble):
    """Test uncertainty quantification methods."""
    print("\nTesting uncertainty estimation...")
    
    ensemble.eval()
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    uncertainty_types = ['variance', 'entropy', 'mutual_info']
    
    for unc_type in uncertainty_types:
        with torch.no_grad():
            probs, uncertainty = ensemble.get_predictive_uncertainty(x, uncertainty_type=unc_type)
            
            assert probs.shape == (batch_size, 8), f"Probs shape: expected (4, 8), got {probs.shape}"
            assert uncertainty.shape == (batch_size,), f"Uncertainty shape: expected (4,), got {uncertainty.shape}"
            assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probabilities not in [0, 1]"
            assert torch.all(torch.abs(probs.sum(dim=1) - 1) < 1e-5), "Probabilities don't sum to 1"
            
            print(f"✓ {unc_type}: probs {probs.shape}, uncertainty {uncertainty.shape}")
            print(f"  Mean uncertainty: {uncertainty.mean().item():.4f}")


def test_member_diversity(ensemble):
    """Test that ensemble members are different."""
    print("\nTesting member diversity...")
    
    ensemble.eval()
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        _, individual_outputs = ensemble.ensemble_predict(x, return_individual=True)
        individual_probs = torch.softmax(individual_outputs, dim=-1)
        
        # Check that not all members give identical predictions
        std_across_members = individual_probs.std(dim=0).mean()
        
        print(f"✓ Standard deviation across members: {std_across_members.item():.4f}")
        
        # They should be different (std > 0)
        assert std_across_members > 0, "All members giving identical predictions - no diversity!"
        print("✓ Members show diversity in predictions")


def test_gradient_flow(ensemble):
    """Test that gradients flow correctly during training."""
    print("\nTesting gradient flow...")
    
    ensemble.train()
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1])
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Test each member
    for i in range(ensemble.num_estimators):
        ensemble.set_active_member(i)
        
        # Forward pass
        output = ensemble(x)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for the active member
        member = ensemble.get_member(i)
        has_grads = False
        for param in member.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        assert has_grads, f"No gradients for member {i}"
        
        # Zero gradients for next iteration
        ensemble.zero_grad()
        
        print(f"✓ Member {i+1}: gradients flowing correctly")
    
    print("✓ Gradient flow verified for all members")


def main():
    print("="*60)
    print("DEEP ENSEMBLE IMPLEMENTATION TEST")
    print("="*60)
    
    try:
        # Create ensemble
        ensemble = test_ensemble_creation()
        
        # Test training mode
        test_training_mode(ensemble)
        
        # Test inference mode
        test_inference_mode(ensemble)
        
        # Test uncertainty estimation
        test_uncertainty_estimation(ensemble)
        
        # Test diversity
        test_member_diversity(ensemble)
        
        # Test gradients
        test_gradient_flow(ensemble)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour Deep Ensemble implementation is working correctly.")
        print("You can now proceed with full training:")
        print("  python src/train.py model=deep_ensemble_classifier")
        print("\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
