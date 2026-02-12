#!/usr/bin/env python
"""
Quick example: Load SNGP model from HF Hub and run inference.

Usage:
    python scripts/example_inference.py
    
Or use as a module:
    from scripts.example_inference import quick_sngp_inference
    results = quick_sngp_inference("wong_sngp_resnet18", image_batch)
"""

import torch
import sys
from pathlib import Path


def quick_sngp_inference(
    model_name: str,
    input_tensor: torch.Tensor,
    repo_id: str = "nirschl-lab/sngp-models",
    device: str = None,
):
    """
    One-liner for SNGP inference from HF Hub.
    
    Args:
        model_name: Model name (e.g., "wong_sngp_resnet18")
        input_tensor: [B, 3, H, W] image batch
        repo_id: HF repo ID
        device: "cuda" or "cpu" (auto-detects if None)
        
    Returns:
        dict with "logits", "variance", "predictions", "confidence"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Add parent to path for src imports
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    from src.models.hf_loader import HFModelLoader
    
    # Load model
    loader = HFModelLoader(repo_id=repo_id)
    model, config = loader.load_model(model_name, device=device)
    
    # Infer
    model.eval()
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        mean_field_logits, raw_logits, pred_var = model(input_tensor)
    
    probs = torch.softmax(mean_field_logits, dim=1)
    predictions = probs.argmax(dim=1)
    confidence = probs.max(dim=1)[0]
    
    return {
        "logits": mean_field_logits.cpu(),
        "variance": pred_var.cpu(),
        "predictions": predictions.cpu(),
        "confidence": confidence.cpu(),
        "probabilities": probs.cpu(),
    }


def quick_baseline_inference(
    model_name: str,
    input_tensor: torch.Tensor,
    repo_id: str = "nirschl-lab/sngp-models",
    device: str = None,
):
    """
    One-liner for Baseline inference from HF Hub.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Add parent to path for src imports
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    from src.models.hf_loader import HFModelLoader
    
    # Load model
    loader = HFModelLoader(repo_id=repo_id)
    model, config = loader.load_model(model_name, device=device)
    
    # Infer
    model.eval()
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
    
    probs = torch.softmax(logits, dim=1)
    predictions = probs.argmax(dim=1)
    confidence = probs.max(dim=1)[0]
    
    return {
        "logits": logits.cpu(),
        "predictions": predictions.cpu(),
        "confidence": confidence.cpu(),
        "probabilities": probs.cpu(),
    }


if __name__ == "__main__":
    print("="*60)
    print("SNGP Model Inference Example")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create dummy batch
    batch = torch.randn(4, 3, 224, 224)
    
    # Test SNGP
    print("Testing SNGP Model: wong_sngp_resnet18")
    print("-" * 60)
    try:
        sngp_results = quick_sngp_inference(
            "wong_sngp_resnet18",
            batch,
            device=device
        )
        
        print(f"Input batch shape: {batch.shape}")
        print(f"Logits shape: {sngp_results['logits'].shape}")
        print(f"Predictions: {sngp_results['predictions'].tolist()}")
        print(f"Confidence: {sngp_results['confidence'].tolist()}")
        print(f"Variance (uncertainty):")
        for i, v in enumerate(sngp_results['variance'].squeeze().tolist()):
            print(f"  Sample {i}: {v:.4f}")
        print("✓ SNGP inference successful!\n")
    except Exception as e:
        print(f"❌ SNGP inference failed: {e}\n")
    
    # Test Baseline
    print("Testing Baseline Model: wong_baseline_resnet18")
    print("-" * 60)
    try:
        baseline_results = quick_baseline_inference(
            "wong_baseline_resnet18",
            batch,
            device=device
        )
        
        print(f"Input batch shape: {batch.shape}")
        print(f"Logits shape: {baseline_results['logits'].shape}")
        print(f"Predictions: {baseline_results['predictions'].tolist()}")
        print(f"Confidence: {baseline_results['confidence'].tolist()}")
        print("✓ Baseline inference successful!\n")
    except Exception as e:
        print(f"❌ Baseline inference failed: {e}\n")
    
    print("="*60)
    print("Inference example complete!")
    print("="*60)
