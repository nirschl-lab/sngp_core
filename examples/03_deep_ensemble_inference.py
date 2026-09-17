"""
Example script demonstrating Deep Ensemble inference and uncertainty quantification.

This script shows how to:
1. Load a trained deep ensemble model
2. Perform inference with uncertainty estimation
3. Visualize predictions and uncertainty
4. Compare ensemble vs individual member predictions

Usage:
    python examples/03_deep_ensemble_inference.py --checkpoint path/to/checkpoint.ckpt
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpointing.io import load_lit_module
from src.models.deep_ensemble_lit_module import DeepEnsembleLitModule


def load_ensemble_model(checkpoint_path: str, device: str = "cuda") -> DeepEnsembleLitModule:
    """Load a trained deep ensemble model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    model = load_lit_module(checkpoint_path, device=device)

    print(f"✓ Loaded ensemble with {model.num_estimators} members")
    return model


def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for inference."""
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict_with_uncertainty(
    model: DeepEnsembleLitModule,
    image: torch.Tensor,
    class_names: list = None,
    uncertainty_type: str = "variance"
) -> dict:
    """
    Perform inference with uncertainty quantification.
    
    Returns:
        dict with:
            - 'probs': Mean predicted probabilities
            - 'pred_class': Predicted class index
            - 'pred_label': Predicted class name (if class_names provided)
            - 'uncertainty': Uncertainty score
            - 'confidence': Max probability (1 - uncertainty for normalized view)
            - 'individual_probs': Predictions from each ensemble member
    """
    device = next(model.parameters()).device
    image = image.to(device)
    
    with torch.no_grad():
        # Get ensemble predictions with uncertainty
        mean_probs, uncertainty = model.net.get_predictive_uncertainty(
            image, 
            uncertainty_type=uncertainty_type
        )
        
        # Get individual member predictions for analysis
        _, individual_logits = model.net.ensemble_predict(image, return_individual=True)
        individual_probs = torch.softmax(individual_logits, dim=-1)
        
        # Predictions
        pred_class = mean_probs.argmax(dim=1).item()
        confidence = mean_probs.max(dim=1).values.item()
        
    results = {
        'probs': mean_probs[0].cpu().numpy(),
        'pred_class': pred_class,
        'pred_label': class_names[pred_class] if class_names else f"Class {pred_class}",
        'uncertainty': uncertainty[0].item(),
        'confidence': confidence,
        'individual_probs': individual_probs[:, 0, :].cpu().numpy(),  # [num_members, num_classes]
    }
    
    return results


def visualize_predictions(
    image_path: str,
    results: dict,
    class_names: list = None,
    save_path: str = None
):
    """Visualize image, predictions, and uncertainty."""
    
    # Load original image
    image = Image.open(image_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title(f"Input Image\nPrediction: {results['pred_label']}")
    axes[0].axis('off')
    
    # 2. Class Probabilities
    num_classes = len(results['probs'])
    class_labels = class_names if class_names else [f"C{i}" for i in range(num_classes)]
    
    axes[1].barh(range(num_classes), results['probs'])
    axes[1].set_yticks(range(num_classes))
    axes[1].set_yticklabels(class_labels)
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f"Ensemble Predictions\nConfidence: {results['confidence']:.3f}")
    axes[1].set_xlim([0, 1])
    
    # 3. Individual Member Predictions (heatmap)
    im = axes[2].imshow(results['individual_probs'], aspect='auto', cmap='YlOrRd')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Ensemble Member')
    axes[2].set_title(f"Member Agreement\nUncertainty: {results['uncertainty']:.3f}")
    axes[2].set_xticks(range(num_classes))
    axes[2].set_xticklabels(class_labels, rotation=45)
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_members(model: DeepEnsembleLitModule, image: torch.Tensor, class_names: list = None):
    """Compare predictions from individual ensemble members."""
    device = next(model.parameters()).device
    image = image.to(device)
    
    print("\n" + "="*60)
    print("INDIVIDUAL ENSEMBLE MEMBER PREDICTIONS")
    print("="*60)
    
    with torch.no_grad():
        _, individual_logits = model.net.ensemble_predict(image, return_individual=True)
        individual_probs = torch.softmax(individual_logits, dim=-1)
    
    num_members = individual_probs.shape[0]
    num_classes = individual_probs.shape[-1]
    
    for i in range(num_members):
        probs = individual_probs[i, 0].cpu().numpy()
        pred_class = probs.argmax()
        confidence = probs[pred_class]
        
        pred_label = class_names[pred_class] if class_names else f"Class {pred_class}"
        
        print(f"\nMember {i+1}:")
        print(f"  Prediction: {pred_label}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Top-3 classes: ", end="")
        
        top3_idx = np.argsort(probs)[-3:][::-1]
        for idx in top3_idx:
            label = class_names[idx] if class_names else f"C{idx}"
            print(f"{label}({probs[idx]:.3f}) ", end="")
        print()
    
    # Ensemble prediction
    mean_probs = individual_probs.mean(dim=0)[0].cpu().numpy()
    pred_class = mean_probs.argmax()
    pred_label = class_names[pred_class] if class_names else f"Class {pred_class}"
    
    print("\n" + "-"*60)
    print("ENSEMBLE PREDICTION (averaged):")
    print(f"  Prediction: {pred_label}")
    print(f"  Confidence: {mean_probs[pred_class]:.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Deep Ensemble Inference Example")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="List of class names (e.g., --class-names cat dog bird)"
    )
    parser.add_argument(
        "--uncertainty-type",
        type=str,
        default="variance",
        choices=["variance", "entropy", "mutual_info"],
        help="Type of uncertainty to compute"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        default=None,
        help="Path to save visualization (optional)"
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_ensemble_model(args.checkpoint, device=args.device)
    
    # Preprocess image
    print(f"\nLoading image: {args.image}")
    image = preprocess_image(args.image)
    
    # Perform inference
    print(f"Running inference with {args.uncertainty_type} uncertainty...")
    results = predict_with_uncertainty(
        model, 
        image, 
        class_names=args.class_names,
        uncertainty_type=args.uncertainty_type
    )
    
    # Print results
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {results['pred_label']}")
    print(f"Confidence:      {results['confidence']:.4f}")
    print(f"Uncertainty:     {results['uncertainty']:.4f}")
    print(f"Uncertainty Type: {args.uncertainty_type}")
    print("="*60)
    
    # Show detailed member predictions
    compare_members(model, image, class_names=args.class_names)
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_predictions(
        args.image,
        results,
        class_names=args.class_names,
        save_path=args.save_viz
    )
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
