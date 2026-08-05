"""
Script to compare different uncertainty quantification methods:
- Baseline (with Monte Carlo Dropout)
- SNGP (Spectral-normalized Neural Gaussian Process)
- Deep Ensemble

This script evaluates all three methods on the same test set and compares:
- Accuracy
- Calibration metrics (ECE, Brier Score)
- Uncertainty quality
- Inference time

Usage:
    python scripts/compare_methods.py \
        --baseline-ckpt checkpoints/baseline.ckpt \
        --sngp-ckpt checkpoints/sngp.ckpt \
        --ensemble-ckpt checkpoints/ensemble.ckpt \
        --data configs/data/your_data.yaml
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
import sys
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline_classification_lit_module import BaselineClassificationLitModule
from src.models.sngp_classification_lit_module import SNGPClassificationLitModule
from src.models.ensemble import DeepEnsembleLitModule


def load_models(args):
    """Load all model checkpoints."""
    models = {}
    
    if args.baseline_ckpt:
        print(f"Loading Baseline model from {args.baseline_ckpt}")
        models['Baseline'] = BaselineClassificationLitModule.load_from_checkpoint(args.baseline_ckpt)
        models['Baseline'].eval()
    
    if args.sngp_ckpt:
        print(f"Loading SNGP model from {args.sngp_ckpt}")
        models['SNGP'] = SNGPClassificationLitModule.load_from_checkpoint(args.sngp_ckpt)
        models['SNGP'].eval()
    
    if args.ensemble_ckpt:
        print(f"Loading Deep Ensemble model from {args.ensemble_ckpt}")
        models['DeepEnsemble'] = DeepEnsembleLitModule.load_from_checkpoint(args.ensemble_ckpt)
        models['DeepEnsemble'].eval()
    
    return models


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: True labels [N]
        n_bins: Number of bins
    
    Returns:
        ECE value
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_size = mask.sum() / len(confidences)
            ece += bin_size * abs(bin_accuracy - bin_confidence)
    
    return ece


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier score (lower is better).
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: True labels [N]
    
    Returns:
        Brier score
    """
    num_classes = probs.shape[1]
    one_hot_labels = np.eye(num_classes)[labels]
    return np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1))


def evaluate_model(model, dataloader, device: str, model_name: str) -> Dict:
    """
    Evaluate a single model on the test set.
    
    Returns dictionary with:
        - accuracy
        - ece
        - brier_score
        - avg_uncertainty (if available)
        - inference_time_per_batch
        - all_probs: predicted probabilities
        - all_labels: true labels
        - all_uncertainties: uncertainty scores (if available)
    """
    print(f"\nEvaluating {model_name}...")
    
    model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    all_uncertainties = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{model_name}"):
            # Unpack batch (assuming standard format)
            if len(batch) == 4:
                img_ids, images, labels, fold = batch
            else:
                images, labels = batch[:2]
            
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Time inference
            start = time.time()
            
            # Get predictions based on model type
            if model_name == "DeepEnsemble" and hasattr(model.net, 'get_predictive_uncertainty'):
                probs, uncertainty = model.net.get_predictive_uncertainty(images, uncertainty_type='variance')
                probs = probs.cpu().numpy()
                uncertainty = uncertainty.cpu().numpy()
                all_uncertainties.extend(uncertainty)
            else:
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                # For SNGP, try to get uncertainty from variance
                if model_name == "SNGP" and hasattr(model.net, 'get_uncertainty'):
                    try:
                        uncertainty = probs.var(axis=1)  # Simple variance of probs
                        all_uncertainties.extend(uncertainty)
                    except:
                        pass
            
            elapsed = time.time() - start
            inference_times.append(elapsed)
            
            all_probs.append(probs)
            all_labels.extend(labels)
    
    # Concatenate results
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    predictions = all_probs.argmax(axis=1)
    accuracy = (predictions == all_labels).mean()
    ece = compute_ece(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels)
    
    results = {
        'accuracy': accuracy,
        'ece': ece,
        'brier_score': brier,
        'inference_time_per_batch': np.mean(inference_times),
        'all_probs': all_probs,
        'all_labels': all_labels,
    }
    
    if all_uncertainties:
        results['avg_uncertainty'] = np.mean(all_uncertainties)
        results['all_uncertainties'] = np.array(all_uncertainties)
    
    return results


def create_comparison_plots(all_results: Dict, save_dir: Path):
    """Create comparison visualizations."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics comparison bar plot
    metrics_to_plot = ['accuracy', 'ece', 'brier_score']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[metric] for name, results in all_results.items()]
        names = list(all_results.keys())
        
        axes[idx].bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71'][:len(names)])
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved metrics comparison to {save_dir / 'metrics_comparison.png'}")
    plt.close()
    
    # 2. Calibration curves
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, (name, results) in enumerate(all_results.items()):
        probs = results['all_probs']
        labels = results['all_labels']
        
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
        
        # Bin confidences
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(confidences, bins) - 1
        
        bin_accs = []
        bin_confs = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accs.append(accuracies[mask].mean())
                bin_confs.append(confidences[mask].mean())
            else:
                bin_accs.append(0)
                bin_confs.append(bin_centers[i])
        
        axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        axes[idx].plot(bin_confs, bin_accs, 'o-', label=name, linewidth=2, markersize=8)
        axes[idx].set_xlabel('Confidence')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].set_title(f'{name}\nECE: {results["ece"]:.4f}')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xlim([0, 1])
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'calibration_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved calibration curves to {save_dir / 'calibration_curves.png'}")
    plt.close()
    
    # 3. Uncertainty distributions (if available)
    models_with_uncertainty = {name: res for name, res in all_results.items() 
                               if 'all_uncertainties' in res}
    
    if models_with_uncertainty:
        fig, axes = plt.subplots(1, len(models_with_uncertainty), 
                                figsize=(5*len(models_with_uncertainty), 5))
        if len(models_with_uncertainty) == 1:
            axes = [axes]
        
        for idx, (name, results) in enumerate(models_with_uncertainty.items()):
            uncertainties = results['all_uncertainties']
            labels = results['all_labels']
            predictions = results['all_probs'].argmax(axis=1)
            correct = predictions == labels
            
            axes[idx].hist(uncertainties[correct], bins=50, alpha=0.5, 
                          label='Correct', density=True)
            axes[idx].hist(uncertainties[~correct], bins=50, alpha=0.5, 
                          label='Incorrect', density=True)
            axes[idx].set_xlabel('Uncertainty')
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{name} Uncertainty Distribution')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'uncertainty_distributions.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved uncertainty distributions to {save_dir / 'uncertainty_distributions.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare uncertainty quantification methods")
    parser.add_argument("--baseline-ckpt", type=str, help="Path to baseline checkpoint")
    parser.add_argument("--sngp-ckpt", type=str, help="Path to SNGP checkpoint")
    parser.add_argument("--ensemble-ckpt", type=str, help="Path to deep ensemble checkpoint")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Validate at least one model is provided
    if not any([args.baseline_ckpt, args.sngp_ckpt, args.ensemble_ckpt]):
        raise ValueError("At least one model checkpoint must be provided")
    
    # Load models
    models = load_models(args)
    
    # Load data config
    print(f"\nLoading data configuration from {args.data_config}")
    cfg = OmegaConf.load(args.data_config)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Evaluate each model
    all_results = {}
    for name, model in models.items():
        results = evaluate_model(model, test_loader, args.device, name)
        all_results[name] = results
        
        print(f"\n{name} Results:")
        print(f"  Accuracy:     {results['accuracy']:.4f}")
        print(f"  ECE:          {results['ece']:.4f}")
        print(f"  Brier Score:  {results['brier_score']:.4f}")
        if 'avg_uncertainty' in results:
            print(f"  Avg Uncertainty: {results['avg_uncertainty']:.4f}")
        print(f"  Inference Time: {results['inference_time_per_batch']*1000:.2f}ms per batch")
    
    # Create summary table
    summary_data = []
    for name, results in all_results.items():
        row = {
            'Method': name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'ECE': f"{results['ece']:.4f}",
            'Brier Score': f"{results['brier_score']:.4f}",
            'Inference Time (ms)': f"{results['inference_time_per_batch']*1000:.2f}",
        }
        if 'avg_uncertainty' in results:
            row['Avg Uncertainty'] = f"{results['avg_uncertainty']:.4f}"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)
    print(f"\n✓ Saved summary to {output_dir / 'comparison_summary.csv'}")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    create_comparison_plots(all_results, output_dir)
    
    print(f"\n✓ All results saved to {output_dir}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
