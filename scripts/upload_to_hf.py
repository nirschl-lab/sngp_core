"""
Script to upload all SNGP and Baseline model checkpoints to Hugging Face Hub.

Before running:
1. Ensure you're logged in to HF: huggingface-cli login
2. Update repo_id to your HF repo
3. Update model_specs with your checkpoint paths and configs
"""

import torch
import os
import sys
from pathlib import Path

# Add repo root to path so src can be imported
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.hf_loader import (
    HFModelUploader,
    SNGPModelConfig,
    BaselineModelConfig,
)
from src.models.sngp.sngp_classifier import SNGPClassifier
from src.models.baseline.baseline_models import BaselineClassifier


def upload_models_to_hf(
    repo_id: str = "nirschl-lab/sngp-models",
):
    """
    Upload all cleaned checkpoints to Hugging Face Hub.
    
    Args:
        repo_id: Your Hugging Face repo ID
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define all models to upload
    model_specs = [
        # Baseline models
        {
            "name": "acevedo_baseline_resnet18",
            "type": "baseline",
            "ckpt_path": "notebooks/cleaned_checkpoints/acevedo_baseline_resnet18/acevedo_baseline_resnet18_cleaned.ckpt",
            "config": {
                "arch": "resnet18",
                "num_classes": 8,
                "dropout_p": 0.5,
                "pretrained": False,
            },
        },
        {
            "name": "wong_baseline_resnet18",
            "type": "baseline",
            "ckpt_path": "notebooks/cleaned_checkpoints/wong_baseline_resnet18/wong_baseline_resnet18_cleaned.ckpt",
            "config": {
                "arch": "resnet18",
                "num_classes": 4,
                "dropout_p": 0.5,
                "pretrained": False,
            },
        },
        # SNGP models
        {
            "name": "acevedo_sngp_resnet18",
            "type": "sngp",
            "ckpt_path": "notebooks/cleaned_checkpoints/acevedo_sngp_resnet18/acevedo_sngp_resnet18_cleaned.ckpt",
            "config": {
                "arch": "resnet18",
                "num_classes": 8,
                "rff_dim": 1024,
                "length_scale": 1.0,
                "ridge_penalty": 1e-3,
                "cov_momentum": 0.999,
                "mean_field": True,
                "pretrained": False,
            },
        },
        {
            "name": "wong_sngp_resnet18",
            "type": "sngp",
            "ckpt_path": "notebooks/cleaned_checkpoints/wong_sngp_resnet18/wong_sngp_resnet18_cleaned.ckpt",
            "config": {
                "arch": "resnet18",
                "num_classes": 4,
                "rff_dim": 1024,
                "length_scale": 1.0,
                "ridge_penalty": 1e-3,
                "cov_momentum": 0.999,
                "mean_field": True,
                "pretrained": False,
            },
        },
    ]
    
    uploader = HFModelUploader(repo_id=repo_id)
    
    for spec in model_specs:
        try:
            model_name = spec["name"]
            model_type = spec["type"]
            ckpt_path = repo_root / spec["ckpt_path"]
            config_dict = spec["config"]
            
            print(f"\n{'='*60}")
            print(f"Uploading: {model_name}")
            print(f"{'='*60}")
            
            # Check if checkpoint exists
            if not ckpt_path.exists():
                print(f"❌ Checkpoint not found: {ckpt_path}")
                continue
            
            # Instantiate model
            if model_type == "sngp":
                model = SNGPClassifier(**config_dict)
            else:  # baseline
                model = BaselineClassifier(**config_dict)
            
            # Load checkpoint
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded checkpoint: {ckpt_path}")
            
            # Upload
            commit_info = uploader.upload_model(
                model=model,
                model_name=model_name,
                model_type=model_type,
                config_dict=config_dict,
                commit_message=f"Upload {model_name} ({model_type})",
            )
            print(f"✓ Uploaded successfully!")
            print(f"  Commit: {commit_info.commit_url}")
            
        except Exception as e:
            print(f"❌ Failed to upload {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Upload complete!")
    print(f"Repository: https://huggingface.co/{repo_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    upload_models_to_hf()
