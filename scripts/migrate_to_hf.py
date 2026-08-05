#!/usr/bin/env python3
"""
Script to migrate Lightning checkpoint to HuggingFace model format.

Usage:
    python scripts/migrate_to_hf.py \
        --checkpoint logs/train/runs/2026-01-29_13-44-38/checkpoints/last.ckpt \
        --output-dir ./hf_checkpoint \
        --arch resnet18 \
        --num-classes 8 \
        --dropout-p 0.5 \
        --push-to-hub org/my-model \
        --hf-token <your-huggingface-token>
"""

import argparse
import torch
import json
from pathlib import Path
from typing import Optional

def migrate_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    arch: str = "resnet18",
    num_classes: int = 8,
    dropout_p: float = 0.5,
    push_to_hub: Optional[str] = None,
    hf_token: Optional[str] = None,
    model_variant: Optional[str] = None,
):
    """Migrate Lightning checkpoint to HuggingFace format."""
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict from Lightning checkpoint
    state_dict = checkpoint["state_dict"]
    
    # Remove 'net.' prefix from all keys (Lightning wraps the model)
    remapped_state_dict = {}
    for key, value in state_dict.items():
        # Remove both 'model.model.' and 'net.' prefixes
        new_key = key.replace("net.", "", 1).replace("model.model.", "model.", 1)
        remapped_state_dict[new_key] = value
    
    print(f"State dict keys (first 5): {list(remapped_state_dict.keys())[:5]}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = {
        "architectures": ["BaselineClassifierForImageClassification"],
        "model_type": "baseline_classifier",
        "arch": arch,
        "num_classes": num_classes,
        "dropout_p": dropout_p,
        "pretrained": False,
    }
    
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Save model weights
    weights_path = output_path / "pytorch_model.bin"
    torch.save(remapped_state_dict, weights_path)
    print(f"Saved weights to: {weights_path}")
    
    # Save model code as separate module
    model_code_path = output_path / "hf_model.py"
    import inspect
    from src.hf_model import BaselineClassifierConfig, BaselineClassifier, BaselineClassifierForImageClassification
    
    # Get the source code
    model_source = inspect.getsource(BaselineClassifierConfig)
    model_source += "\n\n"
    model_source += inspect.getsource(BaselineClassifier)
    model_source += "\n\n"
    model_source += inspect.getsource(BaselineClassifierForImageClassification)
    
    with open(model_code_path, "w") as f:
        f.write(model_source)
    print(f"Saved model code to: {model_code_path}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        try:
            from huggingface_hub import upload_folder
            
            # Determine path_in_repo if model_variant is specified
            path_in_repo = model_variant if model_variant else None
            
            print(f"Pushing to HuggingFace Hub: {push_to_hub}")
            if path_in_repo:
                print(f"  Subdirectory: {path_in_repo}")
            
            upload_folder(
                repo_id=push_to_hub,
                folder_path=str(output_path),
                path_in_repo=path_in_repo,
                token=hf_token,
                repo_type="model",
            )
            print(f"✓ Successfully pushed to: {push_to_hub}")
            if path_in_repo:
                print(f"  Load with: AutoModel.from_pretrained('{push_to_hub}', subfolder='{path_in_repo}', trust_remote_code=True)")
        except ImportError:
            print("⚠ huggingface_hub not installed. Skipping push. Install with: pip install huggingface_hub")
    
    print(f"\n✓ Migration complete! Model saved to: {output_dir}")
    print(f"\nTo use the model:")
    print(f"  # For local use:")
    print(f"  from transformers import AutoModel")
    print(f"  model = AutoModel.from_pretrained('{output_dir}', trust_remote_code=True)")
    print(f"  model.eval()")
    if push_to_hub:
        if model_variant:
            print(f"\n  # For remote use (loaded from subdirectory):")
            print(f"  model = AutoModel.from_pretrained('{push_to_hub}', subfolder='{model_variant}', trust_remote_code=True)")
        else:
            print(f"\n  # For remote use:")
            print(f"  model = AutoModel.from_pretrained('{push_to_hub}', trust_remote_code=True)")
        print(f"  model.eval()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate Lightning checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Lightning checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--output-dir",
        default="./hf_checkpoint",
        help="Output directory for HuggingFace model",
    )
    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"],
        help="Model architecture",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=8,
        help="Number of classes",
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,
        help="Dropout probability",
    )
    parser.add_argument(
        "--push-to-hub",
        help="HuggingFace Hub repo ID (e.g., 'org/my-model')",
    )
    parser.add_argument(
        "--model-variant",
        default=None,
        help="Model variant name for subdirectory (e.g., 'resnet18-v1'). If specified, pushes to repo/variant/",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    
    args = parser.parse_args()
    migrate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        arch=args.arch,
        num_classes=args.num_classes,
        dropout_p=args.dropout_p,
        push_to_hub=args.push_to_hub,
        model_variant=args.model_variant,
        hf_token=args.hf_token,
    )
