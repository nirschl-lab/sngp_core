"""
Utility functions for migrating models to HuggingFace format.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional
from src.hf_model import BaselineClassifierConfig, BaselineClassifierForImageClassification


def extract_config_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Try to extract model config from Lightning checkpoint.
    Returns architecture info if available.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Try to get config from checkpoint
    if "hparams" in checkpoint:
        hparams = checkpoint["hparams"]
        return {
            "arch": hparams.get("model", {}).get("arch", "resnet18"),
            "num_classes": hparams.get("model", {}).get("num_classes", 8),
            "dropout_p": hparams.get("model", {}).get("dropout_p", 0.5),
        }
    
    # Fallback to defaults
    return {
        "arch": "resnet18",
        "num_classes": 8,
        "dropout_p": 0.5,
    }


def remap_state_dict(state_dict: Dict) -> Dict:
    """
    Remap state dict keys to remove Lightning wrapper prefixes.
    
    Examples:
        'net.feature_extractor.layer1...' -> 'feature_extractor.layer1...'
        'model.model.classifier...' -> 'model.classifier...'
    """
    remapped = {}
    for key, value in state_dict.items():
        # Remove common Lightning/wrapper prefixes
        new_key = key.replace("net.", "", 1)  # Remove 'net.' prefix
        new_key = new_key.replace("model.model.", "model.", 1)  # Remove duplicate 'model.'
        remapped[new_key] = value
    
    return remapped


def validate_checkpoint(
    checkpoint_path: str,
    model_config: BaselineClassifierConfig,
    device: str = "cpu",
) -> bool:
    """
    Validate that checkpoint weights match model architecture.
    
    Returns:
        True if checkpoint is valid for this config
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        
        # Remap and load into model
        remapped = remap_state_dict(state_dict)
        
        # Create model
        from src.hf_model import BaselineClassifier
        model = BaselineClassifier(
            arch=model_config.arch,
            num_classes=model_config.num_classes,
            dropout_p=model_config.dropout_p,
        )
        
        # Try to load
        model.load_state_dict(remapped)
        print("✓ Checkpoint validation passed!")
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint validation failed: {e}")
        return False


def create_model_card(
    output_dir: str,
    model_name: str,
    description: str = "",
    usage_example: str = "",
) -> None:
    """
    Create a README.md model card for the HuggingFace model.
    
    Args:
        output_dir: Directory to save README.md
        model_name: Name of the model
        description: Model description
        usage_example: Code example for using the model
    """
    if not usage_example:
        usage_example = """```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("model-id", trust_remote_code=True)
model.eval()

# Inference
input_tensor = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(input_tensor)
```"""
    
    model_card = f"""---
tags:
  - image-classification
  - resnet
  - pytorch
---

# {model_name}

{description}

## Model Details

- **Architecture**: ResNet-based classifier
- **Framework**: PyTorch + HuggingFace Transformers
- **Input**: Images (3, 224, 224)

## Usage

{usage_example}

## Training Details

- Trained using Lightning + Hydra
- Supports Monte Carlo dropout for uncertainty

## License

See LICENSE file.
"""
    
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    print(f"✓ Created model card: {readme_path}")


def test_model_loading(checkpoint_dir: str) -> bool:
    """
    Test that model can be loaded from checkpoint directory.
    
    Returns:
        True if loading successful
    """
    try:
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(checkpoint_dir, trust_remote_code=True)
        print(f"✓ Model loaded successfully from {checkpoint_dir}")
        
        # Test inference
        import torch
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Inference test passed!")
        print(f"  Output type: {type(output)}")
        if isinstance(output, dict):
            print(f"  Output keys: {output.keys()}")
            print(f"  Logits shape: {output['logits'].shape}")
        else:
            print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", default="./hf_checkpoint")
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    
    if args.test:
        test_model_loading(args.model_dir)
    else:
        config = extract_config_from_checkpoint(args.checkpoint)
        print(f"Extracted config: {config}")
        
        validate_checkpoint(
            args.checkpoint,
            BaselineClassifierConfig(**config),
        )
