#!/usr/bin/env python3
"""
Standalone example: loading and using a checkpoint exported by
scripts/hf/export_to_hub.py (generic across baseline/SNGP -- the export bundle is
self-describing via config.json's auto_map, so `AutoModel.from_pretrained` resolves
the right wrapper class automatically).

Requirements:
    pip install torch transformers torchvision
"""

import torch
from pathlib import Path
from transformers import AutoModel


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n1. Loading model from checkpoint...")
    checkpoint_dir = Path("./hf_checkpoint")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}. "
            "Run scripts/hf/export_to_hub.py --ckpt <path> --out ./hf_checkpoint first."
        )

    model = AutoModel.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")

    print("\n2. Preparing input...")
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
    print(f"✓ Input shape: {input_tensor.shape}")

    print("\n3. Running inference...")
    with torch.no_grad():
        output = model(input_tensor)

    print("\n4. Processing outputs...")
    logits = output["logits"]  # (batch_size, num_classes)
    variance = output.get("variance")  # (batch_size, 1), SNGP only

    probabilities = torch.softmax(logits, dim=1)

    print(f"\n5. Results for batch of {batch_size} images:")
    print("=" * 70)
    for i in range(batch_size):
        print(f"\nSample {i + 1}:")
        print(f"  Logits:               {logits[i].cpu().numpy()}")
        print(f"  Probabilities:        {probabilities[i].cpu().numpy()}")
        print(f"  Predicted class:      {probabilities[i].argmax().item()}")
        print(f"  Confidence (max prob):{probabilities[i].max().item():.4f}")
        if variance is not None:
            print(f"  Uncertainty (var):    {variance[i].item():.4f}")
    print("=" * 70)

    if variance is not None:
        print("\n6. Using uncertainty estimates:")
        threshold = variance.median().item()
        confident_mask = variance.squeeze() < threshold
        print(f"  Median uncertainty: {threshold:.4f}")
        print(f"  Confident predictions: {confident_mask.sum().item()}/{batch_size}")
        print(f"  - Indices: {torch.where(confident_mask)[0].tolist()}")


if __name__ == "__main__":
    print("=" * 70)
    print("HuggingFace Export Checkpoint Inference Example")
    print("=" * 70)
    main()
