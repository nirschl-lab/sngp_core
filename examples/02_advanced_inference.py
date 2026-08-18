"""
Advanced inference examples: HuggingFace Hub models, config-driven inference,
multi-checkpoint ensembling, and dataset-wide inference with saved results.

Update the placeholder paths/IDs in each function before running:

    uv run examples/02_advanced_inference.py
"""

import json
import logging
from pathlib import Path
from typing import List

import torch

from src.checkpointing.io import load_net
from src.inference.predict_image import load_image_as_tensor, predict_batch
from src.models.hf_loader import HFModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_huggingface_model():
    """Load a published model straight from the HF Hub (no local checkpoint needed)."""
    logger.info("=== HuggingFace Hub Model Inference ===")

    repo_id = "nirschl-lab/sngp-models"
    model_name = "wong_sngp_resnet18"  # see README.md for the full published-models list
    image_path = "path/to/image.jpg"

    loader = HFModelLoader(repo_id=repo_id)
    model, config = loader.load_model(model_name)
    model.eval()

    x = load_image_as_tensor(image_path)
    with torch.no_grad():
        probs = torch.softmax(model(x).logits, dim=1)

    logger.info(f"Predicted class: {probs.argmax(dim=1).item()}")
    logger.info(f"Confidence: {probs.max().item():.4f}")
    return probs


def example_config_based_inference():
    """Drive inference parameters from a JSON config instead of hardcoded values."""
    logger.info("=== Config-Based Inference ===")

    config = {
        "ckpt_path": "logs/train/runs/<your-run>/checkpoints/best.ckpt",
        "image_dir": "path/to/image/directory",
        "batch_size": 32,
    }
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    image_paths = list(Path(config["image_dir"]).glob("*.jpg"))
    results = predict_batch(config["ckpt_path"], image_paths, batch_size=config["batch_size"])

    logger.info(f"Predicted {len(results['predictions'])} images")
    return results


def example_multi_checkpoint_ensemble():
    """Average predictions across several trained checkpoints (e.g. different seeds
    or backbones) -- for a proper Deep Ensemble *model* trained end-to-end as one
    unit, see docs/DEEP_ENSEMBLES_GUIDE.md instead; this is for combining independently
    trained checkpoints after the fact."""
    logger.info("=== Multi-Checkpoint Ensemble ===")

    ckpt_paths: List[str] = [
        "logs/train/runs/<run-1>/checkpoints/best.ckpt",
        "logs/train/runs/<run-2>/checkpoints/best.ckpt",
        "logs/train/runs/<run-3>/checkpoints/best.ckpt",
    ]
    image_path = "path/to/image.jpg"

    x = load_image_as_tensor(image_path)
    all_probs = []
    for ckpt_path in ckpt_paths:
        net = load_net(ckpt_path, device="cpu")
        with torch.no_grad():
            probs = torch.softmax(net(x).logits, dim=1)
        all_probs.append(probs)
        logger.info(f"{ckpt_path}: {probs.squeeze().numpy()}")

    ensemble_probs = torch.stack(all_probs).mean(dim=0)
    logger.info(f"Ensemble prediction: {ensemble_probs.argmax(dim=1).item()}")
    logger.info(f"Ensemble confidence: {ensemble_probs.max().item():.4f}")
    return ensemble_probs


def example_inference_on_dataset():
    """Run inference over a directory of images and save results to JSON. For metrics
    (accuracy/ECE/...) against labeled data, use the Hydra entrypoint
    `src/inference/infer.py` instead -- this is for unlabeled images only."""
    logger.info("=== Dataset-Wide Inference ===")

    ckpt_path = "logs/train/runs/<your-run>/checkpoints/best.ckpt"
    image_dir = Path("path/to/image/directory")
    output_file = "predictions.json"

    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} images in {image_dir}")

    results = predict_batch(ckpt_path, image_paths, batch_size=32)

    predictions = {
        "predictions": [
            {"image": str(path), "class_id": int(pred), "confidence": float(probs.max())}
            for path, pred, probs in zip(image_paths, results["predictions"], results["probs"])
        ],
        "summary": {
            "total_images": len(image_paths),
            "mean_confidence": float(results["probs"].max(axis=1).mean()) if len(image_paths) else 0.0,
        },
    }

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved results to {output_file}")
    return predictions


if __name__ == "__main__":
    # Update the placeholder paths/IDs in each function above, then uncomment:

    # example_huggingface_model()
    # example_config_based_inference()
    # example_multi_checkpoint_ensemble()
    # example_inference_on_dataset()

    logger.info("Update the placeholder paths above and uncomment an example to run.")
