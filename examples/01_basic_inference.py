"""
Basic inference example: load a checkpoint and make predictions on images.

Update CHECKPOINT_PATH / IMAGE_PATH below before running:

    uv run examples/01_basic_inference.py
"""

import logging
from pathlib import Path

from src.inference.predict_image import predict_batch, predict_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "logs/train/runs/<your-run>/checkpoints/best.ckpt"
IMAGE_PATH = "path/to/image.jpg"
IMAGE_DIR = "path/to/image/directory"


def example_single_image():
    """Predict a single image, printing the predicted class and confidence."""
    logger.info("=== Single Image Inference ===")

    result = predict_image(CHECKPOINT_PATH, IMAGE_PATH)

    logger.info(f"Predicted class: {result['predicted_class']}")
    logger.info(f"Confidence: {result['confidence']:.4f}")
    logger.info(f"Class probabilities: {result['probs']}")
    return result


def example_batch_inference():
    """Predict every .jpg/.png in IMAGE_DIR."""
    logger.info("=== Batch Inference ===")

    image_paths = list(Path(IMAGE_DIR).glob("*.jpg")) + list(Path(IMAGE_DIR).glob("*.png"))
    logger.info(f"Found {len(image_paths)} images")

    results = predict_batch(CHECKPOINT_PATH, image_paths, batch_size=32)

    logger.info(f"Predicted classes: {results['predictions']}")
    logger.info(f"Mean confidence: {results['probs'].max(axis=1).mean():.4f}")
    return results


def example_uncertainty_estimation():
    """MC-Dropout uncertainty for a single image (baseline/MC-dropout checkpoints only
    -- `net.mc_predict` isn't defined for SNGP or Deep Ensemble nets; those get their
    uncertainty from the SNGP predictive variance or ensemble disagreement instead,
    see docs/models/DEEP_ENSEMBLES_GUIDE.md)."""
    logger.info("=== MC-Dropout Uncertainty Estimation ===")

    result = predict_image(CHECKPOINT_PATH, IMAGE_PATH, use_mc_dropout=True, mc_passes=25)

    logger.info(f"Predicted class: {result['predicted_class']} (confidence {result['confidence']:.4f})")
    logger.info(f"Predictive uncertainty (mean std across classes): {result['uncertainty']:.4f}")
    return result


if __name__ == "__main__":
    # Update CHECKPOINT_PATH / IMAGE_PATH / IMAGE_DIR above, then uncomment:

    # example_single_image()
    # example_batch_inference()
    # example_uncertainty_estimation()

    logger.info("Update CHECKPOINT_PATH/IMAGE_PATH/IMAGE_DIR above and uncomment an example to run.")
