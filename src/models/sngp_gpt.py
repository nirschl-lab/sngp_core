"""Backward-compatible module for older checkpoint imports.

Some older checkpoints were serialized with the class path
``src.models.sngp_gpt``. This shim re-exports the current SNGP lightning module
so those checkpoints can still be loaded.
"""

from src.models.sngp.sngp_classifier import (
    RandomFeatureGaussianProcess,
    SNGPClassifier,
)
from src.models.sngp_classification_lit_module import SNGPClassificationLitModule

# Backward-compatible aliases used by legacy checkpoints.
SNGPGPT = SNGPClassificationLitModule
