"""Spectral-norm wrapping used by SNGP nets.

Dependency-light (only `torch`, plus `src.models.backbones` for the compatibility set) so
this file can be vendored into HF export bundles alongside `backbones.py`.
"""
import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.models.backbones import SPECTRAL_NORM_COMPATIBLE


def apply_spectral_norm(module: nn.Module, n_power_iterations: int = 1) -> None:
    """Recursively wrap Conv2d/Linear layers with spectral normalization. Bias and
    BatchNorm layers are left untouched."""
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            if not hasattr(child, "weight_u"):
                setattr(module, name, spectral_norm(child, n_power_iterations=n_power_iterations))
        else:
            apply_spectral_norm(child, n_power_iterations=n_power_iterations)


def assert_spectral_norm_compatible(arch: str) -> None:
    """Raise if `arch` is known to be incompatible with recursive spectral-norm wrapping.

    ViT backbones are excluded: SNGP's spectral normalization has not been validated
    against ViT internals (LayerNorm/attention), unlike resnet's Conv2d/Linear layers.
    """
    if arch not in SPECTRAL_NORM_COMPATIBLE:
        raise ValueError(
            f"Spectral normalization (SNGP) is not supported for backbone '{arch}'. "
            f"Supported backbones: {sorted(SPECTRAL_NORM_COMPATIBLE)}"
        )
