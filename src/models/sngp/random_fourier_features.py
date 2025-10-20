#!/usr/bin/env python3
"""random_fourier_features.py in src/sngp_core/models/sngp.

Implements unscaled Random Fourier Features for Gaussian-process layers. Scaling
is applied externally in the GP layer, following the Edward2 implementation.
Based on:
- Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel
  Machines. In _Neural Information Processing Systems_, 2007.
  https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

Adapted from:
https://github.com/Jmkernes/Spectral-Normalized-Gaussian-Process/blob/main/random_fourier_features.py
https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/gaussian_process.py
https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py
"""


import math
import os
import torch
from torch import nn
from loguru import logger

_SUPPORTED_RBF_KERNEL_TYPES = ["gaussian", "laplacian"]


class RandomFourierFeatures(nn.Module):
    """Random Fourier Feature (RFF) mapping for shift-invariant kernels.

    This layer approximates a translation-invariant kernel K(x, y) ≈ φ(x)ᵀφ(y),
    using a random cosine feature map following Rahimi & Recht (2007).

    **Important:** This version returns *unscaled* features, consistent with the
    TensorFlow Edward2 implementation, where the Rahimi–Recht scaling factor
    √(2 / num_features) is applied externally (in the GP layer).

    Args:
        in_features: Dimensionality of the input features.
        out_features: Number of random Fourier features (num_inducing in GP).
        kernel_type: Either 'gaussian' or 'laplacian'.
        kernel_scale: Length-scale parameter ℓ of the kernel.
        trainable_kernel_scale: Whether ℓ is a trainable parameter.
        use_softplus: Ensures positive length-scale via softplus (optional).
        verbose: Enables debug logging.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_type: str = "gaussian",
        kernel_scale: float = 1.0,
        trainable_kernel_scale: bool = False,
        use_softplus: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        if out_features <= 0:
            raise ValueError(f"`out_features` must be > 0. Given: {out_features}")
        if kernel_type.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
            raise ValueError(
                f"Unsupported kernel type: '{kernel_type}'. Supported types: {_SUPPORTED_RBF_KERNEL_TYPES}."
            )

        if kernel_scale is not None and kernel_scale <= 0.0:
            raise ValueError(
                f"`kernel_scale` must be a positive float. Given: {kernel_scale}"
            )

        self.verbose = bool(verbose or os.getenv("VERBOSE") or os.getenv("DEBUG"))
        self.use_softplus = use_softplus
        self.kernel_type = kernel_type.lower()
        self.in_features = in_features
        self.out_features = out_features

        # === Random weights and biases ===
        if self.kernel_type == "gaussian":
            # Matches Edward2: orthogonalized Gaussian features
            weight = torch.randn(in_features, out_features) / math.sqrt(in_features)
        elif self.kernel_type == "laplacian":
            weight = torch.tan(math.pi * (torch.rand(in_features, out_features) - 0.5))
        else:
            raise ValueError("Unsupported kernel_type: use 'gaussian' or 'laplacian'.")
        self.register_buffer("weight", weight)
        self.register_buffer("bias", torch.rand(out_features) * 2 * math.pi)

        # === Kernel scale (lengthscale) ===
        if trainable_kernel_scale:
            self.kernel_scale = nn.Parameter(torch.tensor(kernel_scale, dtype=torch.float32))
        else:
            self.register_buffer("kernel_scale", torch.tensor(kernel_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes φ(x) = cos(x·(W/ℓ) + b), unscaled."""
        if self.use_softplus:
            kernel_scale = torch.nn.functional.softplus(self.kernel_scale) + 1e-6
        else:
            kernel_scale = torch.nn.functional.relu(self.kernel_scale) + 1e-6

        # ✅ Scale frequencies, not inputs
        weight_scaled = self.weight / kernel_scale

        if x.device != weight_scaled.device:
            weight_scaled = weight_scaled.to(x.device)
            bias = self.bias.to(x.device)
        else:
            bias = self.bias

        features = torch.cos(x @ weight_scaled + bias)

        if self.verbose:
            logger.debug(
                f"RFF.forward: x.shape={x.shape}, W.shape={self.weight.shape}, "
                f"ℓ={kernel_scale.item():.3f}, out.shape={features.shape}"
            )

        return features

