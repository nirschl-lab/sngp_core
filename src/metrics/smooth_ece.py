#!/usr/bin/env python3
"""smooth_ece.py in src/metrics.

Smoothed Expected Calibration Error  (Błasiok & Nakkiran 2023).
"""

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveFloat, field_validator
from pydantic.dataclasses import dataclass
from scipy.special import logit
from scipy.stats import norm


@dataclass
class SmoothECEInput:
    """Validate inputs for SmoothECE."""

    p: Any
    y: Any
    sigma: PositiveFloat  # ensures sigma > 0 automatically

    @field_validator("p", "y", mode="before")
    @classmethod
    def to_numpy(cls, v):
        """Convert input lists/tuples/arrays into 1D numpy float64 arrays."""
        return np.asarray(v, dtype=np.float64).reshape(-1)

    @field_validator("y")
    @classmethod
    def check_arrays(cls, y: np.ndarray, info):
        """Check shape consistency and non-empty arrays."""
        p = info.data.get("p")
        if p is not None:
            if p.size == 0 or y.size == 0:
                raise ValueError("Input arrays must be non-empty.")
            if p.shape != y.shape:
                raise ValueError(f"Shape mismatch: p {p.shape}, y {y.shape}")
            if np.any((p < 0) | (p > 1)):
                raise ValueError("Probabilities p must be in [0, 1].")
            # Clip probabilities for stability
            info.data["p"] = np.clip(p, 1e-6, 1 - 1e-6)

        return y


def smoothed_ece_logit(
    p: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma: float = 0.1,
    n_grid: int = 1000,
    eps: float = 1e-6,
) -> float:
    """Compute logit-smoothed Expected Calibration Error (SmoothECE)."""
    validated = SmoothECEInput(p=p, y=y, sigma=sigma)
    p, y, sigma = validated.p, validated.y, validated.sigma

    logit_p = logit(p)
    residuals = p - y

    # Evaluation grid
    t = np.linspace(eps, 1 - eps, n_grid)
    logit_t = logit(t)

    # Gaussian kernel smoothing in logit space
    logger.debug(f"Using sigma={sigma} for logit-space smoothing.")
    K = norm.pdf((logit_t[:, None] - logit_p[None, :]) / sigma)

    weighted_residuals = K @ residuals
    density = K @ np.ones_like(residuals)

    smoothed_error = np.divide(
        weighted_residuals,
        density + eps,
        out=np.zeros_like(weighted_residuals),
        where=density > 0,
    )

    return float(np.mean(np.abs(smoothed_error)))


def smoothed_ece_logit_search(
    p: NDArray[np.float64],
    y: NDArray[np.float64],
    eps: float = 1e-3,
    n_steps: int = 20,
    search_range: Optional[Tuple[float, float]] = None,
    return_bandwidth: bool = False,
) -> float | Tuple[float, float]:
    """
    Adaptive bandwidth search for SmoothECE.

    Finds sigma* such that smECE_sigma*(p,y) ≈ sigma* (fixed-point).
    """
    # Validate inputs with minimal sigma
    validated = SmoothECEInput(p=p, y=y, sigma=1e-3)
    p, y = validated.p, validated.y

    lo, hi = (1.0, 1e-3) if search_range is None else search_range
    best_val, best_sigma = float("inf"), hi

    for _ in range(n_steps):
        sigma = (lo + hi) / 2
        val = smoothed_ece_logit(p, y, sigma=sigma)
        if val < best_val:
            best_val, best_sigma = val, sigma
        if val < eps:
            hi = sigma
        else:
            lo = sigma

    return (best_val, best_sigma) if return_bandwidth else best_val
