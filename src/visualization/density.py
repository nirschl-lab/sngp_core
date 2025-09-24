#!/usr/bin/env python3
"""density.py in src/visualization.

Minimal kernels and smoothers for calibration plots.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _gaussian(u: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def reflected_kde(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """
    Estimates a kernel density on [0,1] using reflection to reduce boundary bias.

    This function computes a kernel density estimate for the given samples over the specified grid,
    applying Gaussian kernels and reflecting samples at the boundaries to improve accuracy near the edges.

    Args:
        samples: Input data samples, expected in [0, 1].
        grid: Grid points at which to evaluate the density.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        An array of density values evaluated at each grid point.
    """
    # Kernel density on [0,1] via reflection (to reduce boundary bias).
    s = np.clip(samples.reshape(-1), 0.0, 1.0)
    g = np.clip(grid.reshape(-1), 0.0, 1.0)
    if s.size == 0:
        return np.zeros_like(g)

    # reflections around 0 and 1
    diffs = (g[:, None] - s[None, :]) / sigma
    diffs_l = (g[:, None] - (-s)[None, :]) / sigma
    diffs_r = (g[:, None] - (2.0 - s)[None, :]) / sigma

    dens = (_gaussian(diffs) + _gaussian(diffs_l) + _gaussian(diffs_r)).sum(axis=1)
    dens = np.maximum(dens, 0.0)
    # normalize to integrate ~ 1 over grid spacing
    area = dens.sum()
    if area > 0:
        dens = dens / area
    return dens


def nadaraya_watson(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    grid: NDArray[np.float64],
    sigma: float,
    boundary: Literal["reflected", "plain"] = "reflected",
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Estimates the conditional expectation E[y|x≈t] using the Nadaraya–Watson estimator with a Gaussian kernel.

    This function computes a smooth estimate of the expected value of y given x, evaluated at each point in the grid.
    It supports both reflected and plain boundary handling for the kernel.

    Args:
        x: Input data points for the independent variable.
        y: Input data points for the dependent variable.
        grid: Grid points at which to evaluate the conditional expectation.
        sigma: Standard deviation of the Gaussian kernel.
        boundary: Boundary handling mode, either "reflected" or "plain".
        eps: Small value to avoid division by zero.

    Returns:
        An array of estimated conditional expectations at each grid point.
    """
    # Nadaraya–Watson E[y|x≈t] with Gaussian kernel; reflected by default on [0,1].
    x = x.reshape(-1)
    y = y.reshape(-1)
    g = grid.reshape(-1)

    if boundary == "reflected":
        diffs = (g[:, None] - x[None, :]) / sigma
        diffs_l = (g[:, None] - (-x)[None, :]) / sigma
        diffs_r = (g[:, None] - (2.0 - x)[None, :]) / sigma
        K = _gaussian(diffs) + _gaussian(diffs_l) + _gaussian(diffs_r)
    else:
        K = _gaussian((g[:, None] - x[None, :]) / sigma)

    num = K @ y
    den = K.sum(axis=1)
    out = np.divide(num, den + eps, out=np.zeros_like(num), where=den > 0)
    return np.clip(out, 0.0, 1.0)


def density_ticks(
    density: NDArray[np.float64],
    grid: NDArray[np.float64],
    n_ticks: int = 200,
    seed: int = 0,
) -> NDArray[np.float64]:
    """
    Samples x-positions for rug ticks proportional to the provided density over the grid.

    This function generates tick positions for visualization by sampling from the grid according to the density distribution.
    If the density is zero everywhere, ticks are placed uniformly.

    Args:
        density: Array of density values corresponding to the grid.
        grid: Array of grid points from which to sample.
        n_ticks: Number of ticks to sample (default 200).
        seed: Random seed for reproducibility (default 0).

    Returns:
        An array of sampled grid positions for rug ticks.
    """
    # Sample x-positions for rug ticks, proportional to density over the grid
    d = np.maximum(density.reshape(-1), 0.0)
    if d.sum() == 0:
        return np.linspace(0, 1, n_ticks)
    p = d / d.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grid), size=n_ticks, p=p, replace=True)
    return grid[idx]