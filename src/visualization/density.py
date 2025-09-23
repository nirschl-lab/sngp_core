#!/usr/bin/env python3
"""density.py in src/visualization.

Minimal kernels and smoothers for calibration plots.
"""

from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def _gaussian(u: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def reflected_kde(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Kernel density on [0,1] via reflection (to reduce boundary bias)."""
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
    """Nadaraya–Watson E[y|x≈t] with Gaussian kernel; reflected by default on [0,1]."""
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
    """Sample x-positions for rug ticks, proportional to density over the grid."""
    d = np.maximum(density.reshape(-1), 0.0)
    if d.sum() == 0:
        return np.linspace(0, 1, n_ticks)
    p = d / d.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grid), size=n_ticks, p=p, replace=True)
    return grid[idx]


# old code

