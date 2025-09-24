#!/usr/bin/env python3
"""utils.py in src/metrics."""

from typing import Any

import numpy as np
from pydantic import PositiveInt


def _bootstrap_ci_width(
    f: np.ndarray,
    y: np.ndarray,
    func: Any,
    n_resamples: PositiveInt = 200,
    confidence: float = 0.95,
    seed: int = 0,
    **kwargs,
) -> float:
    """Bootstrap confidence interval width for a calibration error metric."""
    # func must be function
    if not callable(func):
        raise ValueError("func must be a callable function")

    # f and y must have same shape
    if f.shape != y.shape:
        raise IndexError("f and y must have the same shape")
    if len(f) == 0:
        raise ValueError("f and y must not be empty")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    f = np.asarray(f, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    rng = np.random.default_rng(seed)
    vals = []
    n = len(f)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals.append(func(f[idx], y[idx], **kwargs))
    vals = np.sort(vals)
    lo = np.percentile(vals, (1 - confidence) / 2 * 100)
    hi = np.percentile(vals, (1 + confidence) / 2 * 100)
    return float(max(hi - vals.mean(), vals.mean() - lo))
