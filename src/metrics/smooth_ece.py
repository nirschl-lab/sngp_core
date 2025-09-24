#!/usr/bin/env python3
"""smooth_ece.py in src/metrics.

Smoothed Expected Calibration Error  (Błasiok & Nakkiran 2023).
"""
import numpy as np
from pydantic import PositiveFloat, PositiveInt
from loguru import logger



def _smooth_round_to_grid(f: np.ndarray, v: np.ndarray, eval_points: PositiveInt) -> np.ndarray:
    """Linear 'splatting' of values v at positions f onto a uniform grid."""
    if eval_points < 2:
        raise ValueError("eval_points must be at least 2.")

    f = np.asarray(f, float).clip(0, 1).reshape(-1)
    v = np.asarray(v, float).reshape(-1)
    assert f.shape == v.shape
    out = np.zeros(eval_points, float)
    # map f to [0, eval_points-1] bins with linear interpolation to neighbors
    x = f * (eval_points - 1)
    lo = np.floor(x).astype(int).clip(0, eval_points - 2)
    hi = lo + 1
    w_hi = x - lo
    w_lo = 1.0 - w_hi
    np.add.at(out, lo, w_lo * v)
    np.add.at(out, hi, w_hi * v)
    return out


def _gaussian_kernel_1d(sigma: PositiveFloat, m: PositiveInt) -> np.ndarray:
    """Gaussian kernel samples, centered at 0.5 on [0,1]."""
    if sigma <= 0:
        raise ZeroDivisionError("sigma must be positive.")

    if m < 1:
        raise ValueError("m must be at least 1.")

    t = np.linspace(0, 1, m)
    return np.exp(-0.5 * (t - 0.5) ** 2 / (sigma**2)) / (np.sqrt(2 * np.pi) * sigma)



def _smooth_ece_interpolated(r_grid: np.ndarray, sigma: PositiveFloat) -> float:
    if sigma <= 0:
        raise ZeroDivisionError("sigma must be positive.")

    ker = _gaussian_kernel_1d(sigma, len(r_grid))
    rs = _reflected_convolve(r_grid, ker)
    return float(np.sum(np.abs(rs)) / len(r_grid))


def _reflected_convolve(values: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """Convolution with reflection."""
    if len(ker) < 1:
        raise ValueError("Kernel length must be at least 1.")
    elif len(ker) == 1:
        logger.warning("Kernel length is 1; convolution is a no-op.")
        return values.copy()

    if len(values) < 1:
        raise ValueError("Values length must be at least 1.")

    if  len(values) < len(ker):
        raise ValueError("Kernel length must be at least as long as values length.")

    m = len(values)
    # correct reflection: flip without double-counting edges
    ext = np.concatenate([np.flip(values)[:-1], values, np.flip(values)[1:]])
    conv = np.convolve(ext, ker, mode="valid")
    return conv[m // 2 : m // 2 + m]


def smECE_fast_compat(
    f: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-3,
    m: PositiveInt = 200,
    return_width: bool = False,
) -> float | tuple[float, float]:
    """Reimplementation of smECE_fast in probability space."""
    f = np.asarray(f, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    assert f.shape == y.shape and f.size > 0

    # coarse grid
    m = 200 if m < 200 else m
    r_grid = _smooth_round_to_grid(f, f - y, eval_points=m) / f.size

    def _maybe_refine(alpha: float):
        nonlocal m, r_grid
        while round(20.0 / alpha) > m:
            m *= 4
            r_grid = _smooth_round_to_grid(f, f - y, eval_points=m) / f.size

    def _check_smooth_ece(alpha: float) -> bool:
        _maybe_refine(alpha)
        return (alpha < eps) or (alpha < _smooth_ece_interpolated(r_grid, alpha))

    # Binary search like orig search_param
    start, end = 1.0, 0.0
    sigma = start
    for _ in range(10):
        mid = 0.5 * (start + end)
        if _check_smooth_ece(mid):
            end, sigma = mid, mid
        else:
            start = mid

    val = _smooth_ece_interpolated(r_grid, sigma)
    return (val, float(sigma)) if return_width else val
