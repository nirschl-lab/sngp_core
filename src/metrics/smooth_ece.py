#!/usr/bin/env python3
"""smooth_ece.py in src/metrics.

Smoothed Expected Calibration Error  (Błasiok & Nakkiran 2023).
"""
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
from loguru import logger
from pydantic import PositiveFloat, PositiveInt


def _smooth_round_to_grid(
    f: np.ndarray, v: np.ndarray, eval_points: PositiveInt
) -> np.ndarray:
    """
    Maps values onto a uniform grid using linear interpolation.

    This function distributes the values `v` at positions `f` onto a uniform grid of length `eval_points`
    using linear 'splatting', ensuring smooth assignment between neighboring bins.

    Args:
        f: Array of positions in [0, 1], shape (N,).
        v: Array of values to distribute, shape (N,).
        eval_points: Number of grid points (must be at least 2).

    Returns:
        A numpy array of shape (eval_points,) with the distributed values.

    Raises:
        ValueError: If eval_points is less than 2.
        AssertionError: If input arrays do not have the same shape.
    """
    # Linear 'splatting' of values v at positions f onto a uniform grid.
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
    """
    Generates a 1D Gaussian kernel centered at 0.5 over the interval [0, 1].

    This function returns a normalized array of length `m` representing samples of a Gaussian
    with standard deviation `sigma`, centered at 0.5, evaluated uniformly over [0, 1].

    Args:
        sigma: Standard deviation of the Gaussian kernel (must be positive).
        m: Number of points in the kernel (must be at least 1).

    Returns:
        A numpy array of shape (m,) containing the Gaussian kernel samples.

    Raises:
        ZeroDivisionError: If sigma is not positive.
        ValueError: If m is less than 1.
    """
    # Gaussian kernel samples, centered at 0.5 on [0,1]
    if sigma <= 0:
        raise ZeroDivisionError("sigma must be positive.")

    if m < 1:
        raise ValueError("m must be at least 1.")

    t = np.linspace(0, 1, m)
    return np.exp(-0.5 * (t - 0.5) ** 2 / (sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def _smooth_ece_interpolated(r_grid: np.ndarray, sigma: PositiveFloat) -> float:
    """
    Computes the smoothed Expected Calibration Error (smECE) for a given grid and smoothing parameter.

    This function applies a Gaussian kernel and reflected convolution to the input grid,
    then returns the mean absolute value as the smoothed calibration error.

    Args:
        r_grid: Input array representing the residual grid.
        sigma: Smoothing parameter (must be positive).

    Returns:
        The smoothed ECE value as a float.

    Raises:
        ZeroDivisionError: If sigma is not positive.
    """
    if sigma <= 0:
        raise ZeroDivisionError("sigma must be positive.")

    ker = _gaussian_kernel_1d(sigma, len(r_grid))
    rs = _reflected_convolve(r_grid, ker)
    return float(np.sum(np.abs(rs)) / len(r_grid))


def _reflected_convolve(values: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """Performs a convolution of the input array with a kernel using reflection at the boundaries.

    This function extends the input array by reflecting its values at both ends to avoid edge effects,
    then applies a 1D convolution with the provided kernel. It returns the central part of the result
    with the same length as the input array.

    Args:
        values: Input array to be convolved.
        ker: 1D kernel array for convolution.

    Returns:
        The convolved array, with the same shape as the input values.

    Raises:
        ValueError: If the kernel or values are empty, or if the kernel is longer than the values.
    """
    if len(ker) < 1:
        raise ValueError("Kernel length must be at least 1.")
    elif len(ker) == 1:
        logger.warning("Kernel length is 1; convolution is a no-op.")
        return values.copy()

    if len(values) < 1:
        raise ValueError("Values length must be at least 1.")

    if len(values) < len(ker):
        raise ValueError("Kernel length must be at least as long as values length.")

    m = len(values)
    # correct reflection: flip without double-counting edges
    ext = np.concatenate([np.flip(values)[:-1], values, np.flip(values)[1:]])
    conv = np.convolve(ext, ker, mode="valid")
    return conv[m // 2 : m // 2 + m]


def _smece_binary_core(
    f: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-3,
    m: PositiveInt = 200,
) -> Tuple[float, float]:
    """smECE Binary classification: returns (value, sigma)."""
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

    # Binary search
    start, end = 1.0, 0.0
    sigma = start
    for _ in range(10):
        mid = 0.5 * (start + end)
        if _check_smooth_ece(mid):
            end, sigma = mid, mid
        else:
            start = mid

    val = _smooth_ece_interpolated(r_grid, sigma)
    return float(val), float(sigma)


def smECE_fast_compat(
    f: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-3,
    m: PositiveInt = 200,
    return_width: bool = False,
    *,
    average: Literal["macro", "weighted"] = "macro",
    return_details: bool = False,
) -> Union[
    float,
    Tuple[float, float],
    Dict[str, Any],
]:
    """
    Smoothed ECE in probability space (binary *and* multiclass).

    Parameters
    ----------
    f : np.ndarray
        - Binary: shape (n,), predicted probabilities for the positive class.
        - Multiclass: shape (n, C), class probabilities over C classes.
    y : np.ndarray
        - Binary: shape (n,), 0/1 labels.
        - Multiclass: shape (n,), integer labels in {0, …, C-1}.
    eps : float
        Fixed-point search threshold for bandwidth.
    m : int
        Initial grid size for discretization (refined automatically).
    return_width : bool
        If True (binary): return (value, sigma).
        If True (multiclass): return (value, sigmas_per_class_mean)  # kept for backward compat.
    average : {"macro","weighted"}
        Aggregation of per-class smECE for multiclass.
    return_details : bool
        If True (multiclass): return a dict with rich fields.

    Returns
    -------
    - Binary:
        value : float
        or (value, sigma) if return_width.
    - Multiclass:
        default: value : float
        if return_width: (value, sigma_bar)  # mean bandwidth just to preserve tuple type
        if return_details: dict with
            {
              "value": float,
              "per_class": np.ndarray[C],
              "sigmas": np.ndarray[C],
              "weights": np.ndarray[C],
              "average": "macro" or "weighted"
            }

    Notes
    -----
    - Multiclass is computed via one-vs-rest reduction:
        f_c = P(Y=c),  y_c = 1{y==c}
      Then aggregate per-class smECE by macro-mean or label-frequency weighting.
    - We do NOT normalize by density (faithful to Apple fast path).
    """
    f = np.asarray(f)
    y = np.asarray(y)

    # --- Binary classification---
    if f.ndim == 1:
        val, sigma = _smece_binary_core(f, y, eps=eps, m=m)
        if return_width:
            return val, sigma
        return val

    # --- Multiclass classification ---
    if f.ndim != 2:
        raise ValueError("f must be 1D (binary) or 2D (n, C) for multiclass.")

    n, C = f.shape
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(
            "For multiclass, y must be shape (n,) with integer class labels."
        )
    y_int = y.astype(int).reshape(-1)
    if (y_int < 0).any() or (y_int >= C).any():
        logger.debug(f"y unique labels: {np.unique(y_int)}")
        logger.debug(f"C={C}")
        raise ValueError("y contains class indices outside [0, C-1].")

    per_class_vals = np.zeros(C, dtype=float)
    per_class_sigmas = np.zeros(C, dtype=float)
    class_weights = np.zeros(C, dtype=float)

    for c in range(C):
        f_c = np.asarray(f[:, c], float).reshape(-1)
        y_c = (y_int == c).astype(float).reshape(-1)
        per_class_vals[c], per_class_sigmas[c] = _smece_binary_core(
            f_c, y_c, eps=eps, m=m
        )
        class_weights[c] = float(y_c.mean())  # prevalence

    if average == "macro":
        value = float(per_class_vals.mean())
    elif average == "weighted":
        if class_weights.sum() == 0:
            value = float(per_class_vals.mean())
        else:
            value = float(np.average(per_class_vals, weights=class_weights))
    else:
        raise ValueError("average must be 'macro' or 'weighted'.")

    if return_details:
        return {
            "value": value,
            "per_class": per_class_vals,
            "sigmas": per_class_sigmas,
            "weights": class_weights,
            "average": average,
        }

    if return_width:
        # For backward compat with binary tuple return, provide an aggregate sigma.
        # We use a weighted mean of sigmas under the same 'average' policy.
        if average == "macro" or class_weights.sum() == 0:
            sigma_bar = float(per_class_sigmas.mean())
        else:
            sigma_bar = float(np.average(per_class_sigmas, weights=class_weights))
        return value, sigma_bar

    return value
