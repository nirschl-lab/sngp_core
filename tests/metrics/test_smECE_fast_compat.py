#!/usr/bin/env python3
"""test_smECE_fast_compat.py in tests/metrics."""

import numpy as np
import pytest

from src.metrics.smooth_ece import (_smooth_ece_interpolated,
                                    _smooth_round_to_grid, smECE_fast_compat)


# Dummy implementations for dependencies (for isolated testing)
def _mock_smooth_round_to_grid(f, v, eval_points):
    # Simple histogram-like binning for test
    out = np.zeros(eval_points, float)
    if len(f) == 0:
        return out
    x = np.clip(f, 0, 1) * (eval_points - 1)
    lo = np.floor(x).astype(int).clip(0, eval_points - 2)
    hi = lo + 1
    w_hi = x - lo
    w_lo = 1.0 - w_hi
    np.add.at(out, lo, w_lo * v)
    np.add.at(out, hi, w_hi * v)
    return out


def _mock_smooth_ece_interpolated(r_grid, sigma):
    # Just return the mean absolute value for test
    return float(np.sum(np.abs(r_grid)) / len(r_grid))


# Patch the dependencies for isolated testing
import sys

sys.modules["src.metrics.smooth_ece"]._smooth_round_to_grid = _mock_smooth_round_to_grid
sys.modules["src.metrics.smooth_ece"]._smooth_ece_interpolated = (
    _mock_smooth_ece_interpolated
)


@pytest.mark.parametrize(
    "f, y, eps, return_width, expected_type, id",
    [
        ([0.1, 0.5, 0.9], [0, 1, 1], 1e-3, False, float, "valid_fl"),
        ([0.1, 0.5, 0.9], [0, 1, 1], 1e-3, True, tuple, "valid_tup"),
        (np.array([0.2, 0.8]), np.array([1, 0]), 1e-2, False, float, "numpy_arr"),
        ((0.3, 0.7), (0, 1), 1e-2, False, float, "tuple_inp"),
        ([0.0, 1.0], [0, 1], 1e-2, False, float, "bound"),
    ],
    ids=["valid_fl", "valid_tup", "numpy_arr", "tuple_inp", "bound"],
)
def test_smECE_fast_compat_happy_and_edge(
    monkeypatch, f, y, eps, return_width, expected_type, id
):
    # Arrange
    monkeypatch.setattr(
        "src.metrics.smooth_ece._smooth_round_to_grid", _smooth_round_to_grid
    )
    monkeypatch.setattr(
        "src.metrics.smooth_ece._smooth_ece_interpolated", _smooth_ece_interpolated
    )

    # Act
    result = smECE_fast_compat(f, y, eps=eps, return_width=return_width)

    # Assert
    if return_width:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
    else:
        assert isinstance(result, float)


@pytest.mark.parametrize(
    "f, y, eps, return_width, expected_exception, id",
    [
        ([0.1, 0.2], [1], 1e-3, False, AssertionError, "shape_mismatch"),
        ([], [], 1e-3, False, AssertionError, "empty_input"),
        ([], [1, 0], 1e-3, False, AssertionError, "f_empty"),
        ([0.1, 0.2], [], 1e-3, False, AssertionError, "y_empty"),
    ],
    ids=["shape_mismatch", "empty_input", "f_empty", "y_empty"],
)
def test_smECE_fast_compat_errors(
    monkeypatch, f, y, eps, return_width, expected_exception, id
):
    # Arrange
    monkeypatch.setattr(
        "src.metrics.smooth_ece._smooth_round_to_grid", _smooth_round_to_grid
    )
    monkeypatch.setattr(
        "src.metrics.smooth_ece._smooth_ece_interpolated", _smooth_ece_interpolated
    )

    # Act & Assert
    with pytest.raises(expected_exception):
        smECE_fast_compat(f, y, eps=eps, return_width=return_width)
