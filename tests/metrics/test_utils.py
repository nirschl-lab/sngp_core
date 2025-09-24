#!/usr/bin/env python3
"""test_reliability.py in tests/metrics."""

import numpy as np
import pytest

from src.metrics.utils import _bootstrap_ci_width


def dummy_metric(f, y, scale=1.0):
    # Simple metric: mean absolute error scaled
    return float(np.mean(np.abs(f - y)) * scale)


@pytest.mark.parametrize(
    "f, y, func, n_resamples, confidence, seed, kwargs, expected_type, id",
    [
        (
            [0.1, 0.5, 0.9],
            [0, 1, 1],
            dummy_metric,
            100,
            0.95,
            42,
            {},
            float,
            "simple_arrays",
        ),
        (
            np.array([0.2, 0.8]),
            np.array([1, 0]),
            dummy_metric,
            50,
            0.99,
            123,
            {},
            float,
            "numpy_arrays_high_conf",
        ),
        (
            [0.1, 0.5, 0.9],
            [0, 1, 1],
            dummy_metric,
            100,
            0.95,
            42,
            {"scale": 2.0},
            float,
            "with_kwargs",
        ),
        ((0.3, 0.7), (0, 1), dummy_metric, 20, 0.9, 7, {}, float, "tuple_inputs"),
        ([0.0, 1.0], [0, 1], dummy_metric, 10, 0.8, 0, {}, float, "boundaries"),
        (
            [0.1, 0.5, 0.9],
            [0, 1, 1],
            dummy_metric,
            1,
            0.95,
            42,
            {},
            float,
            "single_resample",
        ),
    ],
    ids=[
        "simple_arrays",
        "numpy_arrays_high_conf",
        "with_kwargs",
        "tuple_inputs",
        "boundaries",
        "single_resample",
    ],
)
def test_bootstrap_ci_width_happy_and_edge(
    f, y, func, n_resamples, confidence, seed, kwargs, expected_type, id
):
    # Act
    result = _bootstrap_ci_width(
        np.array(f),
        np.array(y),
        func,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        **kwargs
    )

    # Assert
    assert isinstance(result, expected_type)
    assert np.isfinite(result)
    assert result >= 0


@pytest.mark.parametrize(
    "f, y, func, n_resamples, confidence, seed, kwargs, expected_exception, match, id",
    [
        (
            [0.1, 0.5],
            [0, 1],
            42,
            10,
            0.95,
            0,
            {},
            ValueError,
            "func must be a callable function",
            "func_not_callable",
        ),
        (
            [0.1, 0.5],
            [0],
            dummy_metric,
            10,
            0.95,
            0,
            {},
            IndexError,
            "f and y must have the same shape",
            "shape_mismatch",
        ),
        (
            [],
            [],
            dummy_metric,
            10,
            0.95,
            0,
            {},
            ValueError,
            "f and y must not be empty",
            "empty_arrays",
        ),
        (
            [0.1, 0.5],
            [0, 1],
            dummy_metric,
            0,
            0.95,
            0,
            {},
            ValueError,
            "n_resamples must be at least 1",
            "zero_resamples",
        ),
        (
            [0.1, 0.5],
            [0, 1],
            dummy_metric,
            10,
            1.5,
            0,
            {},
            ValueError,
            "confidence must be in \(0, 1\)",
            "confidence_out_of_bounds",
        ),
        (
            [0.1, 0.5],
            [0, 1],
            dummy_metric,
            10,
            0.0,
            0,
            {},
            ValueError,
            "confidence must be in \(0, 1\)",
            "confidence_too_low",
        ),
    ],
    ids=[
        "func_not_callable",
        "shape_mismatch",
        "empty_arrays",
        "zero_resamples",
        "confidence_out_of_bounds",
        "confidence_too_low",
    ],
)
def test_bootstrap_ci_width_errors(
    f, y, func, n_resamples, confidence, seed, kwargs, expected_exception, match, id
):
    # Act & Assert
    if match:
        with pytest.raises(expected_exception, match=match):
            _bootstrap_ci_width(
                np.array(f),
                np.array(y),
                func,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
                **kwargs
            )
    else:
        with pytest.raises(expected_exception):
            _bootstrap_ci_width(
                np.array(f),
                np.array(y),
                func,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
                **kwargs
            )
