"""test_smooth_ece.py in tests/metrics.

Tests for Smoothed Expected Calibration Error (Błasiok & Nakkiran 2023).
"""

import json
from pathlib import Path

# Disable plotting for tests
import matplotlib
import numpy as np
import pandas as pd
import pytest

from src.metrics.smooth_ece import smECE_fast_compat
from src.metrics.utils import _bootstrap_ci_width

matplotlib.use("Agg")

from src.metrics.smooth_ece import (_gaussian_kernel_1d, _reflected_convolve,
                                    _smooth_round_to_grid)


## _smooth_round_to_grid
@pytest.mark.parametrize(
    "f, v, eval_points, expected, id",
    [
        ([0.0, 0.5, 1.0], [1.0, 2.0, 3.0], 5, None, "mid_edges"),
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 4, None, "all_f_zero"),
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], 4, None, "all_f_one"),
        ([0.0, 1 / 3, 2 / 3, 1.0], [1, 2, 3, 4], 4, None, "bin_bound"),
        ([0.0, 0.5, 1.0], [0.0, 0.0, 0.0], 5, np.zeros(5), "v_all_zeros"),
        ([0.0, 0.5, 1.0], [1.0, 2.0, 3.0], 2, np.array([2.0, 4.0]), "eval_points_two"),
        ([], [], 4, np.zeros(4), "empty_input"),
    ],
    ids=[
        "mid_edges",
        "all_f_zero",
        "all_f_one",
        "bin_bound",
        "v_all_zeros",
        "eval_points_one",
        "empty_input",
    ],
)
def test_smooth_round_to_grid_happy_and_edge(f, v, eval_points, expected, id):
    # Act
    result = _smooth_round_to_grid(np.array(f), np.array(v), eval_points)

    # Assert
    if expected is not None:
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)
    else:
        # For cases where expected is None, check sum and shape
        assert result.shape == (eval_points,)
        assert np.isfinite(result).all()
        assert np.all(result >= 0) or np.all(result == 0)


@pytest.mark.parametrize(
    "f, v, eval_points, expected_exception, id",
    [
        ([0.0, 0.5], [1.0], 3, AssertionError, "shape_mismatch"),
        ([0.0, 0.5], [1.0, 2.0], -1, ValueError, "negative_eval_points"),
        ([0.0, 0.5], [1.0, 2.0], 0, ValueError, "zero_eval_points"),
    ],
    ids=["shape_mismatch", "neg_eval_pts", "zero_eval_pts"],
)
def test_smooth_round_to_grid_errors(f, v, eval_points, expected_exception, id):
    # Act & Assert
    with pytest.raises(expected_exception):
        _smooth_round_to_grid(np.array(f), np.array(v), eval_points)


## _gaussian_kernel_1d
@pytest.mark.parametrize(
    "sigma, m, expected_shape, id",
    [
        (0.1, 5, (5,), "std_sig_5"),
        (0.5, 10, (10,), "std_sig_10"),
        (1.0, 1, (1,), "sig1_m1"),
        (0.2, 1000, (1000,), "large_m"),
        (1e-6, 5, (5,), "tiny_sig"),
        (1e6, 5, (5,), "huge_sig"),
    ],
    ids=["std_sig_5", "std_sig_10", "sig1_m1", "large_m", "tiny_sig", "huge_sig"],
)
def test_gaussian_kernel_1d_happy_and_edge(sigma, m, expected_shape, id):
    # Act
    ker = _gaussian_kernel_1d(sigma, m)

    # Assert
    assert isinstance(ker, np.ndarray)
    assert ker.shape == expected_shape
    assert np.all(np.isfinite(ker))
    # For m > 1, max should be at or near the center
    if m > 1:
        center_idx = m // 2
        assert np.argmax(ker) in {center_idx, center_idx - 1}


@pytest.mark.parametrize(
    "sigma, m, expected_exception, id",
    [
        (0.0, 5, ZeroDivisionError, "sigma_zero"),
        (-1.0, 5, ZeroDivisionError, "sigma_neg"),
        (0.1, 0, ValueError, "m_zero"),
        (0.1, -5, ValueError, "m_neg"),
    ],
    ids=["sigma_zero", "sigma_neg", "m_zero", "m_neg"],
)
def test_gaussian_kernel_1d_errors(sigma, m, expected_exception, id):
    # Act & Assert
    with pytest.raises(expected_exception):
        _gaussian_kernel_1d(sigma, m)


## _smooth_ece_interpolated
# TBD


## _reflected_convolve
@pytest.mark.parametrize(
    "values, ker, expected, id",
    [
        ([1, 2, 3], [0.2, 0.6, 0.2], None, "simple_odd"),
        ([1, 2, 3, 4], [0.25, 0.5, 0.25], None, "simple_even"),
        ([0, 0, 0], [1, 0, 0], np.zeros(3), "all_zeros"),
        ([1, 2, 3], [0, 0, 0], np.zeros(3), "ker_zeros"),
        ([42], [1], np.array([42]), "len_one"),
        ([1, 2, 3], [1], np.array([1, 2, 3]), "ker_len_one"),
        ([-1, 0, 1], [0.5, 0, 0.5], None, "negs"),
    ],
    ids=[
        "simple_odd",
        "simple_even",
        "all_zeros",
        "ker_zeros",
        "len_one",
        "ker_len_one",
        "negs",
    ],
)
def test_reflected_convolve_happy_and_edge(values, ker, expected, id):
    # Act
    result = _reflected_convolve(
        np.array(values, dtype=float), np.array(ker, dtype=float)
    )

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(values),)
    assert np.all(np.isfinite(result))
    if expected is not None:
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "values, ker, expected_exception, id",
    [
        ([], [], ValueError, "both_empty"),
        ([], [1, 2, 3], ValueError, "values_emp"),
        ([1, 2, 3], [], ValueError, "ker_emp"),
        ([1], [1, 2, 3], ValueError, "ker_len_gt_vals"),
    ],
    ids=["both_emp", "values_emp", "ker_emp", "ker_len_gt_vals"],
)
def test_reflected_convolve_errors(values, ker, expected_exception, id):
    # Act & Assert
    with pytest.raises(expected_exception):
        _reflected_convolve(np.array(values, dtype=float), np.array(ker, dtype=float))


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset",
    [
        "pop3",
        "densenetblur121d",
        "efficientnet_b1",
        "efficientnet_b3a",
        "ese_vovnet19b_dw",
        "gluon_senet154",
        "mixnet_m",
        "mobilenetv3_large_100",
        "resnet34",
        "resnext50_32x4d",
    ],
)
def test_smooth_ece_and_ci_width(dataset):
    # Arrange

    # Load sample_data.json from the same directory as this test file
    sample_data_filepath = Path(__file__).parent.joinpath("sample_data.json")
    with open(sample_data_filepath, "r") as f:
        sample_data = json.load(f)

    # Act
    # Load data from URL
    url = sample_data[dataset]["url"]
    df = pd.read_csv(url)
    dataset_name = sample_data[dataset]["dataset_name"]

    if "pop3" in dataset_name.lower():
        df = pd.read_csv(url, sep="\s+", header=0)
        obs = df["obs(mm)"]
        df = df.loc[obs.abs() < 100]
        df = df.loc[(df["p24_cat0"] >= 0) & (df["p24_cat0"] <= 1)]
        f = 1.0 - df["p24_cat0"].to_numpy()
        y = (df["obs(mm)"] > 0.2).to_numpy() * 1.0
    elif "solar" in dataset_name.lower():
        f = df["DAFFS"].to_numpy().copy()
        y = df["rlz.C1"].to_numpy().copy()
    elif "imagenet" in dataset_name.lower():
        f = df["confidence"].to_numpy()
        y = (df["true_label"] == df["pred_label"]).to_numpy() * 1.0
    else:
        raise ValueError(f"Unknown dataset format for {dataset_name}")

    # Compute smECE and CI width
    ece_val = smECE_fast_compat(f, y)
    ece_ci_width = _bootstrap_ci_width(f, y, smECE_fast_compat, confidence=0.999)

    # Assert
    assert np.isclose(
        ece_val, sample_data[dataset]["ece_expected"], atol=1e-2
    ), f"ECE mismatch for {dataset}"
    assert np.isclose(
        ece_ci_width, sample_data[dataset]["ece_ci_width"], atol=1e-2
    ), f"CI width mismatch for {dataset}"
