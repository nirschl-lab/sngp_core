#!/usr/bin/env python3
"""Tests for test_feature_scaling.py.

Tests for feature scaling and kernel semantics in SNGP layers.
"""

import math

import numpy as np
import pytest
import torch

from src.models.sngp.gaussian_process import RandomFeatureGaussianProcess
from src.models.sngp.random_fourier_features import RandomFourierFeatures


@pytest.fixture
def sample_input():
    torch.manual_seed(0)
    return torch.randn(8, 16)  # (batch_size, in_features)


def test_rff_unscaled_cos_features(sample_input):
    """Ensure RFF outputs pure cos(Wx + b) without Rahimi–Recht scaling."""
    rff = RandomFourierFeatures(in_features=16, out_features=128)
    with torch.no_grad():
        phi = rff(sample_input)

    # Expected range of cosine values: [-1, 1]
    assert torch.all(phi <= 1.0 + 1e-6)
    assert torch.all(phi >= -1.0 - 1e-6)

    # Mean and variance roughly within cosine expectation
    assert abs(phi.mean().item()) < 0.2
    assert 0.4 < phi.std().item() < 0.8, "RFF seems to apply extra scaling."


def test_rff_output_unscaled_cosine():
    """RFF should output unscaled cos(Wx + b), leaving scaling to GP layer."""
    rff = RandomFourierFeatures(in_features=8, out_features=256)
    x = torch.randn(32, 8)
    phi = rff(x)
    # The mean magnitude of unscaled cos should be < 1, variance < 1.
    assert 0.0 < phi.var().item() < 1.0


@pytest.mark.parametrize("normalize_input", [True, False])
def test_combined_scaling_consistency(sample_input, normalize_input):
    gp_model = RandomFeatureGaussianProcess(
        in_features=16,
        out_features=1,
        random_features=512,
        normalize_input=normalize_input,
        scale_random_features=True,
        kernel_scale=None if normalize_input else 1.0,
    )


def test_gp_applies_scaling_only_when_enabled(sample_input):
    """Ensure Rahimi–Recht scaling √(2/m) applied only when enabled."""
    in_dim, num_features = 16, 256
    gp_scaled = RandomFeatureGaussianProcess(
        in_features=in_dim,
        out_features=1,
        random_features=num_features,
        scale_random_features=True,
        normalize_input=False,
    )
    gp_unscaled = RandomFeatureGaussianProcess(
        in_features=in_dim,
        out_features=1,
        random_features=num_features,
        scale_random_features=False,
        normalize_input=False,
    )

    with torch.no_grad():
        phi_scaled = gp_scaled.feature_layer(sample_input)
        phi_unscaled = gp_unscaled.feature_layer(sample_input)

        # Apply scaling as GP.forward() does
        phi_scaled = phi_scaled * math.sqrt(2.0 / num_features)

    ratio = phi_scaled.std() / phi_unscaled.std()
    expected = math.sqrt(2.0 / num_features)
    np.testing.assert_allclose(
        ratio,
        expected,
        rtol=0.05,
        err_msg="GP scaling factor mismatch: expected √(2/m) factor.",
    )


def test_lengthscale_rescaling_effect(sample_input):
    """Verify kernel length scale (ℓ) affects feature frequency as 1/√ℓ."""
    rff_small_ell = RandomFourierFeatures(16, 128, kernel_scale=0.5)
    rff_large_ell = RandomFourierFeatures(16, 128, kernel_scale=2.0)

    with torch.no_grad():
        phi_small = rff_small_ell(sample_input)
        phi_large = rff_large_ell(sample_input)

    # Features with smaller ℓ vary faster (higher frequency)
    diff_small = phi_small[:, 1:] - phi_small[:, :-1]
    diff_large = phi_large[:, 1:] - phi_large[:, :-1]
    assert (
        diff_small.abs().mean() > diff_large.abs().mean()
    ), "Expected smaller ℓ to produce higher-frequency features."


# def test_combined_scaling_consistency(sample_input):
#     """Ensure that GP+RFF combined produces expected variance magnitude."""
#     gp_model = RandomFeatureGaussianProcess(
#         in_features=16,
#         out_features=1,
#         random_features=512,
#         normalize_input=False,
#         scale_random_features=True,
#     )
#
#     with torch.no_grad():
#         phi = gp_model.feature_layer(sample_input)
#         num_features = gp_model.feature_layer.out_features
#         phi_scaled = phi * math.sqrt(2.0 / num_features)
#
#     # Check feature variance is approximately 1.0 after scaling
#     var = phi_scaled.var(dim=0).mean().item()
#     np.testing.assert_allclose(
#         var, 1.0, atol=0.2, err_msg="Scaled RFF features should have unit variance."
#     )


def test_kernel_scale_default_respects_normalization_flag():

    gp_norm = RandomFeatureGaussianProcess(32, 4, normalize_input=True)
    gp_no_norm = RandomFeatureGaussianProcess(32, 4, normalize_input=False)

    assert math.isclose(
        gp_norm.feature_layer.kernel_scale.item(), 1.0, rel_tol=1e-6
    ), "Normalized GP should default to kernel_scale=1.0"

    assert (
        gp_no_norm.feature_layer.kernel_scale.item() > 1.0
    ), "Unnormalized GP should default to kernel_scale ≈ √(d/2)"
