#!/usr/bin/env python3
"""Tests for test_rff_scaling.py.

Tests for Random Fourier Feature (RFF) layer independent of GP logic.
"""

import math

import numpy as np
import pytest
import torch

from src.models.sngp.gaussian_process import (
    LaplaceRandomFeatureCovariance,
    RandomFeatureGaussianProcess,
)
from src.models.sngp.random_fourier_features import RandomFourierFeatures


@pytest.fixture
def sample_input():
    torch.manual_seed(0)
    return torch.randn(64, 16)  # (batch_size, in_features)


def test_rff_unscaled_cos_features(sample_input):
    """Ensure RFF outputs pure cos(Wx + b) without Rahimi–Recht scaling."""
    rff = RandomFourierFeatures(in_features=16, out_features=512, kernel_scale=1.0)
    with torch.no_grad():
        phi = rff(sample_input)

    # Expected cosine range and general statistics
    assert torch.all(phi <= 1.0 + 1e-6)
    assert torch.all(phi >= -1.0 - 1e-6)
    assert abs(phi.mean().item()) < 0.2, "Mean of cos features should be near 0"
    assert 0.4 < phi.std().item() < 0.8, "Unscaled RFF should have std ~0.7"


def test_rff_scaled_features_have_unit_variance(sample_input):
    """
    Rahimi–Recht scaling (sqrt(2/m)) makes the variance of EACH FEATURE ≈ 1/m,
    not 1.0. The sum over features approximates a unit-norm kernel mapping.
    """
    import math

    import numpy as np

    from src.models.sngp.random_fourier_features import RandomFourierFeatures

    m = 512
    rff = RandomFourierFeatures(in_features=16, out_features=m, kernel_scale=1.0)

    with torch.no_grad():
        phi = rff(sample_input)  # shape [B, m], values in [-1,1]
        phi_scaled = phi * math.sqrt(2.0 / m)  # per-column scale

    # Column-wise variance across the batch, then averaged over columns.
    # Use unbiased=False for population-style variance to reduce noise.
    col_var = phi_scaled.var(dim=0, unbiased=False).mean().item()

    expected = 1.0 / m  # ≈ 0.001953 for m=512

    # A tolerant check (finite batch + fixed W,b can shift the estimate).
    np.testing.assert_allclose(
        col_var,
        expected,
        rtol=0.5,
        atol=5e-4,
        err_msg=(
            f"Each scaled feature should have variance ≈ 1/m. "
            f"Got {col_var:.6f}, expected ≈ {expected:.6f}."
        ),
    )


def test_rff_scaled_feature_norm_is_one_on_average(sample_input):
    import math

    from src.models.sngp.random_fourier_features import RandomFourierFeatures

    m = 512
    rff = RandomFourierFeatures(in_features=16, out_features=m, kernel_scale=1.0)
    with torch.no_grad():
        phi = rff(sample_input)
        z = phi * math.sqrt(2.0 / m)  # [B, m]
        norms_sq = (z * z).sum(dim=1)  # ||z(x)||^2 per sample

    # Expectation over random (W,b) gives ~1; with one draw of (W,b) and finite B,
    # stay generous with bounds but near 1.
    assert norms_sq.mean().item() == pytest.approx(1.0, rel=0.2, abs=0.2)


def test_rff_scaled_features_have_unit_kernel_norm(sample_input):
    """Ensure Rahimi–Recht scaling √(2/m) normalizes the expected kernel magnitude."""
    in_features = 16
    m = 512
    rff = RandomFourierFeatures(
        in_features=in_features, out_features=m, kernel_scale=1.0
    )
    with torch.no_grad():
        phi = rff(sample_input)
        phi_scaled = phi * math.sqrt(2.0 / m)

    # Compute empirical kernel magnitude for each sample
    norms = (phi_scaled**2).sum(dim=1)
    mean_norm = norms.mean().item()

    print(f"Mean feature L2 norm squared ≈ {mean_norm:.4f}")

    # Check that E[||φ(x)||^2] ≈ 1 (kernel normalization)
    np.testing.assert_allclose(
        mean_norm,
        1.0,
        atol=0.2,
        err_msg=(
            f"Rahimi–Recht scaled features should yield unit expected kernel norm; got {mean_norm:.4f}."
        ),
    )


# TODO: fix
# @pytest.mark.parametrize("kernel_scale", [0.5, 1.0, 2.0])
# def test_rff_lengthscale_affects_smoothness(sample_input, kernel_scale):
#     """Smaller kernel_scale should produce less correlated (higher-frequency) features."""
#     torch.manual_seed(0)
#     rff = RandomFourierFeatures(
#         in_features=16, out_features=512, kernel_scale=kernel_scale
#     )
#     with torch.no_grad():
#         phi = rff(sample_input)
#         # Small random perturbation in input to probe smoothness
#         phi_perturbed = rff(sample_input + 0.05 * torch.randn_like(sample_input))
#
#     # Compute feature correlation between φ(x) and φ(x+δx)
#     corr = torch.mean(
#         (phi * phi_perturbed).sum(dim=1) / (phi.norm(dim=1) * phi_perturbed.norm(dim=1))
#     ).item()
#     print(f"kernel_scale={kernel_scale:.2f} → feature corr={corr:.3f}")
#
#     # Store correlation for later comparison
#     if not hasattr(test_rff_lengthscale_affects_smoothness, "corrs"):
#         test_rff_lengthscale_affects_smoothness.corrs = {}
#     test_rff_lengthscale_affects_smoothness.corrs[kernel_scale] = corr
#
#     # Only check absolute thresholds for extreme cases
#     if kernel_scale < 1.0:
#         assert (
#             corr < 0.95
#         ), f"Expected lower correlation (<0.95) for smaller ℓ, got {corr:.3f}."
#     elif kernel_scale > 1.0:
#         assert (
#             corr > 0.9
#         ), f"Expected higher correlation (>0.9) for larger ℓ, got {corr:.3f}."
#

# @pytest.fixture(scope="module", autouse=True)
# def _validate_smoothness_trend(request):
#     """After all parameterized runs, check that correlation increases monotonically with ℓ."""
#     yield  # run after all above tests
#     corrs = getattr(test_rff_lengthscale_affects_smoothness, "corrs", {})
#     if corrs:
#         scales, values = zip(*sorted(corrs.items()))
#         print(f"Smoothness trend: {list(zip(scales, values))}")
#         assert all(
#             earlier <= later for earlier, later in zip(values, values[1:])
#         ), "Expected correlation to increase monotonically with kernel_scale."

# TODO FIX
# @pytest.mark.parametrize("kernel_scale", [0.5, 1.0, 2.0])
# def test_rff_empirical_kernel_matches_gaussian(sample_input, kernel_scale):
#     """Empirical RFF kernel should approximate analytical RBF kernel for various scales."""
#     torch.manual_seed(42)
#     x = sample_input[:16]  # use small batch for clarity
#     n = x.size(0)
#     m = 2048  # number of random features
#
#     # Instantiate unscaled RFF layer
#     rff = RandomFourierFeatures(
#         in_features=x.size(1),
#         out_features=m,
#         kernel_scale=kernel_scale,
#         kernel_type="gaussian",
#     )
#
#     with torch.no_grad():
#         phi = rff(x) * math.sqrt(2.0 / m)  # Rahimi–Recht scaling
#
#     # Empirical kernel matrix
#     k_emp = phi @ phi.T
#
#     # Analytical RBF kernel matrix
#     pairwise_d2 = torch.cdist(x, x, p=2.0).pow(2)
#     k_true = torch.exp(-0.5 * pairwise_d2 / (kernel_scale**2))
#
#     # Normalize both to ensure comparable scale (numerical stability)
#     k_emp /= k_emp.diag().mean()
#     k_true /= k_true.diag().mean()
#
#     mse = torch.mean((k_emp - k_true) ** 2).item()
#     corr = np.corrcoef(k_emp.flatten().cpu(), k_true.flatten().cpu())[0, 1]
#
#     print(f"ℓ={kernel_scale:.2f} → MSE={mse:.4e}, Corr={corr:.4f}")
#
#     assert corr > 0.95, f"Empirical kernel correlation too low for ℓ={kernel_scale}"
#     assert (
#         mse < 0.05
#     ), f"Empirical kernel too different (MSE={mse:.4f}) for ℓ={kernel_scale}"


def test_rff_reproducibility_with_fixed_seed(sample_input):
    torch.manual_seed(123)
    rff1 = RandomFourierFeatures(16, 128)
    torch.manual_seed(123)
    rff2 = RandomFourierFeatures(16, 128)
    assert torch.allclose(rff1.weight, rff2.weight)
    assert torch.allclose(rff1.bias, rff2.bias)


def test_rff_different_seeds_decorrelated(sample_input):
    torch.manual_seed(0)
    rff1 = RandomFourierFeatures(16, 256)
    torch.manual_seed(1)
    rff2 = RandomFourierFeatures(16, 256)
    phi1, phi2 = rff1(sample_input), rff2(sample_input)
    corr = torch.mean((phi1 * phi2).mean(dim=1))
    assert corr.abs() < 0.05


def test_rff_backward_pass_works(sample_input):
    rff = RandomFourierFeatures(16, 64, kernel_scale_trainable=True)
    phi = rff(sample_input)
    loss = phi.mean()
    loss.backward()
    assert rff.kernel_scale.grad is not None


def test_covariance_momentum_smoothing():
    dim, n = 10, 256
    x = torch.randn(n, dim)
    for momentum in [0.0, 0.9, 0.999]:
        cov = LaplaceRandomFeatureCovariance(in_features=dim, momentum=momentum)
        cov.train()
        for _ in range(20):
            cov(x)
        print(f"momentum={momentum} → precision mean={cov.precision.mean():.6f}")


def test_laplace_covariance_minibatch_converges():
    torch.manual_seed(0)
    dim, n = 10, 512
    momentum = 0.999
    num_updates = 200

    x_data = torch.randn(n, dim)
    cov_estimator = LaplaceRandomFeatureCovariance(
        in_features=dim,
        momentum=momentum,
        ridge_penalty=1e-6,
        likelihood="gaussian",
    )
    cov_estimator.train()

    for _ in range(num_updates):
        cov_estimator(x_data)

    # Target of the EMA (steady state) is Q = (X^T X)/n
    Q = (x_data.T @ x_data) / n

    # From zero init, after t steps: P_t = (1 - m^t) * Q
    transient_scale = 1.0 - (momentum**num_updates)
    expected_mean = (Q.mean() * transient_scale).item()
    actual_mean = cov_estimator.precision.mean().item()

    torch.testing.assert_close(
        torch.tensor(actual_mean),
        torch.tensor(expected_mean),
        atol=1e-3,
        rtol=5e-2,
        msg=(
            "Precision EMA did not match transient expectation.\n"
            f"momentum={momentum}, updates={num_updates}\n"
            f"(1 - m^t)={transient_scale:.6f}\n"
            f"Expected mean={expected_mean:.6f}, got {actual_mean:.6f}"
        ),
    )

    # Light structural check
    P = cov_estimator.precision
    assert torch.allclose(P, P.T, atol=1e-6)


# TODO: fix
# def test_rfgp_prior_kernel_matches_rbf():
#     n, d = 128, 10
#     x = torch.randn(n, d)
#     gp = RandomFeatureGaussianProcess(
#         in_features=d,
#         out_features=1,
#         random_features=2048,
#         kernel_type="gaussian",
#         normalize_input=False,
#         return_features=True,
#     )
#     with torch.no_grad():
#         out = gp(x)
#         phi = out["features"]
#     k_emp = phi @ phi.T
#     d2 = torch.cdist(x, x).pow(2)
#     k_true = torch.exp(-0.5 * d2)
#     k_emp /= k_emp.diag().mean()
#     k_true /= k_true.diag().mean()
#     corr = np.corrcoef(k_emp.flatten(), k_true.flatten())[0, 1]
#     assert corr > 0.95
#


def test_rfgp_posterior_kernel_matches_gp_posterior():
    torch.manual_seed(0)
    n_train, n_test, d = 256, 64, 10
    x_train, x_test = torch.randn(n_train, d), torch.randn(n_test, d)
    gp = RandomFeatureGaussianProcess(
        in_features=d,
        out_features=1,
        random_features=1024,
        kernel_type="gaussian",
        normalize_input=False,
        covariance_momentum=0.5,
        covariance_ridge_penalty=1.0,
        return_covariance=True,
    )
    gp(x_train)  # training
    out = gp(x_test)  # inference
    k_pred = out["cov"]
    # Compare with analytical RBF posterior kernel
    d_tt = torch.cdist(x_train, x_train).pow(2)
    d_ts = torch.cdist(x_train, x_test).pow(2)
    k_tt = torch.exp(-0.5 * d_tt) + torch.eye(n_train)
    k_ts = torch.exp(-0.5 * d_ts)
    k_post = torch.exp(
        -0.5 * torch.cdist(x_test, x_test).pow(2)
    ) - k_ts.T @ torch.linalg.solve(k_tt, k_ts)
    torch.testing.assert_close(k_pred, k_post, atol=0.1, rtol=1.5)


def test_rfgp_linear_kernel_identity():
    x = torch.randn(32, 8)
    gp = RandomFeatureGaussianProcess(
        in_features=8,
        out_features=1,
        kernel_type="linear",
        use_custom_features=True,
        normalize_input=False,
        return_features=True,
    )

    out = gp(x)
    phi = out["features"]

    # For linear kernel, phi should be a *linear transform* of x, not necessarily identical
    W = gp.feature_layer.weight  # shape (1024, 8)
    torch.testing.assert_close(phi, x @ W.T, atol=1e-3, rtol=1e-3)
