#!/usr/bin/env python3
"""Tests for test_rff_scaling.py.

Tests for Random Fourier Feature (RFF) layer independent of GP logic.
"""

import math

import numpy as np
import pytest
import torch

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


# def test_rff_scaled_features_have_unit_variance(sample_input):
#     """Ensure Rahimi–Recht scaling √(2/m) produces unit variance in features."""
#     rff = RandomFourierFeatures(in_features=16, out_features=512, kernel_scale=1.0)
#     with torch.no_grad():
#         phi = rff(sample_input)
#         phi_scaled = phi * math.sqrt(2.0 / rff.out_features)
#
#     var = phi_scaled.var(dim=0).mean().item()
#
#     np.testing.assert_allclose(
#         var,
#         1.0,
#         atol=0.2,
#         err_msg=(
#             f"Scaled RFF features should have unit variance; got {var:.4f}.\n"
#             "Hint: use kernel_scale=1.0 when testing normalized inputs."
#         ),
#     )
#

def test_rff_scaled_features_have_unit_kernel_norm(sample_input):
    """Ensure Rahimi–Recht scaling √(2/m) normalizes the expected kernel magnitude."""
    in_features = 16
    m = 512
    rff = RandomFourierFeatures(in_features=in_features, out_features=m, kernel_scale=1.0)
    with torch.no_grad():
        phi = rff(sample_input)
        phi_scaled = phi * math.sqrt(2.0 / m)

    # Compute empirical kernel magnitude for each sample
    norms = (phi_scaled ** 2).sum(dim=1)
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

@pytest.mark.parametrize("kernel_scale", [0.5, 1.0, 2.0])
def test_rff_lengthscale_affects_smoothness(sample_input, kernel_scale):
    """Smaller kernel_scale should produce less correlated (higher-frequency) features."""
    torch.manual_seed(0)
    rff = RandomFourierFeatures(
        in_features=16, out_features=512, kernel_scale=kernel_scale
    )
    with torch.no_grad():
        phi = rff(sample_input)
        # Small random perturbation in input to probe smoothness
        phi_perturbed = rff(sample_input + 0.05 * torch.randn_like(sample_input))

    # Compute feature correlation between φ(x) and φ(x+δx)
    corr = torch.mean(
        (phi * phi_perturbed).sum(dim=1) / (phi.norm(dim=1) * phi_perturbed.norm(dim=1))
    ).item()
    print(f"kernel_scale={kernel_scale:.2f} → feature corr={corr:.3f}")

    # Store correlation for later comparison
    if not hasattr(test_rff_lengthscale_affects_smoothness, "corrs"):
        test_rff_lengthscale_affects_smoothness.corrs = {}
    test_rff_lengthscale_affects_smoothness.corrs[kernel_scale] = corr

    # Only check absolute thresholds for extreme cases
    if kernel_scale < 1.0:
        assert corr < 0.95, (
            f"Expected lower correlation (<0.95) for smaller ℓ, got {corr:.3f}."
        )
    elif kernel_scale > 1.0:
        assert corr > 0.9, (
            f"Expected higher correlation (>0.9) for larger ℓ, got {corr:.3f}."
        )


@pytest.fixture(scope="module", autouse=True)
def _validate_smoothness_trend(request):
    """After all parameterized runs, check that correlation increases monotonically with ℓ."""
    yield  # run after all above tests
    corrs = getattr(test_rff_lengthscale_affects_smoothness, "corrs", {})
    if corrs:
        scales, values = zip(*sorted(corrs.items()))
        print(f"Smoothness trend: {list(zip(scales, values))}")
        assert all(
            earlier <= later for earlier, later in zip(values, values[1:])
        ), "Expected correlation to increase monotonically with kernel_scale."


@pytest.mark.parametrize("kernel_scale", [0.5, 1.0, 2.0])
def test_rff_empirical_kernel_matches_gaussian(sample_input, kernel_scale):
    """Empirical RFF kernel should approximate analytical RBF kernel for various scales."""
    torch.manual_seed(42)
    x = sample_input[:16]  # use small batch for clarity
    n = x.size(0)
    m = 2048  # number of random features

    # Instantiate unscaled RFF layer
    rff = RandomFourierFeatures(
        in_features=x.size(1),
        out_features=m,
        kernel_scale=kernel_scale,
        kernel_type="gaussian",
    )

    with torch.no_grad():
        phi = rff(x) * math.sqrt(2.0 / m)  # Rahimi–Recht scaling

    # Empirical kernel matrix
    k_emp = phi @ phi.T

    # Analytical RBF kernel matrix
    pairwise_d2 = torch.cdist(x, x, p=2.0).pow(2)
    k_true = torch.exp(-0.5 * pairwise_d2 / (kernel_scale**2))

    # Normalize both to ensure comparable scale (numerical stability)
    k_emp /= k_emp.diag().mean()
    k_true /= k_true.diag().mean()

    mse = torch.mean((k_emp - k_true) ** 2).item()
    corr = np.corrcoef(k_emp.flatten().cpu(), k_true.flatten().cpu())[0, 1]

    print(f"ℓ={kernel_scale:.2f} → MSE={mse:.4e}, Corr={corr:.4f}")

    assert corr > 0.95, f"Empirical kernel correlation too low for ℓ={kernel_scale}"
    assert (
        mse < 0.05
    ), f"Empirical kernel too different (MSE={mse:.4f}) for ℓ={kernel_scale}"


# import os
# import matplotlib.pyplot as plt
# import math
# import torch
# import numpy as np
# from src.models.sngp.random_fourier_features import RandomFourierFeatures
#
#
# def _visualize_kernel_comparison(k_emp: torch.Tensor, k_true: torch.Tensor, kernel_scale: float):
#     """Helper for visualizing empirical vs. analytical kernel matrices."""
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#     k_diff = (k_emp - k_true).abs()
#
#     im0 = axes[0].imshow(k_true.cpu(), cmap="viridis")
#     axes[0].set_title(f"Analytical RBF (ℓ={kernel_scale})")
#     plt.colorbar(im0, ax=axes[0], fraction=0.046)
#
#     im1 = axes[1].imshow(k_emp.cpu(), cmap="viridis")
#     axes[1].set_title("Empirical RFF Kernel")
#     plt.colorbar(im1, ax=axes[1], fraction=0.046)
#
#     im2 = axes[2].imshow(k_diff.cpu(), cmap="magma")
#     axes[2].set_title("|Emp - True|")
#     plt.colorbar(im2, ax=axes[2], fraction=0.046)
#
#     for ax in axes:
#         ax.set_xlabel("Sample index")
#         ax.set_ylabel("Sample index")
#
#     plt.suptitle("RFF vs Analytical Kernel Comparison", fontsize=14)
#     plt.tight_layout()
#     plt.show()
#
#
# @pytest.mark.parametrize("kernel_scale", [0.5, 1.0, 2.0])
# def test_rff_empirical_kernel_matches_gaussian(sample_input, kernel_scale):
#     """Empirical RFF kernel should approximate analytical RBF kernel for various scales."""
#     torch.manual_seed(42)
#     x = sample_input[:16]  # small batch for clarity
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
#     k_true = torch.exp(-0.5 * pairwise_d2 / (kernel_scale ** 2))
#
#     # Normalize both for numerical stability
#     k_emp /= k_emp.diag().mean()
#     k_true /= k_true.diag().mean()
#
#     mse = torch.mean((k_emp - k_true) ** 2).item()
#     corr = np.corrcoef(k_emp.flatten().cpu(), k_true.flatten().cpu())[0, 1]
#
#     print(f"ℓ={kernel_scale:.2f} → MSE={mse:.4e}, Corr={corr:.4f}")
#
#     # Optional visualization for manual inspection
#     if os.getenv("DEBUG_VISUAL", "0").lower() in {"1", "true", "yes"}:
#         _visualize_kernel_comparison(k_emp, k_true, kernel_scale)
#
#     assert corr > 0.95, f"Empirical kernel correlation too low for ℓ={kernel_scale}"
#     assert mse < 0.05, f"Empirical kernel too different (MSE={mse:.4f}) for ℓ={kernel_scale}"
