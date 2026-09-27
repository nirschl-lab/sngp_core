"""`src/metrics/gp_evidence.py`: the eigen-form evidence against a brute-force N x N
Gaussian, and the type-II fixed point."""
import math

import pytest
import torch

from src.metrics.gp_evidence import evidence_basis, log_evidence, median_heuristic, optimize_hyperparameters
from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess


def _data(n=60, m=12, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n, m, generator=g, dtype=torch.float64)
    y = torch.randint(0, k, (n,), generator=g)
    return phi, y


def _brute_force(phi, y, k, alpha, noise):
    """Sum over outputs of log N(y_k; 0, s I + Phi Phi^T / alpha)."""
    cov = noise * torch.eye(len(phi), dtype=torch.float64) + phi @ phi.T / alpha
    mvn = torch.distributions.MultivariateNormal(torch.zeros(len(phi), dtype=torch.float64), covariance_matrix=cov)
    onehot = torch.nn.functional.one_hot(y, k).double()
    return sum(mvn.log_prob(onehot[:, j]).item() for j in range(k))


@pytest.mark.parametrize("alpha,noise", [(1.0, 1.0), (0.3, 2.0), (5.0, 0.1)])
def test_matches_the_brute_force_gaussian(alpha, noise):
    phi, y = _data()
    basis = evidence_basis(phi, y, num_classes=3, batch=7)  # batching must not matter
    assert log_evidence(basis, alpha, noise) == pytest.approx(_brute_force(phi, y, 3, alpha, noise), rel=1e-9)


def test_more_features_than_points():
    phi, y = _data(n=10, m=25)
    basis = evidence_basis(phi, y, num_classes=3)
    assert log_evidence(basis, 0.7, 0.5) == pytest.approx(_brute_force(phi, y, 3, 0.7, 0.5), rel=1e-8)


def test_type2_optimum_is_a_local_maximum():
    phi, y = _data(n=200, m=20)
    phi = phi + 0.5 * torch.nn.functional.one_hot(y, 3).double() @ torch.randn(3, 20, dtype=torch.float64)
    basis = evidence_basis(phi, y, num_classes=3)
    alpha, noise, best = optimize_hyperparameters(basis)
    assert best >= log_evidence(basis, 1.0, 1.0)
    for da, ds in [(1.1, 1), (1 / 1.1, 1), (1, 1.1), (1, 1 / 1.1)]:
        assert best >= log_evidence(basis, alpha * da, noise * ds) - 1e-9


def test_evidence_of_the_head_features_reflects_the_length_scale():
    """The evidence is a function of the head's own features, so rebuilding the head at
    l and feeding it the inputs is the same as feeding x / (l / l0) to the l0 head."""
    torch.manual_seed(0)
    x = torch.randn(50, 8, dtype=torch.float32)
    y = torch.randint(0, 4, (50,))
    kw = dict(in_dim=8, num_classes=4, rff_dim=16, normalize_input=False, scale_random_features=False)
    torch.manual_seed(1)
    at_2 = RandomFeatureGaussianProcess(length_scale=2.0, **kw)
    torch.manual_seed(1)
    at_1 = RandomFeatureGaussianProcess(length_scale=1.0, **kw)
    b2 = evidence_basis(at_2._features(x), y, num_classes=4)
    b1 = evidence_basis(at_1._features(x / 2.0), y, num_classes=4)
    assert log_evidence(b2, 1.0, 1.0) == pytest.approx(log_evidence(b1, 1.0, 1.0), rel=1e-5)


def test_median_heuristic_scales_with_the_data():
    x = torch.randn(500, 4)
    assert median_heuristic(3.0 * x, n_sub=300) == pytest.approx(3.0 * median_heuristic(x, n_sub=300), rel=1e-6)
    # Two standard Gaussian points in d dims are sqrt(2d) apart in expectation.
    assert median_heuristic(torch.randn(2000, 64), n_sub=1000) == pytest.approx(math.sqrt(128), rel=0.05)
