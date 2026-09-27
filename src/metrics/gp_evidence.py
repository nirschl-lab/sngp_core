"""Log marginal likelihood (evidence) of the random-feature GP, for choosing its length
scale from in-distribution data alone.

Under the Gaussian likelihood the SNGP head is Bayesian linear regression on the random
features: targets `Y = Phi B + eps` with one-hot `Y` [N, K], prior `B ~ N(0, I / alpha)`
per column and noise `eps ~ N(0, s I)`. Its posterior precision

    A = alpha I + Phi^T Phi / s

is the head's `ridge * I + sum_i phi_i phi_i^T` at `alpha = ridge`, `s = 1`: the same
matrix whose inverse gives the SNGP predictive variance. The evidence is closed form
(Bishop, PRML eq. 3.86, summed over the K independent outputs):

    log p(Y) = -NK/2 log(2 pi s) + Km/2 log alpha - K/2 log|A|
               - 1/(2s) (||Y||^2 - tr(Y^T Phi A^-1 Phi^T Y) / s)

Everything depends on the data only through the eigenvalues `lambda_i` of `Phi^T Phi` and
the projections `c_i^2 = ||u_i^T Phi^T Y||^2`, so one float64 `eigh` per feature map makes
every `(alpha, s)` O(m). MacKay's fixed-point updates then give type-II ML values for both.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class EvidenceBasis:
    """The sufficient statistics of `(Phi, Y)` for the evidence, all float64."""

    eigvals: torch.Tensor  # [m], eigenvalues of Phi^T Phi (clamped at 0)
    c2: torch.Tensor  # [m], ||u_i^T Phi^T Y||^2 summed over the K outputs
    yy: float  # ||Y||_F^2
    n: int
    k: int

    @property
    def m(self) -> int:
        return self.eigvals.numel()


@torch.no_grad()
def evidence_basis(phi: torch.Tensor, y: torch.Tensor, num_classes: Optional[int] = None,
                   batch: int = 8192) -> EvidenceBasis:
    """Accumulate `Phi^T Phi` and `Phi^T Y` in float64 (in batches) and diagonalise.

    phi: [N, m] features; y: [N] integer labels, or [N, K] targets.
    """
    if y.dim() == 1:
        y = torch.nn.functional.one_hot(y.long(), num_classes).to(torch.float64)
    y = y.to(device=phi.device, dtype=torch.float64)
    m = phi.shape[1]
    gram = torch.zeros(m, m, dtype=torch.float64, device=phi.device)
    proj = torch.zeros(m, y.shape[1], dtype=torch.float64, device=phi.device)
    for i in range(0, len(phi), batch):
        p = phi[i : i + batch].double()
        gram += p.T @ p
        proj += p.T @ y[i : i + batch]
    eigvals, eigvecs = torch.linalg.eigh(gram)
    c2 = (eigvecs.T @ proj).pow(2).sum(dim=1)
    return EvidenceBasis(eigvals=eigvals.clamp(min=0.0), c2=c2, yy=float(y.pow(2).sum()), n=len(phi), k=y.shape[1])


def log_evidence(basis: EvidenceBasis, alpha: float, noise: float) -> float:
    """log p(Y | Phi, alpha, s) for prior precision `alpha` and noise variance `s`."""
    a = alpha + basis.eigvals / noise
    fit = (basis.yy - (basis.c2 / a).sum().item() / noise) / noise
    return (
        -0.5 * basis.n * basis.k * math.log(2 * math.pi * noise)
        + 0.5 * basis.k * basis.m * math.log(alpha)
        - 0.5 * basis.k * torch.log(a).sum().item()
        - 0.5 * fit
    )


def optimize_hyperparameters(basis: EvidenceBasis, alpha: float = 1.0, noise: float = 1.0, iters: int = 500,
                             tol: float = 1e-10) -> Tuple[float, float, float]:
    """Type-II ML `(alpha, s)` by MacKay's fixed-point updates (Bishop, PRML 3.5.2):

        gamma = sum_i (lambda_i / s) / (alpha + lambda_i / s)   (effective number of parameters)
        alpha = K gamma / ||M||^2,    s = ||Y - Phi M||^2 / (K (N - gamma))

    with `M = A^-1 Phi^T Y / s` the posterior mean. Returns `(alpha, s, log_evidence)`.
    """
    lam, c2 = basis.eigvals, basis.c2
    for _ in range(iters):
        a = alpha + lam / noise
        gamma = ((lam / noise) / a).sum().item()
        mean_sq = (c2 / (noise * a).pow(2)).sum().item()  # ||M||^2
        resid = basis.yy - 2 * (c2 / (noise * a)).sum().item() + (lam * c2 / (noise * a).pow(2)).sum().item()
        new_alpha = basis.k * gamma / max(mean_sq, 1e-300)
        new_noise = max(resid, 1e-300) / (basis.k * (basis.n - gamma))
        done = abs(new_alpha - alpha) <= tol * alpha and abs(new_noise - noise) <= tol * noise
        alpha, noise = new_alpha, new_noise
        if done:
            break
    return alpha, noise, log_evidence(basis, alpha, noise)


@torch.no_grad()
def median_heuristic(x: torch.Tensor, n_sub: int = 4096, seed: int = 0) -> float:
    """Median pairwise Euclidean distance over a random subsample of `x` [N, d] -- the
    classic data-driven RBF length scale."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:n_sub].to(x.device)
    d = torch.cdist(x[idx].double(), x[idx].double())
    iu = torch.triu_indices(len(idx), len(idx), offset=1, device=x.device)
    return float(d[iu[0], iu[1]].median())
