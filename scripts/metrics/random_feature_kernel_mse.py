"""Kernel-approximation MSE of SNGP's random-feature couplings, for cos and positive features.

`random_feature_type=simrf` (simplex random features, Reid et al. 2023, arXiv:2301.13856)
ties `orf` when measured on the cos features SNGP actually uses. That is consistent with the
paper: SimRF's MSE-optimality is proven for *positive* random features (PRFs), not
trigonometric ones. This offline sweep decides whether a PRF feature map in the GP layer is
worth building, by answering three questions:

  1. Sanity -- in the paper's regime (small d, ||x||/l ~ 1) does our sampler reproduce
     simrf < orf < rff for PRFs? If not, the sampler is suspect, not the idea.
  2. Transfer -- does that gap survive at the SNGP head width (d = 512)? The simplex's
     angle gap from ORF, -1/(d-1) vs 0, shrinks as 1/d.
  3. Feasibility -- at the protocol length scale (LayerNormed inputs, ||x|| = sqrt(d),
     l = sqrt(2), so rho = ||x||/l = sqrt(d/2) ~ 16 at d = 512), are PRFs usable in float32,
     and how does their error compare to cos features?

The projection is drawn by the production sampler,
`RandomFeatureGaussianProcess._sample_projection`, so what is measured is what training would
use. Inputs are expressed in kernel units (x <- x / l), and the target is
k(x, y) = exp(-||x - y||^2 / 2). The feature maps are:

  cos         sqrt(2/m) cos(W^T x + b), b ~ U(0, 2pi)              -- matches `_features`
  positive    (1/sqrt(m)) exp(W^T x - ||x||^2)                     -- Performer's FAVOR+ PRF
  hyperbolic  (1/sqrt(2m)) [exp(W^T x), exp(-W^T x)] exp(-||x||^2) -- FAVOR+ "++" variant

All three are unbiased. Because every coupling is Haar-rotation-invariant, the error depends
only on (rho, r) with r = ||x - y|| / l. Each (rho, r) cell averages 8 pairs at that geometry.

Read the `bias` column with care at large rho. PRFs stay unbiased, but their variance grows
like exp(||x + y||^2), so the sample mean stops converging at any feasible seed count and
`bias` drifts to ~ -k. That drift is itself the answer to question 3, not a bug. The cells at
r = 2 rho have y = -x, so x + y = 0 and the PRF estimate is exact with zero SE. At small rho,
the `bias` column is a live unbiasedness check. It is how this sweep caught
`_sample_projection` drawing non-Haar rotations (the QR sign convention, now fixed): ORF with
positive features was biased at ~40% of the rho <= 1 cells, and cos features were blind to it.

Emits one row per (d, m, rho, r, coupling, feature_map) to `--csv`, plus a per-(d, m,
feature_map, rho) summary next to it (`*_summary.csv`). The figures are drawn by
`src/visualization/random_feature_kernel_mse.py`.

Usage:

    uv run python scripts/metrics/random_feature_kernel_mse.py \\
        --csv figures/random_feature_kernel_mse/random_feature_kernel_mse.csv
"""
import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import rootutils
import torch
from loguru import logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess  # noqa: E402

COUPLINGS: Tuple[str, ...] = ("rff", "orf", "simrf")
FEATURE_MAPS: Tuple[str, ...] = ("cos", "positive", "hyperbolic")
PAIRS_PER_CELL = 8
N_R = 8
R_CAP = 4.0  # exp(-4^2/2) ~ 3e-4: past this the kernel is ~0 and relative error is meaningless
FP32_SEEDS = 20


def rho_grid(d: int) -> List[float]:
    """Kernel-unit input norms: a Performer-like range plus the SNGP protocol value."""
    return sorted({0.5, 1.0, 2.0, 4.0, math.sqrt(d / 2)})


def make_pairs(d: int, rhos: Sequence[float], generator: torch.Generator) -> Tuple[torch.Tensor, ...]:
    """Build `PAIRS_PER_CELL` pairs per (rho, r) cell with ||x|| = ||y|| = rho and
    ||x - y|| = r, for r on `N_R` points in (0, min(2 rho, R_CAP)].

    Returns (x, y, rho, r) flattened over cells x pairs; `rho`/`r` label each pair.
    """
    xs, ys, rho_col, r_col = [], [], [], []
    for rho in rhos:
        r_max = min(2 * rho, R_CAP)
        for r in np.linspace(r_max / N_R, r_max, N_R):
            # ||x - y||^2 = 2 rho^2 (1 - cos theta)
            cos_t = 1.0 - r**2 / (2 * rho**2)
            sin_t = math.sqrt(max(0.0, 1.0 - cos_t**2))
            u = torch.randn(PAIRS_PER_CELL, d, dtype=torch.float64, generator=generator)
            u = u / u.norm(dim=1, keepdim=True)
            v = torch.randn(PAIRS_PER_CELL, d, dtype=torch.float64, generator=generator)
            v = v - (v * u).sum(1, keepdim=True) * u
            v = v / v.norm(dim=1, keepdim=True)
            xs.append(rho * u)
            ys.append(rho * (cos_t * u + sin_t * v))
            rho_col += [rho] * PAIRS_PER_CELL
            r_col += [float(r)] * PAIRS_PER_CELL
    return torch.cat(xs), torch.cat(ys), torch.tensor(rho_col), torch.tensor(r_col)


def kernel_estimates(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
    """k_hat(x_i, y_i) for every feature map, in float64.

    The PRF estimates are computed as exp of the summed exponents rather than as a product
    of features, so float64 never overflows on intermediate features. Whether the
    production form survives float32 is measured separately by `fp32_bad_fraction`.
    """
    m = W.shape[1]
    px, py = x @ W, y @ W  # [P, m]
    nx = x.pow(2).sum(1, keepdim=True)
    ny = y.pow(2).sum(1, keepdim=True)
    s = px + py
    return {
        "cos": (2.0 / m) * (torch.cos(px + b) * torch.cos(py + b)).sum(1),
        "positive": torch.exp(s - nx - ny).mean(1),
        "hyperbolic": 0.5 * (torch.exp(s - nx - ny) + torch.exp(-s - nx - ny)).mean(1),
    }


def fp32_bad_fraction(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per input: 1 if its float32 feature vector, in the production (product) form, has
    a non-finite entry or has underflowed to all zeros; otherwise 0."""
    x32, W32, b32 = x.float(), W.float(), b.float()
    m = W.shape[1]
    proj = x32 @ W32
    nx = x32.pow(2).sum(1, keepdim=True)
    feats = {
        "cos": math.sqrt(2.0 / m) * torch.cos(proj + b32),
        "positive": torch.exp(proj - nx) / math.sqrt(m),
        "hyperbolic": torch.cat([torch.exp(proj - nx), torch.exp(-proj - nx)], 1) / math.sqrt(2 * m),
    }
    return {k: ((~torch.isfinite(f)).any(1) | (f == 0).all(1)).double() for k, f in feats.items()}


def sweep_one(d: int, m: int, coupling: str, n_seeds: int, x, y) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per-seed squared error, signed error and fp32-bad flags, all [n_seeds, P]."""
    sq = {fm: [] for fm in FEATURE_MAPS}
    err = {fm: [] for fm in FEATURE_MAPS}
    bad = {fm: [] for fm in FEATURE_MAPS}
    exact = torch.exp(-(x - y).pow(2).sum(1) / 2)
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        W = RandomFeatureGaussianProcess._sample_projection(d, m, coupling, torch.float64)
        b = 2 * math.pi * torch.rand(m, dtype=torch.float64)
        for fm, k_hat in kernel_estimates(x, y, W, b).items():
            sq[fm].append((k_hat - exact) ** 2)
            err[fm].append(k_hat - exact)
        if seed < FP32_SEEDS:
            flags_x, flags_y = fp32_bad_fraction(x, W, b), fp32_bad_fraction(y, W, b)
            for fm in FEATURE_MAPS:
                bad[fm].append(torch.maximum(flags_x[fm], flags_y[fm]))
    return {
        fm: {"sq": torch.stack(sq[fm]), "err": torch.stack(err[fm]), "bad": torch.stack(bad[fm]), "exact": exact}
        for fm in FEATURE_MAPS
    }


def to_rows(d: int, m: int, coupling: str, stats, rho_col, r_col, protocol_rho: float) -> List[dict]:
    rows = []
    cells = sorted(set(zip(rho_col.tolist(), r_col.tolist())))
    for rho, r in cells:
        idx = ((rho_col == rho) & (r_col == r)).nonzero().squeeze(1)
        for fm, s in stats.items():
            per_seed_sq = s["sq"][:, idx].mean(1)  # average the pairs within a seed
            per_seed_err = s["err"][:, idx].mean(1)
            n = per_seed_sq.shape[0]
            k = math.exp(-r**2 / 2)
            mse = per_seed_sq.mean().item()
            rows.append(
                dict(
                    d=d,
                    m=m,
                    rho=rho,
                    rho_is_protocol=math.isclose(rho, protocol_rho),
                    r=r,
                    k_exact=k,
                    coupling=coupling,
                    feature_map=fm,
                    n_seeds=n,
                    mse=mse,
                    mse_se=(per_seed_sq.std(unbiased=True) / math.sqrt(n)).item(),
                    rel_mse=mse / k**2,
                    bias=per_seed_err.mean().item(),
                    bias_se=(per_seed_err.std(unbiased=True) / math.sqrt(n)).item(),
                    fp32_bad_frac=s["bad"][:, idx].mean().item(),
                )
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (d, m, feature_map, rho): geometric-mean MSE ratio vs ORF over the r grid, and
    how many r points each side wins by more than 2 combined SE."""
    out = []
    for (d, m, fm, rho), g in df.groupby(["d", "m", "feature_map", "rho"]):
        by = {c: g[g.coupling == c].sort_values("r").reset_index(drop=True) for c in COUPLINGS}
        orf = by["orf"]
        row = dict(d=d, m=m, feature_map=fm, rho=rho, rho_is_protocol=bool(g.rho_is_protocol.iloc[0]))
        for c in ("simrf", "rff"):
            other = by[c]
            ratio = other.mse / orf.mse
            gap_se = np.sqrt(other.mse_se**2 + orf.mse_se**2)
            row[f"mse_ratio_{c}_orf"] = float(np.exp(np.log(ratio).mean()))
            row[f"n_r_{c}_better_sig"] = int(((orf.mse - other.mse) > 2 * gap_se).sum())
            row[f"n_r_orf_better_than_{c}_sig"] = int(((other.mse - orf.mse) > 2 * gap_se).sum())
        row["median_rel_mse_orf"] = float(orf.rel_mse.median())
        row["fp32_bad_frac_max"] = float(g.fp32_bad_frac.max())
        row["max_abs_bias_over_se"] = float((g.bias.abs() / g.bias_se.clip(lower=1e-300)).max())
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="tidy CSV, one row per (d, m, rho, r, coupling, feature_map)")
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 64, 512])
    ap.add_argument("--rff-dims", type=int, nargs="+", default=[64, 256, 1024], help="m, the number of features")
    ap.add_argument("--seeds-small", type=int, default=500, help="seeds for d < 256")
    ap.add_argument("--seeds-large", type=int, default=200, help="seeds for d >= 256 (QR-bound)")
    ap.add_argument("--seeds", type=int, default=None, help="override both seed counts (smoke runs)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    t0 = time.time()
    for d in args.dims:
        n_seeds = args.seeds or (args.seeds_small if d < 256 else args.seeds_large)
        rhos = rho_grid(d)
        # Fixed pair geometry per d, shared by every coupling and m, so the comparisons
        # are across couplings on identical inputs.
        x, y, rho_col, r_col = make_pairs(d, rhos, torch.Generator().manual_seed(10_000 + d))
        for m in args.rff_dims:
            for coupling in COUPLINGS:
                stats = sweep_one(d, m, coupling, n_seeds, x, y)
                rows += to_rows(d, m, coupling, stats, rho_col, r_col, math.sqrt(d / 2))
                logger.info(f"d={d} m={m} {coupling:>5}: {n_seeds} seeds done ({time.time() - t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    summary = summarize(df)
    summary_path = csv_path.with_name(csv_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote {csv_path} ({len(df)} rows) and {summary_path}")

    m_max = max(args.rff_dims)
    for _, s in summary[summary.m == m_max].iterrows():
        logger.info(
            f"d={s.d:>4} m={m_max} {s.feature_map:>10} rho={s.rho:6.2f}{'*' if s.rho_is_protocol else ' '} | "
            f"simrf/orf={s.mse_ratio_simrf_orf:.3f} (sig better/worse at {s.n_r_simrf_better_sig}/"
            f"{s['n_r_orf_better_than_simrf_sig']} of {N_R} r) | rff/orf={s.mse_ratio_rff_orf:.3f} | "
            f"orf rel_mse={s.median_rel_mse_orf:.2e} | fp32 bad={s.fp32_bad_frac_max:.2f}"
        )


if __name__ == "__main__":
    main()
