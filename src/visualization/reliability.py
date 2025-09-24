#!/usr/bin/env python3
"""reliability.py in src/visualization.

Reliability diagram plotting utilities (smoothed ECE with kernel regression).
"""
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV


from src.metrics.smooth_ece import smECE_fast_compat
from src.visualization.density import reflected_kde, nadaraya_watson
from src.visualization.style import set_default_style

# === BINNED ECE ===
def prepare_rel_diagram_binned(f: np.ndarray, y: np.ndarray, nbins: int = 15):
    # TODO: use GridSearchCV to find optimal nbins based on ECE minimization

    f = np.asarray(f, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    # Bin indices
    bin_idx = np.floor(f * nbins).astype(int).clip(0, nbins - 1)
    buckets = np.zeros(nbins)
    sizes = np.zeros(nbins, int)
    for i, yi in zip(bin_idx, y):
        buckets[i] += yi
        sizes[i] += 1
    preds = np.array([b / s if s > 0 else 0 for b, s in zip(buckets, sizes)])

    # avoid division by zero
    if sizes.sum() == 0 or sizes.max() == 0:
        raise ZeroDivisionError("All sizes are zero, cannot compute alphas or ECE.")

    alphas = sizes / sizes.max()
    ece = np.abs(buckets - sizes * preds).sum() / len(f) if sizes.sum() > 0 else 0.0

    t = np.linspace(0, 1, nbins)
    return {"t": t, "mu": preds, "buckets": preds, "alphas": alphas, "ece": ece}


def plot_rel_diagram_binned(diagram: dict, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    nbins = len(diagram["buckets"])
    for i, yb in enumerate(diagram["buckets"]):
        lb, ub = i / nbins, (i + 1) / nbins
        ax.stairs([yb], [lb, ub], fill=True, color="gray", alpha=diagram["alphas"][i])

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("f")
    ax.set_ylabel("E[y|f]")
    ax.text(0.05, 0.9, f"ECE$_{{{nbins}}}: {diagram['ece']:.3f}$")
    return fig, ax


def rel_diagram_binned(f: np.ndarray, y: np.ndarray, nbins: int = 15, fig=None, ax=None):
    return plot_rel_diagram_binned(prepare_rel_diagram_binned(f, y, nbins), fig=fig, ax=ax)


# === SMOOTHED ECE ===
def prepare_rel_diagram_smoothed(
    f: np.ndarray,
    y: np.ndarray,
    sigma: float = 0.1,
    n_bootstrap: int = 500,
    report_ce: bool = True,
    report_ce_ci: bool = True,
    num_mesh: int = 1000
) -> Dict[str, Any]:
    """
    Prepare calibration data for a smoothed reliability diagram.

    Args:
        f: Predicted probabilities in [0,1].
        y: True binary labels {0,1}.
        sigma: Kernel bandwidth for Nadaraya–Watson smoothing.
        n_bootstrap: Number of bootstrap resamples for CIs.
        report_ce: Whether to compute smECE.
        report_ce_ci: Whether to compute bootstrap CI for smECE.

    Returns:
        Dictionary with mesh, main curve, densities, calibration error, etc.
    """
    f = np.asarray(f, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    assert f.shape == y.shape and f.size > 0

    mesh = np.linspace(0, 1, num=num_mesh)
    outputs: Dict[str, Any] = {"mesh": mesh}

    # Main smoothed calibration curve
    mu = nadaraya_watson(f, y, mesh, sigma=sigma, boundary="reflected")
    outputs["mu"] = mu

    # Bootstrapped confidence bands
    if n_bootstrap > 0:
        rng = np.random.default_rng(0)
        mus = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(f), size=len(f))
            mu_b = nadaraya_watson(f[idx], y[idx], mesh, sigma=sigma)
            mus.append(mu_b)
        mus = np.array(mus)
        outputs["lower"] = np.percentile(mus, 2.5, axis=0)
        outputs["upper"] = np.percentile(mus, 97.5, axis=0)

    # Unconditional density
    outputs["density"] = reflected_kde(f, mesh, sigma)

    # Conditional densities
    outputs["densities"] = [
        reflected_kde(f[y == 0], mesh, sigma),
        reflected_kde(f[y == 1], mesh, sigma),
    ]

    # Calibration error
    if report_ce:
        ce, sigma_opt = smECE_fast_compat(f, y, return_width=True)
        outputs["ce"] = ce
        outputs["sigma_opt"] = sigma_opt
        if report_ce_ci:
            vals = []
            rng = np.random.default_rng(1)
            for _ in range(200):
                idx = rng.integers(0, len(f), size=len(f))
                vals.append(smECE_fast_compat(f[idx], y[idx]))
            lo, hi = np.percentile(vals, [2.5, 97.5])
            outputs["ce_ci_width"] = max(outputs["ce"] - lo, hi - outputs["ce"])

    # Subsample for rug ticks
    rng = np.random.default_rng(42)
    idx = rng.choice(len(f), size=min(200, len(f)), replace=False)
    outputs["f_samp"], outputs["y_samp"] = f[idx], y[idx]

    return outputs


def plot_rel_diagram_smoothed(
    diagram: Dict[str, Any],
    ax: plt.Axes | None = None,
    color: str = "red",
    show_density_ticks: bool = True,
    show_confidence_band: bool = True,
    show_diagonal: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a smoothed reliability diagram from prepared data.

    Args:
        diagram: Dictionary returned by prepare_rel_diagram_smoothed.
        ax: Optional Matplotlib axis to draw on.
        color: Line color for main curve.
        show_density_ticks: Whether to add rug ticks for prediction density.
        show_confidence_band: Whether to shade bootstrap CI.
        show_diagonal: Whether to show perfect calibration diagonal.

    Returns:
        (fig, ax) tuple of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    set_default_style()

    t = diagram["mesh"]
    mu = diagram["mu"]

    # Diagonal (perfect calibration)
    if show_diagonal:
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)

    # Confidence bands
    if show_confidence_band and "upper" in diagram and "lower" in diagram:
        ax.fill_between(t, diagram["lower"], diagram["upper"], color="gray", alpha=0.3)

    # Main smoothed calibration curve
    ax.plot(t, mu, color=color, lw=2, label="Smoothed calibration")

    # Density ticks
    if show_density_ticks and "f_samp" in diagram:
        fs, ys = diagram["f_samp"], diagram["y_samp"]
        ax.vlines(fs[ys == 0], 0, 0.02, colors="blue", lw=0.5, alpha=0.7)
        ax.vlines(fs[ys == 1], 0, 0.02, colors="green", lw=0.5, alpha=0.7)

    # Labels and axes
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Calibration error annotation
    if "ce" in diagram:
        ce = diagram["ce"]
        if "ce_ci_width" in diagram:
            wid = diagram["ce_ci_width"]
            ax.text(0.05, 0.9, f"smECE = {ce:.3f} ± {wid:.3f}")
        else:
            ax.text(0.05, 0.9, f"smECE = {ce:.3f}")

    return fig, ax

def rel_diagram_smoothed(f: np.ndarray, y: np.ndarray, **kwargs):
    return plot_rel_diagram_smoothed(prepare_rel_diagram_smoothed(f, y, **kwargs))