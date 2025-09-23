#!/usr/bin/env python3
"""reliability.py in src/visualization.

Reliability diagram plotting utilities (smoothed ECE with kernel regression).
Refactored for the new repository structure.
"""
from typing import Any, Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.metrics.smooth_ece import smoothed_ece_logit
from src.visualization.density import density_ticks, nadaraya_watson, reflected_kde
from src.visualization.style import set_default_style


def _bootstrap_ci_width(
    f: np.ndarray,
    y: np.ndarray,
    func,
    n_resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
    **kwargs,
) -> float:
    """Bootstrap confidence interval width for a calibration error metric."""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(f)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals.append(func(f[idx], y[idx], **kwargs))
    vals = np.sort(vals)
    lo = np.percentile(vals, (1 - confidence) / 2 * 100)
    hi = np.percentile(vals, (1 + confidence) / 2 * 100)
    return float(max(hi - vals.mean(), vals.mean() - lo))


def prepare_rel_diagram(
    f: np.ndarray,
    y: np.ndarray,
    sigma: float = 0.2,
    n_bootstrap: int = 200,
    report_ce: bool = True,
    report_ce_ci: bool = True,
) -> Dict[str, Any]:
    """
    Prepare calibration data for reliability diagram.

    Args:
        f: predicted probabilities in [0,1]
        y: true binary labels {0,1}
        sigma: kernel bandwidth for smoothing
        n_bootstrap: number of bootstrap resamples for CI
        report_ce: whether to compute smoothed ECE
        report_ce_ci: whether to compute CI via bootstrap

    Returns:
        Dictionary with mesh, main curve, densities, calibration error, etc.
    """
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    assert f.shape == y.shape

    mesh = np.linspace(0, 1, 200)
    outputs: Dict[str, Any] = {"mesh": mesh}

    # Main smoothed calibration curve
    mu = nadaraya_watson(f, y, mesh, sigma=sigma, boundary="reflected")
    outputs["mu"] = mu

    # Bootstrapped confidence intervals
    if n_bootstrap > 0:
        mus = []
        rng = np.random.default_rng(0)
        n = len(f)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
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
        ce = smoothed_ece_logit(f, y, sigma=sigma)
        outputs["ce"] = ce
        if report_ce_ci:
            wid = _bootstrap_ci_width(f, y, smoothed_ece_logit, sigma=sigma)
            outputs["ce_ci_width"] = wid

    # Subsample for rug ticks
    rng = np.random.default_rng(42)
    idx = rng.choice(len(f), size=min(200, len(f)), replace=False)
    outputs["f_samp"] = f[idx]
    outputs["y_samp"] = y[idx]

    return outputs
#
#
# def plot_rel_diagram(
#     diagram: Dict[str, Any],
#     ax: plt.Axes | None = None,
#     color: str = "red",
#     show_density_ticks: bool = True,
#     show_confidence_band: bool = True,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Plot a reliability diagram from prepared calibration data.
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6, 6))
#     else:
#         fig = ax.figure
#
#     set_default_style()
#
#     t = diagram["mesh"]
#     mu = diagram["mu"]
#
#     # Reference diagonal
#     ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
#
#     # Confidence bands
#     if show_confidence_band and "upper" in diagram and "lower" in diagram:
#         ax.fill_between(t, diagram["lower"], diagram["upper"], color="gray", alpha=0.3)
#
#     # Main smoothed line
#     ax.plot(t, mu, color=color, lw=2, label="Smoothed calibration")
#
#     # Density ticks
#     if show_density_ticks and "f_samp" in diagram:
#         fs, ys = diagram["f_samp"], diagram["y_samp"]
#         ax.vlines(fs[ys == 0], 0, 0.02, colors="blue", lw=0.5, alpha=0.7)
#         ax.vlines(fs[ys == 1], 0, 0.02, colors="green", lw=0.5, alpha=0.7)
#
#     # Labels
#     ax.set_xlabel("Predicted probability")
#     ax.set_ylabel("Empirical accuracy")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect("equal")
#
#     # Calibration error annotation
#     if "ce" in diagram:
#         ce = diagram["ce"]
#         if "ce_ci_width" in diagram:
#             wid = diagram["ce_ci_width"]
#             ax.text(0.05, 0.9, f"smECE = {ce:.3f} ± {wid:.3f}")
#         else:
#             ax.text(0.05, 0.9, f"smECE = {ce:.3f}")
#
#     return fig, ax
#
#
# def rel_diagram(
#     f: np.ndarray,
#     y: np.ndarray,
#     sigma: float = 0.2,
#     n_bootstrap: int = 200,
#     report_ce: bool = True,
#     report_ce_ci: bool = True,
#     ax: plt.Axes | None = None,
#     fig: plt.Figure | None = None,
#     color: str = "red",
#     show_density_ticks: bool = True,
#     show_confidence_band: bool = True,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """Convenience wrapper: prepare and plot reliability diagram."""
#     # --- data preparation ---
#     diagram = prepare_rel_diagram(
#         f,
#         y,
#         sigma=sigma,
#         n_bootstrap=n_bootstrap,
#         report_ce=report_ce,
#         report_ce_ci=report_ce_ci,
#     )
#
#     # --- plotting ---
#     return plot_rel_diagram(
#         diagram,
#         ax=ax,
#         color=color,
#         show_density_ticks=show_density_ticks,
#         show_confidence_band=show_confidence_band,
#     )



# def prepare_rel_diagram(
#     f: np.ndarray,
#     y: np.ndarray,
#     sigma: float = 0.2,
#     n_bootstrap: int = 200,
#     report_ce: bool = True,
#     report_ce_ci: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Prepare calibration data for reliability diagram.
#
#     Args:
#         f: predicted probabilities in [0,1]
#         y: true binary labels {0,1}
#         sigma: kernel bandwidth for smoothing
#         n_bootstrap: number of bootstrap resamples for confidence band
#         report_ce: whether to compute smoothed ECE
#         report_ce_ci: whether to compute CI via bootstrap
#
#     Returns:
#         Dictionary with mesh, main curve, densities, calibration error, etc.
#     """
#     f = np.asarray(f, dtype=np.float64).reshape(-1)
#     y = np.asarray(y, dtype=np.float64).reshape(-1)
#     assert f.shape == y.shape
#
#     mesh = np.linspace(0, 1, 200)
#     outputs: Dict[str, Any] = {"mesh": mesh}
#
#     # Main smoothed calibration curve (via kernel regression)
#     # Here we use Gaussian kernel smoothing manually
#     z = f.reshape(1, -1)
#     X = mesh.reshape(-1, 1)
#     K = norm.pdf((X - z) / sigma)
#     mu = (K @ y) / (K.sum(axis=1) + 1e-12)
#     outputs["mu"] = mu
#
#     # Bootstrapped confidence intervals
#     if n_bootstrap > 0:
#         mus = []
#         rng = np.random.default_rng(0)
#         n = len(f)
#         for _ in range(n_bootstrap):
#             idx = rng.integers(0, n, size=n)
#             f_b, y_b = f[idx], y[idx]
#             Kb = norm.pdf((X - f_b.reshape(1, -1)) / sigma)
#             mu_b = (Kb @ y_b) / (Kb.sum(axis=1) + 1e-12)
#             mus.append(mu_b)
#         mus = np.array(mus)
#         outputs["lower"] = np.percentile(mus, 2.5, axis=0)
#         outputs["upper"] = np.percentile(mus, 97.5, axis=0)
#
#     # Density of predictions
#     outputs["density"] = np.histogram(f, bins=50, range=(0, 1), density=True)[0]
#     outputs["densities"] = reflected_kde(f, y, x_eval=mesh, sigma=sigma)
#
#     # Calibration error
#     if report_ce:
#         ce = smoothed_ece_logit(f, y, sigma=sigma)
#         outputs["ce"] = ce
#         if report_ce_ci:
#             ci = bootstrap_confidence_interval(f, y, smoothed_ece_logit, sigma=sigma)
#             outputs["ce_ci_width"] = ci
#
#     return outputs


def plot_rel_diagram(
    diagram: Dict[str, Any],
    ax: plt.Axes | None = None,
    color: str = "red",
    show_density_ticks: bool = True,
    show_confidence_band: bool = True,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a reliability diagram from prepared calibration data.

    Args:
        diagram: dictionary from prepare_rel_diagram
        ax: optional matplotlib axis
        color: line color
        show_density_ticks: whether to show tick marks for prediction density
        show_confidence_band: whether to shade bootstrap CIs

    Returns:
        (fig, ax) matplotlib figure and axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    set_default_style()

    t = diagram["mesh"]
    mu = diagram["mu"]

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)

    # Confidence bands
    if show_confidence_band and "upper" in diagram and "lower" in diagram:
        ax.fill_between(t, diagram["lower"], diagram["upper"], color="gray", alpha=0.3)

    # Main smoothed line
    ax.plot(t, mu, color=color, lw=2, label="Smoothed calibration")

    # Density ticks
    if show_density_ticks and "f_samp" in diagram:
        fs, ys = diagram["f_samp"], diagram["y_samp"]
        ax.vlines(fs[ys == 0], 0, 0.02, colors="blue", lw=0.5, alpha=0.7)
        ax.vlines(fs[ys == 1], 0, 0.02, colors="green", lw=0.5, alpha=0.7)

    # Labels
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


def rel_diagram(
    f: np.ndarray,
    y: np.ndarray,
    sigma: float = 0.2,
    n_bootstrap: int = 200,
    report_ce: bool = True,
    report_ce_ci: bool = True,
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
    color: str = "red",
    show_density_ticks: bool = True,
    show_confidence_band: bool = True,
    show_diagonal: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convenience wrapper: prepare and plot reliability diagram.

    Args:
        f: Predicted probabilities in [0,1].
        y: Binary labels (0 or 1).
        sigma: Kernel bandwidth for smoothing.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
        report_ce: Whether to compute calibration error (smECE).
        report_ce_ci: Whether to compute confidence interval for smECE.
        ax: Optional matplotlib axis to plot on.
        fig: Optional matplotlib figure to attach to.
        color: Line/point color for calibration curve.
        show_density_ticks: Add rug ticks for density visualization.
        show_confidence_band: Add shaded bootstrap confidence intervals.
        show_diagonal: Plot perfect calibration diagonal.

    Returns:
        (fig, ax) of the plot.
    """
    # --- data preparation ---
    diagram = prepare_rel_diagram(
        f,
        y,
        sigma=sigma,
        n_bootstrap=n_bootstrap,
        report_ce=report_ce,
        report_ce_ci=report_ce_ci,
    )

    # --- plotting ---
    fig, ax = plot_rel_diagram(
        diagram,
        ax=ax,
        fig=fig,
        color=color,
        show_density_ticks=show_density_ticks,
        show_confidence_band=show_confidence_band,
        show_diagonal=show_diagonal,
        **kwargs,
    )
    return fig, ax

