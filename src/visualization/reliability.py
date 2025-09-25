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
def prepare_rel_diagram_binned(
    f: np.ndarray,
    y: np.ndarray,
    nbins: int = 15,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Prepare calibration data for a binned reliability diagram.

    Supports binary and multiclass (one-vs-rest).

    Args:
        f: Predicted probabilities.
            - Binary: shape (n,)
            - Multiclass: shape (n, C)
        y: Labels.
            - Binary: shape (n,)
            - Multiclass: shape (n,), int labels {0,…,C-1}
        nbins: Number of bins.
        average: Aggregation for smECE ("macro" or "weighted").

    Returns:
        Dictionary with per-class curves, densities, and aggregated ECE.
    """
    # TODO: use GridSearchCV to find optimal nbins based on ECE minimization

    f = np.asarray(f, float)
    y = np.asarray(y, int).reshape(-1)

    outputs: Dict[str, Any] = {"nbins": nbins}

    # === Binary case ===
    if f.ndim == 1:
        assert f.shape == y.shape

        bin_idx = np.floor(f * nbins).astype(int).clip(0, nbins - 1)
        buckets = np.zeros(nbins)
        sizes = np.zeros(nbins, int)
        for i, yi in zip(bin_idx, y):
            buckets[i] += yi
            sizes[i] += 1
        preds = np.array([b / s if s > 0 else 0 for b, s in zip(buckets, sizes)])

        alphas = sizes / sizes.max() if sizes.max() > 0 else np.zeros_like(sizes)
        ece = np.sum(np.abs(preds - (np.arange(nbins) + 0.5) / nbins) * sizes) / len(f)

        outputs["mu"] = [preds]
        outputs["alphas"] = [alphas]
        outputs["ece"] = ece
        return outputs

    # === Multiclass case ===
    n, C = f.shape
    curves, alphas_all, eces, class_weights = [], [], [], np.zeros(C)

    for c in range(C):
        f_c = f[:, c]
        y_c = (y == c).astype(float)

        bin_idx = np.floor(f_c * nbins).astype(int).clip(0, nbins - 1)
        buckets = np.zeros(nbins)
        sizes = np.zeros(nbins, int)
        for i, yi in zip(bin_idx, y_c):
            buckets[i] += yi
            sizes[i] += 1
        preds = np.array([b / s if s > 0 else 0 for b, s in zip(buckets, sizes)])

        alphas = sizes / sizes.max() if sizes.max() > 0 else np.zeros_like(sizes)
        ece_c = np.sum(np.abs(preds - (np.arange(nbins) + 0.5) / nbins) * sizes) / n

        curves.append(preds)
        alphas_all.append(alphas)
        eces.append(ece_c)
        class_weights[c] = y_c.mean()

    outputs["mu"] = curves
    outputs["alphas"] = alphas_all
    outputs["per_class_ece"] = np.array(eces)
    outputs["class_weights"] = class_weights

    if average == "macro":
        outputs["ece"] = np.mean(eces)
    elif average == "weighted" and class_weights.sum() > 0:
        outputs["ece"] = np.average(eces, weights=class_weights)
    else:
        outputs["ece"] = np.mean(eces)

    return outputs


def plot_rel_diagram_binned(
    diagram: Dict[str, Any],
    ax: plt.Axes | None = None,
    colors: list[str] | None = None,
    title: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a binned reliability diagram (binary or multiclass).
    Uses legends for class labels and per-class ECEs, positioned outside the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    else:
        fig = ax.figure

    set_default_style()

    nbins = diagram["nbins"]
    mus = diagram["mu"]

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3, label="Perfect calibration")

    handles, labels = [], []

    # === Binary ===
    if len(mus) == 1:
        preds = mus[0]
        for i, yb in enumerate(preds):
            lb, ub = i / nbins, (i + 1) / nbins
            ax.stairs([yb], [lb, ub], fill=True, color="gray", alpha=0.6)

        labels.append(f"ECE$_{{{nbins}}}$ = {diagram['ece']:.3f}")
        handles.append(plt.Line2D([0], [0], color="gray", lw=2))

    # === Multiclass ===
    else:
        C = len(mus)
        if colors is None:
            colors = plt.cm.tab10.colors if C <= 10 else plt.cm.tab20.colors

        for c, preds in enumerate(mus):
            color = colors[c % len(colors)]
            for i, yb in enumerate(preds):
                lb, ub = i / nbins, (i + 1) / nbins
                ax.stairs([yb], [lb, ub], fill=True, color=color, alpha=0.3)

            ce_c = diagram["per_class_ece"][c]
            handles.append(plt.Line2D([0], [0], color=color, lw=2))
            labels.append(f"Class {c}: {ce_c:.3f}")

        # Aggregate ECE
        handles.append(plt.Line2D([0], [0], color="black", lw=0))
        labels.append(f"Aggregate ECE = {diagram['ece']:.3f}")

    # Place legend outside plot (right side)
    if labels:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=True,
        )

    # Labels and axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")

    if title:
        ax.set_title(title)

    # fig.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for legend
    return fig, ax



def rel_diagram_binned(f: np.ndarray, y: np.ndarray, nbins: int = 15, **kwargs):
    return plot_rel_diagram_binned(prepare_rel_diagram_binned(f, y, nbins), **kwargs)


# === SMOOTHED ECE ===
def prepare_rel_diagram_smoothed(
    f: np.ndarray,
    y: np.ndarray,
    sigma: float = 0.1,
    n_bootstrap: int = 500,
    report_ce: bool = True,
    report_ce_ci: bool = True,
    num_mesh: int = 1000,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Prepare calibration data for a smoothed reliability diagram.

    Supports both binary and multiclass (one-vs-rest) inputs.

    Args:
        f: Predicted probabilities.
            - Binary: shape (n,), probabilities for positive class.
            - Multiclass: shape (n, C), probabilities for each class.
        y: True labels.
            - Binary: shape (n,), values in {0,1}.
            - Multiclass: shape (n,), integer class labels in {0, …, C-1}.
        sigma: Kernel bandwidth for Nadaraya–Watson smoothing.
        n_bootstrap: Number of bootstrap resamples for CIs.
        report_ce: Whether to compute smECE.
        report_ce_ci: Whether to compute bootstrap CI for smECE.
        average: Aggregation for smECE in multiclass ("macro" or "weighted").

    Returns:
        Dictionary with mesh, curves, densities, calibration error, etc.
    """
    f = np.asarray(f, float)
    y = np.asarray(y, int).reshape(-1)

    mesh = np.linspace(0, 1, num=num_mesh)
    outputs: Dict[str, Any] = {"mesh": mesh}

    # === Binary case ===
    if f.ndim == 1:
        assert f.shape == y.shape and f.size > 0
        mu = nadaraya_watson(f, y, mesh, sigma=sigma, boundary="reflected")
        outputs["mu"] = [mu]

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

        outputs["density"] = [reflected_kde(f, mesh, sigma)]
        outputs["densities"] = [
            reflected_kde(f[y == 0], mesh, sigma),
            reflected_kde(f[y == 1], mesh, sigma),
        ]

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

        rng = np.random.default_rng(42)
        idx = rng.choice(len(f), size=min(200, len(f)), replace=False)
        outputs["f_samp"], outputs["y_samp"] = f[idx], y[idx]

        return outputs

    # === Multiclass case ===
    n, C = f.shape
    curves, densities, ces, sigmas = [], [], [], []
    class_weights = np.zeros(C)

    for c in range(C):
        f_c = f[:, c]
        y_c = (y == c).astype(float)

        mu_c = nadaraya_watson(f_c, y_c, mesh, sigma=sigma, boundary="reflected")
        curves.append(mu_c)
        densities.append(reflected_kde(f_c, mesh, sigma))

        if report_ce:
            ce_c, sigma_c = smECE_fast_compat(f_c, y_c, return_width=True)
            ces.append(ce_c)
            sigmas.append(sigma_c)
            class_weights[c] = y_c.mean()

    outputs["mu"] = curves
    outputs["density"] = densities
    outputs["per_class_ce"] = np.array(ces)
    outputs["per_class_sigma"] = np.array(sigmas)
    outputs["class_weights"] = class_weights

    if report_ce:
        if average == "macro":
            outputs["ce"] = np.mean(ces)
        elif average == "weighted" and class_weights.sum() > 0:
            outputs["ce"] = np.average(ces, weights=class_weights)
        else:
            outputs["ce"] = np.mean(ces)

    return outputs


def plot_rel_diagram_smoothed(
    diagram: Dict[str, Any],
    ax: plt.Axes | None = None,
    colors: list[str] | None = None,
    show_confidence_band: bool = True,
    show_diagonal: bool = True,
    title: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a smoothed reliability diagram from prepared data (binary or multiclass).
    Uses external legends (like binned), with harmonized colors.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    else:
        fig = ax.figure

    set_default_style()

    t = diagram["mesh"]
    mus = diagram["mu"]

    # Choose consistent color palette
    if colors is None:
        C = len(mus)
        colors = plt.cm.tab10.colors if C <= 10 else plt.cm.tab20.colors

    # Perfect calibration diagonal
    if show_diagonal:
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3, label="Perfect calibration")

    handles, labels = [], []

    # === Binary case ===
    if len(mus) == 1:
        mu = mus[0]
        if show_confidence_band and "upper" in diagram and "lower" in diagram:
            ax.fill_between(
                t, diagram["lower"], diagram["upper"],
                color="gray", alpha=0.3, label="95% CI"
            )
        ax.plot(t, mu, color=colors[0], lw=2, label="Smoothed calibration")

        if "ce" in diagram:
            ce_text = f"smECE = {diagram['ce']:.3f}"
            if "ce_ci_width" in diagram:
                ce_text += f" ± {diagram['ce_ci_width']:.3f}"
            handles.append(plt.Line2D([0], [0], color=colors[0], lw=2))
            labels.append(ce_text)

    # === Multiclass case ===
    else:
        C = len(mus)
        for c, mu in enumerate(mus):
            color = colors[c % len(colors)]
            ax.plot(t, mu, lw=2, color=color)
            ce_c = diagram.get("per_class_ce", [None] * C)[c]
            if ce_c is not None:
                handles.append(plt.Line2D([0], [0], color=color, lw=2))
                labels.append(f"Class {c}: {ce_c:.3f}")

        # Add aggregate smECE
        if "ce" in diagram:
            handles.append(plt.Line2D([0], [0], color="black", lw=0))
            labels.append(f"Aggregate smECE = {diagram['ce']:.3f}")

    # Legend outside the plot
    if labels:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),  # slightly outside
            borderaxespad=0,
            frameon=True,
        )

    # Labels and axes
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)

    # Leave space on the right for the legend
    # fig.tight_layout(rect=[0, 0, 0.8, 1])

    return fig, ax

def rel_diagram_smoothed(f: np.ndarray, y: np.ndarray, **kwargs):
    return plot_rel_diagram_smoothed(prepare_rel_diagram_smoothed(f, y, **kwargs))