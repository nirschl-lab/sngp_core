#!/usr/bin/env python3
"""Kernel-approximation error of random-feature couplings (rff / orf / simrf) under cos and
positive feature maps.

Consumes the tidy CSV and summary written by
`scripts/metrics/random_feature_kernel_mse.py` and draws two figures:

  * `*_mse_vs_distance` -- rows are input dim d, columns are feature map. Each panel shows MSE
    against r = ||x - y|| / l at a fixed rho = ||x|| / l and m, one line per coupling, with
    +/- 1 SE bands. It shows *where* in distance a coupling helps.
  * `*_ratio_vs_rho` -- one panel per feature map, the geometric-mean MSE ratio simrf/orf
    (solid) and rff/orf (dashed) against rho, one color per d, with the protocol rho marked.
    Below 1 means better than ORF. This is the figure that answers "does SimRF help, and does
    it survive at the SNGP head width and length scale".
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
from matplotlib.figure import Figure

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

# Wong 2011 palette, as in style.py
COUPLING_COLORS = {"rff": "#949494", "orf": "#0173B2", "simrf": "#D55E00"}
DIM_COLORS = ["#029E73", "#CC78BC", "#DE8F05", "#56B4E9"]
FEATURE_MAP_TITLES = {"cos": "cos (current SNGP)", "positive": "positive (FAVOR+)", "hyperbolic": "hyperbolic (FAVOR++)"}


def plot_mse_vs_distance(df: pd.DataFrame, rho: float, m: int) -> Figure:
    dims = sorted(df.d.unique())
    fms = [f for f in FEATURE_MAP_TITLES if f in set(df.feature_map)]
    fig, axes = plt.subplots(len(dims), len(fms), figsize=(5 * len(fms), 3.6 * len(dims)), squeeze=False)
    sel = df[(df.m == m) & ((df.rho - rho).abs() < 1e-9)]
    # r = 2 rho is y = -x: PRFs are exact there (MSE ~ 1e-33), which would flatten every
    # other point on a log axis. It carries no comparison between couplings, so drop it.
    sel = sel[(sel.r - 2 * sel.rho).abs() > 1e-9]
    for i, d in enumerate(dims):
        for j, fm in enumerate(fms):
            ax = axes[i, j]
            for c, color in COUPLING_COLORS.items():
                g = sel[(sel.d == d) & (sel.feature_map == fm) & (sel.coupling == c)].sort_values("r")
                ax.plot(g.r, g.mse, color=color, label=c, lw=2)
                ax.fill_between(g.r, (g.mse - g.mse_se).clip(lower=g.mse.min() * 1e-3), g.mse + g.mse_se, color=color, alpha=0.2)
            ax.set_yscale("log")
            ax.set_title(f"d={d} · {FEATURE_MAP_TITLES[fm]}", fontsize=12)
            if i == len(dims) - 1:
                ax.set_xlabel(r"$r = \|x-y\| / \ell$")
            if j == 0:
                ax.set_ylabel("kernel MSE")
    axes[0, 0].legend(fontsize=11)
    fig.suptitle(rf"Kernel MSE vs distance ($\rho = \|x\|/\ell = {rho:g}$, m = {m})", y=1.0)
    fig.tight_layout()
    return fig


def plot_ratio_vs_rho(summary: pd.DataFrame, m: int) -> Figure:
    fms = [f for f in FEATURE_MAP_TITLES if f in set(summary.feature_map)]
    fig, axes = plt.subplots(1, len(fms), figsize=(5.5 * len(fms), 4.4), squeeze=False, sharey=True)
    sel = summary[summary.m == m]
    for j, fm in enumerate(fms):
        ax = axes[0, j]
        for k, d in enumerate(sorted(sel.d.unique())):
            g = sel[(sel.d == d) & (sel.feature_map == fm)].sort_values("rho")
            color = DIM_COLORS[k % len(DIM_COLORS)]
            ax.plot(g.rho, g.mse_ratio_simrf_orf, "-o", color=color, lw=2, label=f"simrf/orf, d={d}")
            ax.plot(g.rho, g.mse_ratio_rff_orf, "--", color=color, lw=1.2, alpha=0.7, label=f"rff/orf, d={d}")
            proto = g[g.rho_is_protocol]
            ax.scatter(proto.rho, proto.mse_ratio_simrf_orf, marker="*", s=220, color=color, zorder=5, edgecolor="k")
        ax.axhline(1.0, color="k", lw=1)
        if fm != "cos":
            # PRF variance grows like exp(||x + y||^2): past rho ~ 2 every coupling returns
            # k_hat ~ 0 (relative MSE >= 1), so a ratio near 1 there means "all fail equally".
            ax.axvspan(2.0, ax.get_xlim()[1], color="0.85", zorder=0)
            ax.text(2.2, 0.12, "PRF estimates\ncollapse here", fontsize=10, color="0.35")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\rho = \|x\| / \ell$  (★ = SNGP protocol)")
        ax.set_title(FEATURE_MAP_TITLES[fm], fontsize=12)
    axes[0, 0].set_ylabel("MSE ratio vs ORF (geo-mean over r)")
    axes[0, -1].legend(fontsize=9, loc="best")
    fig.suptitle(f"Does coupling beat ORF? (m = {m}; below 1 = better)", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = ROOT / "figures" / "random_feature_kernel_mse"
    parser.add_argument("--csv", default=str(default_dir / "random_feature_kernel_mse.csv"))
    parser.add_argument("--figures-dir", default=str(default_dir))
    parser.add_argument("--rho", type=float, default=1.0, help="rho for the MSE-vs-distance figure")
    parser.add_argument("--m", type=int, default=None, help="number of features to plot; default the largest")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    summary = pd.read_csv(csv_path.with_name(csv_path.stem + "_summary.csv"))
    m = args.m or int(df.m.max())
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    set_default_style()
    for stem, fig in (
        ("random_feature_kernel_mse_vs_distance", plot_mse_vs_distance(df, args.rho, m)),
        ("random_feature_kernel_mse_ratio_vs_rho", plot_ratio_vs_rho(summary, m)),
    ):
        fig.savefig(figures_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
