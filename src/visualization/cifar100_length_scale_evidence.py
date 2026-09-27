#!/usr/bin/env python3
"""CIFAR-100 GP evidence over the length scale, against the variance's OOD AUROC.

Consumes the summary CSVs written by `scripts/metrics/cifar100_length_scale_evidence.py`
(any `*_summary.csv` in the figures dir, so the rff_dim 16384 run can live in its own file)
and its per-seed median-heuristic rows.

  * Left pair: type-II log evidence per datum on train and on held-out val, relative to its
    maximum per rff_dim (so the curves share an axis), mean +/- 1 std over 3 backbone seeds.
  * Right pair: the decoupled study's variance AUROC vs CIFAR-10 / SVHN at the same l.
  * Dashed vertical lines: the evidence argmax per rff_dim; grey dotted: median heuristic.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

# Wong 2011 palette, as in style.py and the decoupled-variance figure: one hue per rff_dim.
DIM_COLORS = {1024: "#0173B2", 4096: "#D55E00", 16384: "#029E73"}
PANELS = (
    ("log_ev_type2", "Type-II log evidence / N, train (Δ from max) ↑"),
    ("log_ev_val_type2", "Type-II log evidence / N, val (Δ from max) ↑"),
    ("auroc_var_cifar10", "Variance AUROC vs CIFAR-10 ↑"),
    ("auroc_var_svhn", "Variance AUROC vs SVHN ↑"),
)


def plot(summary: pd.DataFrame, l_median: float) -> Figure:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.0 * len(PANELS), 3.8))
    argmax = {m: g.loc[g["log_ev_type2_mean"].idxmax(), "length_scale"] for m, g in summary.groupby("rff_dim")}
    for ax, (metric, title) in zip(axes, PANELS):
        for m, color in DIM_COLORS.items():
            g = summary[summary.rff_dim == m].sort_values("length_scale")
            if g.empty:
                continue
            if metric.startswith("log_ev"):
                y, err = g[f"{metric}_mean"] - g[f"{metric}_mean"].max(), g[f"{metric}_std"]
            else:
                g = g.dropna(subset=[f"{metric}_mean"])
                y, err = g[f"{metric}_mean"], None
            ax.errorbar(g.length_scale, y, yerr=err, color=color, marker="o", ms=4, lw=1.5, capsize=2,
                        label=f"rff_dim {m}")
            ax.axvline(argmax[m], color=color, ls="--", lw=1.0, alpha=0.8)
        ax.axvline(l_median, color="0.5", ls=":", lw=1.3, label=f"median heuristic (ℓ = {l_median:.1f})")
        if metric.startswith("auroc"):
            ax.axhline(0.5, color="0.7", lw=4, alpha=0.4, zorder=0)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ls = sorted(summary.length_scale.unique())
        ax.set_xticks(ls)
        ax.set_xticklabels([f"{v:g}" for v in ls], fontsize=8)
        ax.set_xlabel("length scale ℓ", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("CIFAR-100 SpecReg: GP evidence over ℓ vs variance OOD AUROC (frozen backbones, 3 seeds; "
                 "dashed = evidence argmax)", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = ROOT / "figures" / "cifar100_length_scale_evidence"
    parser.add_argument("--figures-dir", default=str(default_dir))
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    summary = pd.concat([pd.read_csv(p) for p in sorted(figures_dir.glob("*_summary.csv"))], ignore_index=True)
    per_seed = pd.concat([pd.read_csv(p) for p in sorted(figures_dir.glob("*_per_seed.csv"))], ignore_index=True)
    l_median = per_seed[per_seed.kind == "median"].drop_duplicates("seed").length_scale.mean()

    set_default_style()
    fig = plot(summary, l_median)
    fig.savefig(figures_dir / "cifar100_length_scale_evidence.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "cifar100_length_scale_evidence.pdf", bbox_inches="tight")
    print(f"Saved figure to: {figures_dir}")


if __name__ == "__main__":
    main()
