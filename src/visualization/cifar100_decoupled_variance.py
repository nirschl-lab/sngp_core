#!/usr/bin/env python3
"""CIFAR-100 decoupled GP variance: OOD detection vs the variance head's length scale.

Consumes the summary CSV written by `scripts/metrics/cifar100_decoupled_variance.py`.
Logits come from the trained l = 20 SpecReg head, so accuracy is fixed; only the variance is
recomputed, post hoc, from a second random-feature map at length scale l.

  * Left pair: variance-only AUROC vs CIFAR-10 / SVHN against l, one line per rff_dim
    (mean +/- 1 std over the three backbone seeds). Dashed: the original head's MSP; dotted:
    the original head's own (l = 20) variance; grey band: chance.
  * Right pair: FPR@95%TPR for the same scores (lower is better).
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

# Wong 2011 palette, as in style.py: one hue per rff_dim.
DIM_COLORS = {1024: "#0173B2", 4096: "#D55E00", 16384: "#029E73"}
PANELS = (
    ("auroc_var", "cifar10", "Variance AUROC vs CIFAR-10 ↑"),
    ("auroc_var", "svhn", "Variance AUROC vs SVHN ↑"),
    ("fpr95_var", "cifar10", "Variance FPR@95 vs CIFAR-10 ↓"),
    ("fpr95_var", "svhn", "Variance FPR@95 vs SVHN ↓"),
)
MSP_REF = {"auroc_var": "auroc_msp", "fpr95_var": "fpr95_msp"}


def plot(summary: pd.DataFrame) -> Figure:
    dec = summary[summary.variance == "decoupled"]
    orig = summary[summary.variance == "original"].iloc[0]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.0 * len(PANELS), 3.8))
    for ax, (metric, ood, title) in zip(axes, PANELS):
        for m, color in DIM_COLORS.items():
            g = dec[dec.rff_dim == m].sort_values("length_scale")
            ax.errorbar(g.length_scale, g[f"{metric}_{ood}_mean"], yerr=g[f"{metric}_{ood}_std"],
                        color=color, marker="o", ms=4, lw=1.5, capsize=2, label=f"rff_dim {m}")
        ax.axhline(orig[f"{MSP_REF[metric]}_{ood}_mean"], color="0.3", ls="--", lw=1.1, label="MSP (trained head)")
        ax.axhline(orig[f"{metric}_{ood}_mean"], color="0.3", ls=":", lw=1.1, label="own variance (trained head, ℓ = 20)")
        if metric == "auroc_var":
            ax.axhline(0.5, color="0.7", lw=4, alpha=0.4, zorder=0)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ls = sorted(dec.length_scale.unique())
        ax.set_xticks(ls)
        ax.set_xticklabels([f"{v:g}" for v in ls], fontsize=9)
        ax.set_xlabel("variance-head length scale ℓ", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("CIFAR-100 SpecReg: GP variance at a decoupled length scale (logits fixed, 3 seeds)", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = ROOT / "figures" / "cifar100_decoupled_variance"
    parser.add_argument("--summary", default=str(default_dir / "cifar100_decoupled_variance_summary.csv"))
    parser.add_argument("--figures-dir", default=str(default_dir))
    args = parser.parse_args()

    set_default_style()
    fig = plot(pd.read_csv(args.summary))
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "cifar100_decoupled_variance.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "cifar100_decoupled_variance.pdf", bbox_inches="tight")
    print(f"Saved figure to: {figures_dir}")


if __name__ == "__main__":
    main()
