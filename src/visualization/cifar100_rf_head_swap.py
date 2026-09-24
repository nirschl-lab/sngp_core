#!/usr/bin/env python3
"""CIFAR-100 head-swap: calibration and OOD detection for cos / positive / hyperbolic random
features x ORF / SimRF, on frozen SpecReg backbones.

Consumes the summary CSV written by `scripts/metrics/cifar100_rf_head_swap.py`.

  * Top row: each metric for the six feature-map x coupling arms at the recipe length scale
    (l = 20), mean +/- 1 std across the three backbone seeds. The dashed line is the original
    (end-to-end trained) SpecReg head scored through the same pipeline. The retrained cos/orf
    control is the reference for every comparison; the original head is only a sanity check
    that the head-only protocol reproduces it.
  * Bottom row: the same metrics against l (10 / 20 / 40, i.e. rho ~ 1.15 / 0.57 / 0.29),
    one line per arm.

Accuracy is shown as a control: it cannot move under the mean-field correction, so any
spread there is the head itself, not calibration.
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

# Wong 2011 palette, as in style.py: one hue per feature map; coupling by line/marker style.
FM_COLORS = {"cos": "#0173B2", "positive": "#D55E00", "hyperbolic": "#029E73"}
COUPLING_STYLE = {"orf": dict(marker="o", ls="-"), "simrf": dict(marker="D", ls="--")}
PANELS = (
    ("nll", "NLL (λ* on val) ↓"),
    ("smece", "smECE ↓"),
    ("acc", "Accuracy (control)"),
    ("auroc_msp_cifar10", "MSP AUROC vs CIFAR-10 ↑"),
    ("auroc_msp_svhn", "MSP AUROC vs SVHN ↑"),
    ("auroc_ds_svhn", "DS AUROC vs SVHN ↑"),
    ("auroc_var_svhn", "GP-variance AUROC vs SVHN ↑"),
)
RECIPE_L = 20.0


def plot(summary: pd.DataFrame) -> Figure:
    retrained = summary[summary.arm == "retrained"]
    original = summary[summary.arm == "original_head"].iloc[0]
    arms = [(fm, c) for fm in FM_COLORS for c in COUPLING_STYLE if ((retrained.feature_map == fm) & (retrained.coupling == c)).any()]
    fig, axes = plt.subplots(2, len(PANELS), figsize=(3.6 * len(PANELS), 7.6))
    at_recipe = retrained[retrained.length_scale == RECIPE_L]
    for j, (metric, title) in enumerate(PANELS):
        ax = axes[0, j]
        for i, (fm, c) in enumerate(arms):
            r = at_recipe[(at_recipe.feature_map == fm) & (at_recipe.coupling == c)]
            if r.empty:
                continue
            ax.errorbar(
                i, r[f"{metric}_mean"].iloc[0], yerr=r[f"{metric}_std"].iloc[0], color=FM_COLORS[fm],
                marker=COUPLING_STYLE[c]["marker"], ms=7, capsize=3, lw=1.5,
                mfc=FM_COLORS[fm] if c == "orf" else "white",
            )
        ax.axhline(original[f"{metric}_mean"], color="0.4", ls=":", lw=1.2)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([f"{fm[:3]}/{c}" for fm, c in arms], rotation=60, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)

        ax = axes[1, j]
        for fm, c in arms:
            g = retrained[(retrained.feature_map == fm) & (retrained.coupling == c)].sort_values("length_scale")
            ax.plot(g.length_scale, g[f"{metric}_mean"], color=FM_COLORS[fm], lw=1.6, ms=5, label=f"{fm}/{c}", **COUPLING_STYLE[c])
        ax.axhline(original[f"{metric}_mean"], color="0.4", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())  # else a stray "3x10^1" label
        ax.set_xticks(sorted(retrained.length_scale.unique()))
        ax.set_xticklabels([f"{v:g}" for v in sorted(retrained.length_scale.unique())], fontsize=9)
        ax.set_xlabel("length scale ℓ", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
    axes[0, 0].set_ylabel(f"ℓ = {RECIPE_L:g} (±1 std, 3 seeds)", fontsize=10)
    axes[1, 0].set_ylabel("vs ℓ (mean of 3 seeds)", fontsize=10)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.4", ls=":"))
    labels.append("original SpecReg head")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("CIFAR-100 head-swap on frozen SpecReg backbones: random-feature map × coupling", y=1.0)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = ROOT / "figures" / "cifar100_rf_head_swap"
    parser.add_argument("--summary", default=str(default_dir / "cifar100_rf_head_swap_summary.csv"))
    parser.add_argument("--figures-dir", default=str(default_dir))
    args = parser.parse_args()

    set_default_style()
    fig = plot(pd.read_csv(args.summary))
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = "cifar100_rf_head_swap"
    fig.savefig(figures_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    print(f"Saved figure to: {figures_dir}")


if __name__ == "__main__":
    main()
