#!/usr/bin/env python3
"""CIFAR-100 `evidence_ls` benchmark: Baseline / SNGP / SNGP+SpecReg / SNGP+SpecReg+evidence l.

Consumes the per-seed CSV written by `scripts/metrics/cifar100_evidence_ls_report.py`. Every
number is on test at that row's single validation-fitted knob (lambda for the SNGP arms, T for
the baseline). One column per arm: the three seeds as dots, their mean as a horizontal bar.
The baseline has no GP variance, so it is absent from the variance panels.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from matplotlib.figure import Figure

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

# Wong 2011 palette, as in style.py.
ARMS = {
    "baseline": ("Baseline", "#949494"),
    "sngp": ("SNGP", "#0173B2"),
    "specreg": ("+SpecReg", "#029E73"),
    "els": ("+SpecReg\n+evidence ℓ", "#D55E00"),
}
PANELS = (
    ("acc", "Accuracy"),
    ("nll", "NLL ↓"),
    ("smece", "smECE ↓"),
    ("auroc_msp_cifar10", "MSP AUROC vs CIFAR-10 ↑"),
    ("auroc_msp_svhn", "MSP AUROC vs SVHN ↑"),
    ("auroc_ds_svhn", "DS AUROC vs SVHN ↑"),
    ("auroc_var_cifar10", "GP-variance AUROC vs CIFAR-10 ↑"),
    ("auroc_var_svhn", "GP-variance AUROC vs SVHN ↑"),
)


def plot(df: pd.DataFrame) -> Figure:
    arms = [a for a in ARMS if a in set(df.arm)]
    fig, axes = plt.subplots(2, len(PANELS) // 2, figsize=(3.4 * len(PANELS) // 2, 7.6))
    for ax, (metric, title) in zip(axes.ravel(), PANELS):
        for i, arm in enumerate(arms):
            vals = df.loc[df.arm == arm, metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            color = ARMS[arm][1]
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.plot(i + jitter, vals, ls="none", marker="o", ms=6, color=color, alpha=0.85)
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=color, lw=2.2)
        if metric.startswith("auroc_var"):
            ax.axhline(0.5, color="0.6", ls=":", lw=1.0)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARMS[a][0] for a in arms], fontsize=9)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle(
        "CIFAR-100 / WRN-28-10, last.ckpt, 3 seeds; one val-fit knob per row (λ for SNGP, T for baseline); "
        "dotted = chance",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = ROOT / "figures" / "cifar100_evidence_ls"
    parser.add_argument("--csv", default=str(default_dir / "cifar100_evidence_ls_per_seed.csv"))
    parser.add_argument("--figures-dir", default=str(default_dir))
    args = parser.parse_args()

    set_default_style()
    fig = plot(pd.read_csv(args.csv))
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "cifar100_evidence_ls.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "cifar100_evidence_ls.pdf", bbox_inches="tight")
    print(f"Saved figure to: {figures_dir}")


if __name__ == "__main__":
    main()
