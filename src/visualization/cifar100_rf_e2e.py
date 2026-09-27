#!/usr/bin/env python3
"""CIFAR-100 `rf_e2e` study: SpecReg trained end-to-end per random-feature head, one seed.

Consumes the CSV written by `scripts/metrics/cifar100_rf_e2e_report.py` and plots the
`val_fit` rows (lambda fitted on val, scored on test). One point per run: colour is the
feature map, marker the coupling (filled = orf, open = simrf). The dotted vertical line
separates the two heads (l = 20 recipe | l = 2 paper), which also differ in sigma^2 and
lambda, and the dashed horizontal line is the l = 20 cos/orf control. Accuracy is a
control: the mean-field correction cannot move it.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
from matplotlib.figure import Figure

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

# Wong 2011 palette, as in style.py and cifar100_rf_head_swap.py.
FM_COLORS = {"cos": "#0173B2", "positive": "#D55E00", "hyperbolic": "#029E73"}
COUPLING_MARKER = {"orf": "o", "simrf": "D"}
PANELS = (
    ("acc", "Accuracy (control)"),
    ("nll", "NLL (λ* on val) ↓"),
    ("smece", "smECE ↓"),
    ("auroc_msp_cifar10", "MSP AUROC vs CIFAR-10 ↑"),
    ("auroc_msp_svhn", "MSP AUROC vs SVHN ↑"),
    ("auroc_ds_svhn", "DS AUROC vs SVHN ↑"),
    ("auroc_var_svhn", "GP-variance AUROC vs SVHN ↑"),
)
CONTROL = "control_specreg_s12345"


def plot(df: pd.DataFrame) -> Figure:
    rows = df[df.lambda_kind == "val_fit"].reset_index(drop=True)
    control = rows[rows.label == CONTROL].iloc[0]
    split = (rows["head"] == "l20recipe").sum() - 0.5
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.2 * len(PANELS), 4.2))
    for ax, (metric, title) in zip(axes, PANELS):
        for i, r in rows.iterrows():
            ax.plot(
                i, r[metric], ls="none", marker=COUPLING_MARKER[r.coupling], ms=8,
                color=FM_COLORS[r.feature_map], mfc=FM_COLORS[r.feature_map] if r.coupling == "orf" else "white",
                mew=1.6,
            )
        ax.axhline(control[metric], color="0.4", ls="--", lw=1.0)
        ax.axvline(split, color="0.6", ls=":", lw=1.0)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(
            [f"{'ℓ2' if h == 'l2paper' else 'ℓ20'} {fm[:3]}/{c}" for h, fm, c in zip(rows["head"], rows.feature_map, rows.coupling)],
            rotation=60, fontsize=9,
        )
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", ms=7, label=fm) for fm, c in FM_COLORS.items()]
    handles += [
        plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=7, label="orf"),
        plt.Line2D([], [], color="0.3", marker="D", mfc="white", ls="none", ms=7, label="simrf"),
        plt.Line2D([], [], color="0.4", ls="--", label="ℓ20 cos/orf control"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("CIFAR-100 SpecReg, end-to-end per random-feature head (seed 12345, last.ckpt)", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = ROOT / "figures" / "cifar100_rf_e2e"
    parser.add_argument("--csv", default=str(default_dir / "cifar100_rf_e2e.csv"))
    parser.add_argument("--figures-dir", default=str(default_dir))
    args = parser.parse_args()

    set_default_style()
    fig = plot(pd.read_csv(args.csv))
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "cifar100_rf_e2e.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "cifar100_rf_e2e.pdf", bbox_inches="tight")
    print(f"Saved figure to: {figures_dir}")


if __name__ == "__main__":
    main()
