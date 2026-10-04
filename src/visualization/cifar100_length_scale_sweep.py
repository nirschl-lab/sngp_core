#!/usr/bin/env python3
"""CIFAR-100 length-scale sweep: the GP head without LayerNorm (Muon wd 0.1 + SGD aux, cosine), seed 12345.

Consumes the per-seed CSV written by `scripts/metrics/cifar100_evidence_ls_report.py` (test metrics at
the val-fitted λ) and the feature-scale CSV of `scripts/metrics/cifar100_gp_head_diag.py`. One line over
ℓ = 2 / 3.5 / 7 / 14 / 20 (log axis); the GP-input LayerNorm arm (ℓ = 20) is a hollow marker, and
cosine SGD SpecReg (ℓ = 7) a dashed reference line.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
from matplotlib.figure import Figure

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

# report arm key -> (ℓ, diag label)
SWEEP = {
    "cos_muonsgd01_l2": (2.0, "cos_muonsgd_l2_wd0.1_s12345"),
    "cos_muonsgd01_l3.5": (3.5, "cos_muonsgd_l3.5_wd0.1_s12345"),
    "cos_muonsgd01": (7.0, "cos_muonsgd_wd0.1_s12345"),
    "cos_muonsgd01_l14": (14.0, "cos_muonsgd_l14_wd0.1_s12345"),
    "cos_muonsgd01_l20": (20.0, "cos_muonsgd_l20_wd0.1_s12345"),
}
LN = ("cos_muonsgd_ln01", 20.0, "cos_muonsgd_ln_l20_wd0.1_s12345")
REF = ("cos_specreg", "cos_specreg_l7_s12345")
# Wong 2011 palette, as in style.py.
COLOR, LN_COLOR, REF_COLOR = "#0173B2", "#D55E00", "#949494"
PANELS = (
    ("acc", "Accuracy", "report"),
    ("nll", "NLL ↓", "report"),
    ("fitted", "val-fit λ*", "report"),
    ("h_in_norm_over_l", "‖h_in‖ / ℓ (kernel input scale)", "diag"),
    ("auroc_msp_cifar10", "MSP AUROC vs CIFAR-10 ↑", "report"),
    ("auroc_msp_svhn", "MSP AUROC vs SVHN ↑", "report"),
    ("auroc_ds_svhn", "DS AUROC vs SVHN ↑", "report"),
    ("auroc_var_svhn", "GP-variance AUROC vs SVHN ↑", "report"),
)


def plot(report: pd.DataFrame, diag: pd.DataFrame) -> Figure:
    rep = report.set_index("arm")
    dg = diag.set_index("label")
    ls = [SWEEP[a][0] for a in SWEEP]
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.8))
    for ax, (metric, title, src) in zip(axes.ravel(), PANELS):
        if src == "report":
            ys = [rep.loc[a, metric] for a in SWEEP]
            ln_y, ref_y = rep.loc[LN[0], metric], rep.loc[REF[0], metric]
        else:
            ys = [dg.loc[SWEEP[a][1], metric] for a in SWEEP]
            ln_y, ref_y = dg.loc[LN[2], metric], dg.loc[REF[1], metric]
        ax.plot(ls, ys, marker="o", ms=6, lw=1.8, color=COLOR, label="no LayerNorm")
        ax.plot([LN[1]], [ln_y], ls="none", marker="o", ms=9, mfc="none", mew=2, color=LN_COLOR,
                label="GP-input LayerNorm, ℓ = 20")
        ax.axhline(ref_y, color=REF_COLOR, ls="--", lw=1.2, label="SGD SpecReg, ℓ = 7")
        if metric.startswith("auroc_var"):
            ax.axhline(0.5, color="0.6", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_xticks(ls)
        ax.set_xticklabels([f"{v:g}" for v in ls], fontsize=9)
        ax.minorticks_off()
        ax.set_xlabel("ℓ", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "CIFAR-100 / WRN-28-10, Muon wd 0.1 + SGD aux, cosine, last.ckpt, seed 12345; "
        "test metrics at the val-fit λ; dotted = chance",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = ROOT / "figures" / "cifar100_cosine"
    parser.add_argument("--csv", default=str(default_dir / "cifar100_cosine_per_seed.csv"))
    parser.add_argument("--diag-csv", default=str(default_dir / "cifar100_gp_head_diag.csv"))
    parser.add_argument("--figures-dir", default=str(default_dir))
    args = parser.parse_args()

    set_default_style()
    fig = plot(pd.read_csv(args.csv), pd.read_csv(args.diag_csv))
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "cifar100_length_scale_sweep.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "cifar100_length_scale_sweep.pdf", bbox_inches="tight")
    print(f"Saved figure to: {figures_dir}")


if __name__ == "__main__":
    main()
