#!/usr/bin/env python3
"""Calibration and OOD-AUROC sensitivity to SNGP's `mean_field_factor` on CIFAR-100.

Consumes the per-seed tidy CSV written by
`scripts/metrics/cifar100_mean_field_sweep.py` and draws eight panels against lambda:
NLL / smECE / Brier / accuracy on the top row, and MSP and Dempster-Shafer OOD AUROC
against CIFAR-10 (near) and SVHN (far) on the bottom.

The figure exists to make one thing legible: **calibration and far-OOD detection want
different lambdas.** NLL and smECE have an interior optimum around lambda ~ 20-50, while
far-OOD AUROC keeps climbing past it and near-OOD AUROC falls. The dotted line on the MSP
panels is the `lambda -> infinity` limit those curves converge to -- at that limit the MSP
ranking is logit-margin / sigma, and dividing the margin by the GP standard deviation helps
far-OOD while hurting near-OOD.

The accuracy panel is a control, not a result. The correction divides every logit of an
example by one positive scalar, so argmax is invariant and the line must be flat; a visibly
sloping accuracy panel means the sweep is wrong.

Two deliberate departures from the sibling sweep figure,
`spectral_norm_bound_curve.py`:

  * **A true log x-axis**, where that module plots its swept values at even spacing. Its
    reason is that it has a handful of discrete bounds whose numeric spacing would imply a
    sensitivity its single-seed data cannot support. Here lambda is sampled densely (30
    log-spaced points) and *is* a continuous knob, so the honest rendering is the real axis.
  * **Bands, not error bars.** Three seeds at 30 lambdas would be 90 caps per panel; a
    `fill_between` of +/- 1 std across seeds reads as the continuous uncertainty it is.

`lambda = 0` is in the CSV but is not drawn separately: at the grid's left end
(lambda = 0.1) the mean shrink factor is 1.001, so the curve's left edge already *is* the
uncorrected model to three decimal places.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import METHOD_COLORS, set_default_style  # noqa: E402

PRODUCTION_LAMBDA = 7.5

# Sweep group key -> the canonical model name `METHOD_COLORS`/`MODEL_ROW_ORDER` use, so this
# figure's colors match every other method-comparison figure in the project.
GROUP_TO_MODEL: Dict[str, str] = {
    "sngp": "SNGP",
    "specreg": "SNGP + Spectral Reg",
    "specreg_trace": "SNGP + Spectral Reg (trace-logistic)",
    "baseline": "Baseline Classifier",
}

# (column, panel title, whether lower is better) -- the last only drives the axis note.
PANELS: Tuple[Tuple[str, str], ...] = (
    ("nll", "NLL"),
    ("smece", "smECE (top-label)"),
    ("brier", "Brier"),
    ("acc", "Accuracy (control — must be flat)"),
    ("auroc_msp_cifar10", "MSP AUROC vs CIFAR-10 (near-OOD)"),
    ("auroc_msp_svhn", "MSP AUROC vs SVHN (far-OOD)"),
    ("auroc_ds_cifar10", "DS AUROC vs CIFAR-10 (near-OOD)"),
    ("auroc_ds_svhn", "DS AUROC vs SVHN (far-OOD)"),
)

# The lambda -> infinity limit is only defined for the MSP panels: it comes from softmax
# flattening, which is what turns the top-1 probability into a margin/sigma ranking. DS has
# no such limit -- it collapses toward 0.5 for every example instead.
ASYMPTOTE_PANELS = frozenset({"auroc_msp_cifar10", "auroc_msp_svhn"})

# Calibration blows up by an order of magnitude past lambda ~ 1e2, which on an auto-scaled
# axis flattens the interior optimum near lambda ~ 20-50 into an invisible wobble -- i.e. it
# hides the one thing these panels exist to show. They are clipped to the informative range
# instead, with the curve left to exit the top of the panel and the excursion stated in the
# panel title, so the truncation is declared rather than merely visible.
CLIPPED_PANELS = frozenset({"nll", "smece", "brier"})
CLIP_HEADROOM = 1.22


def load_sweep(csv_path: Path) -> pd.DataFrame:
    """Read the per-seed sweep CSV, keeping every `kind` for the caller to split."""
    df = pd.read_csv(csv_path)
    if "kind" not in df.columns:
        raise ValueError(
            f"{csv_path} has no 'kind' column -- it looks like the seed-averaged CSV from "
            "cifar100_predictive_links.py, which carries no per-seed rows and so cannot "
            "support an error band. Regenerate with scripts/metrics/cifar100_mean_field_sweep.py."
        )
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across training seeds, per (group, lambda).

    `std` here is the spread across independently trained runs -- the only thing `±` ever
    means in this family of reports. Nothing is resampled and nothing is bootstrapped.
    """
    sweep = df[df["kind"] == "sweep"]
    metrics = [c for c, _ in PANELS]
    grouped = sweep.groupby(["group", "display", "lambda"])[metrics]
    mean = grouped.mean()
    std = grouped.std(ddof=1).rename(columns=lambda c: f"{c}_std")
    return mean.join(std).reset_index()


def _draw_panel(
    ax,
    agg: pd.DataFrame,
    df: pd.DataFrame,
    metric: str,
    groups: Sequence[str],
) -> None:
    for group in groups:
        sub = agg[(agg["group"] == group) & (agg["lambda"] > 0)].sort_values("lambda")
        if sub.empty or metric not in sub:
            continue
        model = GROUP_TO_MODEL.get(group, group)
        color = METHOD_COLORS.get(model, None)
        mean = sub[metric]
        ax.plot(sub["lambda"], mean, color=color, label=model, linewidth=1.8)
        std = sub.get(f"{metric}_std")
        if std is not None and std.notna().any():
            ax.fill_between(
                sub["lambda"], mean - std, mean + std, color=color, alpha=0.18, linewidth=0
            )

    baseline = df[(df["kind"] == "baseline")]
    if not baseline.empty and metric in baseline and baseline[metric].notna().any():
        ax.axhline(
            float(baseline[metric].iloc[0]),
            color=METHOD_COLORS["Baseline Classifier"],
            linestyle="--",
            linewidth=1.4,
            label="Baseline Classifier (no λ axis)",
        )

    if metric in ASYMPTOTE_PANELS:
        asym = df[(df["kind"] == "asymptote")]
        if not asym.empty and metric in asym and asym[metric].notna().any():
            for group in groups:
                vals = asym[asym["group"] == group][metric].dropna()
                if vals.empty:
                    continue
                ax.axhline(
                    float(vals.mean()),
                    color=METHOD_COLORS.get(GROUP_TO_MODEL.get(group, group)),
                    linestyle=":",
                    linewidth=1.2,
                    label="λ → ∞ limit (margin / σ)",
                )

    ax.axvline(PRODUCTION_LAMBDA, color="0.35", linestyle="-.", linewidth=1.0,
               label=f"λ = {PRODUCTION_LAMBDA:g} (current)")
    ax.set_xscale("log")

    if metric in ASYMPTOTE_PANELS:
        # `axhline` does not participate in autoscaling, so an asymptote just below the data
        # lands on the axis edge and reads as a frame rather than as the limit the curve is
        # heading for. Widen the view to hold it.
        asym = df[df["kind"] == "asymptote"][metric].dropna() if "kind" in df else []
        if len(asym):
            lo, hi = ax.get_ylim()
            pad = 0.04 * (hi - lo)
            ax.set_ylim(min(lo, float(asym.min()) - pad), max(hi, float(asym.max()) + pad))


def _clip_axis(ax, agg: pd.DataFrame, df: pd.DataFrame, metric: str, title: str) -> str:
    """Focus a calibration panel on its interior optimum; return the annotated title.

    The ceiling is set from the *uncorrected* end of the sweep rather than from the data
    maximum, so every panel is clipped at "a bit worse than doing nothing" -- a reference
    the reader already has -- instead of at an arbitrary fraction of a blow-up.
    """
    sweep = agg[agg["lambda"] > 0]
    if sweep.empty or metric not in sweep:
        return title

    left = float(sweep.loc[sweep["lambda"].idxmin(), metric])
    baseline = df[df["kind"] == "baseline"]
    reference = max(
        left,
        float(baseline[metric].iloc[0]) if not baseline.empty and metric in baseline else left,
    )
    low = float(sweep[metric].min())
    high = reference * CLIP_HEADROOM
    peak = float(sweep[metric].max())

    ax.set_ylim(low - 0.08 * (high - low), high)
    return f"{title} — y clipped, rises to {peak:.2f}" if peak > high else title


def plot_mean_field_sweep(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    suptitle: Optional[str] = None,
) -> Figure:
    """Eight panels of metric-vs-lambda. Returns the figure; saves nothing."""
    set_default_style()
    groups = [g for g in ("sngp", "specreg", "specreg_trace") if g in set(agg["group"])]

    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.5))
    for ax, (metric, title) in zip(axes.flat, PANELS):
        _draw_panel(ax, agg, df, metric, groups)
        if metric in CLIPPED_PANELS:
            title = _clip_axis(ax, agg, df, metric, title)
        ax.set_title(title, fontsize=13)

    # Accuracy is the control: pin a visible window around the value so a flat line reads as
    # flat rather than as amplified float noise on an auto-scaled axis.
    acc_ax = axes.flat[3]
    acc_vals = agg["acc"].dropna()
    if not acc_vals.empty:
        center = float(acc_vals.mean())
        acc_ax.set_ylim(center - 0.05, center + 0.05)

    # Bottom margin fits a two-row legend under the x-label. One row was enough while the
    # figure had two arms; a third pushes the legend into the label at this width.
    fig.tight_layout(rect=(0, 0.15, 1, 0.95 if suptitle else 1.0))
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.supxlabel(
        "mean-field factor $\\lambda$   (log scale; at the left edge, $\\lambda=10^{-2}$, no arm is "
        "meaningfully corrected)",
        y=0.095,
    )

    # One legend for the whole figure, de-duplicated: every panel draws the same series, and
    # the asymptote entry appears once per arm.
    seen: Dict[str, Line2D] = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        frameon=False,
        loc="lower center",
        ncol=min(len(seen), 3),
        bbox_to_anchor=(0.5, 0.0),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sweep-csv",
        required=True,
        help="per-seed CSV from scripts/metrics/cifar100_mean_field_sweep.py",
    )
    parser.add_argument("--figures-dir", default=str(ROOT / "figures" / "mean_field_sweep"))
    parser.add_argument(
        "--no-suptitle", action="store_true", help="drop the title (for paper insets)"
    )
    args = parser.parse_args()

    df = load_sweep(Path(args.sweep_csv))
    agg = aggregate(df)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = "cifar100_calibration_and_ood_vs_mean_field_factor"

    agg.to_csv(figures_dir / f"{stem}.csv", index=False)

    n_seeds = df[df["kind"] == "sweep"].groupby("group")["seed_label"].nunique().max()
    fig = plot_mean_field_sweep(
        df,
        agg,
        suptitle=None
        if args.no_suptitle
        else f"CIFAR-100 / WideResNet-28-10 — sensitivity to the mean-field factor "
        f"(mean $\\pm$ 1 std over {int(n_seeds)} training seeds)",
    )
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure and tidy CSV to: {figures_dir}")


if __name__ == "__main__":
    main()
