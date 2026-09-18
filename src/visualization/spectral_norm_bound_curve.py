"""spectral_norm_bound_curve.py in src/visualization.

Classification metrics vs. SNGP's spectral norm bound `c` (Liu et al. 2022 eq. 15), read
from the `metrics.json` of one inference sweep -- one run directory per swept bound, all
against the same dataset/fold. Every run in such a sweep differs only in `c`, so the
curves are read straight off `metrics.json` and no re-scoring of `predictions.csv` is
needed. One panel per requested metric, side by side on a shared bound axis.

`spectral_norm_bound=None` is **not** a point on the `c` axis and is never plotted as
one. A float `c` bounds the spectral norm (`W <- c * W / sigma_hat` only when
`sigma_hat > c`, so `sigma <= c` and `W` is left untouched otherwise), whereas `None`
is stock `torch.nn.utils.spectral_norm` -- every wrapped weight divided by its estimated
spectral norm, so `sigma == 1` exactly, scaling weights *up* as readily as down (see
`src/models/components/spectral_norm.py`). It is a different normalization regime, not
the `c -> 1` limit, so it is drawn as a horizontal reference line instead.

Error bars are +/- 1 SEM and appear only on the metrics that have one. Of the metrics
plotted here only `acc` does: `acc`/`nll`/`brier` are a mean over per-sample values and
carry an exact `_sem` in `metrics.json`, while macro precision/recall/F1 are count-based,
have no per-sample decomposition, and deliberately get none -- they are drawn as bare
lines rather than given a resampled stand-in. A sweep at one seed also carries no
run-to-run spread, so even where bars exist they are within-checkpoint only.

Usage:
    uv run src/visualization/spectral_norm_bound_curve.py \\
        --run-dir /data1/.../infer/sngp_classifier_acevedo_snb_ablation/2026-09-17_15-06-32 \\
        --dataset acevedo --fold test
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
from matplotlib.figure import Figure

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

# `hydra.sweep.subdir=spectral_norm_bound_${model.net.spectral_norm_bound}` (see the
# ablation's multirun.yaml), carried over verbatim as the inference run_name's leaf.
_BOUND_DIR_PATTERN = re.compile(r"^spectral_norm_bound_(None|[0-9.]+)$")

BOUNDED_COLOR = "#0173B2"
UNBOUNDED_COLOR = "#D55E00"

DEFAULT_METRICS = ["acc", "f1", "precision", "recall"]
METRIC_LABELS = {
    "acc": "Accuracy",
    "f1": "F1 (macro)",
    "precision": "Precision (macro)",
    "recall": "Recall (macro)",
    "f1_micro": "F1 (micro)",
    "precision_micro": "Precision (micro)",
    "recall_micro": "Recall (micro)",
    "ece": "ECE",
    "nll": "NLL",
    "brier": "Brier",
    "auroc": "AUROC",
    "auprc": "AUPRC",
}


def _parse_bound(dir_name: str) -> Optional[float]:
    """The swept `c` out of a sweep subdirectory name; `None` for the unbounded
    (`sigma == 1`) run. Raises on a directory that isn't a sweep subdir at all."""
    match = _BOUND_DIR_PATTERN.match(dir_name)
    if match is None:
        raise ValueError(f"Not a spectral_norm_bound sweep subdirectory: {dir_name!r}")
    raw = match.group(1)
    return None if raw == "None" else float(raw)


def load_sweep_metrics(
    run_dir: Path, metrics: Sequence[str] = tuple(DEFAULT_METRICS)
) -> pd.DataFrame:
    """One row per swept bound: `bound` (NaN for the unbounded run), each requested
    metric, and each metric's `_sem`/`_std` where `metrics.json` carries them.

    Only `acc`, `nll` and `brier` are a mean over per-sample values, so only those three
    get a `_sem`/`_std` -- AUROC/AUPRC/ECE and macro precision/recall/F1 are rank-, bin-
    or count-based and deliberately have none (see docs/results/ACEVEDO_RESULTS.md). The
    columns are left absent rather than filled with zeros when a metric has no dispersion.
    """
    rows = []
    for metrics_json in sorted(run_dir.glob("spectral_norm_bound_*/metrics.json")):
        bound = _parse_bound(metrics_json.parent.name)
        data = json.loads(metrics_json.read_text())
        row = {"bound": bound, "n_samples": data.get("n_samples")}
        for metric in metrics:
            if metric not in data:
                raise KeyError(f"{metrics_json} has no {metric!r} (keys: {sorted(data)})")
            row[metric] = data[metric]
            for suffix in ("std", "sem"):
                key = f"{metric}_{suffix}"
                if key in data:
                    row[key] = data[key]
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No spectral_norm_bound_*/metrics.json under {run_dir}")
    return pd.DataFrame(rows).sort_values("bound", na_position="last").reset_index(drop=True)


def _draw_panel(ax, df: pd.DataFrame, metric: str) -> None:
    """One metric's curve: the bounded runs over evenly-spaced grid positions, plus the
    unbounded (`sigma == 1`) run as a horizontal reference band."""
    sem_col = f"{metric}_sem"
    bounded = df[df["bound"].notna()].sort_values("bound")
    unbounded = df[df["bound"].isna()]

    positions = range(len(bounded))
    yerr = bounded[sem_col] if sem_col in bounded else None
    ax.errorbar(
        positions,
        bounded[metric],
        yerr=yerr,
        marker="o",
        capsize=3,
        color=BOUNDED_COLOR,
        label="Bounded spectral norm ($\\sigma \\leq c$)",
    )

    if not unbounded.empty:
        value = float(unbounded[metric].iloc[0])
        ax.axhline(
            value,
            color=UNBOUNDED_COLOR,
            linestyle="--",
            label="Unbounded, $\\sigma \\equiv 1$ (bound=None)",
        )
        if sem_col in unbounded:
            sem = float(unbounded[sem_col].iloc[0])
            ax.axhspan(value - sem, value + sem, color=UNBOUNDED_COLOR, alpha=0.12, linewidth=0)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{b:g}" for b in bounded["bound"]])
    label = METRIC_LABELS.get(metric, metric)
    ax.set_title(f"{label} ($\\pm$ 1 SEM)" if yerr is not None else label, fontsize=14)


def plot_bound_curves(
    df: pd.DataFrame,
    metrics: Sequence[str] = tuple(DEFAULT_METRICS),
    suptitle: Optional[str] = None,
) -> Figure:
    """A row of metric-vs-`c` panels, one per metric, in the order given.

    The swept bounds are plotted at even spacing rather than at their numeric values.
    A to-scale axis (linear or log) collides the 0.9/0.95/1.0 labels and, worse, renders
    the 0.95 -> 1.0 step as a near-vertical cliff purely because those two bounds sit
    close together -- visually asserting a sensitivity the single-seed data cannot
    support. Even spacing reads the grid as the ordered set of configurations it is; the
    shared x label says so.

    Each panel keeps its own y scale: the metrics share a range here but need not, and a
    shared axis would flatten whichever one varies least.
    """
    set_default_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 5.0), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        _draw_panel(ax, df, metric)

    if suptitle:
        fig.suptitle(suptitle)

    # The shared x label and the shared legend both live in the strip below the axes, so
    # tight_layout reserves it (rect) and the two are anchored apart inside it -- laying
    # them out by default puts the legend straight through the label.
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.supxlabel("Spectral norm bound $c$ (swept grid, evenly spaced)", y=0.085)

    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: "Unbounded" in labels[i])
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", required=True, help="Sweep root holding one spectral_norm_bound_*/ per bound."
    )
    parser.add_argument(
        "--dataset", default="acevedo", help="Dataset the sweep was run on (title/filename only)."
    )
    parser.add_argument(
        "--fold", default="test", help="Fold the sweep was run on (title/filename only)."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="metrics.json keys to plot, one panel each, left to right in this order.",
    )
    parser.add_argument(
        "--figures-dir", default=str(ROOT / "figures" / "spectral_norm_bound_ablation")
    )
    args = parser.parse_args()

    df = load_sweep_metrics(Path(args.run_dir), metrics=args.metrics)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_stem = args.metrics[0] if len(args.metrics) == 1 else "metrics"
    stem = f"{args.dataset}_{args.fold}_{metric_stem}_vs_spectral_norm_bound"
    df.to_csv(figures_dir / f"{stem}.csv", index=False)

    n_samples = df["n_samples"].dropna()
    n_note = f" (n = {int(n_samples.iloc[0]):,})" if not n_samples.empty else ""
    fig = plot_bound_curves(
        df,
        metrics=args.metrics,
        suptitle=f"SNGP on {args.dataset.capitalize()} {args.fold}{n_note}",
    )
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure and tidy CSV to: {figures_dir}")


if __name__ == "__main__":
    main()
