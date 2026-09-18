"""spectral_norm_bound_curve.py in src/visualization.

Accuracy vs. SNGP's spectral norm bound `c` (Liu et al. 2022 eq. 15), read from the
`metrics.json` of one inference sweep -- one run directory per swept bound, all against
the same dataset/fold. Every run in such a sweep differs only in `c`, so the curve is
read straight off `metrics.json` and no re-scoring of `predictions.csv` is needed.

`spectral_norm_bound=None` is **not** a point on the `c` axis and is never plotted as
one. A float `c` bounds the spectral norm (`W <- c * W / sigma_hat` only when
`sigma_hat > c`, so `sigma <= c` and `W` is left untouched otherwise), whereas `None`
is stock `torch.nn.utils.spectral_norm` -- every wrapped weight divided by its estimated
spectral norm, so `sigma == 1` exactly, scaling weights *up* as readily as down (see
`src/models/components/spectral_norm.py`). It is a different normalization regime, not
the `c -> 1` limit, so it is drawn as a horizontal reference line instead.

Error bars are +/- 1 SEM of the mean per-sample accuracy (`acc_sem` in `metrics.json`,
i.e. the binomial standard error over the fold's n samples). That is the exact
dispersion of a single checkpoint's accuracy, not a resampling estimate -- a sweep at
one seed carries no run-to-run spread, so points closer together than these bars are not
separated by the data.

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
from typing import Optional

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


def _parse_bound(dir_name: str) -> Optional[float]:
    """The swept `c` out of a sweep subdirectory name; `None` for the unbounded
    (`sigma == 1`) run. Raises on a directory that isn't a sweep subdir at all."""
    match = _BOUND_DIR_PATTERN.match(dir_name)
    if match is None:
        raise ValueError(f"Not a spectral_norm_bound sweep subdirectory: {dir_name!r}")
    raw = match.group(1)
    return None if raw == "None" else float(raw)


def load_sweep_metrics(run_dir: Path, metric: str = "acc") -> pd.DataFrame:
    """One row per swept bound: `bound` (NaN for the unbounded run), `metric`, and its
    `_sem`/`_std` when `metrics.json` carries them.

    Only `acc`, `nll` and `brier` are a mean over per-sample values, so only those three
    get a `_sem`/`_std` -- AUROC/AUPRC/ECE/macro-F1 are rank-, bin- or count-based and
    deliberately have none (see docs/results/ACEVEDO_RESULTS.md). The columns are left
    absent rather than filled with zeros when the metric has no dispersion.
    """
    rows = []
    for metrics_json in sorted(run_dir.glob("spectral_norm_bound_*/metrics.json")):
        bound = _parse_bound(metrics_json.parent.name)
        data = json.loads(metrics_json.read_text())
        if metric not in data:
            raise KeyError(f"{metrics_json} has no {metric!r} (keys: {sorted(data)})")
        row = {"bound": bound, metric: data[metric], "n_samples": data.get("n_samples")}
        for suffix in ("std", "sem"):
            key = f"{metric}_{suffix}"
            if key in data:
                row[key] = data[key]
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No spectral_norm_bound_*/metrics.json under {run_dir}")
    return pd.DataFrame(rows).sort_values("bound", na_position="last").reset_index(drop=True)


def plot_bound_curve(
    df: pd.DataFrame,
    metric: str = "acc",
    metric_label: str = "Accuracy",
    title: Optional[str] = None,
) -> Figure:
    """Metric vs. `c` over evenly-spaced grid positions, with the unbounded
    (`sigma == 1`) run as a horizontal reference band rather than a point on the axis.

    The swept bounds are plotted at even spacing rather than at their numeric values.
    A to-scale axis (linear or log) collides the 0.9/0.95/1.0 labels and, worse, renders
    the 0.95 -> 1.0 step as a near-vertical cliff purely because those two bounds sit
    close together -- visually asserting a sensitivity the single-seed data cannot
    support. Even spacing reads the grid as the ordered set of configurations it is; the
    axis label says so.
    """
    set_default_style()
    fig, ax = plt.subplots(figsize=(8, 6))

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
    ax.set_xlabel("Spectral norm bound $c$ (swept grid, evenly spaced)")
    ax.set_ylabel(f"{metric_label} ($\\pm$ 1 SEM)" if yerr is not None else metric_label)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: "Unbounded" in labels[i])
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()
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
    parser.add_argument("--metric", default="acc", help="metrics.json key to plot.")
    parser.add_argument("--metric-label", default="Accuracy")
    parser.add_argument(
        "--figures-dir", default=str(ROOT / "figures" / "spectral_norm_bound_ablation")
    )
    args = parser.parse_args()

    df = load_sweep_metrics(Path(args.run_dir), metric=args.metric)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.dataset}_{args.fold}_{args.metric}_vs_spectral_norm_bound"
    df.to_csv(figures_dir / f"{stem}.csv", index=False)

    n_samples = df["n_samples"].dropna()
    n_note = f" (n = {int(n_samples.iloc[0]):,})" if not n_samples.empty else ""
    fig = plot_bound_curve(
        df,
        metric=args.metric,
        metric_label=args.metric_label,
        title=f"SNGP on {args.dataset.capitalize()} {args.fold}{n_note}",
    )
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure and tidy CSV to: {figures_dir}")


if __name__ == "__main__":
    main()
