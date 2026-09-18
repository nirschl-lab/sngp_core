"""predictive_entropy.py in src/visualization.

Per-dataset Shannon-entropy (nats) KDE comparison from inference `predictions.csv`
files -- compares a model's predictive entropy on its in-distribution dataset against
one or more out-of-distribution datasets.

Usage:
    uv run src/visualization/predictive_entropy.py \\
        --run-dir /data1/maheswararao/experiments/uncertaity-aware-ml/infer/baseline_classifier_acevedo/2026-08-25_14-31-36 \\
        --indist acevedo \\
        --outdist jung kather2016 kather2018 nirschl2018 tang wong \\
        --name baseline_acevedo_entropy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.auc import _parse_class_probs  # noqa: E402
from src.visualization.style import DATASET_COLORS, set_default_style  # noqa: E402


def _raw_entropy_from_probs(probs: list[float], eps: float = 1e-12) -> float:
    """Shannon entropy in nats, unnormalized.

    Unlike `src.metrics.auc._normalized_entropy` (divides by log(num_classes) for a
    0-1 OOD score), this keeps the raw nats scale.
    """
    p = np.asarray(probs, dtype=float)
    return float(-np.sum(p * np.log(p + eps)))


def load_entropy_frame(
    predictions_csv: Path,
    n_total: int = 500,
    seed: int = 42,
    fold: str | None = None,
) -> pd.DataFrame:
    """Load a predictions.csv, class-balance-sample up to n_total rows, and add an `entropy` column."""
    if not predictions_csv.exists():
        raise FileNotFoundError(f"No predictions.csv at {predictions_csv}")

    df = pd.read_csv(predictions_csv)

    if fold is not None:
        if "fold" not in df.columns:
            raise KeyError(f"{predictions_csv} has no 'fold' column to filter on.")
        df = df[df["fold"] == fold]
        if df.empty:
            raise ValueError(f"No rows with fold={fold!r} in {predictions_csv}")

    if "target" not in df.columns:
        raise KeyError(f"{predictions_csv} is missing the 'target' column needed for class-balanced sampling.")

    groups = [g for _, g in df.groupby("target")]
    per_class = max(1, n_total // len(groups))
    sampled = pd.concat([g.sample(n=min(per_class, len(g)), random_state=seed) for g in groups])
    shortfall = n_total - len(sampled)
    if shortfall > 0:
        remaining = df.drop(sampled.index)
        if len(remaining) > 0:
            topup = remaining.sample(n=min(shortfall, len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, topup])

    sampled = sampled.copy()
    class_probs = sampled["class_probs"].apply(_parse_class_probs)
    if any(v is None for v in class_probs):
        raise ValueError(f"Failed to parse `class_probs` for some rows in {predictions_csv}")
    sampled["entropy"] = class_probs.apply(_raw_entropy_from_probs)
    return sampled


def plot_entropy_kde(
    entropy_by_label: dict[str, np.ndarray],
    in_dist_labels: set[str],
    bw_method: str | float = "scott",
    gridsize: int = 512,
    pad: float = 0.05,
) -> plt.Figure:
    """KDE overlay of entropy per dataset -- solid line for in-distribution labels, dashed for the rest."""
    set_default_style()

    global_min = min(float(np.min(s)) for s in entropy_by_label.values())
    global_max = max(float(np.max(s)) for s in entropy_by_label.values())
    if np.isclose(global_min, global_max):
        span = max(1.0, abs(global_min)) * 0.1
        global_min -= span
        global_max += span
    xr = global_max - global_min
    x = np.linspace(global_min - xr * pad, global_max + xr * pad, gridsize)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values in entropy_by_label.items():
        is_in_dist = label in in_dist_labels
        line_style = "-" if is_in_dist else "--"
        alpha = 1.0 if is_in_dist else 0.9
        color = DATASET_COLORS.get(label)

        uniq = np.unique(values)
        if uniq.size < 2 or np.isclose(np.var(values), 0.0):
            ax.axvline(
                float(uniq[0]), color=color, linestyle=line_style, linewidth=2.0, alpha=0.5, label=f"{label} (no KDE)"
            )
            continue

        kde = gaussian_kde(values, bw_method=bw_method)
        y = kde.evaluate(x)
        ax.plot(x, y, color=color, linewidth=2.0, linestyle=line_style, alpha=alpha, label=label)
        ax.fill_between(x, y, 0, color=color, alpha=0.15)

    ax.set_xlabel("Entropy (nats)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)

    legend1 = ax.legend(frameon=False, fontsize=14)
    ax.add_artist(legend1)

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2),
    ]
    # Above the axes, so it can never land on the dataset legend (which matplotlib places
    # at its "best" free spot) -- the two used to overlap in every published figure.
    ax.legend(
        custom_lines,
        ["In-distribution", "Out-of-distribution"],
        frameon=False,
        fontsize=14,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        borderaxespad=0.2,
    )

    ax.margins(x=0)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive-entropy KDE comparison across datasets.")
    parser.add_argument(
        "--run-dir", required=True, type=Path, help="Directory whose immediate subfolders hold predictions.csv."
    )
    parser.add_argument(
        "--indist", required=True, nargs="+", help="Folder name(s) under --run-dir treated as in-distribution."
    )
    parser.add_argument(
        "--outdist", required=True, nargs="+", help="Folder name(s) under --run-dir treated as out-of-distribution."
    )
    parser.add_argument("--name", required=True, help="Output filename stem.")
    parser.add_argument("--figures-dir", default=str(ROOT / "figures" / "predictive_entropy"))
    parser.add_argument("--fold", default=None, help="Filter rows to this fold before sampling (default: none).")
    parser.add_argument("--n-total", type=int, default=500, help="Class-balanced sample size per dataset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dist = list(args.indist)
    out_dist = list(args.outdist)

    overlap = set(in_dist) & set(out_dist)
    if overlap:
        raise ValueError(f"Names cannot be in both --indist and --outdist: {sorted(overlap)}")

    all_names = in_dist + out_dist
    if len(all_names) != len(set(all_names)):
        raise ValueError(f"Duplicate dataset name(s): {all_names}")

    entropy_by_label: dict[str, np.ndarray] = {}
    for name in all_names:
        predictions_csv = args.run_dir / name / "predictions.csv"
        df = load_entropy_frame(predictions_csv, n_total=args.n_total, seed=args.seed, fold=args.fold)
        entropy_by_label[name] = df["entropy"].to_numpy()
        print(f"{name}: {len(df)} rows, mean entropy = {df['entropy'].mean():.4f} nats")

    fig = plot_entropy_kde(entropy_by_label, in_dist_labels=set(in_dist))

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{args.name}.png"
    pdf_path = figures_dir / f"{args.name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
