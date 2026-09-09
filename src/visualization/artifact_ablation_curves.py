"""artifact_ablation_curves.py in src/visualization.

Ablation across the Acevedo artifact-simulation axes: config axis (`count`=1..5 pasted
artifact overlays) and procedural axis (`severity`=1..5 graded acquisition degradation).
Plots NLL vs. count/severity, one line per model, read from each axis-value's
`artifact_quantification_summary.csv` (written by
`src/paper_helpers/ood_metrics/render_artifact_results_tables.py` /
`src/metrics/artifact_quantification.py::quantify_artifact_impact`; run paths in
`configs/paper_helpers/acevedo_artifact_axis_paths.yaml`). The config-axis figure adds a
slim panel above the NLL panel showing mean `percent_pixels_affected` per count -- already
computed per-sample by `ArtifactHFDataset` and present in each run's `predictions.csv` --
since that's a property of the simulator's overlay draw, not of the model (near-identical
across all 5 checkpoints' predictions.csv for the same count, since the simulator's seed is
content-derived per image, see docs/DATASETS.md). Procedural effects are whole-frame, not
masked, so there's no comparable coverage number for that axis.

Usage:
    uv run src/visualization/artifact_ablation_curves.py \\
        --config configs/paper_helpers/acevedo_artifact_axis_paths.yaml \\
        --output-dir csv/artifact_quantification/acevedo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
import seaborn as sns
import yaml

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

MODEL_ROW_ORDER = ["Baseline Classifier", "Deep Ensemble", "Monte Carlo Dropout", "SNGP", "SNGP Ensemble"]
MODEL_COLORS = dict(zip(MODEL_ROW_ORDER, sns.color_palette("colorblind", len(MODEL_ROW_ORDER))))
N_VALUES = [1, 2, 3, 4, 5]


def _axis_key(axis: str, n: int) -> str:
    if axis == "config":
        return "config_axis" if n == 1 else f"config_axis_count_{n}"
    return "procedural_axis" if n == 1 else f"procedural_axis_severity_{n}"


def load_nll_curve(output_dir: Path, axis: str) -> pd.DataFrame:
    """One row per (model, n) with the axis-value's `artifact_nll`."""
    rows = []
    for n in N_VALUES:
        summary_csv = output_dir / _axis_key(axis, n) / "artifact_quantification_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing {summary_csv} -- run render_artifact_results_tables.py first")
        df = pd.read_csv(summary_csv)[["model", "artifact_nll"]].copy()
        df["n"] = n
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_pixel_coverage(axis_paths: dict) -> pd.DataFrame:
    """Config axis only: mean/min/max `percent_pixels_affected` per count, across models."""
    rows = []
    for n in N_VALUES:
        model_paths = axis_paths[_axis_key("config", n)]["models"]
        means = [
            pd.read_csv(paths["artifact"], usecols=["percent_pixels_affected"])["percent_pixels_affected"].mean()
            for paths in model_paths.values()
        ]
        rows.append({"n": n, "mean_pct": sum(means) / len(means), "min_pct": min(means), "max_pct": max(means)})
    return pd.DataFrame(rows)


def plot_config_axis(nll_df: pd.DataFrame, coverage_df: pd.DataFrame, save_dir: Path) -> None:
    set_default_style()
    fig, (ax_top, ax_main) = plt.subplots(
        2, 1, figsize=(8, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08}, layout="constrained"
    )

    ax_top.plot(coverage_df["n"], coverage_df["mean_pct"], color="0.3", marker="o", lw=2)
    ax_top.fill_between(coverage_df["n"], coverage_df["min_pct"], coverage_df["max_pct"], color="0.3", alpha=0.15)
    ax_top.set_ylabel("Pixels affected (%)")
    ax_top.set_title("Config axis: NLL vs. overlay count")

    for model in MODEL_ROW_ORDER:
        sub = nll_df[nll_df["model"] == model].sort_values("n")
        ax_main.plot(sub["n"], sub["artifact_nll"], marker="o", label=model, color=MODEL_COLORS[model])
    ax_main.set_xlabel("count (pasted artifact overlays)")
    ax_main.set_ylabel("NLL (artifact stream)")
    ax_main.set_xticks(N_VALUES)
    ax_main.legend(frameon=False, loc="upper left")

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "config_axis_nll_vs_count.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "config_axis_nll_vs_count.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_procedural_axis(nll_df: pd.DataFrame, save_dir: Path) -> None:
    set_default_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ROW_ORDER:
        sub = nll_df[nll_df["model"] == model].sort_values("n")
        ax.plot(sub["n"], sub["artifact_nll"], marker="o", label=model, color=MODEL_COLORS[model])
    ax.set_xlabel("severity (graded acquisition degradation)")
    ax.set_ylabel("NLL (artifact stream)")
    ax.set_xticks(N_VALUES)
    ax.set_title("Procedural axis: NLL vs. severity")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "procedural_axis_nll_vs_severity.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "procedural_axis_nll_vs_severity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the axis-paths YAML sidecar.")
    parser.add_argument(
        "--output-dir", required=True, help="Root holding each axis-value's artifact_quantification_summary.csv."
    )
    parser.add_argument("--figures-dir", default=str(ROOT / "figures" / "artifact_ablation"))
    args = parser.parse_args()

    axis_paths = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config_nll = load_nll_curve(output_dir, "config")
    procedural_nll = load_nll_curve(output_dir, "procedural")
    coverage = load_pixel_coverage(axis_paths)

    config_nll.to_csv(figures_dir / "config_axis_nll_vs_count.csv", index=False)
    coverage.to_csv(figures_dir / "config_axis_pixel_coverage_vs_count.csv", index=False)
    procedural_nll.to_csv(figures_dir / "procedural_axis_nll_vs_severity.csv", index=False)

    plot_config_axis(config_nll, coverage, figures_dir)
    plot_procedural_axis(procedural_nll, figures_dir)
    print(f"Saved figures and tidy CSVs to: {figures_dir}")


if __name__ == "__main__":
    main()
