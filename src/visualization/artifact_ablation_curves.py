"""artifact_ablation_curves.py in src/visualization.

Ablation across the Acevedo artifact-simulation axes: config axis (`count`=0..5, 0 = real
image, 1..5 = pasted artifact overlays) and procedural axis (`severity`=0..5, 0 = real
image, 1..5 = graded acquisition degradation). Plots NLL vs. count/severity, one line per
model. The `n=1..5` points read `artifact_nll` from each axis-value's
`artifact_quantification_summary.csv` (written by
`src/paper_helpers/ood_metrics/render_artifact_results_tables.py` /
`src/metrics/artifact_quantification.py::quantify_artifact_impact`; run paths in
`configs/paper_helpers/acevedo_artifact_axis_paths.yaml`); the `n=0` anchor reads
`real.nll` straight from each checkpoint's existing `real_baseline/metrics.json` (shared
by both axes, no new inference/aggregation needed). The config-axis figure overlays mean
`percent_pixels_affected` (+/- 1 std across the ~3420 test images) per count on a second
y-axis -- already computed per-sample by `ArtifactHFDataset` and present in each run's
predictions.csv -- since that's a property of the simulator's overlay draw, not of the
model (bit-identical across all 5 checkpoints' predictions.csv for the same count, since
the simulator's seed is content-derived per image, see docs/DATASETS.md; 0 at n=0 by
definition -- the real stream never runs the simulator, so it has no such column at all).
Procedural effects are whole-frame, not masked, so there's no comparable coverage number
for that axis.

Usage:
    uv run src/visualization/artifact_ablation_curves.py \\
        --config configs/paper_helpers/acevedo_artifact_axis_paths.yaml \\
        --output-dir csv/artifact_quantification/acevedo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
import seaborn as sns
import yaml

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import MODEL_ROW_ORDER, order_models, set_default_style  # noqa: E402

# One fixed color per canonical model, so the same model reads as the same color in every
# figure regardless of which subset a given sidecar contains.
MODEL_COLORS = dict(zip(MODEL_ROW_ORDER, sns.color_palette("colorblind", len(MODEL_ROW_ORDER))))
N_VALUES = [1, 2, 3, 4, 5]


def _models_in(nll_df: pd.DataFrame) -> list[str]:
    """The models this sidecar actually has, in canonical order."""
    return order_models(nll_df["model"].unique())


def _color_for(model: str, models: list[str]):
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    # A model outside the canonical list: a stable extra palette entry, by position.
    extras = [m for m in models if m not in MODEL_COLORS]
    palette = sns.color_palette("colorblind", len(MODEL_COLORS) + len(extras))
    return palette[len(MODEL_COLORS) + extras.index(model)]


def _axis_key(axis: str, n: int) -> str:
    if axis == "config":
        return "config_axis" if n == 1 else f"config_axis_count_{n}"
    return "procedural_axis" if n == 1 else f"procedural_axis_severity_{n}"


def load_real_nll(axis_paths: dict) -> pd.DataFrame:
    """The `n=0` anchor (real, unperturbed images): `real.nll` from each checkpoint's
    `real_baseline/metrics.json`, one row per model. That file is identical for both axes
    (same checkpoint, same clean test set), so this is computed once and shared."""
    rows = []
    for model, paths in axis_paths["config_axis"]["models"].items():
        metrics_json = Path(paths["real_baseline"]).with_name("metrics.json")
        data = json.loads(metrics_json.read_text())
        rows.append({"model": model, "artifact_nll": data["real.nll"], "n": 0})
    return pd.DataFrame(rows)


def load_nll_curve(output_dir: Path, axis: str, axis_paths: dict) -> pd.DataFrame:
    """One row per (model, n) with the axis-value's NLL, `n=0` (real) through `n=5`."""
    rows = [load_real_nll(axis_paths)]
    for n in N_VALUES:
        summary_csv = output_dir / _axis_key(axis, n) / "artifact_quantification_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing {summary_csv} -- run render_artifact_results_tables.py first")
        df = pd.read_csv(summary_csv)[["model", "artifact_nll"]].copy()
        df["n"] = n
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_pixel_coverage(axis_paths: dict) -> pd.DataFrame:
    """Config axis only: mean +/- std `percent_pixels_affected` per count, across the test
    images (from one representative model -- the column is bit-identical across all 5 for
    a given count). `n=0` (real images) is 0 by definition -- the real stream never runs
    the simulator, so `real_baseline/predictions.csv` has no `percent_pixels_affected`
    column at all."""
    reference_model = order_models(axis_paths["config_axis"]["models"])[0]
    rows = [{"n": 0, "mean_pct": 0.0, "std_pct": 0.0}]
    for n in N_VALUES:
        artifact_csv = axis_paths[_axis_key("config", n)]["models"][reference_model]["artifact"]
        pct = pd.read_csv(artifact_csv, usecols=["percent_pixels_affected"])["percent_pixels_affected"]
        rows.append({"n": n, "mean_pct": pct.mean(), "std_pct": pct.std()})
    return pd.DataFrame(rows)


ALL_N = [0, *N_VALUES]


def plot_config_axis(nll_df: pd.DataFrame, coverage_df: pd.DataFrame, save_dir: Path) -> None:
    set_default_style()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax2 = ax.twinx()

    models = _models_in(nll_df)
    for model in models:
        sub = nll_df[nll_df["model"] == model].sort_values("n")
        ax.plot(sub["n"], sub["artifact_nll"], marker="o", label=model, color=_color_for(model, models))
    ax.set_xlabel("count (0 = real image, 1-5 = pasted artifact overlays)")
    ax.set_ylabel("NLL (artifact stream)")
    ax.set_xticks(ALL_N)

    cov = coverage_df.sort_values("n")
    ax2.errorbar(
        cov["n"],
        cov["mean_pct"],
        yerr=cov["std_pct"],
        color="0.3",
        linestyle="--",
        marker="s",
        capsize=3,
        label="Pixels affected (%, +/-1 std)",
    )
    ax2.set_ylabel("Pixels affected (%)")
    ax2.set_ylim(bottom=0)

    ax.set_title("Config axis: NLL and pixel coverage vs. overlay count")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "config_axis_nll_vs_count.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "config_axis_nll_vs_count.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_procedural_axis(nll_df: pd.DataFrame, save_dir: Path) -> None:
    set_default_style()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    models = _models_in(nll_df)
    for model in models:
        sub = nll_df[nll_df["model"] == model].sort_values("n")
        ax.plot(sub["n"], sub["artifact_nll"], marker="o", label=model, color=_color_for(model, models))
    ax.set_xlabel("severity (0 = real image, 1-5 = graded acquisition degradation)")
    ax.set_ylabel("NLL (artifact stream)")
    ax.set_xticks(ALL_N)
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

    config_nll = load_nll_curve(output_dir, "config", axis_paths)
    procedural_nll = load_nll_curve(output_dir, "procedural", axis_paths)
    coverage = load_pixel_coverage(axis_paths)

    config_nll.to_csv(figures_dir / "config_axis_nll_vs_count.csv", index=False)
    coverage.to_csv(figures_dir / "config_axis_pixel_coverage_vs_count.csv", index=False)
    procedural_nll.to_csv(figures_dir / "procedural_axis_nll_vs_severity.csv", index=False)

    plot_config_axis(config_nll, coverage, figures_dir)
    plot_procedural_axis(procedural_nll, figures_dir)
    print(f"Saved figures and tidy CSVs to: {figures_dir}")


if __name__ == "__main__":
    main()
