"""artifact_confidence_histograms.py in src/visualization.

Max-softmax-probability (confidence) histograms across the artifact-simulation sweep:
how each model's confidence distribution moves as the shift grows, split into correct
and incorrect predictions. A model that handles shift well pushes its *incorrect*
predictions toward low confidence as the count/severity increases; an overconfident one
keeps them stacked near 1.0.

Layouts (`--layout`):

- `grid` (default): rows = models (`order_models`), columns = clean + each level. Each
  cell is a stacked histogram (correct bottom, incorrect top, so bar height is the full
  distribution) in percent of that cell's samples, with the clean distribution as a
  dashed outline for reference and the cell's accuracy / mean confidence annotated.
- `overlay`: one panel per level, every model's full distribution drawn as an outline in
  its `METHOD_COLORS` color.

With `--split-levels` (default on) every column/panel is also written as its own figure
on the same bins and y-limit, so the per-level files line up when arranged side by side
in a draft.

Level 0 ("clean") is each checkpoint's `real_baseline` predictions.csv; levels >= 1 are
the `artifact` predictions.csv of the sidecar's `_axis_key(axis, n)` entry. Several
sidecars can be merged with `--config a.yaml b.yaml` (first one wins on a duplicate model
within an axis key). Probabilities are `class_probs` exactly as inference wrote them --
for SNGP-family runs that is the mean-field predictive at the factor used at inference
time; nothing is re-fitted here.

Usage:
    uv run src/visualization/artifact_confidence_histograms.py \
        --config configs/paper_helpers/acevedo_artifact_axis_paths.yaml \
                 configs/paper_helpers/acevedo_specreg_artifact_axis_paths.yaml \
        --axis config \
        --models "Baseline Classifier" "Deep Ensemble" "Monte Carlo Dropout" SNGP \
                 "SNGP + Spectral Reg"
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import seaborn as sns
import yaml
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.io import load_predictions  # noqa: E402
from src.visualization.artifact_ablation_curves import N_VALUES, _axis_key  # noqa: E402
from src.visualization.style import (  # noqa: E402
    METHOD_COLORS,
    order_models,
    set_default_style,
)

Axis = Literal["config", "procedural"]
YScale = Literal["linear", "log"]

LEVEL_NAME = {"config": "Count", "procedural": "Severity"}

# Correct/incorrect split, from the same Wong 2011 palette as style.py. Models are told
# apart by row (grid) here, not by color, so these two do not collide with METHOD_COLORS.
CORRECT_COLOR = "#56B4E9"
INCORRECT_COLOR = "#D55E00"
CLEAN_OUTLINE = dict(color="0.15", lw=0.8, ls="--")

# "Confidently wrong" threshold for the summary table.
HIGH_CONF = 0.9

# Print sizing: the project default (font.size 14, 8x6 in) is for slides/notebooks; these
# figures are laid out at ~7.2 in (double column), so text is set at print size here.
PRINT_RC = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.titlepad": 4,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
}
CELL_W, CELL_H = 1.15, 1.0  # inches per grid cell
ROW_LABEL_WIDTH = 12  # characters before a model name wraps onto a second line


# --------------------------------------------------------------------------- loading


def load_axis_paths(configs: list[Path]) -> dict:
    """Merge several axis-path sidecars: per axis key, the union of their `models`.

    The first sidecar to define a model under a given axis key wins; a later duplicate
    is skipped with a warning (e.g. the specreg sidecar's procedural-axis "SNGP" is the
    c=4.0 bound-ablation run, not the protocol SNGP of the 5-model sidecar).
    """
    merged: dict = {}
    for config in configs:
        sidecar = yaml.safe_load(Path(config).read_text()) or {}
        skipped: dict[str, list[str]] = {}
        for axis_key, entry in sidecar.items():
            target = merged.setdefault(axis_key, {"label": entry.get("label", ""), "models": {}})
            for model, paths in entry.get("models", {}).items():
                if model in target["models"]:
                    skipped.setdefault(model, []).append(axis_key)
                    continue
                target["models"][model] = paths
        for model, keys in skipped.items():
            logger.warning(
                f"{config}: '{model}' already defined by an earlier sidecar under "
                f"{', '.join(keys)} -- keeping the earlier one."
            )
    return merged


def available_models(axis_paths: dict, axis: Axis) -> list[str]:
    return order_models(axis_paths.get(_axis_key(axis, 1), {}).get("models", {}))


def load_confidence_grid(
    axis_paths: dict, axis: Axis, models: list[str], levels: list[int]
) -> tuple[pd.DataFrame, int]:
    """Tidy `(model, level, confidence, correct)` frame over level 0 (clean) + `levels`,
    plus the number of classes (needed for the `1/K` lower edge of the bins)."""
    cache: dict[str, pd.DataFrame] = {}
    num_classes: set[int] = set()
    missing: list[str] = []
    frames = []

    def _read(csv_path: str) -> pd.DataFrame:
        if csv_path not in cache:
            frame = load_predictions(csv_path)
            num_classes.add(frame.num_classes)
            cache[csv_path] = frame.df[["confidence", "correct"]].astype(
                {"confidence": float, "correct": bool}
            )
        return cache[csv_path]

    for model in models:
        for level in [0, *levels]:
            entry = axis_paths.get(_axis_key(axis, max(level, 1)), {}).get("models", {}).get(model)
            if entry is None:
                missing.append(f"{model} @ {LEVEL_NAME[axis].lower()}={level}")
                continue
            df = _read(entry["real_baseline"] if level == 0 else entry["artifact"])
            frames.append(df.assign(model=model, level=level))

    if missing:
        raise KeyError(
            f"No sidecar entry for {len(missing)} (model, level) arm(s): " + "; ".join(missing)
        )
    if len(num_classes) != 1:
        raise ValueError(
            f"Prediction CSVs disagree on the number of classes: {sorted(num_classes)}"
        )
    out = pd.concat(frames, ignore_index=True)[["model", "level", "confidence", "correct"]]
    return out, num_classes.pop()


# --------------------------------------------------------------------------- statistics


def confidence_bins(num_classes: int, n_bins: int) -> np.ndarray:
    """Equal-width bins on `[1/K, 1]`: max-softmax can never fall below the uniform floor."""
    return np.linspace(1.0 / num_classes, 1.0, n_bins + 1)


def _pct_hist(values: np.ndarray, bins: np.ndarray, n_total: int) -> np.ndarray:
    counts, _ = np.histogram(np.clip(values, bins[0], bins[-1]), bins=bins)
    return 100.0 * counts / max(n_total, 1)


def histogram_table(df: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    """Percent of each (model, level) cell's samples per bin, split correct/incorrect --
    the exact numbers drawn, written alongside the figure."""
    rows = []
    for (model, level), cell in df.groupby(["model", "level"], sort=False):
        conf, correct = cell["confidence"].to_numpy(), cell["correct"].to_numpy()
        pct_c = _pct_hist(conf[correct], bins, len(cell))
        pct_i = _pct_hist(conf[~correct], bins, len(cell))
        for j in range(len(bins) - 1):
            rows.append(
                {
                    "model": model,
                    "level": level,
                    "bin_left": bins[j],
                    "bin_right": bins[j + 1],
                    "pct_correct": pct_c[j],
                    "pct_incorrect": pct_i[j],
                    "pct_total": pct_c[j] + pct_i[j],
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, level) sample statistics -- all exact, no resampling."""

    def _one(cell: pd.DataFrame) -> pd.Series:
        conf, correct = cell["confidence"], cell["correct"]
        return pd.Series(
            {
                "n": len(cell),
                "acc": correct.mean(),
                "mean_conf": conf.mean(),
                "mean_conf_correct": conf[correct].mean(),
                "mean_conf_incorrect": conf[~correct].mean(),
                f"frac_wrong_conf_ge_{HIGH_CONF}": ((~correct) & (conf >= HIGH_CONF)).mean(),
            }
        )

    return (
        df.groupby(["model", "level"], sort=False)[["confidence", "correct"]]
        .apply(_one)
        .reset_index()
    )


def _y_limits(hist: pd.DataFrame, yscale: YScale) -> tuple[float, float]:
    top = hist["pct_total"].max()
    if yscale == "log":
        positive = hist.loc[hist["pct_total"] > 0, "pct_total"]
        return 0.5 * positive.min(), 2.0 * top
    return 0.0, 1.08 * top


# --------------------------------------------------------------------------- plotting


def _level_title(axis: Axis, level: int) -> str:
    return "Clean" if level == 0 else f"{LEVEL_NAME[axis]} = {level}"


def _row_label(model: str) -> str:
    return textwrap.fill(model, ROW_LABEL_WIDTH)


def _model_color(model: str, models: list[str]):
    if model in METHOD_COLORS:
        return METHOD_COLORS[model]
    extras = [m for m in models if m not in METHOD_COLORS]
    return sns.color_palette("colorblind", 10)[(len(METHOD_COLORS) + extras.index(model)) % 10]


def plot_cell(
    ax: plt.Axes,
    hist: pd.DataFrame,
    summary: pd.Series,
    bins: np.ndarray,
    clean_total: np.ndarray | None,
) -> None:
    """One (model, level) cell: stacked correct/incorrect bars, clean outline, stats."""
    pct_c = hist["pct_correct"].to_numpy()
    pct_t = hist["pct_total"].to_numpy()
    ax.stairs(pct_c, bins, fill=True, color=CORRECT_COLOR, lw=0)
    ax.stairs(pct_t, bins, baseline=pct_c, fill=True, color=INCORRECT_COLOR, lw=0)
    if clean_total is not None:
        ax.stairs(clean_total, bins, **CLEAN_OUTLINE)
    ax.text(
        0.04,
        0.95,
        f"acc {summary['acc']:.1%}\nconf {summary['mean_conf']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        linespacing=1.15,
    )


def _grid_legend_handles() -> list:
    return [
        Patch(color=CORRECT_COLOR, label="Correct"),
        Patch(color=INCORRECT_COLOR, label="Incorrect"),
        Line2D([], [], label="Clean (all)", **CLEAN_OUTLINE),
    ]


def _style_axes(ax: plt.Axes, bins: np.ndarray, ylim: tuple[float, float], yscale: YScale) -> None:
    ax.set_xlim(bins[0], bins[-1])
    ax.set_yscale(yscale)
    ax.set_ylim(*ylim)
    # No tick on the 1/K floor: it would sit against the neighbouring cell's "1".
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([".25", ".5", ".75", "1"])
    ax.grid(axis="x", visible=False)


def build_grid_figure(
    hist: pd.DataFrame,
    summary: pd.DataFrame,
    models: list[str],
    levels: list[int],
    axis: Axis,
    bins: np.ndarray,
    yscale: YScale = "linear",
    row_labels: bool = True,
    legend: bool = True,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Rows = models, columns = `levels` (0 = clean). Shared x and y across every cell."""
    ylim = ylim or _y_limits(hist, yscale)
    fig, axes = plt.subplots(
        len(models),
        len(levels),
        figsize=(
            CELL_W * len(levels) + (0.55 if row_labels else 0.2),
            CELL_H * len(models) + 0.75,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    hist_idx = hist.set_index(["model", "level"]).sort_index()
    summ_idx = summary.set_index(["model", "level"])
    for i, model in enumerate(models):
        clean_total = hist_idx.loc[(model, 0), "pct_total"].to_numpy()
        for j, level in enumerate(levels):
            ax = axes[i, j]
            plot_cell(
                ax,
                hist_idx.loc[(model, level)],
                summ_idx.loc[(model, level)],
                bins,
                clean_total if level != 0 else None,
            )
            _style_axes(ax, bins, ylim, yscale)
            if i == 0:
                ax.set_title(_level_title(axis, level))
            if j == 0 and row_labels:
                ax.set_ylabel(_row_label(model), fontweight="bold")
            if j != 0 or not row_labels:
                ax.tick_params(labelleft=row_labels and j == 0)

    fig.supxlabel("Max softmax probability", fontsize=8)
    if row_labels:
        fig.supylabel("% of samples", fontsize=8)
    if legend:
        fig.legend(
            handles=_grid_legend_handles(),
            loc="outside upper center",
            ncol=3,
            handlelength=1.4,
            columnspacing=1.2,
        )
    return fig


def build_overlay_figure(
    hist: pd.DataFrame,
    models: list[str],
    levels: list[int],
    axis: Axis,
    bins: np.ndarray,
    yscale: YScale = "linear",
    ylim: tuple[float, float] | None = None,
    legend: bool = True,
    ylabel: bool = True,
) -> plt.Figure:
    """One panel per level, every model's full distribution as an outline."""
    ylim = ylim or _y_limits(hist, yscale)
    fig, axes = plt.subplots(
        1,
        len(levels),
        figsize=(1.6 * len(levels) + 0.3, 2.2),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    hist_idx = hist.set_index(["model", "level"]).sort_index()
    for ax, level in zip(axes[0], levels):
        for model in models:
            pct = hist_idx.loc[(model, level), "pct_total"].to_numpy()
            ax.stairs(pct, bins, color=_model_color(model, models), lw=1.1, label=model)
        _style_axes(ax, bins, ylim, yscale)
        ax.set_title(_level_title(axis, level))
    if ylabel:
        axes[0, 0].set_ylabel("% of samples")
    fig.supxlabel("Max softmax probability", fontsize=8)
    if legend:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(len(models), 5),
            handlelength=1.4,
        )
    return fig


# --------------------------------------------------------------------------- CLI


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {stem.with_suffix('.png')} and {stem.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", nargs="+", required=True, help="Axis-paths YAML sidecar(s); merged."
    )
    parser.add_argument("--axis", choices=["config", "procedural"], default="config")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Subset of sidecar model names (default: all)."
    )
    parser.add_argument("--levels", nargs="+", type=int, default=N_VALUES)
    parser.add_argument("--layout", choices=["grid", "overlay"], default="grid")
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--split-levels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write one figure per level (same bins and y-limit as the full figure).",
    )
    parser.add_argument(
        "--row-labels",
        choices=["first", "all"],
        default="first",
        help="Per-level grid files: model names on the clean file only, or on every file.",
    )
    parser.add_argument("--figures-dir", default=str(ROOT / "figures" / "confidence_histograms"))
    parser.add_argument("--stem-prefix", default="acevedo")
    args = parser.parse_args()

    axis_paths = load_axis_paths([Path(c) for c in args.config])
    present = available_models(axis_paths, args.axis)
    models = order_models(args.models) if args.models else present
    unknown = [m for m in models if m not in present]
    if unknown:
        raise SystemExit(
            f"Model(s) not in the merged sidecars' {args.axis} axis: {unknown}. Have: {present}"
        )

    levels = [0, *args.levels]
    df, num_classes = load_confidence_grid(axis_paths, args.axis, models, args.levels)
    bins = confidence_bins(num_classes, args.bins)
    hist = histogram_table(df, bins)
    summary = summarize(df)
    ylim = _y_limits(hist, args.yscale)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stem_prefix}_{args.axis}_confidence_{args.layout}"
    if args.yscale == "log":
        stem += "_log"
    hist.to_csv(figures_dir / f"{stem}_hist.csv", index=False)
    summary.to_csv(figures_dir / f"{stem}_summary.csv", index=False)
    logger.info(f"Wrote {figures_dir / stem}_{{hist,summary}}.csv")

    set_default_style()
    with mpl.rc_context(PRINT_RC):
        if args.layout == "grid":
            _save(
                build_grid_figure(
                    hist, summary, models, levels, args.axis, bins, args.yscale, ylim=ylim
                ),
                figures_dir / stem,
            )
            if args.split_levels:
                for level in levels:
                    labels = args.row_labels == "all" or level == 0
                    fig = build_grid_figure(
                        hist,
                        summary,
                        models,
                        [level],
                        args.axis,
                        bins,
                        args.yscale,
                        row_labels=labels,
                        legend=False,
                        ylim=ylim,
                    )
                    _save(fig, figures_dir / f"{stem}_level{level}")
                # The shared legend, once, as its own file for the draft layout.
                fig = plt.figure(figsize=(3.0, 0.3))
                fig.legend(handles=_grid_legend_handles(), loc="center", ncol=3, handlelength=1.4)
                _save(fig, figures_dir / f"{stem}_legend")
        else:
            _save(
                build_overlay_figure(
                    hist, models, levels, args.axis, bins, args.yscale, ylim=ylim
                ),
                figures_dir / stem,
            )
            if args.split_levels:
                for level in levels:
                    fig = build_overlay_figure(
                        hist,
                        models,
                        [level],
                        args.axis,
                        bins,
                        args.yscale,
                        ylim=ylim,
                        legend=level == 0,
                        ylabel=level == 0,
                    )
                    _save(fig, figures_dir / f"{stem}_level{level}")


if __name__ == "__main__":
    main()
