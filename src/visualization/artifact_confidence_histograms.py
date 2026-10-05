"""artifact_confidence_histograms.py in src/visualization.

Max-softmax-probability (confidence) histograms across the artifact-simulation sweep:
how each model's confidence distribution moves as the shift grows, split into correct
and incorrect predictions. A model that handles shift well pushes its *incorrect*
predictions toward low confidence as the count/severity increases; an overconfident one
keeps them piled up near 1.0.

Layouts (`--layout`):

- `grid` (default): rows = models (`order_models`), columns = clean + each level, with
  the cell's accuracy / NLL / error count annotated. `--bars` picks how a cell is drawn:
  - `mirrored` (default): correct predictions above the zero line, incorrect below it,
    each in percent of its *own* group. Errors are a few percent of a cell's samples, so
    on a shared scale they vanish under the correct mass at 1.0 (linear) or turn into
    single-sample spikes (log); normalizing each group separately shows *where in
    confidence* the errors sit, which is the question under shift. The clean level's two
    group distributions are mirrored the same way as a dashed reference.
  - `overlap` / `outline` / `stacked`: both groups in percent of all the cell's samples,
    incorrect drawn over correct, outlines only, or incorrect stacked on correct; the clean
    total distribution is the dashed reference. These also support `--yscale log`.
- `overlay`: one panel per level, every model's full distribution drawn as a translucent
  fill in its `METHOD_COLORS` color.

Outputs go to `figures/confidence_histograms/<stem-prefix>/` (the full figure + CSVs);
with `--split-levels` (default on) every column/panel and the legend are also written to
its `parts/` subfolder, on the same bins and y-limit, so the per-level files line up when
arranged side by side in a draft.

Level 0 ("clean") is each checkpoint's `real_baseline` predictions.csv; levels >= 1 are
the `artifact` predictions.csv of the sidecar's `_axis_key(axis, n)` entry. Several
sidecars can be merged with `--config a.yaml b.yaml` (first one wins on a duplicate model
within an axis key).

Probabilities: every SNGP-family run (any CSV with `raw_logits`) is re-scored at one
common mean-field factor, `--mean-field-factor` (default pi/8, the probit approximation):
`softmax(raw_logits / sqrt(1 + factor * variance))` via
`src.metrics.posthoc_calibration.mean_field_scale`, so a run's calibrated or
inference-time factor does not leak into the comparison. Dividing all logits of a sample by
one scalar leaves the argmax unchanged, so accuracy is identical to the CSV's. Runs with
no mean-field step (Baseline, Deep Ensemble, MC Dropout) use `class_probs` as written.
`--as-inferred` skips the re-scoring and plots `class_probs` as written for every run.

Usage:
    uv run src/visualization/artifact_confidence_histograms.py \
        --config configs/paper_helpers/acevedo_artifact_axis_paths.yaml \
                 configs/paper_helpers/acevedo_specreg_artifact_axis_paths.yaml \
                 configs/paper_helpers/acevedo_muon_artifact_axis_paths.yaml \
        --axis config \
        --models "Baseline Classifier" "Deep Ensemble" "Monte Carlo Dropout" SNGP \
                 "SNGP + Spectral Reg" "SNGP + Muon"
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import seaborn as sns
import torch
import yaml
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.dispersion import per_sample_nll  # noqa: E402
from src.metrics.io import (  # noqa: E402
    load_predictions,
    probs_array,
    raw_logits_array,
    uncertainty_array,
)
from src.metrics.posthoc_calibration import mean_field_scale  # noqa: E402
from src.visualization.artifact_ablation_curves import N_VALUES, _axis_key  # noqa: E402
from src.visualization.style import (  # noqa: E402
    METHOD_COLORS,
    order_models,
    set_default_style,
)

Axis = Literal["config", "procedural"]
YScale = Literal["linear", "log"]
Bars = Literal["mirrored", "overlap", "outline", "stacked"]

LEVEL_NAME = {"config": "Count", "procedural": "Severity"}

# The probit-approximation constant: the one mean-field factor every SNGP-family run is
# re-scored at for plots and reports.
PROBIT_FACTOR = math.pi / 8

# Correct/incorrect split: Wong 2011 blue and vermilion. The figure is about the incorrect
# predictions, so correct is the lighter layer (lower fill opacity, thinner outline) and
# incorrect carries the emphasis. Models are told apart by row (grid), not by color, so
# these do not collide with METHOD_COLORS.
CORRECT_STYLE = {"color": "#0072B2", "alpha": 0.45, "lw": 0.6}
INCORRECT_STYLE = {"color": "#D55E00", "alpha": 0.75, "lw": 1.0}
CHARCOAL = "#444444"
CLEAN_OUTLINE = dict(color=CHARCOAL, lw=0.8, ls="--")
# Model fills in the `overlay` layout.
DEFAULT_ALPHA = 0.5

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
    "grid.color": "#E5E5E5",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
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


def confidence_and_correct(
    csv_path: str, mean_field_factor: float | None
) -> tuple[pd.DataFrame, int, float]:
    """`(confidence, correct, nll)` per sample, the number of classes, and the mean-field
    factor applied (NaN when the run has no mean-field step or `mean_field_factor` is
    None, i.e. `class_probs` used as written). `nll` is `per_sample_nll` -- the definition
    `metrics.json` uses -- under the same probabilities the confidence comes from."""
    frame = load_predictions(csv_path)
    targets = frame.df["target"].astype(int).to_numpy()

    def _nll(probs: np.ndarray) -> np.ndarray:
        return per_sample_nll(torch.as_tensor(probs), torch.as_tensor(targets)).numpy()

    if mean_field_factor is None or "raw_logits" not in frame.capabilities:
        df = frame.df[["confidence", "correct"]].astype({"confidence": float, "correct": bool})
        df["nll"] = _nll(probs_array(frame))
        return df, frame.num_classes, float("nan")

    raw = torch.as_tensor(raw_logits_array(frame), dtype=torch.float64)
    var = torch.as_tensor(uncertainty_array(frame), dtype=torch.float64)
    probs = torch.softmax(mean_field_scale(raw, var, mean_field_factor), dim=1).numpy()
    pred = probs.argmax(axis=1)
    # A per-sample positive rescaling cannot move the argmax; if it does, the CSV's
    # raw_logits/uncertainty are not the pair the head's mean-field step consumed.
    stored_pred = probs_array(frame).argmax(axis=1)
    if not np.array_equal(pred, stored_pred):
        raise ValueError(
            f"{csv_path}: re-scored argmax differs from class_probs' on "
            f"{int((pred != stored_pred).sum())} row(s) -- raw_logits/uncertainty mismatch."
        )
    df = pd.DataFrame(
        {
            "confidence": probs.max(axis=1),
            "correct": pred == targets,
            "nll": _nll(probs),
        }
    )
    return df, frame.num_classes, float(mean_field_factor)


def load_confidence_grid(
    axis_paths: dict,
    axis: Axis,
    models: list[str],
    levels: list[int],
    mean_field_factor: float | None = PROBIT_FACTOR,
) -> tuple[pd.DataFrame, int]:
    """Tidy `(model, level, confidence, correct, nll, mean_field_factor)` frame over level 0
    (clean) + `levels`, plus the number of classes (for the `1/K` lower edge of the bins)."""
    cache: dict[str, tuple[pd.DataFrame, float]] = {}
    num_classes: set[int] = set()
    missing: list[str] = []
    frames = []

    for model in models:
        for level in [0, *levels]:
            entry = axis_paths.get(_axis_key(axis, max(level, 1)), {}).get("models", {}).get(model)
            if entry is None:
                missing.append(f"{model} @ {LEVEL_NAME[axis].lower()}={level}")
                continue
            csv_path = entry["real_baseline"] if level == 0 else entry["artifact"]
            if csv_path not in cache:
                df, k, factor = confidence_and_correct(csv_path, mean_field_factor)
                num_classes.add(k)
                cache[csv_path] = (df, factor)
            df, factor = cache[csv_path]
            frames.append(df.assign(model=model, level=level, mean_field_factor=factor))

    if missing:
        raise KeyError(
            f"No sidecar entry for {len(missing)} (model, level) arm(s): " + "; ".join(missing)
        )
    if len(num_classes) != 1:
        raise ValueError(
            f"Prediction CSVs disagree on the number of classes: {sorted(num_classes)}"
        )
    out = pd.concat(frames, ignore_index=True)
    columns = ["model", "level", "confidence", "correct", "nll", "mean_field_factor"]
    return out[columns], num_classes.pop()


# --------------------------------------------------------------------------- statistics


def confidence_bins(num_classes: int, n_bins: int) -> np.ndarray:
    """Equal-width bins on `[1/K, 1]`: max-softmax can never fall below the uniform floor."""
    return np.linspace(1.0 / num_classes, 1.0, n_bins + 1)


def _pct_hist(values: np.ndarray, bins: np.ndarray, n_total: int) -> np.ndarray:
    counts, _ = np.histogram(np.clip(values, bins[0], bins[-1]), bins=bins)
    return 100.0 * counts / max(n_total, 1)


def histogram_table(df: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    """Per (model, level) cell and bin, split correct/incorrect -- the exact numbers drawn,
    written alongside the figure. `pct_correct`/`pct_incorrect`/`pct_total` are percent of
    all the cell's samples; `pct_of_correct`/`pct_of_incorrect` are percent of that group
    alone (each sums to 100, or to 0 for an empty group) -- what `mirrored` draws."""
    rows = []
    for (model, level), cell in df.groupby(["model", "level"], sort=False):
        conf, correct = cell["confidence"].to_numpy(), cell["correct"].to_numpy()
        pct_c = _pct_hist(conf[correct], bins, len(cell))
        pct_i = _pct_hist(conf[~correct], bins, len(cell))
        of_c = _pct_hist(conf[correct], bins, int(correct.sum()))
        of_i = _pct_hist(conf[~correct], bins, int((~correct).sum()))
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
                    "pct_of_correct": of_c[j],
                    "pct_of_incorrect": of_i[j],
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
                "n_incorrect": int((~correct).sum()),
                "mean_field_factor": cell["mean_field_factor"].iloc[0],
                "acc": correct.mean(),
                "nll": cell["nll"].mean(),
                "mean_conf": conf.mean(),
                "mean_conf_correct": conf[correct].mean(),
                "mean_conf_incorrect": conf[~correct].mean(),
                f"frac_wrong_conf_ge_{HIGH_CONF}": ((~correct) & (conf >= HIGH_CONF)).mean(),
            }
        )

    return (
        df.groupby(["model", "level"], sort=False)[
            ["confidence", "correct", "nll", "mean_field_factor"]
        ]
        .apply(_one)
        .reset_index()
    )


def _y_limits(hist: pd.DataFrame, yscale: YScale, bars: Bars) -> tuple[float, float]:
    """Shared y-range across every cell. `mirrored`: correct up, incorrect down, each to
    its own group's tallest bar. Otherwise the tallest drawn bar -- the total when stacked,
    the larger of the two groups (or the clean outline, which is always a total)."""
    if bars == "mirrored":
        return -1.1 * hist["pct_of_incorrect"].max(), 1.1 * hist["pct_of_correct"].max()
    tops = ["pct_total"] if bars == "stacked" else ["pct_correct", "pct_incorrect"]
    clean = hist.loc[hist["level"] == 0, "pct_total"]
    drawn = pd.concat([hist[c] for c in tops] + [clean])
    if yscale == "log":
        return 0.5 * drawn[drawn > 0].min(), 2.0 * drawn.max()
    return 0.0, 1.08 * drawn.max()


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
    clean: pd.DataFrame | None,
    bars: Bars = "mirrored",
    ylim: tuple[float, float] | None = None,
) -> None:
    """One (model, level) cell: correct/incorrect histograms, clean reference, stats.

    `mirrored`: correct (% of correct) above zero, incorrect (% of incorrect) below.
    `overlap`: correct drawn first (light), incorrect on top of it (strong). `outline`: the
    two outlines alone. `stacked`: incorrect stacked on correct. `clean` is the model's
    level-0 rows of the histogram table (None for the clean column itself).
    """
    if bars == "mirrored":
        _plot_mirrored(ax, hist, clean, bins, ylim or _y_limits(hist, "linear", bars))
    else:
        _plot_shared_scale(ax, hist, clean, bins, bars)
    stats = (
        f"acc {summary['acc']:.1%}\nNLL {summary['nll']:.3f}\n"
        f"{int(summary['n_incorrect'])} errors"
    )
    ax.text(
        0.04,
        0.96,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        linespacing=1.1,
    )


def _draw(ax: plt.Axes, values, bins, style: dict, baseline=0, fill: bool = True) -> None:
    if fill:
        ax.stairs(
            values,
            bins,
            baseline=baseline,
            fill=True,
            color=style["color"],
            alpha=style["alpha"],
            lw=0,
        )
    ax.stairs(values, bins, baseline=baseline, color=style["color"], lw=style["lw"])


def _plot_mirrored(
    ax: plt.Axes,
    hist: pd.DataFrame,
    clean: pd.DataFrame | None,
    bins: np.ndarray,
    ylim: tuple[float, float],
) -> None:
    """Each half on its own scale, drawn to equal height: correct mass piles up to ~90% in
    the last bin while the errors' tallest bin is far lower, so one shared scale would
    squash the incorrect half -- the part the figure is about. Axes run -1..1; ticks are
    relabelled in each half's own percent."""
    lim_i, lim_c = -ylim[0], ylim[1]
    _draw(ax, hist["pct_of_correct"].to_numpy() / lim_c, bins, CORRECT_STYLE)
    _draw(ax, -hist["pct_of_incorrect"].to_numpy() / lim_i, bins, INCORRECT_STYLE)
    ax.axhline(0, color=CHARCOAL, lw=0.6)
    if clean is not None:
        ax.stairs(clean["pct_of_correct"].to_numpy() / lim_c, bins, **CLEAN_OUTLINE)
        ax.stairs(-clean["pct_of_incorrect"].to_numpy() / lim_i, bins, **CLEAN_OUTLINE)

    def _nice(limit: float) -> list[float]:
        ticks = mpl.ticker.MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10]).tick_values(0, limit)
        return [t for t in ticks if 0 < t < limit]

    up, down = _nice(lim_c), _nice(lim_i)
    positions = [-t / lim_i for t in reversed(down)] + [0.0] + [t / lim_c for t in up]
    labels = [f"{t:g}" for t in reversed(down)] + ["0"] + [f"{t:g}" for t in up]
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))


def _plot_shared_scale(
    ax: plt.Axes, hist: pd.DataFrame, clean: pd.DataFrame | None, bins: np.ndarray, bars: Bars
) -> None:
    pct_c = hist["pct_correct"].to_numpy()
    pct_i = hist["pct_incorrect"].to_numpy()
    base = pct_c if bars == "stacked" else 0
    top = pct_c + pct_i if bars == "stacked" else pct_i
    _draw(ax, pct_c, bins, CORRECT_STYLE, fill=bars != "outline")
    _draw(ax, top, bins, INCORRECT_STYLE, baseline=base, fill=bars != "outline")
    if clean is not None:
        ax.stairs(clean["pct_total"].to_numpy(), bins, **CLEAN_OUTLINE)


def _grid_legend_handles(bars: Bars = "mirrored") -> list:
    def _patch(style: dict, label: str) -> Patch:
        face = mpl.colors.to_rgba(style["color"], style["alpha"] if bars != "outline" else 0.0)
        return Patch(facecolor=face, edgecolor=style["color"], lw=style["lw"], label=label)

    mirrored = bars == "mirrored"
    return [
        _patch(CORRECT_STYLE, "Correct (above)" if mirrored else "Correct"),
        _patch(INCORRECT_STYLE, "Incorrect (below)" if mirrored else "Incorrect"),
        Line2D([], [], label="Clean" if mirrored else "Clean (all)", **CLEAN_OUTLINE),
    ]


def _style_axes(ax: plt.Axes, bins: np.ndarray, ylim: tuple[float, float], yscale: YScale) -> None:
    ax.set_xlim(bins[0], bins[-1])
    # set_yscale resets the tick locator/formatter -- skip it when nothing changes, so the
    # mirrored layout's per-half tick labels survive.
    if ax.get_yscale() != yscale:
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
    bars: Bars = "mirrored",
    row_labels: bool = True,
    legend: bool = True,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Rows = models, columns = `levels` (0 = clean). Shared x and y across every cell."""
    ylim = ylim or _y_limits(hist, yscale, bars)
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
        clean = hist_idx.loc[(model, 0)]
        for j, level in enumerate(levels):
            ax = axes[i, j]
            plot_cell(
                ax,
                hist_idx.loc[(model, level)],
                summ_idx.loc[(model, level)],
                bins,
                clean if level != 0 else None,
                bars=bars,
                ylim=ylim,
            )
            _style_axes(ax, bins, (-1.0, 1.0) if bars == "mirrored" else ylim, yscale)
            if i == 0:
                ax.set_title(_level_title(axis, level))
            if j == 0 and row_labels:
                ax.set_ylabel(_row_label(model), fontweight="bold")
            if j != 0 or not row_labels:
                ax.tick_params(labelleft=row_labels and j == 0)

    fig.supxlabel("Max softmax probability", fontsize=8)
    if row_labels:
        fig.supylabel(
            (
                "% of correct \u2191 / % of incorrect \u2193"
                if bars == "mirrored"
                else "% of samples"
            ),
            fontsize=8,
        )
    if legend:
        fig.legend(
            handles=_grid_legend_handles(bars),
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
    alpha: float = DEFAULT_ALPHA,
    ylim: tuple[float, float] | None = None,
    legend: bool = True,
    ylabel: bool = True,
) -> plt.Figure:
    """One panel per level, every model's full distribution as a translucent fill."""
    ylim = ylim or _y_limits(hist, yscale, "stacked")
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
            color = _model_color(model, models)
            ax.stairs(pct, bins, fill=True, color=color, alpha=alpha, lw=0, label=model)
            ax.stairs(pct, bins, color=color, lw=0.8)
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
            ncol=min(len(models), 6),
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
    parser.add_argument(
        "--bars", choices=["mirrored", "overlap", "outline", "stacked"], default="mirrored"
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="Model fill opacity (overlay layout)."
    )
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--mean-field-factor",
        type=float,
        default=PROBIT_FACTOR,
        help="Mean-field factor every SNGP-family run is re-scored at (default pi/8).",
    )
    parser.add_argument(
        "--as-inferred",
        action="store_true",
        help="Skip the re-scoring: plot class_probs exactly as inference wrote them.",
    )
    parser.add_argument(
        "--split-levels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write one figure per level + the legend to <figures-dir>/parts/.",
    )
    parser.add_argument(
        "--row-labels",
        choices=["first", "all"],
        default="first",
        help="Per-level grid files: model names on the clean file only, or on every file.",
    )
    parser.add_argument("--stem-prefix", default="acevedo")
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Default: figures/confidence_histograms/<stem-prefix>.",
    )
    args = parser.parse_args()

    axis_paths = load_axis_paths([Path(c) for c in args.config])
    present = available_models(axis_paths, args.axis)
    models = order_models(args.models) if args.models else present
    unknown = [m for m in models if m not in present]
    if unknown:
        raise SystemExit(
            f"Model(s) not in the merged sidecars' {args.axis} axis: {unknown}. Have: {present}"
        )

    factor = None if args.as_inferred else args.mean_field_factor
    levels = [0, *args.levels]
    df, num_classes = load_confidence_grid(axis_paths, args.axis, models, args.levels, factor)
    bins = confidence_bins(num_classes, args.bins)
    hist = histogram_table(df, bins)
    summary = summarize(df)
    bars = args.bars if args.layout == "grid" else "stacked"
    ylim = _y_limits(hist, args.yscale, bars)

    rescored = sorted(summary.loc[summary["mean_field_factor"].notna(), "model"].unique())
    logger.info(
        "class_probs as written for every run"
        if factor is None
        else f"Re-scored at mean-field factor {factor:.4f}: {rescored}"
    )

    figures_dir = Path(
        args.figures_dir or ROOT / "figures" / "confidence_histograms" / args.stem_prefix
    )
    parts_dir = figures_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stem_prefix}_{args.axis}_confidence_{args.layout}"
    if bars == "mirrored" and args.yscale == "log":
        raise SystemExit("--bars mirrored draws incorrect below zero; it needs --yscale linear.")
    if args.layout == "grid" and bars != "mirrored":
        stem += f"_{bars}"
    if args.yscale == "log":
        stem += "_log"
    if factor is None:
        stem += "_as_inferred"
    hist.to_csv(figures_dir / f"{stem}_hist.csv", index=False)
    summary.to_csv(figures_dir / f"{stem}_summary.csv", index=False)
    logger.info(f"Wrote {figures_dir / stem}_{{hist,summary}}.csv")

    set_default_style()
    with mpl.rc_context(PRINT_RC):
        if args.layout == "grid":
            common = dict(yscale=args.yscale, bars=bars, ylim=ylim)
            _save(
                build_grid_figure(hist, summary, models, levels, args.axis, bins, **common),
                figures_dir / stem,
            )
            if args.split_levels:
                for level in levels:
                    fig = build_grid_figure(
                        hist,
                        summary,
                        models,
                        [level],
                        args.axis,
                        bins,
                        row_labels=args.row_labels == "all" or level == 0,
                        legend=False,
                        **common,
                    )
                    _save(fig, parts_dir / f"{stem}_level{level}")
                # The shared legend, once, as its own file for the draft layout.
                fig = plt.figure(figsize=(3.2, 0.3))
                fig.legend(
                    handles=_grid_legend_handles(bars),
                    loc="center",
                    ncol=3,
                    handlelength=1.4,
                )
                _save(fig, parts_dir / f"{stem}_legend")
        else:
            common = dict(yscale=args.yscale, alpha=args.alpha, ylim=ylim)
            _save(
                build_overlay_figure(hist, models, levels, args.axis, bins, **common),
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
                        legend=level == 0,
                        ylabel=level == 0,
                        **common,
                    )
                    _save(fig, parts_dir / f"{stem}_level{level}")


if __name__ == "__main__":
    main()
