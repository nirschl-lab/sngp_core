"""artifact_severity_curves.py in src/visualization.

Standard classification-model ROC (-> AUROC), precision-recall (-> AUPR), and
risk-coverage (-> AURC) curves for the Acevedo artifact-simulation sweep -- true class
vs. predicted probability, computed directly on each axis-value's artifact-stream
`predictions.csv` (not the real-vs-artifact "detection" framing in
`src/metrics/artifact_quantification.py`). One figure per (axis, metric): a row of 5
panels (`MODEL_ROW_ORDER`, one per model), each overlaying count/severity 1-5 as
separate curves so the effect of artifact severity on each model's own classification
performance is directly comparable within a panel and across panels.

AUROC/AUPR use one-vs-rest micro-averaging (label-binarize the true class, ravel
against the full probability matrix, single `roc_curve`/`precision_recall_curve` call)
-- the same construction `multi_class_ROC.py::plot_roc_curve` uses per-class, collapsed
to one line per severity level so 5 severities overlay legibly in one panel. AURC
reuses `torch_uncertainty.metrics.classification.AURC`'s own `.plot()` method (the
same class `src/metrics/selective_classification.py::aurc()` wraps for the scalar
value already published in `artifact_quantification_summary.csv`) rather than
re-deriving the risk-coverage curve by hand.

Reads the same `configs/paper_helpers/acevedo_artifact_axis_paths.yaml` sidecar as
`artifact_ablation_curves.py` -- only each axis-value's `artifact` predictions.csv is
needed (no real-baseline pairing), loaded via the canonical
`src.metrics.io.load_predictions`.

Usage:
    uv run src/visualization/artifact_severity_curves.py \
        --config configs/paper_helpers/acevedo_artifact_axis_paths.yaml \
        --figures-dir figures/artifact_severity_curves
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch_uncertainty.metrics.classification import AURC

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.io import load_predictions, probs_array  # noqa: E402
from src.visualization.artifact_ablation_curves import N_VALUES, _axis_key  # noqa: E402
from src.visualization.style import order_models, set_default_style  # noqa: E402

Axis = Literal["config", "procedural"]
Metric = Literal["auroc", "aupr", "aurc"]

AXIS_LABEL = {"config": "count", "procedural": "severity"}
SEVERITY_CMAP = plt.cm.viridis


def _severity_colors(n_values: list[int] = N_VALUES) -> list:
    """Darker = more severe, shared across every panel/figure for a consistent read."""
    return [SEVERITY_CMAP(i / max(1, len(n_values) - 1)) for i in range(len(n_values))]


def load_artifact_probs_targets(
    axis_paths: dict, axis: Axis, n: int, model: str
) -> tuple[np.ndarray, np.ndarray]:
    """`(probs [N,C], targets [N])` for one (axis, severity/count value, model)."""
    csv_path = axis_paths[_axis_key(axis, n)]["models"][model]["artifact"]
    frame = load_predictions(csv_path)
    targets = frame.df["target"].astype(int).to_numpy()
    return probs_array(frame), targets


def _one_hot(targets: np.ndarray, num_classes: int) -> np.ndarray:
    """`label_binarize` returns a single column for exactly 2 classes -- expand that
    to proper one-hot so raveling against the (N,2) probability matrix lines up."""
    y_bin = label_binarize(targets, classes=list(range(num_classes)))
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return y_bin


def micro_roc(probs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One-vs-rest, micro-averaged ROC curve: ravel every class's binarized label and
    predicted probability into one pool before computing a single `roc_curve`."""
    y_bin = _one_hot(targets, probs.shape[1])
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    auroc = roc_auc_score(y_bin.ravel(), probs.ravel())
    return fpr, tpr, float(auroc)


def micro_pr(probs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One-vs-rest, micro-averaged precision-recall curve, same raveling as `micro_roc`."""
    y_bin = _one_hot(targets, probs.shape[1])
    precision, recall, _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    aupr = average_precision_score(y_bin.ravel(), probs.ravel())
    return precision, recall, float(aupr)


def plot_roc_panel(
    ax: plt.Axes, data_by_n: dict[int, tuple[np.ndarray, np.ndarray]], axis: Axis
) -> None:
    colors = _severity_colors()
    for n, color in zip(N_VALUES, colors):
        probs, targets = data_by_n[n]
        fpr, tpr, auroc = micro_roc(probs, targets)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{AXIS_LABEL[axis]}={n} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, loc="lower right")


def plot_pr_panel(
    ax: plt.Axes, data_by_n: dict[int, tuple[np.ndarray, np.ndarray]], axis: Axis
) -> None:
    colors = _severity_colors()
    for n, color in zip(N_VALUES, colors):
        probs, targets = data_by_n[n]
        precision, recall, aupr = micro_pr(probs, targets)
        label = f"{AXIS_LABEL[axis]}={n} (AUPR={aupr:.3f})"
        ax.plot(recall, precision, color=color, lw=2, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7, loc="lower left")


def plot_rc_panel(
    ax: plt.Axes, data_by_n: dict[int, tuple[np.ndarray, np.ndarray]], axis: Axis
) -> float:
    """Draws the panel's 5 severity curves and returns the true max risk (%) among
    them -- `AURC.plot()` sets its own `ylim` internally on every call, sized only to
    that one call's curve, so the axis's ylim after the loop reflects only the last
    severity plotted, not the panel's actual max. The caller uses the returned value
    to set one correct, shared `ylim` across every panel in the figure."""
    ax.set_prop_cycle(color=_severity_colors())
    for n in N_VALUES:
        probs, targets = data_by_n[n]
        probs_t = torch.as_tensor(probs, dtype=torch.float32)
        targets_t = torch.as_tensor(targets, dtype=torch.long)
        metric = AURC()
        metric.update(probs_t, targets_t)
        score = float(metric.compute())
        label = f"{AXIS_LABEL[axis]}={n} (AURC={score:.3f})"
        metric.plot(ax=ax, plot_value=False, name=label)
    ax.legend(fontsize=7, loc="upper left")
    return max(line.get_ydata().max() for line in ax.lines)


PANEL_FN = {"auroc": plot_roc_panel, "aupr": plot_pr_panel, "aurc": plot_rc_panel}
METRIC_TITLE = {
    "auroc": "ROC (AUROC)",
    "aupr": "Precision-Recall (AUPR)",
    "aurc": "Risk-Coverage (AURC)",
}


def build_figure(axis: Axis, metric: Metric, axis_paths: dict) -> plt.Figure:
    # One panel per model the sidecar actually has, in canonical order (a single-model
    # sidecar gives a single panel).
    models = order_models(axis_paths[_axis_key(axis, 1)]["models"])
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.6 * len(models), 5),
        sharex=(metric != "aurc"),
        sharey=(metric != "aurc"),
        squeeze=False,
    )
    axes = axes[0]
    panel_fn = PANEL_FN[metric]
    panel_maxes = []
    for ax, model in zip(axes, models):
        data_by_n = {n: load_artifact_probs_targets(axis_paths, axis, n, model) for n in N_VALUES}
        panel_maxes.append(panel_fn(ax, data_by_n, axis))
        ax.set_title(model, fontsize=12)

    if metric == "aurc":
        # AURC.plot() sets its own ylim per call (see plot_rc_panel's docstring), so
        # give every panel the same, correctly-sized range only after all are drawn.
        shared_max = max(panel_maxes) * 1.05
        for ax in axes:
            ax.set_ylim(0, shared_max)

    fig.suptitle(f"{axis.capitalize()} axis: {METRIC_TITLE[metric]} vs. {AXIS_LABEL[axis]}")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to the axis-paths YAML sidecar.")
    parser.add_argument(
        "--figures-dir", default=str(ROOT / "figures" / "artifact_severity_curves")
    )
    args = parser.parse_args()

    axis_paths = yaml.safe_load(Path(args.config).read_text())
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    set_default_style()
    for axis in ("config", "procedural"):
        for metric in ("auroc", "aupr", "aurc"):
            fig = build_figure(axis, metric, axis_paths)
            stem = figures_dir / f"{axis}_axis_{metric}_curves"
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {stem.with_suffix('.png')} and {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
