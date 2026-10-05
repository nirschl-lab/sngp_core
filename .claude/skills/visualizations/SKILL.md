---
name: visualizations
description: Produce publication figures for this project (reliability/calibration curves, multi-class ROC, probability histograms, Dempster-Shafer uncertainty plots, OOD-AUROC comparisons, artifact-degradation plots). Use whenever the user wants a paper/poster figure, not a live training dashboard.
---

# Visualizations

Scope: offline, reusable figure-generation code for research publications. Not the
live training-time plots (calibration curve/ROC/histogram/DS-uncertainty logged by
`src/callbacks/test_artifacts_callback.py` during `trainer.test(...)` — those already
exist and run automatically; only touch that callback if the *training-time* diagnostic
set itself needs to change).

## Ask first

1. Figure type: reliability/calibration curve, multi-class ROC, probability
   histogram, Dempster-Shafer uncertainty, OOD-AUROC comparison bar/line chart, or
   artifact-degradation plot (accuracy/confidence drop, entropy shift)?
2. Input data: which prediction CSV(s) (see the `metrics` skill's
   `references/csv_schema.md` for the two schemas in use)?
3. Single model or multi-method overlay (Baseline/MC-Dropout/SNGP/Ensemble on one
   plot)?
4. Target venue sizing (single- vs double-column), DPI, output formats (PDF for
   vector/print, PNG for slides/preview).

## Reference implementations

- `src/visualization/style.py::set_default_style(use_tex=False)` — call this first in
  any new figure script; it sets the shared matplotlib/seaborn rcParams
  (whitegrid, pastel palette, font sizes). There is **no** existing per-method color
  mapping (Baseline/MC-Dropout/SNGP/Ensemble don't have fixed colors yet) — if a new
  figure needs consistent colors across methods, define the mapping once in
  `style.py` (add a `METHOD_COLORS` dict) rather than hardcoding colors per-script, so
  later figures can reuse it.
- `src/visualization/plot_ece.py::plot_calibration_curve` — the live training-time
  reliability diagram (also directly reusable offline).
- `src/visualization/reliability.py` — richer, offline-oriented reliability-diagram
  variants (binned and kernel-smoothed) with confidence intervals — more figure
  options than `plot_ece.py`'s live version.
- `src/visualization/multi_class_ROC.py::plot_roc_curve`,
  `plot_prob_histograms.py::single_model_probability_histogram`,
  `dempster_shafer_uncertainity_plot.py::DempsterShaferUncertaintyPlot` — the other
  three live-training-time plots, all directly reusable offline.
- `src/visualization/uncertainity.py` — confidence-vs-accuracy and
  rejection-classification diagrams (not currently wired into training, offline-only).
- `src/visualization/artifact_confidence_histograms.py` — offline, multi-model
  max-softmax-probability histograms across the artifact sweep (models x clean +
  count/severity levels, correct/incorrect stacked, clean outline); merges several
  axis-path sidecars and writes per-level files for draft layout.
- `src/visualization/density.py` — kernel-density helpers (reflected KDE,
  Nadaraya-Watson) used by the smoothed reliability diagrams; reusable for any other
  density-based plot.
- `src/metrics/artifact_quantification.py` — the most
  "productionized" example: takes `output_dir` as an argument, saves to
  `output_dir/plots/`, generates ROC curves, uncertainty-shift KDEs, and comparison
  bar charts. Use this as the template for a new multi-method comparison figure.

## Conventions

- Every plotting function takes data in and **returns a `Figure`** — no `plt.show()`,
  no file I/O inside the plotting function itself. Saving happens in the caller (a
  script under `src/paper_helpers/figures/` or a `__main__` block), so the same
  function works for training-time W&B logging (`TestArtifactsCallback` passes the
  returned figure to `wandb.Image(fig)`) and offline paper-figure generation.
- Save both a vector format (PDF or SVG) for the paper and PNG for quick preview/slides.
- New standalone figure scripts go in `src/paper_helpers/figures/<figure_name>.py`
  with a thin `if __name__ == "__main__":` CLI — not a notebook, so they're
  diffable/reusable (this mirrors how `artifact_quantification.py` already works).
- Close figures after saving (`plt.close(fig)`) in any script that generates many —
  see `TestArtifactsCallback._log_figures` for the pattern.

## Never

- Call `plt.show()` or hardcode a save path inside a reusable plotting function.
- Duplicate an existing plot function's logic in a notebook — import from
  `src/visualization/` or `src/paper_helpers/` instead. Notebook-vs-`src/` duplication
  of exactly this kind (an older `notebooks/metrics/plot_auroc.py` reimplementing ROC
  plotting) was already found and removed once during this project's migration; don't
  reintroduce the pattern.
