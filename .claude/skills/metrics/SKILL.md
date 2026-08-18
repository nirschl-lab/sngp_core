---
name: metrics
description: Compute offline/research metrics from prediction CSVs -- cross-dataset OOD AUROC, calibration (smooth-ECE), Dempster-Shafer uncertainty, artifact-robustness quantification. Use for research-figure-driving numbers, not per-step training metrics (those are automatic via torchmetrics).
---

# Metrics

This project has a deliberate two-tier metrics split — know which tier a request
belongs to before writing anything:

1. **Online/per-step metrics**: `torchmetrics` objects living inside
   `src/models/lit_module_base.py` (accuracy, ECE, precision/recall/F1, NLL). These
   are automatic during training/eval — logged via `self.log(...)`, visible in
   W&B/console. Nothing to do here unless adding a *new* per-step metric, which means
   editing `LitModuleBase`, not this skill.
2. **Offline/research metrics**: standalone numpy/pandas functions in `src/metrics/`
   and `src/paper_helpers/`, driven by prediction CSVs written to disk (see
   `references/csv_schema.md` for the two CSV schemas in use and which consumer wants
   which). This skill is about tier 2.

## Ask first

1. Which metric: cross-dataset OOD AUROC, calibration (smooth-ECE), Dempster-Shafer
   uncertainty, or artifact-robustness quantification?
2. ID vs OOD dataset(s), or real-vs-artifact-simulated pair?
3. Which methods to compare (Baseline / MC-Dropout / SNGP / Deep Ensemble) and where
   their prediction CSVs live?
4. Confidence-score mode: max-softmax-probability (`"msp"`) vs entropy (`"entropy"`)?
   (SNGP predictive variance is a third uncertainty signal, handled separately —
   see `artifact_quantification.py`.)
5. Output directory.

## Reference implementations

- `src/metrics/auc.py` — `AUROC` (single ID/OOD pair), `AUROC_across_dataset` (one ID
  dataset vs many OOD datasets, bootstrapped over 10 seeds).
- `src/paper_helpers/ood_metrics/runner.py::run_ood_comparison` — the generalized,
  parametrized cross-dataset comparison across methods. **Use this, don't write a new
  copy-paste script.** `acevedo.py`/`kather2018.py`/`wong.py` in the same package are
  now just config-only call sites — follow that pattern for a new ID dataset.
- `src/paper_helpers/ood_metrics/artifact_quantification.py::quantify_artifact_impact`
  — paired real/artifact-simulated robustness metrics (accuracy drop, confidence drop,
  entropy rise, prediction flip rate) plus unpaired detection metrics (AUROC/AUPR/
  FPR@95TPR). Already well-parametrized; use directly.
- `src/metrics/smooth_ece.py`, `src/metrics/calibration_losses.py` — smooth/soft-binned
  ECE and calibration-loss terms, usable as post-hoc diagnostics or (already wired)
  auxiliary training losses.
- `src/metrics/dempster_shafer_uncertainity.py` — standalone Dempster-Shafer
  uncertainty from logits.

## Doing the work

- **New in-distribution dataset for OOD comparison**: add a new
  `src/paper_helpers/ood_metrics/<dataset>.py` following the existing three's shape —
  a `run_ood_comparison(dataset=..., methods={...}, ood_datasets=[...], out_dir=...)`
  call, nothing else. If you're about to write a loop over datasets/methods/AUROC
  calls again, stop — that loop already exists in `runner.py`.
- **Artifact-robustness numbers**: call `quantify_artifact_impact` with a
  `{method_name: csv_path}` map (see its docstring / existing call sites for the
  exact CSV columns it needs — schema differs from the OOD-comparison one, check
  `references/csv_schema.md`).
- **Calibration**: `src/metrics/smooth_ece.py` functions operate on
  `(confidences, correctness)` arrays directly — pull those from either CSV schema's
  `prediction_prob_score`/`confidence` + `true_bin_label`/`(prediction==target)` columns.

## Never

- Write a new per-ID-dataset script that reimplements the `AUROC_across_dataset` loop
  — that duplication (3 near-identical ~60-line scripts) is exactly what this
  project's migration consolidated into `runner.py`.
- Assume the two CSV schemas are interchangeable — check `references/csv_schema.md`.
- Add a new metric computation inline inside a notebook if it's meant to feed a
  paper figure — put it in `src/metrics/` or `src/paper_helpers/` so it's reusable
  and testable (`tests/paper_helpers/test_ood_runner.py` is the smoke-test pattern to
  follow — this tier gets smoke coverage only, not full behavioral tests, per
  CLAUDE.md §7 tier 4).
