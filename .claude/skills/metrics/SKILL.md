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
   see `src/metrics/artifact_quantification.py`.)
5. Output directory.

## Reference implementations

- `src/metrics/io.py` — shared loader for both CSV schemas: `load_predictions(path,
  fold=..., require=...)` returns a `PredictionFrame` (normalized `confidence`/
  `correct`/`probs`/`entropy_norm`/`entropy_nats`/`stream_canonical` columns, plus a
  `capabilities` set telling you what's actually in this file -- `"logits"` only on
  runs written after `predictions_csv_schema: 2`, `"member_logits"` only when
  `infer.save.save_member_logits` was on, `"paired_streams"` only for a real+artifact
  run). Also has the canonical `parse_float_list`/`normalized_entropy`/
  `shannon_entropy_nats`/`filter_fold`/`canonicalize_stream` -- **use these, don't
  re-derive them**; `auc.py`'s `_parse_class_probs`/`_normalized_entropy` and
  `artifact_quantification.py`'s `_parse_class_probs`/`_normalized_entropy_from_probs`
  are now thin aliases for exactly this reason.
- `src/metrics/auc.py` — `AUROC` (single ID/OOD pair), `AUROC_across_dataset` (one ID
  dataset vs many OOD datasets, bootstrapped over 10 seeds). Takes a `fold_policy`
  (`OodFoldPolicy` from `io.py`) -- defaults to `LEGACY_ISBI_FOLD_POLICY`, which
  filters the ID frame to `fold=='test'` but leaves OOD unfiltered; that asymmetry is
  load-bearing for every already-published number, so don't change the default to
  "fix" it. Pass `SYMMETRIC_FOLD_POLICY` for a new analysis that wants both frames
  filtered the same way.
- `src/paper_helpers/ood_metrics/runner.py::run_ood_comparison` — the generalized,
  parametrized cross-dataset comparison across methods. **Use this, don't write a new
  copy-paste script.** `acevedo.py`/`kather2018.py`/`wong.py` in the same package are
  now just config-only call sites — follow that pattern for a new ID dataset.
- `src/metrics/artifact_quantification.py::quantify_artifact_impact`
  — paired real/artifact-simulated robustness metrics (accuracy drop, confidence drop,
  entropy rise, prediction flip rate) plus unpaired detection metrics (AUROC/AUPR/
  FPR@95TPR). Already well-parametrized; use directly.
- `src/metrics/smooth_ece.py`, `src/metrics/calibration_losses.py` — smooth/soft-binned
  ECE and calibration-loss terms, usable as post-hoc diagnostics or (already wired)
  auxiliary training losses.
- `src/metrics/dempster_shafer_uncertainity.py` — standalone Dempster-Shafer
  uncertainty from logits.
- `src/metrics/registry.py` + `src/metrics/registered.py` — a metric registry
  (`METRIC_REGISTRY`, `register_metric`, `get_metric`, mirroring
  `src/models/registry.py`'s shape) mapping a plain name to a function
  `MetricContext -> list[MetricRow]`. `registered.py` is where the four current
  metrics (`ood_auroc_msp`, `ood_auroc_entropy`, `dempster_shafer`, `basic_stats`)
  live; importing `src.metrics` at all populates the registry (see `registered.py`'s
  docstring), so nothing special is needed to "activate" it. A `MetricSpec.requires`
  (a subset of `PredictionFrame.capabilities`) lets a caller check whether a run can
  even run a given metric before calling it -- `dempster_shafer` declares
  `requires={"logits"}`, so it's the right pattern to copy for a new metric that only
  some runs can support.
- `src/metrics/run_metrics.py` — runs every registered metric against a set of
  `predictions.csv` files (explicit paths and/or a recursive `--glob`) in one command,
  writing/upserting `csv/run_metrics/metrics_long.csv`. Each run's OOD candidates come
  from sibling dataset directories under the same `run_dir` on disk. This is the thing
  to reach for when a request is "compute X across a batch of runs", not a new
  per-checkpoint script — see `docs/METRICS_GUIDE.md`'s "Batch metrics from prediction
  CSVs" section, including "Adding a new metric to already-computed runs" for the
  register-then-rerun recipe.

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
  `(confidences, correctness)` arrays directly — `load_predictions(path).df[["confidence",
  "correct"]]` gives you both, already normalized across either CSV schema.
- **A new metric that reads a prediction CSV directly** (not through
  `run_ood_comparison`/`quantify_artifact_impact`): start from `load_predictions`, not
  a fresh `pd.read_csv` + your own parser. Check `frame.capabilities` (or pass
  `require=[...]`) before reaching for `logits_array`/`member_logits_array` — a run
  written before its needed column existed should fail with a clear
  `MissingPredictionData`, not a bare `KeyError` partway through.
- **A metric meant to run across every tracked inference run in one command** (not a
  one-off analysis): write it as a `MetricContext -> list[MetricRow]` function in
  `src/metrics/registered.py` and decorate it with `@register_metric(...)`, following
  `dempster_shafer`'s shape. Don't write a new standalone argparse script for this —
  that's exactly the per-metric-script sprawl the registry replaces.

## Never

- Write a new per-ID-dataset script that reimplements the `AUROC_across_dataset` loop
  — that duplication (3 near-identical ~60-line scripts) is exactly what this
  project's migration consolidated into `runner.py`.
- Assume the two CSV schemas are interchangeable — check `references/csv_schema.md`.
- Re-derive `class_probs`/`class_logits` parsing, entropy, or fold-filtering inline —
  `src/metrics/io.py` is that canonical implementation now; a third copy is exactly
  the duplication this file already had to clean up once
  (`artifact_quantification.py`'s `_build_stream_df` still has its own copy of the
  stream-canonicalization step deliberately, not the probability-parsing one — see
  that module's imports).
- Add a new metric computation inline inside a notebook if it's meant to feed a
  paper figure — put it in `src/metrics/` or `src/paper_helpers/` so it's reusable
  and testable (`tests/paper_helpers/test_ood_runner.py` is the smoke-test pattern to
  follow — this tier gets smoke coverage only, not full behavioral tests, per
  CLAUDE.md §7 tier 4).
