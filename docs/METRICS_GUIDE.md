# Metrics Guide

Offline/research metric scripts live in [src/metrics/](../src/metrics/) — driven by
prediction CSVs from inference/eval runs, not by live training (that's the
`torchmetrics`/`LitModuleBase` tier, see `docs/DEVELOPMENT.md`'s "Metrics &
Visualization" section). [src/metrics/io.py](../src/metrics/io.py) is the shared
loader for both prediction-CSV schemas (`load_predictions` → a `PredictionFrame` with
normalized `confidence`/`correct`/`probs`/`entropy_norm`/`entropy_nats` columns and a
`capabilities` set) that everything else in this directory builds on rather than
re-parsing `class_probs` from scratch; `src/metrics/auc.py`'s `AUROC`/
`AUROC_across_dataset` are the shared AUROC building blocks.

Untested-by-policy does **not** apply here (unlike `src/visualization/**`) — `src/metrics/**`
is covered by `pytest tests/metrics/` per `.claude/rules/testing.md`.

## Batch metrics from prediction CSVs

[src/metrics/run_metrics.py](../src/metrics/run_metrics.py) runs every registered
metric from `src/metrics/registry.py` (or a chosen subset) against a set of
`predictions.csv` files. Point it at explicit paths and/or a recursive glob — no
config file listing runs required; each run's identity and OOD candidates are
inferred straight from the on-disk layout (`<run_dir>/<dataset_name>/predictions.csv`,
see [OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md)) — every sibling dataset directory under the
same `run_dir` becomes an OOD frame for `needs_ood` metrics automatically:

```bash
uv run src/metrics/run_metrics.py --predictions /path/to/run/acevedo/predictions.csv
uv run src/metrics/run_metrics.py --glob '/data1/.../infer/**/predictions.csv'
uv run src/metrics/run_metrics.py --glob '...' --metrics basic_stats dempster_shafer
```

Writes one long-format `csv/run_metrics/metrics_long.csv` (columns: `label, run_dir,
dataset, scope, metric, value, std, value_str, status, note`) — **never**
`csv/ood_metrics/*.csv`, which stays `calculate_ood_metrics.py`'s own published
output. A file that can't support a requested metric gets a `status=skipped` row with
a `note` explaining why (missing a required column, or no sibling OOD datasets on
disk) rather than being silently absent or crashing the rest of the batch — one bad
OOD sibling only drops that dataset's row, not the whole run. Re-running after adding
a new metric to `registered.py` **upserts** into the existing output CSV (rows keyed
on `label`/`metric`/`scope`) instead of overwriting it, so you don't have to re-point
it at every historical run just to pick up the new metric.

### Adding a new metric to already-computed runs

You already have `predictions.csv` files on disk from past inference runs and want a
new metric's values added without recomputing or disturbing anything else:

1. Write the metric as a `MetricContext -> list[MetricRow]` function in
   [src/metrics/registered.py](../src/metrics/registered.py) and decorate it with
   `@register_metric("your_metric_name", ...)` — see `dempster_shafer` there for the
   shape, including `requires=`/`needs_ood=` if the metric only applies to some runs.
2. Re-run `run_metrics.py` over the same predictions.csv files as before, restricted
   to just the new metric so nothing else is recomputed:

   ```bash
   uv run src/metrics/run_metrics.py \
     --glob '/data1/.../infer/**/predictions.csv' \
     --metrics your_metric_name
   ```

Because the output is upserted, this only adds/updates rows for `your_metric_name` in
`csv/run_metrics/metrics_long.csv` — every other metric's already-computed rows for
those same runs are left untouched. This does **not** touch each run's own
`metrics.json` (that's `infer.py`'s per-run output, written once at inference time) —
`csv/run_metrics/metrics_long.csv` is the one file this workflow updates.

## Cross-dataset OOD AUROC

[src/metrics/calculate_ood_metrics.py](../src/metrics/calculate_ood_metrics.py) computes
AUROC for how well a single checkpoint's predictive uncertainty separates its
in-distribution dataset from one or more out-of-distribution datasets — both max-softmax-
probability (`confidence` column) and normalized-entropy (`class_probs` column) scores are
computed in one pass, so you get both without picking a score mode upfront.

It reads from the per-dataset inference output layout described in
[docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md) — `<run_dir>/<dataset_name>/predictions.csv` — and
expects `--indist`/`--outdist` to name the dataset folders directly (not full paths), same
convention as [src/visualization/predictive_entropy.py](../src/visualization/predictive_entropy.py):

```bash
uv run src/metrics/calculate_ood_metrics.py \
  --run-dir /data1/maheswararao/experiments/uncertaity-aware-ml/infer/baseline_classifier_acevedo/2026-08-25_14-31-36 \
  --indist acevedo \
  --outdist jung kather2016 kather2018 nirschl2018 tang wong \
  --out-dir csv/ood_metrics \
  --name baseline_acevedo
```

Notes:

- `--indist` takes exactly one dataset name — AUROC needs a single in-distribution
  reference per checkpoint (unlike `predictive_entropy.py`'s KDE overlay, which can show
  several in-distribution lines at once).
- AUROC is computed as a mean ± std across 10 fixed seeds (`src/metrics/auc.py`'s `seeds`
  constant), sampling `--sample-rate` rows per seed (default 1000) — matches the sampling
  methodology of the project's earlier `src/paper_helpers/ood_metrics/` scripts, for
  reproducibility with prior paper numbers.
- Output: `<out-dir>/<name>.csv` (`--out-dir` defaults to `csv/ood_metrics/` at the repo
  root, matching the existing `csv/dataset_stats/`/`csv/isbi_test_files/` convention).
- This script is unrelated to `src/paper_helpers/ood_metrics/` (`acevedo.py`/`kather2018.py`/
  `wong.py`/`runner.py`) — those target an older, now-partially-stale CSV layout
  (`csv/final/<method>/<dataset>.csv`, one flat file per dataset written by
  `TestArtifactsCallback` during training-time testing) and are left as-is.
