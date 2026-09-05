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

## Run manifest

[configs/runs/infer_manifest.yaml](../configs/runs/infer_manifest.yaml) is the
machine-readable record of every tracked inference run — which checkpoint, which
inference run-folder, which datasets it was evaluated against, whether it's an
ensemble or MC-Dropout. `docs/MASTER_INFER_RESULTS_PATH.md` and
`docs/MASTER_CHECKPONT_PATHS.md` are **generated from it** (a header in each says so)
— edit the manifest, not those docs directly.

[src/metrics/manifest.py](../src/metrics/manifest.py) loads/validates it and
regenerates the two docs:

```bash
uv run src/metrics/manifest.py --validate           # checks + confirms the docs are current
uv run src/metrics/manifest.py --write-master-docs  # regenerates both docs after an edit
```

`--validate` checks (with no GPU): labels are unique and filename-safe, each
`id_dataset` is one of its own `eval_datasets`, the run-id embedded in `run_dir`
matches the one in `ckpt`, every `run_dir` exists with a `predictions.csv` for each of
its `eval_datasets`/`paired_streams`, and nothing on disk under a `run_dir` is
unclaimed. Resolving real paths (everything except the label/id_dataset/run-id
checks) needs `EXPERIMENTS_HOME`/`PROJECT_NAME` set (see `env_example`) or an explicit
`--infer-root`.

## Batch metrics across every tracked run

[src/metrics/run_manifest_metrics.py](../src/metrics/run_manifest_metrics.py) runs
every registered metric from `src/metrics/registry.py` (or a chosen subset) against
every manifest entry (or a chosen subset):

```bash
uv run src/metrics/run_manifest_metrics.py
uv run src/metrics/run_manifest_metrics.py --runs baseline_acevedo sngp_acevedo
uv run src/metrics/run_manifest_metrics.py --metrics basic_stats dempster_shafer
```

Writes one long-format `csv/run_metrics/metrics_long.csv` (columns: `label, method,
train_dataset, id_dataset, scope, metric, value, std, value_str, status, note`) --
**never** `csv/ood_metrics/*.csv`, which stays `calculate_ood_metrics.py`'s own
published output (confirmed to match exactly: a test reproduces every published
`csv/ood_metrics/*.csv` value byte-for-byte from this driver). A run that can't
support a requested metric gets a `status=skipped` row with a `note` explaining why
(missing a required column, or no OOD datasets in that manifest entry) rather than
being silently absent or crashing the rest of the batch -- one bad file for one OOD
dataset only drops that dataset's row, not the whole run.

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
