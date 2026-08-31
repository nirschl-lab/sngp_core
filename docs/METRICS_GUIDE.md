# Metrics Guide

Offline/research metric scripts live in [src/metrics/](../src/metrics/) — driven by
prediction CSVs from inference/eval runs, not by live training (that's the
`torchmetrics`/`LitModuleBase` tier, see `docs/DEVELOPMENT.md`'s "Metrics &
Visualization" section). `src/metrics/auc.py` holds the shared building blocks
(`AUROC`, `_parse_class_probs`, `_normalized_entropy`) that scripts in this directory
reuse rather than re-deriving.

Untested-by-policy does **not** apply here (unlike `src/visualization/**`) — `src/metrics/**`
is covered by `pytest tests/metrics/` per `.claude/rules/testing.md`.

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
