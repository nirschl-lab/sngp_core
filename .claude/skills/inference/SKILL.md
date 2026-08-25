---
name: inference
description: Run inference with a trained checkpoint on one of the project's datasets, including sweeps across checkpoints/datasets and artifact (paired real/simulated) inference. Use for any request to run predictions, evaluate a checkpoint on data, or check a checkpoint's architecture.
---

# Inference

This project has **one** inference entrypoint, `src/inference/infer.py`, driven by
Hydra (`configs/infer.yaml`). Do not write a new inference script — every request in
scope here is expressible as an invocation of that entrypoint (see
`docs/INFERENCE_GUIDE.md` for the full flag reference).

## Ask first

1. Checkpoint(s): a path, a glob, or a `wandb-artifact://` URI?
2. Dataset(s): one of the 7 (`acevedo`, `wong`, `tang`, `kather2018`, `jung`,
   `nirschl`, `kather2016` — see CLAUDE.md §1 for HF ids/class counts), or artifact
   (paired real/simulated) mode?
3. Fold: `train | validation | test | all`?
4. Output directory.
5. MC-Dropout passes, if applicable (baseline checkpoints only).
6. Whether to compute metrics (accuracy/ECE/precision/recall/f1/nll) against labels,
   or just produce predictions.

## Before running: check the checkpoint

Read the checkpoint's own metadata **before** constructing the command — this catches
a checkpoint/dataset mismatch immediately instead of after a slow run:

```bash
uv run python -c "
from src.checkpointing.io import read_meta
meta = read_meta('<ckpt_path>')
print(meta)
"
```

This reports `lit_module` (which family), `net_spec` (arch, and family-specific
hyperparameters), and `num_classes` — cross-check `num_classes` against the dataset
you're about to run on.

If `read_meta` raises about a missing `sngp_core` block, the checkpoint predates this
project's checkpoint-contract rewrite and needs a one-time migration:
```bash
uv run scripts/checkpoints/migrate_checkpoints.py --in <path> --out <dir> [--arch <arch>]
```
`--arch` is required only for legacy SNGP checkpoints (they didn't record their
backbone arch).

## Running

```bash
uv run src/inference/infer.py \
  ckpt_path=<path> \
  data=<dataset|artifact_image_classifier> \
  fold=<train|validation|test|all> \
  save_path=<output_dir>
```
`<dataset>` selects a per-dataset config from `configs/data/` (`acevedo`, `tang`,
`wong`, `kather2018`, `kather2016`, `jung`, `nirschl2018`) — each already carries the
right `dataset_name`/`num_classes`/`class_to_idx`, so no manual
`data.datamodule.dataset_name=...`/`num_classes=...` overrides are needed. For a
cross-dataset OOD sweep that overrides `data.datamodule.dataset_name` directly (see
`references/commands.md`), also pass `data.datamodule.class_to_idx=null` — otherwise
the datamodule's config-vs-dataset consistency check will fail once the overridden
dataset no longer matches the base config's own class map.

Key `infer.runtime.*` overrides (see `configs/infer/runtime/default.yaml`):
`device`, `batch_size_override`, `strict` (strict state_dict loading), `use_mc_dropout`,
`mc_passes`. There is deliberately no model-architecture override — the checkpoint is
authoritative for that.

For artifact (paired real/simulated-artifact) inference, use
`data=artifact_image_classifier` — the paired-batch handling is automatic, no separate
flag or script needed. `infer.save.save_images=true` saves the paired real/artifact
images alongside predictions.

`references/commands.md` has real recorded invocations (cross-dataset OOD tests,
`fold=all` sweeps, MC-Dropout runs) — treat as examples of shape, not literal copies;
always confirm `ckpt_path`/`save_path` against what actually exists first.

## Sweeps (multiple datasets/checkpoints)

For a sweep, loop over the checkpoint × dataset combinations, writing to
`<out>/<method>/<dataset>/` per combination — check each checkpoint's `read_meta()`
first (a mismatched `num_classes` between checkpoint and dataset silently skips
metrics rather than erroring, see `docs/INFERENCE_GUIDE.md`'s note on
`_check_metric_compatibility`). After each run, read `<out>/.../metrics.json` and
summarize as a table rather than re-parsing `predictions.csv`.

## Outputs

Per run: `predictions.csv`, `metrics.json` (if `infer.metrics.enabled`), `images/` (if
`infer.save.save_images=true`, artifact mode only). `predictions.csv`'s schema is
documented in the `metrics` skill's `references/csv_schema.md` — it differs slightly
from the schema `TestArtifactsCallback` writes during training-time testing.

## Never

- Write a new inference script or notebook cell that duplicates `infer.py`'s logic.
- Manually construct a model via `hydra.utils.instantiate` for inference — always go
  through `src.checkpointing.io.load_net`/`load_lit_module` (what `infer.py` itself
  uses), or `src/inference/predict_image.py` for a quick single/batch-image script.
