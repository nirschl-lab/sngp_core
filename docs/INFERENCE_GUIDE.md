# Inference Guide

This project uses a single inference entrypoint:

- [src/inference/infer.py](src/inference/infer.py)

The artifact path is handled internally by batch-format detection. You do not need to run a separate script.

## Config Layout

Inference is controlled by [configs/infer.yaml](configs/infer.yaml) and config groups:

- data config: [configs/data](configs/data)
- infer runtime config (device, batch size, MC-dropout knobs): [configs/infer/runtime/default.yaml](configs/infer/runtime/default.yaml)
- infer metrics config: [configs/infer/metrics/default.yaml](configs/infer/metrics/default.yaml)
- infer save config: [configs/infer/save/default.yaml](configs/infer/save/default.yaml)

There is deliberately **no** model-architecture config group. The checkpoint is
authoritative for what model it holds (see `src/checkpointing/io.py`) -- you only ever
need to point at a checkpoint and a dataset, never separately tell inference which
architecture to build.

Top-level fields in [configs/infer.yaml](configs/infer.yaml):

- fold: train, val, or test
- ckpt_path: checkpoint path (required)
- save_path: output directory root
- tags, seed, task_name

## Basic Usage

Run inference with default infer config and command-line overrides:

```bash
uv run src/inference/infer.py \
  --save-path /absolute/path/to/output \
  ckpt_path=/absolute/path/to/model.ckpt
```

The same output location can also be provided through Hydra config overrides:

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  save_path=/absolute/path/to/output
```

Select a data config from [configs/data](configs/data):

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  data=acevedo
```

Choose split/fold:

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  fold=test
```

## Artifact Inference

Use the artifact data config to trigger paired real/artifact processing:

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  data=artifact_image_classifier
```

Enable image saving for artifact runs:

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  data=artifact_image_classifier \
  infer.save.save_images=true \
  infer.save.max_images_to_save=128
```

## Outputs

By default, outputs are written to:

```
save_path/<run_name>/
```

- `save_path` defaults to `${paths.log_dir}/infer` (override with `save_path=...` or
  `--save-path` to point elsewhere).
- `<run_name>` defaults to an auto-derived `<model>/<ckpt_run_id>/<dataset>`, read straight from
  the checkpoint and the dataset used for inference (`<model>` is the checkpoint's own
  architecture identity, `<ckpt_run_id>` is the training run's timestamp extracted from
  `ckpt_path`) -- set `infer.save.run_name=...` explicitly to override it. Note this is keyed on
  checkpoint + dataset only, not `fold`, so rerunning against a different fold overwrites the
  previous outputs in place unless you set `run_name` explicitly.

See [docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md) for the full picture, including how this fits
alongside train/eval output directories and a note on the one architecture-family whose
auto-derived name looks different from its `model.name`.

Files produced (depending on infer.save flags):

- predictions.csv
- metrics.json
- images/ (artifact mode when image saving is enabled)

## Useful Overrides

Strict checkpoint loading (error instead of warn on missing/unexpected state_dict keys):

```bash
infer.runtime.strict=true
```

Run on CPU:

```bash
infer.runtime.device=cpu
```

MC dropout for compatible models:

```bash
infer.runtime.use_mc_dropout=true infer.runtime.mc_passes=20
```

Override the dataloader batch size:

```bash
infer.runtime.batch_size_override=32
```

## Notes

- One entrypoint is intentional for consistency and easier automation.
- Artifact-specific logic is selected automatically when the dataloader returns paired artifact batches.
