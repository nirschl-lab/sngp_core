# Inference Guide

This project uses a single inference entrypoint for dataset-wide, metrics-producing
runs:

- [src/inference/infer.py](src/inference/infer.py)

The artifact path is handled internally by batch-format detection. You do not need to
run a separate script.

For a single image (or a handful) without going through Hydra/a datamodule, use
[src/inference/predict_image.py](src/inference/predict_image.py) instead --
`predict_image`/`predict_batch` are thin wrappers around the same canonical checkpoint
loader. See [examples/README.md](examples/README.md). Everything below is about
`infer.py`.

## Config Layout

Inference is controlled by [configs/infer.yaml](configs/infer.yaml) and config groups:

- data config: [configs/data](configs/data) -- **required, no default** (`data: ???`
  in `configs/infer.yaml`); omitting `data=...` fails fast with a
  `ConfigCompositionException` listing the available options
- infer runtime config (device, batch size, MC-dropout knobs): [configs/infer/runtime/default.yaml](configs/infer/runtime/default.yaml)
- infer metrics config: [configs/infer/metrics/default.yaml](configs/infer/metrics/default.yaml)
- infer save config: [configs/infer/save/default.yaml](configs/infer/save/default.yaml)

There is deliberately **no** model-architecture config group. The checkpoint is
authoritative for what model it holds (see `src/checkpointing/io.py`) -- you only ever
need to point at a checkpoint and a dataset, never separately tell inference which
architecture to build.

Top-level fields in [configs/infer.yaml](configs/infer.yaml):

- `data`: dataset config group from `configs/data/` (**required, no default** --
  see above)
- `ckpt_path`: checkpoint path (required)
- `fold`: `train`, `validation` (`val` also accepted), `test`, or `all` (train + val +
  test combined, not just the test split); defaults to `test`
- `save_path`: output directory root
- `seed`: if set, seeds `numpy`/`random`/`torch`/Lightning before inference runs
  (`src/utils/random_seed.py::set_random_seed`) -- the only source of non-determinism
  this reaches is MC-Dropout's stochastic forward passes; deterministic checkpoints
  (baseline without MC-Dropout, SNGP, Deep Ensemble) are unaffected either way.
  `null` (default) leaves the run unseeded.
- tags, task_name (cosmetic/logging only)

## Basic Usage

`data=<dataset>` is **required on every invocation** -- there is no default dataset,
so it must be passed explicitly (options: `acevedo`, `wong`, `tang`, `kather2018`,
`kather2016`, `jung`, `nirschl2018`, or `artifact_image_classifier`, from
[configs/data](configs/data)).

The minimal invocation only needs a checkpoint and a dataset -- `save_path` is
optional and defaults to `${paths.log_dir}/infer/<run_name>`, with `<run_name>`
auto-derived from the checkpoint and dataset (see [Outputs](#outputs) below):

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  data=acevedo
```

To point outputs somewhere specific, pass either the `--save-path` CLI flag or the
equivalent `save_path=...` Hydra override -- both do the same optional thing:

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  data=acevedo \
  --save-path /absolute/path/to/output
```

Choose split/fold (`train | validation | test | all`; `all` runs train+val+test
combined, not just the test split; defaults to `test` if omitted):

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  data=acevedo \
  fold=all
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

Override the dataloader batch size:

```bash
infer.runtime.batch_size_override=32
```

## MC-Dropout

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/baseline_model.ckpt \
  data=acevedo \
  infer.runtime.use_mc_dropout=true infer.runtime.mc_passes=20
```

- **Baseline checkpoints only.** MC-Dropout is implemented as `mc_predict()` on
  `BaselineClassifier` ([src/models/baseline/baseline_models.py](src/models/baseline/baseline_models.py))
  and dispatched via `hasattr(model, "mc_predict")`
  ([src/inference/infer.py](src/inference/infer.py)). Setting
  `infer.runtime.use_mc_dropout=true` against an SNGP or Deep Ensemble checkpoint is a
  **silent no-op** -- it falls through to a plain forward pass with no warning. Those
  families already expose predictive uncertainty natively (SNGP: `net(x).variance`;
  Deep Ensemble: member disagreement, see
  [docs/DEEP_ENSEMBLES_GUIDE.md](DEEP_ENSEMBLES_GUIDE.md)), so they don't need
  MC-Dropout.
- `infer.runtime.mc_passes` (default `10`) is the number of stochastic forward passes
  averaged per batch; runtime scales roughly linearly with it.
- The resulting per-sample predictive std is written to predictions.csv's
  `uncertainty` column, same column SNGP/ensemble uncertainty is written to.
- Set `infer.save.save_member_logits=true` to additionally persist the raw per-pass
  logits (`[T, C]` per row, JSON-encoded) needed for epistemic/aleatoric
  decomposition -- off by default since it multiplies row size by roughly `T` (same
  flag, same column shape, for a Deep Ensemble's per-member logits). Set `seed` above
  if you need the run to be reproducible.

## Notes

- One entrypoint is intentional for consistency and easier automation.
- Artifact-specific logic is selected automatically when the dataloader returns paired artifact batches.
- A mismatched `num_classes` between the checkpoint and the dataset's datamodule (or
  targets outside the expected class range) does not raise -- `run_inference` silently
  skips metric computation and logs a warning instead
  (`_check_metric_compatibility`/`_skip_metrics` in
  [src/inference/infer.py](src/inference/infer.py)); `predictions.csv` is still
  written, but `metrics.json` comes back empty. Check the log output (or the
  checkpoint's `read_meta()` output beforehand) rather than assuming an empty
  `metrics.json` means zero test samples.
