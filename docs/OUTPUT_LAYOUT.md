# Output Directory Layout

How training/eval/inference outputs are laid out on disk under `$EXPERIMENTS_HOME`, how
`task_name` drives that layout, and the three unrelated config fields that all happen to be
called "name."

## Table of Contents

1. [Directory layout](#1-directory-layout)
2. [How `task_name` and run-id are generated](#2-how-task_name-and-run-id-are-generated)
3. [Training outputs](#3-training-outputs)
4. [Eval outputs](#4-eval-outputs)
5. [Inference outputs](#5-inference-outputs)
6. [Three "name" fields -- don't conflate](#6-three-name-fields----dont-conflate)
7. [Known quirks](#7-known-quirks)

## 1. Directory layout

Everything lives under `${EXPERIMENTS_HOME}/${PROJECT_NAME}/` (both from `.env`; see
[`configs/paths/default.yaml`](../configs/paths/default.yaml)). Train, eval, and infer each get
their own top-level subtree, and each run's directory name now encodes which model family and
dataset it belongs to, not just a timestamp:

```
$EXPERIMENTS_HOME/$PROJECT_NAME/
├── train/<model.name>_<data.name>/runs/<run_id>/
│   ├── run.log
│   ├── .hydra/                    # resolved config snapshot
│   ├── wandb/                     # local wandb run files (if logger=wandb)
│   ├── csv/                       # test_artifacts prediction CSV, if log_csv=true
│   └── checkpoints/
│       ├── best.ckpt
│       └── last.ckpt
├── eval/<model.name>_<data.name>/runs/<run_id>/
│   ├── run.log
│   ├── .hydra/
│   └── csv/
└── infer/<data.name>/<run_name>/
    ├── predictions.csv
    ├── metrics.json
    └── images/                    # artifact mode only, if infer.save.save_images=true
```

`train/` and `eval/` multirun/HPO sweeps (`scripts/hpo/sweep.sh`) land at the equivalent
`multiruns/<run_id>/<job.num>/` shape automatically -- `hydra.sweep.dir` interpolates the same
`${task_name}` as `hydra.run.dir`, so no separate configuration was needed for that.

## 2. How `task_name` and run-id are generated

`run_id` is a plain wall-clock timestamp, not a wandb run id or Hydra job number.
[`configs/hydra/default.yaml`](../configs/hydra/default.yaml):

```yaml
run:
  dir: ${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}
sweep:
  dir: ${paths.log_dir}/${task_name}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}
  subdir: ${hydra.job.num}
```

so `<run_id>` is `%Y-%m-%d_%H-%M-%S` at process start, and `<task_name>` determines everything
above `runs/`:

- **Training** -- [`configs/train.yaml`](../configs/train.yaml):
  `task_name: "train/${model.name}_${data.name}"`.
- **Eval** -- [`configs/eval.yaml`](../configs/eval.yaml):
  `task_name: "eval/${model.name}_${data.name}"`.
- **Inference** -- [`configs/infer.yaml`](../configs/infer.yaml):
  `task_name: "infer/${data.name}"`. This field is set for a clean printed/logged config only --
  `src/inference/infer.py` never runs a real `@hydra.main` job (see
  [§7](#7-known-quirks)), so it does **not** drive `hydra.run.dir` the way it does for train/eval.

`${model.name}` and `${data.name}` are plain config fields resolved at Hydra composition time --
see [§6](#6-three-name-fields----dont-conflate) for exactly which fields these are and which ones
they are *not*.

Because `task_name` now contains `/`, the job log file itself (`run.log`) is a flat, fixed name
rather than `${task_name}.log` -- [`configs/hydra/default.yaml`](../configs/hydra/default.yaml)'s
`job_logging.handlers.file.filename` doesn't create intermediate directories, so a slash in that
value raises `FileNotFoundError` at job-logging setup, before the task function ever runs. The run
directory's own path already fully encodes the task identity, so nothing is lost by the log
filename being generic.

## 3. Training outputs

Checkpoints, the resolved-config snapshot, wandb's local run files, and (if
`callbacks.test_artifacts.log_csv=true`) the prediction CSV all live together under one
`train/<model.name>_<data.name>/runs/<run_id>/` directory, via `${paths.output_dir}`
(`= ${hydra:runtime.output_dir}`, i.e. `hydra.run.dir`):

- [`configs/callbacks/default.yaml`](../configs/callbacks/default.yaml):
  `model_checkpoint.dirpath: ${paths.output_dir}/checkpoints`.
- [`configs/callbacks/test_artifacts.yaml`](../configs/callbacks/test_artifacts.yaml):
  `csv_save_path: ${paths.output_dir}/csv`.
- [`configs/logger/wandb.yaml`](../configs/logger/wandb.yaml): `save_dir: ${paths.output_dir}`.

## 4. Eval outputs

Same shape as training, under `eval/` instead of `train/` -- `src/eval.py` is Hydra-composed the
same way `src/train.py` is, so it gets a real `hydra.run.dir` and every `${paths.output_dir}`
consumer above applies identically.

## 5. Inference outputs

[`src/inference/infer.py`](../src/inference/infer.py) resolves the output directory in two
layers. `configs/infer.yaml` sets a dataset-scoped base:

```yaml
save_path: ${paths.log_dir}/infer/${data.name}
```

Then, unless the caller sets `infer.save.run_name` explicitly, `run_inference()` derives a default
run-folder name from the checkpoint itself
(`src/inference/infer.py::derive_default_run_name`):

```
<net_spec name>_<ckpt_run_id>_<fold>
```

- `<net_spec name>` comes from `read_meta(ckpt_path).net_spec["name"]` -- the checkpoint's own
  `NET_REGISTRY` key (see [§6](#6-three-name-fields----dont-conflate) for why this can differ from
  `model.name`).
- `<ckpt_run_id>` is this project's own run-id timestamp
  (`\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}`), regex-extracted from `ckpt_path` -- since a checkpoint's
  path already contains its training run's `<run_id>`
  (`.../train/<model.name>_<data.name>/runs/<run_id>/checkpoints/last.ckpt`), the inference output
  folder name lets you trace results back to the exact training run that produced the checkpoint,
  with no separate bookkeeping. Falls back to the current timestamp if the checkpoint predates
  this convention or was renamed/moved.
- `<fold>` is the normalized `cfg.fold` (`train`/`validation`/`test`/`all`).

`infer.save.run_name` remains a manual override -- set it explicitly to bypass the auto-derived
name entirely.

## 6. Three "name" fields -- don't conflate

| Field | Where | Example | Drives |
|---|---|---|---|
| `data.name` | `configs/data/*.yaml` top level | `acevedo` | `task_name` (train/eval/infer), infer's `save_path` |
| `model.name` | `configs/model/*.yaml` top level | `sngp_classifier` | `task_name` (train/eval) |
| `net_spec["name"]` | `NET_REGISTRY` key, stamped into every checkpoint (`src/models/registry.py`) | `sngp_classifier`, but **`deep_ensemble`** (not `deep_ensemble_classifier`) | infer's auto-derived run-folder name |
| `name` | `configs/experiment/*.yaml` top level (`_global_`) | `acevedo_sngp_resnet18` | `logger.wandb.name`/`group` only -- never on disk |

The `net_spec["name"]` vs `model.name` mismatch for deep ensembles is real and permanent: the
registry key (`"deep_ensemble"`) is baked into every already-saved deep-ensemble checkpoint's
`net_spec`, and renaming it would break `build_net()` for those checkpoints. So a deep-ensemble
training run lands at `train/deep_ensemble_classifier_<dataset>/...`, but that same checkpoint's
auto-derived inference folder reads `infer/<dataset>/deep_ensemble_<ckpt_run_id>_<fold>/` -- the
"deep_ensemble" vs "deep_ensemble_classifier" difference there is expected, not a bug.

## 7. Known quirks

- **`infer.yaml`'s `task_name` is inert for `hydra.run.dir` purposes.** `src/inference/infer.py`
  composes its config via `hydra.compose(..., return_hydra_config=False)` inside
  `initialize_config_dir` -- it never runs a real `@hydra.main` job, so nothing creates a
  `${paths.log_dir}/infer/.../runs/<run_id>/` bookkeeping directory the way train/eval get one.
  `task_name` is kept for a clean printed/logged config, but the actual inference *results* tree
  is entirely determined by `save_path`/`infer.save.run_name` as described in [§5](#5-inference-outputs).
- **The deep-ensemble name asymmetry** described in [§6](#6-three-name-fields----dont-conflate).
- **HPO sweep dirs inherit the new layout for free.** `configs/hparams_search/*.yaml` only
  reference the unrelated top-level `name` field for their Optuna storage path
  (`${paths.log_dir}/optuna/${name}_hpo.db`); the sweep *output* directory
  (`hydra.sweep.dir`) picks up `train/<model.name>_<data.name>/multiruns/...` automatically since
  it interpolates the same `${task_name}` as `hydra.run.dir`.
