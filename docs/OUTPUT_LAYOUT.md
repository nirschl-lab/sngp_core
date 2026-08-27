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
6. [Ensemble-member parallel training outputs](#6-ensemble-member-parallel-training-outputs)
7. [Three "name" fields -- don't conflate](#7-three-name-fields----dont-conflate)
8. [Known quirks](#8-known-quirks)

## 1. Directory layout

Everything lives under `${EXPERIMENTS_HOME}/${PROJECT_NAME}/` (both from `.env`; see
[`configs/paths/default.yaml`](../configs/paths/default.yaml)). Train, eval, and infer each get
their own top-level subtree, and each run's directory name now encodes which model family and
dataset it belongs to, not just a timestamp:

```
$EXPERIMENTS_HOME/$PROJECT_NAME/
├── train/<model.name>_<data.name>/
│   ├── runs/<run_id>/                       # an ordinary single-process training run
│   │   ├── run.log
│   │   ├── .hydra/                          # resolved config snapshot
│   │   ├── wandb/                           # local wandb run files (if logger=wandb)
│   │   ├── csv/                             # test_artifacts prediction CSV, if log_csv=true
│   │   └── checkpoints/
│   │       ├── best.ckpt
│   │       └── last.ckpt
│   └── ensemble_members/<run_id>/           # a parallel batch of N members, see §6
│       ├── member_0/
│       │   ├── run.log
│       │   ├── .hydra/
│       │   └── checkpoints/best.ckpt
│       ├── member_1/
│       │   └── ...
│       └── member_1.log, member_2.log, ...  # each member's captured stdout/stderr
├── eval/<model.name>_<data.name>/runs/<run_id>/
│   ├── run.log
│   ├── .hydra/
│   └── csv/
└── infer/<net_spec name>/<train dataset>/<ckpt_run_id>/<data.name>/
    ├── predictions.csv
    ├── metrics.json
    └── images/                    # artifact mode only, if infer.save.save_images=true
```

`train/` and `eval/` multirun/HPO sweeps (`scripts/hpo/sweep.sh`) land at the equivalent
`multiruns/<run_id>/<job.num>/` shape automatically -- `hydra.sweep.dir` interpolates the same
`${task_name}` as `hydra.run.dir`, so no separate configuration was needed for that.
`train/<model.name>_<data.name>/ensemble_members/` is a sibling of `runs/` under the same
model+dataset root, but is constructed directly by a bash script rather than driven by
`task_name` -- see [§6](#6-ensemble-member-parallel-training-outputs).

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
  [§8](#8-known-quirks)), so it does **not** drive `hydra.run.dir` the way it does for train/eval,
  and its `${data.name}`-first shape doesn't match the actual on-disk inference layout described in
  [§5](#5-inference-outputs) (`<net_spec name>/<train dataset>/<ckpt_run_id>/<data.name>`, model/run
  first).

`${model.name}` and `${data.name}` are plain config fields resolved at Hydra composition time --
see [§7](#7-three-name-fields----dont-conflate) for exactly which fields these are and which ones
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
layers. `configs/infer.yaml` sets a dataset-agnostic base:

```yaml
save_path: ${paths.log_dir}/infer
```

Then, unless the caller sets `infer.save.run_name` explicitly, `run_inference()` derives a default
run-folder path from the checkpoint and the dataset used for inference
(`src/inference/infer.py::derive_default_run_name`):

```
<net_spec name>/<train dataset>/<ckpt_run_id>/<data.name>
```

- `<net_spec name>` comes from `read_meta(ckpt_path).net_spec["name"]` -- the checkpoint's own
  `NET_REGISTRY` key (see [§7](#7-three-name-fields----dont-conflate) for why this can differ from
  `model.name`). When the checkpoint's `net_spec` carries a nested `base_model_spec` dict (true
  today only for Deep Ensemble checkpoints -- see [§6](#6-ensemble-member-parallel-training-outputs)),
  `base_model_spec["name"]` (e.g. `baseline_classifier` or `sngp_classifier`) is appended as a
  suffix: `deep_ensemble_baseline_classifier`, `deep_ensemble_sngp_classifier`, etc. Without this,
  every Deep Ensemble checkpoint would produce the identical `deep_ensemble` segment regardless of
  member architecture, relying entirely on `<ckpt_run_id>` below to keep a baseline-member
  ensemble's results apart from an sngp-member ensemble's -- fragile, since that value isn't
  derived from checkpoint content (see the fallback caveat immediately below).
- `<train dataset>` is `read_meta(ckpt_path).dataset_name` -- the dataset the checkpoint was
  *trained* on, stamped in at training time (`LitModuleBase.on_save_checkpoint`) from the training
  datamodule's own `dataset_name` attribute, and validated to agree across all members when a Deep
  Ensemble checkpoint is assembled (`scripts/ensemble/assemble_ensemble_checkpoint.py`). That
  attribute is the full HF repo id (e.g. `nirschl-lab/wong_et_al_2022`), not path-safe as-is (see
  the `name:` vs `datamodule.dataset_name:` comment in any `configs/data/*.yaml`), so
  `derive_default_run_name` strips the org prefix before using it as a directory segment. This is
  what lets two checkpoints of the same architecture trained on different datasets land in separate
  folders instead of only being distinguishable by an opaque `<ckpt_run_id>` timestamp. Omitted
  entirely -- not replaced with an "unknown" placeholder -- for checkpoints that predate this
  field or were saved without a datamodule attached.
- `<ckpt_run_id>` is this project's own run-id timestamp
  (`\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}`), regex-extracted from `ckpt_path` -- since a checkpoint's
  path already contains its training run's `<run_id>`
  (`.../train/<model.name>_<data.name>/runs/<run_id>/checkpoints/last.ckpt`), the inference output
  folder name lets you trace results back to the exact training run that produced the checkpoint,
  with no separate bookkeeping. Falls back to the current timestamp if the checkpoint predates
  this convention or was renamed/moved.
- `<data.name>` is the dataset inference was run against -- the same value that scoped `save_path`
  before this layout changed. Not to be confused with `<train dataset>` above: this is the eval-time
  dataset, which may differ from the training dataset (e.g. a cross-dataset OOD sweep).

Note: the folder is keyed on checkpoint + dataset only, not `fold` -- rerunning the same checkpoint
against the same dataset with a different `fold` (or a second time with the same fold) overwrites
the previous `predictions.csv`/`metrics.json` in place. Set `infer.save.run_name` explicitly (a
manual override that bypasses the auto-derived path entirely) if you need distinct folders per
fold or per rerun.

## 6. Ensemble-member parallel training outputs

[`scripts/ensemble/train_members_parallel.sh`](../scripts/ensemble/train_members_parallel.sh)
trains N Deep Ensemble members as independent, fully parallel `baseline_classifier` runs (one
ordinary `train.py` invocation per GPU) instead of `DeepEnsembleLitModule`'s sequential
single-run cycling, then
[`scripts/ensemble/assemble_ensemble_checkpoint.py`](../scripts/ensemble/assemble_ensemble_checkpoint.py)
combines the resulting checkpoints into one `DeepEnsemble` checkpoint. Each member is trained by
its own `train.py` invocation with `hydra.run.dir=<explicit path>` passed directly, bypassing
`task_name`/`hydra.run.dir`'s own templating for that one run -- but the *root* those explicit
paths are built under still lands at `train/<model.name>_<data.name>/ensemble_members/<run_id>/`,
the same `train/<model.name>_<data.name>/` prefix an ordinary `runs/<run_id>/` training run for
that model+dataset uses (see [§1](#1-directory-layout)). All N members from one invocation land
together under one shared `<run_id>` instead of scattering across N separate
`runs/<own-timestamp>/` directories -- co-location is what lets
`assemble_ensemble_checkpoint.py --members-dir <root>` auto-discover `member_0/`, `member_1/`, ...
without being told each one's path individually.

`<model.name>_<data.name>` is resolved by composing the experiment config once at the top of the
script (not reused from `$EXPERIMENT`, the experiment config's filename, since that can differ
from the model/dataset it actually points at). Each `member_<i>/` subdirectory is an ordinary
Hydra run dir (same `checkpoints/`, `.hydra/`, `run.log` shape as [§3](#3-training-outputs)), plus
a `member_<i>.log` one level up capturing that member's full stdout/stderr (the script's own
process-supervision log, separate from Hydra's `run.log` inside each member's directory).

By default (no `--out` given), the assembled `ensemble.ckpt` lands inside the `ensemble_members/`
run it was built from, at `ensemble_members/<run_id>/checkpoints/ensemble.ckpt` -- a sibling of
`member_0/`, `member_1/`, ... under the same `<run_id>`, matching the `checkpoints/` subdir
convention `runs/<run_id>/` and each `member_<i>/` already use
(`derive_default_out_path` in `assemble_ensemble_checkpoint.py`). Pass `--out <path>` explicitly
to override -- `assemble()` warns (but still proceeds) if that path already has a checkpoint at
it, so re-assembling in place isn't silent. Either way, its `net_spec["name"]` is `"deep_ensemble"`
(assembled via `DeepEnsemble.spec`, the same registry key a real `DeepEnsembleLitModule` run
produces), even though every member checkpoint underneath it is individually a
`"baseline_classifier"` (or `"sngp_classifier"`) -- consistent with the
[§7](#7-three-name-fields----dont-conflate) asymmetry, just constructed by hand here instead of by
`DeepEnsembleLitModule`. The checkpoint's `net_spec["base_model_spec"]["name"]` retains that
underlying architecture, though, and `src/inference/infer.py`'s auto-derived output layout
([§5](#5-inference-outputs)) reads it: inference against this checkpoint resolves to
`infer/deep_ensemble_baseline_classifier/<train dataset>/<run_id>/<dataset>/` (or
`..._sngp_classifier/...`), not a bare `infer/deep_ensemble/<run_id>/<dataset>/` -- so a
baseline-member ensemble and an sngp-member ensemble never collide onto the same inference output
folder, even if their `<run_id>`s happened to coincide.

## 7. Three "name" fields -- don't conflate

| Field | Where | Example | Drives |
|---|---|---|---|
| `data.name` | `configs/data/*.yaml` top level | `acevedo` | `task_name` (train/eval/infer), infer's auto-derived run-folder path (leaf segment) |
| `model.name` | `configs/model/*.yaml` top level | `sngp_classifier` | `task_name` (train/eval) |
| `net_spec["name"]` | `NET_REGISTRY` key, stamped into every checkpoint (`src/models/registry.py`) | `sngp_classifier`, but **`deep_ensemble`** (not `deep_ensemble_classifier`) | infer's auto-derived run-folder path (top segment, plus a `base_model_spec["name"]` suffix when present -- see [§5](#5-inference-outputs)) |
| `name` | `configs/experiment/*.yaml` top level (`_global_`) | `acevedo_sngp_resnet18` | `logger.wandb.name`/`group` only -- never on disk |

The `net_spec["name"]` vs `model.name` mismatch for deep ensembles is real and permanent: the
registry key (`"deep_ensemble"`) is baked into every already-saved deep-ensemble checkpoint's
`net_spec`, and renaming it would break `build_net()` for those checkpoints. So a deep-ensemble
training run lands at `train/deep_ensemble_classifier_<dataset>/...`, but that same checkpoint's
auto-derived inference folder reads `infer/deep_ensemble_<base-architecture>/<ckpt_run_id>/<dataset>/`
-- the "deep_ensemble" vs "deep_ensemble_classifier" difference there is expected, not a bug. What
used to be a bug -- `net_spec["name"]` alone being identical for every Deep Ensemble checkpoint,
regardless of member architecture -- is mitigated by the `base_model_spec["name"]` suffix
described in [§5](#5-inference-outputs).

## 8. Known quirks

- **`infer.yaml`'s `task_name` is inert for `hydra.run.dir` purposes.** `src/inference/infer.py`
  composes its config via `hydra.compose(..., return_hydra_config=False)` inside
  `initialize_config_dir` -- it never runs a real `@hydra.main` job, so nothing creates a
  `${paths.log_dir}/infer/.../runs/<run_id>/` bookkeeping directory the way train/eval get one.
  `task_name` is kept for a clean printed/logged config, but the actual inference *results* tree
  is entirely determined by `save_path`/`infer.save.run_name` as described in [§5](#5-inference-outputs).
- **The deep-ensemble name asymmetry** described in [§7](#7-three-name-fields----dont-conflate) --
  the `net_spec["name"]`/`model.name` mismatch itself is permanent, but the inference-path
  collision it used to cause between differently-membered ensembles is mitigated by the
  `base_model_spec["name"]` suffix.
- **`ensemble_members/<run_id>/` is a sibling of `runs/<run_id>/`, but each member's own run dir
  is built by an explicit `hydra.run.dir=` override, not `task_name`** -- described in
  [§6](#6-ensemble-member-parallel-training-outputs); deliberate, not an oversight.
- **HPO sweep dirs inherit the new layout for free.** `configs/hparams_search/*.yaml` only
  reference the unrelated top-level `name` field for their Optuna storage path
  (`${paths.log_dir}/optuna/${name}_hpo.db`); the sweep *output* directory
  (`hydra.sweep.dir`) picks up `train/<model.name>_<data.name>/multiruns/...` automatically since
  it interpolates the same `${task_name}` as `hydra.run.dir`.
