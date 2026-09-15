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
└── infer/[deep_ensemble_|mc_]<netname_dataset>/<ckpt_run_id>/<data.name>/
    ├── predictions.csv
    ├── metrics.json
    ├── run.json                   # provenance sidecar, if infer.save.save_run_json=true (default)
    └── images/                    # artifact mode only, if infer.save.save_images=true
```

`train/` and `eval/` Hydra multiruns (`-m`) land at the equivalent
`multiruns/<run_id>/<job.num>/` shape automatically -- `hydra.sweep.dir` interpolates the same
`${task_name}` as `hydra.run.dir`, so no separate configuration was needed for that.
Hyperparameter-search trials (`scripts/hpo/sweep.sh`, W&B sweeps) are ordinary single runs
whose preset overrides `hydra.run.dir` to
`train/<model.name>_<data.name>/sweeps/<experiment name>_hpo/<run_id>_<wandb_run_id>/`
(`configs/hparams_search/<family>.yaml`): the W&B run id suffix keeps concurrent agents
from colliding on a same-second timestamp, and `sweeps/` keeps `runs/` for real training runs.
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
  `task_name: "train/${model.name}_${dataset_label:${data.name},${data.datamodule.institution}}"`.
- **Eval** -- [`configs/eval.yaml`](../configs/eval.yaml):
  `task_name: "eval/${model.name}_${dataset_label:${data.name},${data.datamodule.institution}}"`.
- **Inference** -- [`configs/infer.yaml`](../configs/infer.yaml):
  `task_name: "infer/${dataset_label:${data.name},${data.datamodule.institution}}"`. This field is
  set for a clean printed/logged config only -- `src/inference/infer.py` never runs a real
  `@hydra.main` job (see [§8](#8-known-quirks)), so it does **not** drive `hydra.run.dir` the way it
  does for train/eval, and its dataset-first shape doesn't match the actual on-disk inference layout
  described in [§5](#5-inference-outputs)
  (`[deep_ensemble_|mc_]<netname_dataset>/<ckpt_run_id>/<data.name>[_<institution>]`, model/run
  first).

`dataset_label` (`src/utils/resolvers.py`, a custom OmegaConf resolver registered as an import-time
side effect of `src.utils`) appends `data.datamodule.institution` onto `data.name` whenever an
institution filter is set -- `wong` + `institution=upitt` becomes `wong_upitt` -- so a
per-institution train/eval run gets its own directory automatically instead of collapsing into the
shared `<model.name>_<data.name>/` root every other institution (and the unfiltered, all-institution
run) also writes to; leaving `institution: null` (the default) leaves the path unchanged. Inference's
own output-folder scoping ([§5](#5-inference-outputs)) applies the identical rule directly in Python
(`derive_default_run_name`'s `data_name` argument), not through this cosmetic `task_name` field, since
`src/inference/infer.py` resolves its output directory itself rather than via `hydra.run.dir`. This
replaces the earlier convention of manually passing `data.name=<dataset>_<institution>` on the CLI to
avoid the collision (see the Wong-UCDavis entries in
[docs/MASTER_CHECKPONT_PATHS.md](MASTER_CHECKPONT_PATHS.md) /
[docs/MASTER_INFER_RESULTS_PATH.md](MASTER_INFER_RESULTS_PATH.md), which predate this fix and used
that manual override).

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
[deep_ensemble_|mc_]<netname_dataset>/<ckpt_run_id>/<data.name>
```

- `<netname_dataset>` is parsed straight out of `ckpt_path` by `_extract_netname_dataset` --
  this project's own `<model.name>_<data.name>` training-output segment
  (`train/<model.name>_<data.name>/{runs,ensemble_members}/<run_id>/...`, see [§3](#3-training-outputs)
  and [§6](#6-ensemble-member-parallel-training-outputs)), e.g. `baseline_classifier_acevedo` or
  `sngp_classifier_wong` -- regex-extracted, not reconstructed from checkpoint metadata, so it
  already carries both architecture and training dataset in one string. Falls back to the
  checkpoint's own `net_spec["name"]` (with a `logger.warning`) when `ckpt_path` doesn't match that
  convention -- e.g. a `wandb-artifact://` download (resolves to `<artifact_dir>/model.ckpt`, no
  `runs`/`ensemble_members` segment) or a manually renamed/moved checkpoint.
- A `deep_ensemble_` prefix is added when `read_meta(ckpt_path).net_spec["name"] == "deep_ensemble"`
  -- the checkpoint's own `NET_REGISTRY` key (see [§7](#7-three-name-fields----dont-conflate) for why
  this can differ from `model.name`) -- *unless* `<netname_dataset>` already starts with
  `deep_ensemble` (true only for a checkpoint from `DeepEnsembleLitModule`'s `sequential`
  single-run training strategy, whose own `model.name` is `deep_ensemble_classifier`), which would
  otherwise double up into `deep_ensemble_deep_ensemble_classifier_...`. For an assembled Deep
  Ensemble (see [§6](#6-ensemble-member-parallel-training-outputs)), `<netname_dataset>` is read
  straight from the underlying member's own training path, so a baseline-member ensemble
  (`deep_ensemble_baseline_classifier_acevedo`) and an sngp-member ensemble
  (`deep_ensemble_sngp_classifier_acevedo`) never collide -- no separate metadata lookup needed for
  that disambiguation.
- Otherwise, an `mc_` prefix is added when MC-Dropout is enabled for the inference run
  (`infer.runtime.use_mc_dropout`), so MC-Dropout runs don't overwrite a plain inference run's
  `predictions.csv`/`metrics.json` in the same folder. Deep Ensemble beats MC-Dropout when both are
  set -- `use_mc_dropout` is a no-op on ensemble checkpoints anyway (`mc_predict` is only defined on
  `BaselineClassifier`), so the folder name reflects checkpoint identity, not an inert toggle.
- `<ckpt_run_id>` is this project's own run-id timestamp
  (`\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}`), regex-extracted from `ckpt_path` -- since a checkpoint's
  path already contains its training run's `<run_id>`
  (`.../train/<model.name>_<data.name>/runs/<run_id>/checkpoints/last.ckpt`), the inference output
  folder name lets you trace results back to the exact training run that produced the checkpoint,
  with no separate bookkeeping. Falls back to the current timestamp if the checkpoint predates
  this convention or was renamed/moved.
- `<data.name>` is the dataset inference was run against -- the same value that scoped `save_path`
  before this layout changed. Not to be confused with the *training* dataset baked into
  `<netname_dataset>` above: this is the eval-time dataset, which may differ (e.g. a cross-dataset
  OOD sweep). When `data.datamodule.institution` is set for this inference run, it's appended here
  too (`dataset_label(cfg.data.name, cfg.data.datamodule.institution)`, the same rule §2 describes
  for `task_name`, applied directly in Python since `run_inference()` never goes through
  `hydra.run.dir`) -- e.g. testing a Wong-trained checkpoint against just the UPitt institution
  subset lands at `.../wong_upitt/` rather than colliding with a full-`wong` or other-institution
  run in a shared `.../wong/` folder. `provenance["dataset_name"]` in `run.json` records this same
  institution-scoped value, not the bare `data.name`.

Note: the folder is keyed on checkpoint + dataset only, not `fold` -- rerunning the same checkpoint
against the same dataset with a different `fold` (or a second time with the same fold) overwrites
the previous `predictions.csv`/`metrics.json` in place. Set `infer.save.run_name` explicitly (a
manual override that bypasses the auto-derived path entirely) if you need distinct folders per
fold or per rerun.

Each run also writes a `run.json` provenance sidecar (`src/inference/records.py::write_outputs`,
gated by `infer.save.save_run_json`, default on) -- the checkpoint path, its `net_spec`, `fold`,
resolved `infer.{runtime,metrics,save}`, `member_source`/`num_members` (`"ensemble"`/`"mc_dropout"`/
`null`, and how many), the exact `predictions.csv` column list, and a row count. It is the only
place the checkpoint-to-run binding is recorded on disk -- see [§8](#8-known-quirks) for why
nothing else records it -- and is written last, from `_finalize`, so a run that crashed partway
leaves no `run.json` behind either.

`run.json["predictions_csv_schema"]` is `3` for runs written by current code: schema 2 added
`class_logits`/`raw_logits`, schema 3 the uncertainty columns (`predictive_entropy`,
`confidence_margin`, `dempster_shafer`, `uncertainty_kind`, and the
`total_entropy`/`aleatoric_entropy`/`mutual_information` decomposition). Check it before
assuming a column exists -- neither set can be derived from an older CSV, so an older run needs
re-inference. `metrics.json` from the same code additionally carries `<name>_std`/`<name>_sem`
for `acc`/`nll`/`brier` and an `n_samples` count (`<stream>.`-prefixed in artifact mode, so
each stream reports its own).

### Artifact-mode axis sweeps

A count/severity sweep (`data.datamodule.artifact_count`/`artifact_severity`, see
[docs/DATASETS.md](DATASETS.md#artifact-robustness-evaluation)) sets `infer.save.run_name`
explicitly rather than relying on the auto-derived path above, since every variant shares one
`<netname_dataset>/<ckpt_run_id>` and would otherwise collide in the same
`acevedo_artifact/` folder. Convention:

```
<netname_dataset>/<ckpt_run_id>/acevedo_artifact/
  real_baseline/predictions.csv          # infer.save.streams=[real] -- once per checkpoint
  config/count_<N>/predictions.csv       # infer.save.streams=[artifact], artifact axis only
  procedural/severity_<N>/predictions.csv  # infer.save.streams=[artifact], procedural axis only
```

`real_baseline/` holds the real/clean stream, run once (it's bit-identical for every
count/severity variant against the same checkpoint, so re-running and resaving it per sweep
value would be pure duplication -- see DATASETS.md's "Saving sweep results"). Every other
subfolder holds only the artifact stream for its one config; `quantify_artifact_impact`'s
`real_csv_map` param joins a sweep CSV against `real_baseline/predictions.csv` when both
streams are needed together.

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
underlying architecture, though -- and unlike a sequential `DeepEnsembleLitModule` run, that
architecture is never derived from checkpoint metadata for path-naming purposes here: the assembled
ckpt's own path (`train/<member's model.name>_<data.name>/ensemble_members/<run_id>/checkpoints/ensemble.ckpt`)
already contains `<member's model.name>_<data.name>` before `/ensemble_members/`, which
`src/inference/infer.py`'s auto-derived output layout ([§5](#5-inference-outputs)) reads directly via
`_extract_netname_dataset`, then prepends `deep_ensemble_` (from `net_spec["name"] == "deep_ensemble"`):
inference against this checkpoint resolves to `infer/deep_ensemble_baseline_classifier_<dataset>/<run_id>/<dataset>/`
(or `..._sngp_classifier_<dataset>/...`), not a bare `infer/deep_ensemble/<run_id>/<dataset>/` -- so a
baseline-member ensemble and an sngp-member ensemble never collide onto the same inference output
folder, even if their `<run_id>`s happened to coincide.

## 7. Three "name" fields -- don't conflate

| Field | Where | Example | Drives |
|---|---|---|---|
| `data.name` | `configs/data/*.yaml` top level | `acevedo` | `task_name` (train/eval/infer), infer's auto-derived run-folder path (leaf segment) |
| `model.name` | `configs/model/*.yaml` top level | `sngp_classifier` | `task_name` (train/eval); indirectly drives infer's auto-derived run-folder path too, since `<netname_dataset>` ([§5](#5-inference-outputs)) is parsed straight out of the `<model.name>_<data.name>` training-output path segment |
| `net_spec["name"]` | `NET_REGISTRY` key, stamped into every checkpoint (`src/models/registry.py`) | `sngp_classifier`, but **`deep_ensemble`** (not `deep_ensemble_classifier`) | infer's auto-derived run-folder path -- only the `deep_ensemble_` prefix decision and the `<netname_dataset>` fallback value (when `ckpt_path` doesn't match this project's own convention), not the primary segment -- see [§5](#5-inference-outputs) |
| `name` | `configs/experiment/*.yaml` top level (`_global_`) | `acevedo_sngp_resnet18` | `logger.wandb.name`/`group` only -- never on disk |

The `net_spec["name"]` vs `model.name` mismatch for deep ensembles is real and permanent: the
registry key (`"deep_ensemble"`) is baked into every already-saved deep-ensemble checkpoint's
`net_spec`, and renaming it would break `build_net()` for those checkpoints. So a *sequential*
deep-ensemble training run lands at `train/deep_ensemble_classifier_<dataset>/...`, and that same
checkpoint's auto-derived inference folder reads `infer/deep_ensemble_classifier_<dataset>/<ckpt_run_id>/<dataset>/`
(no doubled `deep_ensemble_` prefix, since `<netname_dataset>` already starts with "deep_ensemble" --
see [§5](#5-inference-outputs)) -- the "deep_ensemble" vs "deep_ensemble_classifier" difference there
is expected, not a bug. An *assembled* deep-ensemble checkpoint (see
[§6](#6-ensemble-member-parallel-training-outputs)) has no such collision to begin with, since its
path never contains "deep_ensemble" at all -- `<netname_dataset>` is read from the underlying
member's own training path (e.g. `baseline_classifier_<dataset>`), and the `deep_ensemble_` prefix
is added on top from `net_spec["name"]` alone.

## 8. Known quirks

- **`infer.yaml`'s `task_name` is inert for `hydra.run.dir` purposes.** `src/inference/infer.py`
  composes its config via `hydra.compose(..., return_hydra_config=False)` inside
  `initialize_config_dir` -- it never runs a real `@hydra.main` job, so nothing creates a
  `${paths.log_dir}/infer/.../runs/<run_id>/` bookkeeping directory the way train/eval get one.
  `task_name` is kept for a clean printed/logged config, but the actual inference *results* tree
  is entirely determined by `save_path`/`infer.save.run_name` as described in [§5](#5-inference-outputs).
- **The deep-ensemble name asymmetry** described in [§7](#7-three-name-fields----dont-conflate) --
  the `net_spec["name"]`/`model.name` mismatch itself is permanent, but the inference-path
  collision it used to cause between differently-membered ensembles doesn't arise for *assembled*
  ensembles, since `<netname_dataset>` is read from the member's own training path rather than
  reconstructed from checkpoint metadata; it's avoided for *sequential* ensembles by skipping the
  `deep_ensemble_` prefix when `<netname_dataset>` already starts with "deep_ensemble" (see
  [§5](#5-inference-outputs)).
- **`ensemble_members/<run_id>/` is a sibling of `runs/<run_id>/`, but each member's own run dir
  is built by an explicit `hydra.run.dir=` override, not `task_name`** -- described in
  [§6](#6-ensemble-member-parallel-training-outputs); deliberate, not an oversight.
- **HPO trial dirs sit under `train/<model.name>_<data.name>/sweeps/<name>_hpo/`.**
  `configs/hparams_search/*.yaml` override `hydra.run.dir` to
  `${paths.log_dir}/${task_name}/sweeps/${name}_hpo/${now:...}_${oc.env:WANDB_RUN_ID,local}`
  -- the same `${task_name}` prefix as `runs/`, a `sweeps/` segment so proxy-budget trial
  checkpoints are never mistaken for real training runs, and the W&B run id (exported by
  `wandb agent`; `local` for a by-hand run) so several agents starting in the same second
  cannot collide.
