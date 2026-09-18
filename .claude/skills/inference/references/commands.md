# Recorded inference commands

Migrated from the pre-migration scratch files `artifact_testing.md` (repo root) and
`docs/artifact_code.md`, updated to the post-Stage-3 config keys (`infer.runtime.*`,
no `infer/model=...` group anymore -- the checkpoint is authoritative for
architecture, see `src/checkpointing/io.py`). Kept as a record of real invocations
this project has used, not as documentation to copy verbatim -- always confirm
`ckpt_path` and `save_path` against the current run before reusing one of these.

## Datamodule sanity check

```bash
uv run src/data/artifact_image_datamodule.py -n 100
```

## Artifact inference (paired real/artifact), per in-distribution dataset

Acevedo, baseline:
```bash
uv run src/inference/infer.py ckpt_path=./checkpoints/acevedo_baseline_resnet18/model.ckpt \
  data=artifact_image_classifier save_path=<output_dir>/acevedo_baseline/ \
  infer.save.save_images=true
```

Acevedo, baseline + MC-Dropout:
```bash
uv run src/inference/infer.py ckpt_path=./checkpoints/acevedo_baseline_resnet18/model.ckpt \
  data=artifact_image_classifier save_path=<output_dir>/acevedo_mc/ \
  infer.save.save_images=true infer.runtime.use_mc_dropout=true infer.runtime.mc_passes=20
```

Acevedo, SNGP:
```bash
uv run src/inference/infer.py ckpt_path=./checkpoints/acevedo_sngp_resnet18/model.ckpt \
  data=artifact_image_classifier save_path=<output_dir>/acevedo_sngp/ \
  infer.save.save_images=true
```

## Artifact/procedural axis sweep (count, severity), without resaving the real stream

Two independent axes as of the `d0b2ae7` package rev -- see
[docs/DATASETS.md](../../../docs/DATASETS.md#artifact-robustness-evaluation). The real/clean
stream is bit-identical across every count/severity variant against one checkpoint, so it's run
once (`infer.save.streams=[real]`) and every swept variant writes only the artifact stream
(`infer.save.streams=[artifact]`) -- narrowing `infer.save.streams` also skips the excluded
stream's forward pass entirely, not just its CSV row. Same convention for `run_name` shape,
Acevedo baseline shown:

```bash
# Once per checkpoint: shared real-stream baseline (clean arm -- both axes off)
uv run src/inference/infer.py ckpt_path=<ckpt> data=artifact_image_classifier \
  data.datamodule.artifact_config_path=none data.datamodule.artifact_procedural_config=none \
  infer.save.streams=[real] \
  infer.save.run_name=baseline_classifier_acevedo/<ckpt_run_id>/acevedo_artifact/real_baseline

# Artifact axis only: count=1 overlay, procedural axis off
uv run src/inference/infer.py ckpt_path=<ckpt> data=artifact_image_classifier \
  data.datamodule.artifact_config_path=artifact_balanced data.datamodule.artifact_procedural_config=none \
  data.datamodule.artifact_count=1 infer.save.streams=[artifact] \
  infer.save.run_name=baseline_classifier_acevedo/<ckpt_run_id>/acevedo_artifact/config/count_1

# Procedural axis only: severity=1 grading, artifact axis off
uv run src/inference/infer.py ckpt_path=<ckpt> data=artifact_image_classifier \
  data.datamodule.artifact_config_path=none data.datamodule.artifact_procedural_config=procedural_ood \
  data.datamodule.artifact_severity=1 infer.save.streams=[artifact] \
  infer.save.run_name=baseline_classifier_acevedo/<ckpt_run_id>/acevedo_artifact/procedural/severity_1
```

## Whole evaluation suite for one checkpoint, in parallel over GPU lanes

`scripts/inference/run_eval_suite_parallel.sh` runs the 7 datasets of `run_all_datasets.sh`
(ID + cross-dataset OOD, `fold=test`) **and** the 11 artifact arms of `run_artifact_axes.sh`
as 18 independent `infer.py` jobs, round-robined over GPU lanes (`CUDA_VISIBLE_DEVICES=<id>`
per lane, one log per lane under `<run>/lane_logs/`). Pass the checkpoint's auto-derived run
name as the prefix so the tree is byte-compatible with a sequential sweep:

```bash
scripts/inference/run_eval_suite_parallel.sh <ckpt> \
  sngp_specreg_classifier_acevedo/<ckpt_run_id> 0 1 2 3 -- data.datamodule.num_workers=8
```

`DRY_RUN=1` prints the lane plan, finished jobs (a `metrics.json` exists) are skipped unless
`FORCE=1`, `DATASETS=`/`LEVELS=` narrow the suite, and anything after `--` goes to `infer.py`.
Cap `num_workers` per job when several lanes share one node (the compositing arms are
CPU-bound). First used for the spectral-regularization pilot, 2026-09-18 (~18 jobs, 4 lanes).

Wong (4 classes), baseline / MC-Dropout / SNGP -- same pattern, override the dataset:
```bash
uv run src/inference/infer.py ckpt_path=./checkpoints/wong_sngp_resnet18/model.ckpt \
  data=artifact_image_classifier \
  data.datamodule.dataset_name=nirschl-lab/wong_et_al_2022 data.datamodule.num_classes=4 \
  save_path=<output_dir>/wong_sngp/ infer.save.save_images=true
```

## Cross-dataset (OOD) evaluation

Test Tang data on a checkpoint trained on Wong (in-distribution model, out-of-distribution data):
```bash
uv run src/inference/infer.py \
  ckpt_path=./checkpoints/wong_sngp_resnet18/model.ckpt \
  data=tang \
  save_path=<output_dir>/tang_data_wong_sngp_model/ \
  infer.save.save_images=false
```

Same, but test on **all** folds (not just the test split) and with MC-Dropout:
```bash
uv run src/inference/infer.py \
  ckpt_path=./checkpoints/wong_baseline_resnet18/model.ckpt \
  data=tang \
  save_path=<output_dir>/tang_data_wong_baseline_MC_model_all_folds/ \
  infer.save.save_images=false \
  infer.runtime.use_mc_dropout=true infer.runtime.mc_passes=20 \
  fold=all
```

`fold=all` tests on train+validation+test combined instead of just the test split.

Each `configs/data/<dataset>.yaml` carries its own `dataset_name`/`num_classes`/
`class_to_idx`, so picking the dataset to test on is just `data=<dataset>` now -- no
`data.datamodule.dataset_name=...`/`num_classes=...` overrides needed (that older
pattern still works via a group-composed base like `data=acevedo` plus overrides, but
requires also passing `data.datamodule.class_to_idx=null`, since the config's own
class map would otherwise be checked against whatever dataset actually loads and fail
loudly on a mismatch -- see `scripts/eval/*.sh`'s cross-dataset sweep scripts for that
pattern in the Hydra-multirun case).

## In-distribution performance

```bash
uv run src/inference/infer.py \
  ckpt_path=./checkpoints/tang_baseline_resnet18/model.ckpt \
  data=tang \
  save_path=<output_dir>/indist_performance/tang/baseline_model/ \
  infer.save.save_images=false
```
