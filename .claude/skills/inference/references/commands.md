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
  data=image_classifier \
  data.datamodule.dataset_name=nirschl-lab/tang_et_al_2019 \
  data.datamodule.num_classes=4 \
  save_path=<output_dir>/tang_data_wong_sngp_model/ \
  infer.save.save_images=false
```

Same, but test on **all** folds (not just the test split) and with MC-Dropout:
```bash
uv run src/inference/infer.py \
  ckpt_path=./checkpoints/wong_baseline_resnet18/model.ckpt \
  data=image_classifier \
  data.datamodule.dataset_name=nirschl-lab/tang_et_al_2019 \
  data.datamodule.num_classes=4 \
  save_path=<output_dir>/tang_data_wong_baseline_MC_model_all_folds/ \
  infer.save.save_images=false \
  infer.runtime.use_mc_dropout=true infer.runtime.mc_passes=20 \
  fold=all
```

`fold=all` tests on train+validation+test combined instead of just the test split.

## In-distribution performance

```bash
uv run src/inference/infer.py \
  ckpt_path=./checkpoints/tang_baseline_resnet18/model.ckpt \
  data=image_classifier \
  data.datamodule.dataset_name=nirschl-lab/tang_et_al_2019 \
  data.datamodule.num_classes=4 \
  save_path=<output_dir>/indist_performance/tang/baseline_model/ \
  infer.save.save_images=false
```
