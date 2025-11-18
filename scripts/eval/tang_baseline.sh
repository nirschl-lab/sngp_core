#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="image_classifier"


# wong baseline model path
CKPT_PATH="nirschl-lab/SNGP/model-0he9e3zd:v1"
MODEL="timm_basic_classifier"


WANDB_GROUP="csv_files"
EXPERIMENT_NAME="Model: Baseline_Tang Test_Data: Wong"
ACCELERATOR="gpu"
BATCH_SIZE=2048
N_CLASSES=4

# DATASET_NAMES=(
#     "nirschl-lab/tang_et_al_2019"
#     "nirschl-lab/wong_et_al_2022"
#     "nirschl-lab/jung_et_al_2022"
#     "nirschl-lab/nirschl_et_al_2018"
#     "nirschl-lab/kather_et_al_2016"
#     "nirschl-lab/acevedo_et_al_2020"
# )

DATASET_NAMES=(
    "nirschl-lab/tang_et_al_2019"
    "nirschl-lab/wong_et_al_2022"
)


DATASET_NAME=$(IFS=,; echo "${DATASET_NAMES[*]}")

# DATASET_NAME="nirschl-lab/wong_et_al_2022"

uv run src/eval.py \
	-m \
	logger=${LOGGER} \
	ckpt_path=${CKPT_PATH} \
	data=${DATA} \
	data.datamodule.batch_size=${BATCH_SIZE} \
	data.datamodule.dataset_name=${DATASET_NAME} \
    data.datamodule.num_classes=$N_CLASSES \
	model=${MODEL} \
    model.net.num_classes=$N_CLASSES \
	logger.wandb.group="${WANDB_GROUP}" \
	+logger.wandb.name="${EXPERIMENT_NAME}" \
    trainer.accelerator="${ACCELERATOR}" \
	data.datamodule.test_all_folds=True \
	model.log_csv=True