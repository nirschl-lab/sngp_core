#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="acevedo_image_classifier"

# tang baseline model path
# CKPT_PATH="logs/train/runs/2025-09-25_14-46-15/checkpoints/epoch16.ckpt"
# MODEL="timm_basic_classifier"

# tang sngp model path
# CKPT_PATH="logs/train/runs/2025-09-26_09-43-40/checkpoints/epoch17.ckpt"
# MODEL="timm_sngp_classifier"

# wong sngp model path
# WANDB_ARTIFACT="nirschl-lab/SNGP/model-bpsugy0j:v1"
# MODEL="timm_sngp_classifier"

# wong baseline model path
CKPT_PATH="logs/train/runs/2025-09-25_16-46-25/checkpoints/epoch46.ckpt"
MODEL="timm_basic_classifier"


WANDB_GROUP="baseline_testing"
EXPERIMENT_NAME="testing_baseline_wong"
ACCELERATOR="gpu"
BATCH_SIZE=2048
N_CLASSES=4

# DATASET_NAMES=(
#     "nirschl-lab/nirschl_et_al_2018"
#     "nirschl-lab/wong_et_al_2022"
#     "nirschl-lab/jung_et_al_2022"
#     "nirschl-lab/kather_et_al_2016"
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