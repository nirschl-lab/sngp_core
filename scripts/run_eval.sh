#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="acevedo_image_classifier"

CKPT_PATH="logs/train/runs/2025-09-17_15-02-56/checkpoints/epoch_043.ckpt"
MODEL="timm_sngp_classifier"

# CKPT_PATH="logs/train/runs/2025-09-17_15-52-06/checkpoints/epoch_045.ckpt"
# MODEL="timm_basic_classifier"

WANDB_GROUP="dev_features"
EXPERIMENT_NAME="testing1"
ACCELERATOR=gpu
BATCH_SIZE=2048


# DATASET_NAMES=(
#     "nirschl-lab/nirschl_et_al_2018"
#     "nirschl-lab/wong_et_al_2022"
#     "nirschl-lab/jung_et_al_2022"
#     "nirschl-lab/kather_et_al_2016"
# )
# DATASET_NAME=$(IFS=,; echo "${DATASET_NAMES[*]}")

# DATASET_NAME='nirschl-lab/nirschl_et_al_2018','nirschl-lab/wong_et_al_2022','nirschl-lab/jung_et_al_2022','nirschl-lab/kather_et_al_2016'
	
# DATASET_NAME=nirschl-lab/acevedo_et_al_2020
DATASET_NAME=nirschl-lab/tang_et_al_2019

uv run src/eval.py \
	logger=${LOGGER} \
	ckpt_path=${CKPT_PATH} \
	data=${DATA} \
	data.datamodule.batch_size=${BATCH_SIZE} \
	data.datamodule.dataset_name=${DATASET_NAME} \
	model=${MODEL} \
	logger.wandb.group="${WANDB_GROUP}" \
	+logger.wandb.name="${EXPERIMENT_NAME}" \
    trainer.accelerator=$ACCELERATOR \
