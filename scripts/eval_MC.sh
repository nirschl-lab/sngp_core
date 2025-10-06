#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="acevedo_image_classifier"

# CKPT_PATH="nirschl-lab/SNGP/model-4d9a70bz:v1"
CKPT_PATH="nirschl-lab/SNGP/model-gqlfhp0n:v1"
MODEL="timm_MC_classifier"

WANDB_GROUP="dev_features"
EXPERIMENT_NAME="testing1"
ACCELERATOR=gpu
BATCH_SIZE=2048
N_CLASSES=4

TEST_NAME="trail"
LOG_CSV=False

# DATASET_NAMES=(
#     "nirschl-lab/nirschl_et_al_2018"
#     "nirschl-lab/wong_et_al_2022"
#     "nirschl-lab/jung_et_al_2022"
#     "nirschl-lab/kather_et_al_2016"
# )
# DATASET_NAME=$(IFS=,; echo "${DATASET_NAMES[*]}")

# DATASET_NAME='nirschl-lab/nirschl_et_al_2018','nirschl-lab/wong_et_al_2022','nirschl-lab/jung_et_al_2022','nirschl-lab/kather_et_al_2016'
	
# DATASET_NAME=nirschl-lab/acevedo_et_al_2020

# DATASET_NAME=nirschl-lab/tang_et_al_2019
DATASET_NAME=nirschl-lab/wong_et_al_2022

uv run src/eval.py \
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
    trainer.accelerator=$ACCELERATOR \
	model.test_name=$TEST_NAME \
	model.log_csv=$LOG_CSV \
    model.use_mc=True
