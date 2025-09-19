#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"
# CKPT_PATH="logs/train/runs/2025-09-17_15-02-56/checkpoints/epoch_043.ckpt"
CKPT_PATH="logs/train/runs/2025-09-17_15-52-06/checkpoints/epoch_045.ckpt"
DATA="acevedo_image_classifier"
# MODEL="timm_sngp_classifier"
MODEL="timm_basic_classifier"
WANDB_GROUP="testdata_acevedo"
ACCELERATOR=gpu
BATCH_SIZE=2048
DATASET_NAME='nirschl-lab/jung_et_al_2022','nirschl-lab/kather_et_al_2016'
# DATASET_NAME=nirschl-lab/acevedo_et_al_2020
# DATASET_NAME=nirschl-lab/tang_et_al_2019

uv run src/eval.py -m \
	logger=${LOGGER} \
	ckpt_path=${CKPT_PATH} \
	data=${DATA} \
	data.datamodule.batch_size=${BATCH_SIZE} \
	data.datamodule.dataset_name=${DATASET_NAME} \
	model=${MODEL} \
	logger.wandb.group="${WANDB_GROUP}" \
	trainer.max_epochs=$MAX_EPOCHS \
    trainer.accelerator=$ACCELERATOR \
