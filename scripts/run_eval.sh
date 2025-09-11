#!/bin/bash
# Script to run evaluation with specific parameters

# Parameters
LOGGER="wandb"
CKPT_PATH="artifacts/model-tq26lk4l:v1/model.ckpt"
DATA="acevedo"
MODEL="custom_sngp"
WANDB_GROUP="testdata_acevedo"

uv run src/eval.py \
	logger=${LOGGER} \
	ckpt_path=${CKPT_PATH} \
	data=${DATA} \
	model=${MODEL} \
	logger.wandb.group="${WANDB_GROUP}"