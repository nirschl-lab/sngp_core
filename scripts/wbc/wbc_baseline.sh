#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="wbc_classifier"
DATA="image_classifier"

# wong baseline model path
CKPT_PATH="nirschl-lab/wbc_challenge/model-mf3inmum:v0" # no class weights
# CKPT_PATH="nirschl-lab/wbc_challenge/model-y1x1ly65:v1" #swin focal loss

MODEL="wbc_classifier"
# MODEL="swin_baseline_classifier"

WANDB_PROJECT="wbc_challenge"
WANDB_GROUP="test_baseline_resnet18"
EXPERIMENT_NAME="Test: baseline_resnet18_acevedo"
ACCELERATOR="gpu"
BATCH_SIZE=2048
N_CLASSES=13
model_name="resnet18"
# model_name="swin_t"
csv_save_path="csv/wbc/swin_baseline/"
data_dir='/data1/shared/data/wbc-bench-2026/'

# Define tags as a variable
TAGS="[\"baseline\", \"resnet18\", \"CE\"]"

uv run src/predict.py \
    logger=${LOGGER} \
    ckpt_path=${CKPT_PATH} \
    data=${DATA} \
    data.datamodule.test_batch_size=${BATCH_SIZE} \
    model=${MODEL} \
    model.class_weights=null \
    model.net.arch=${model_name} \
    model.net.num_classes=$N_CLASSES \
    model.net.pretrained=false \
    model.log_csv=True \
    model.csv_save_path="${csv_save_path}" \
    trainer.accelerator="${ACCELERATOR}" \
    logger.wandb.group="${WANDB_GROUP}" \
    +logger.wandb.name="${EXPERIMENT_NAME}" \
    logger.wandb.tags="${TAGS}" \
	++logger.wandb.project="${WANDB_PROJECT}"