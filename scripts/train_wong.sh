#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# --------- User-configurable parameters ---------
MODEL="timm_sngp_classifier"
DATA="acevedo_image_classifier"
MAX_EPOCHS=50
LR=1e-4
CALLBACKS="default"
HIST_BINS=10
WANDB_LOG_MODEL=True
ACCELERATOR=gpu
BATCH_SIZE=256

DATASET="nirschl-lab/wong_et_al_2022"
N_CLASSES=4

WANDB_GROUP="training_fixing_gpu"
EXPERIMENT_NAME="sngp_wong"

# --------- Script ---------
# Usage: ./run_basic.sh [extra hydra args]

CUDA_VISIBLE_DEVICES=2 uv run src/train.py \
    model=$MODEL \
    data=$DATA \
    data.datamodule.dataset_name=$DATASET \
    data.datamodule.num_classes=$N_CLASSES \
    data.datamodule.batch_size=$BATCH_SIZE \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.accelerator=$ACCELERATOR \
    model.optimizer.lr=$LR \
    model.hist_bins=$HIST_BINS \
    model.net.num_classes=$N_CLASSES \
    callbacks=$CALLBACKS \
    logger.wandb.log_model=$WANDB_LOG_MODEL \
    logger.wandb.group="${WANDB_GROUP}" \
	+logger.wandb.name="${EXPERIMENT_NAME}" \
    "$@"