#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# --------- User-configurable parameters ---------
MODEL="timm_basic_classifier"
DATA="acevedo_image_classifier"
MAX_EPOCHS=10
LR=1e-4
CALLBACKS="default"
HIST_BINS=10
WANDB_LOG_MODEL=False
ACCELERATOR=gpu
BATCH_SIZE=128

# --------- Script ---------
# Usage: ./run_basic.sh [extra hydra args]

uv run src/train.py \
    model=$MODEL \
    data=$DATA \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.accelerator=$ACCELERATOR \
    model.optimizer.lr=$LR \
    data.datamodule.batch_size=$BATCH_SIZE \
    callbacks=$CALLBACKS \
    model.hist_bins=$HIST_BINS \
    logger.wandb.log_model=$WANDB_LOG_MODEL \
    "$@"