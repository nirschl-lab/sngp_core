#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# --------- User-configurable parameters ---------
MODEL="custom_sngp"
DATA="acevedo"
MAX_EPOCHS=10
LR=1e-4
CALLBACKS="default"
HIST_BINS=10
WANDB_LOG_MODEL=True

# --------- Script ---------
# Usage: ./run_basic.sh [extra hydra args]

uv run src/train.py \
    model=$MODEL \
    data=$DATA \
    trainer.max_epochs=$MAX_EPOCHS \
    model.optimizer.lr=$LR \
    callbacks=$CALLBACKS \
    model.hist_bins=$HIST_BINS \
    logger.wandb.log_model=$WANDB_LOG_MODEL \
    "$@"