#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="image_classifier"


# wong baseline model path
# CKPT_PATH="nirschl-lab/SNGP/model-aajrgqcs:v0"
# CKPT_PATH="logs/train/runs/2025-10-13_14-53-57/checkpoints/epoch_33_step_8160_val_acc_0.95.ckpt"
# CKPT_PATH="nirschl-lab/sngp_isbi/model-1xyrcta1:v1" #old loss
CKPT_PATH="nirschl-lab/tang_expts/model-lay0jrby:v1"
MODEL="timm_sngp_classifier"

# Define tags as a variable
TAGS="[\"tang_et_al_2019\", \"SNGP\" ]"

WANDB_GROUP="Testing"
WANDB_PROJECT="tang_expts"
EXPERIMENT_NAME="Test: Tang_sngp_"
ACCELERATOR="gpu"
BATCH_SIZE=2048
N_CLASSES=4

DATASET_NAMES=(
    "nirschl-lab/tang_et_al_2019"
    "nirschl-lab/wong_et_al_2022"
    "nirschl-lab/jung_et_al_2022"
    "nirschl-lab/nirschl_et_al_2018"
    "nirschl-lab/kather_et_al_2016"
	"nirschl-lab/kather_et_al_2018"
    "nirschl-lab/acevedo_et_al_2020"
)

# DATASET_NAMES=(
#     "nirschl-lab/tang_et_al_2019"
#     "nirschl-lab/wong_et_al_2022"
# )

DATASET_NAME=$(IFS=,; echo "${DATASET_NAMES[*]}")

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
	model.log_csv=True \
	logger.wandb.tags="${TAGS}" \
    model.use_mean_field_logits=True \
	model.log_test_metrics=False \
	++logger.wandb.project="${WANDB_PROJECT}"