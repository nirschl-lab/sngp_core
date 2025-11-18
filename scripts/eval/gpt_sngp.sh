#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="image_classifier"

# CKPT_PATH="nirschl-lab/sorted_experiments/model-4xerenbd:v2"
CKPT_PATH="nirschl-lab/sorted_experiments/model-b6idzep0:v2"

MODEL="sngp_classifier"
arch="resnet18"

WANDB_PROJECT="sorted_experiments"
WANDB_GROUP="test_tang_sngp_resnet18_CW"
EXPERIMENT_NAME="Test: Acevedo_sngp_"
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

# Define tags as a variable
TAGS="[\"tang_et_al_2019\", \"SNGP\", \"test\"]"

DATASET_NAME=$(IFS=,; echo "${DATASET_NAMES[*]}")

# DATASET_NAME='nirschl-lab/acevedo_et_al_2020'

uv run src/eval.py \
	-m \
	logger=${LOGGER} \
	ckpt_path=${CKPT_PATH} \
	data=${DATA} \
	data.datamodule.batch_size=${BATCH_SIZE} \
	data.datamodule.dataset_name=${DATASET_NAME} \
    data.datamodule.num_classes=$N_CLASSES \
	model=${MODEL} \
    model.net.arch=${arch} \
    model.net.num_classes=$N_CLASSES \
	model.net.pretrained=false \
	logger.wandb.group="${WANDB_GROUP}" \
	+logger.wandb.name="${EXPERIMENT_NAME}" \
    trainer.accelerator="${ACCELERATOR}" \
	data.datamodule.test_all_folds=True \
	model.log_csv=True \
	logger.wandb.tags="${TAGS}" \
	model.use_mean_field_logits=True \
	model.log_test_metrics=False \
	model.class_weights=null \
	++logger.wandb.project="${WANDB_PROJECT}"