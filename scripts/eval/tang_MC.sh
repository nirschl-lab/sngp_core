#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="image_classifier"


CKPT_PATH="nirschl-lab/sorted_experiments/model-vjfwa59j:v2"
# CKPT_PATH="nirschl-lab/sorted_experiments/model-6a8xclyg:v2" #no class weights
MODEL="baseline_classifier"

WANDB_PROJECT="sorted_experiments"
WANDB_GROUP="test_tang_baseline_resnet18"
EXPERIMENT_NAME="Test: tang_baseline"
ACCELERATOR="gpu"
BATCH_SIZE=2048
N_CLASSES=4
model_name="resnet18"

# Define tags as a variable
TAGS="[\"acevedo_et_al_2020\", \"baseline\", \"test\"]"

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
# DATASET_NAME='nirschl-lab/acevedo_et_al_2020'

# DATASET_NAME="nirschl-lab/acevedo_et_al_2020"
uv run src/eval.py \
	-m \
    logger=${LOGGER} \
    ckpt_path=${CKPT_PATH} \
    data=${DATA} \
    data.datamodule.batch_size=${BATCH_SIZE} \
    data.datamodule.dataset_name=${DATASET_NAME} \
    data.datamodule.num_classes=$N_CLASSES \
    model=${MODEL} \
    model.net.arch=${model_name} \
    model.net.num_classes=$N_CLASSES \
    model.net.pretrained=false \
    model.use_mc=False \
    logger.wandb.group="${WANDB_GROUP}" \
    +logger.wandb.name="${EXPERIMENT_NAME}" \
    trainer.accelerator="${ACCELERATOR}" \
    data.datamodule.test_all_folds=False \
    model.log_csv=True \
    logger.wandb.tags="${TAGS}" \
    model.mc_passes=10 \
    model.log_test_metrics=False\
	++logger.wandb.project="${WANDB_PROJECT}"