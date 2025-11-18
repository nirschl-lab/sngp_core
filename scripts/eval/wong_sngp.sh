#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="image_classifier"

CKPT_PATH="nirschl-lab/final_experiments/model-d7ro3p3n:v2"

MODEL="sngp_classifier"

WANDB_PROJECT="final_experiments"
WANDB_GROUP="test_wong_sngp_resnet18"
EXPERIMENT_NAME="Test: wong_sngp_"
ACCELERATOR="gpu"
BATCH_SIZE=2048
N_CLASSES=4
model_name="resnet18"
csv_save_path="csv/final/wong_sngp/"

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
TAGS="[\"wong_et_al_2022\", \"SNGP\", \"test\"]"

DATASET_NAME=$(IFS=,; echo "${DATASET_NAMES[*]}")

# DATASET_NAME='nirschl-lab/acevedo_et_al_2020'
DATASET_NAME="nirschl-lab/wong_et_al_2022"
uv run src/eval.py \
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
	logger.wandb.group="${WANDB_GROUP}" \
	+logger.wandb.name="${EXPERIMENT_NAME}" \
    trainer.accelerator="${ACCELERATOR}" \
	data.datamodule.test_all_folds=False \
	model.log_csv=False \
	model.csv_save_path="${csv_save_path}" \
	logger.wandb.tags="${TAGS}" \
	model.use_mean_field_logits=True \
	model.log_test_metrics=True \
	model.class_weights=null \
	++logger.wandb.project="${WANDB_PROJECT}"