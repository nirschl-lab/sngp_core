#!/bin/bash
# Script to run evaluation with specific parameters
set -e

# Parameters
LOGGER="wandb"

DATA="image_classifier"

# CKPT_PATH="nirschl-lab/sorted_experiments/model-leolaoi5:v2"
# CKPT_PATH="nirschl-lab/sorted_experiments/model-4cggbjor:v2" #10000 samples
CKPT_PATH="nirschl-lab/final_experiments/model-xxsowjqw:v1"

MODEL="sngp_classifier"

WANDB_PROJECT="final_experiments"
WANDB_GROUP="test_kather2018_sngp_resnet18"
EXPERIMENT_NAME="Test: kather2018_sngp"
ACCELERATOR="gpu"
BATCH_SIZE=2048
N_CLASSES=9
model_name="resnet18"
csv_save_path="csv/final/kather2018_sngp/"

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
TAGS="[\"kather_et_al_2018\", \"SNGP\", \"test\"]"

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
    model.net.arch=${model_name} \
    model.net.num_classes=$N_CLASSES \
	model.net.pretrained=false \
	logger.wandb.group="${WANDB_GROUP}" \
	+logger.wandb.name="${EXPERIMENT_NAME}" \
    trainer.accelerator="${ACCELERATOR}" \
	data.datamodule.test_all_folds=True \
	model.log_csv=True \
	model.csv_save_path="${csv_save_path}" \
	logger.wandb.tags="${TAGS}" \
	model.use_mean_field_logits=True \
	model.log_test_metrics=False \
	model.class_weights=null \
	++logger.wandb.project="${WANDB_PROJECT}"