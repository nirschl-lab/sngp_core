#!/bin/bash
# Inference + validation lambda fit for the `rf_e2e` study (scripts/tmux/cifar100_rf_e2e.sh,
# which trained them and ran no inference). Splits: CIFAR-100 (ID), CIFAR-10 (near-OOD),
# SVHN (far-OOD) -- the ones scripts/metrics/cifar100_rf_e2e_report.py reads.
#
# `last.ckpt` (epoch 249), the CIFAR comparison point (docs/checkpoints/CIFAR_CHECKPOINTS.md).
# `infer.save.run_name` MUST carry the dataset, or the three datasets overwrite each other;
# `<tag>_<label>__<dataset>` is what cifar100_overnight_report.py::_infer_dirname resolves.
#
# The two heads were trained at different mean-field factors (pi/8 vs 7.5), so lambda is
# refit per run on VALIDATION (calibrate_checkpoint.py --dry-run: nothing written) and every
# number is reported on test. The l = 20 cos/orf control (existing SpecReg seed-12345 run,
# already inferred as infer/cifar100last_specreg__<dataset>) gets the same fit.
#
# 4 GPU lanes, ~15-20 min total.
#
#   scripts/tmux/cifar100_rf_e2e_infer.sh [STAMP]
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STAMP="${1:-2026-09-26_16-55-28}"
TAG="rf_e2e_${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
RUNS="${ROOT}/train/rf_e2e/${STAMP}"
LOGS="${ROOT}/tmux_logs/rf_e2e_infer_${STAMP}"
mkdir -p "${LOGS}"
CONTROL_CKPT="${ROOT}/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt"

LABELS=(l2paper_cos_orf l2paper_cos_simrf l20recipe_positive_orf l20recipe_positive_simrf
        l20recipe_hyperbolic_orf l20recipe_hyperbolic_simrf)
DATASETS=(cifar100 cifar10 svhn)

echo "tag:  ${TAG}"
echo "logs: ${LOGS}"

# fit_lambda <label> <ckpt> <experiment> <gpu>
fit_lambda() {
  CUDA_VISIBLE_DEVICES="$4" uv run python scripts/checkpoints/calibrate_checkpoint.py \
      --ckpt "$2" --experiment "$3" --split val --dry-run \
      > "${LOGS}/lambda_$1.log" 2>&1 \
    || echo "[$(date +%H:%M:%S)] FAILED lambda fit $1 (see ${LOGS}/lambda_$1.log)"
}

lane() {
  local gpu="$1"; shift
  for label in "$@"; do
    local ckpt="${RUNS}/${label}/checkpoints/last.ckpt" exp=sngp_specreg_cifar100
    [[ "${label}" == l2paper_* ]] && exp=sngp_specreg_cifar100_rf
    [[ -f "${ckpt}" ]] || { echo "MISSING ${ckpt}" >&2; continue; }
    for ds in "${DATASETS[@]}"; do
      echo "[$(date +%H:%M:%S)] ${label} / ${ds} (gpu ${gpu})"
      CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/inference/infer.py \
          ckpt_path="${ckpt}" data="${ds}" fold=test \
          "infer.save.run_name=${TAG}_${label}__${ds}" \
          data.datamodule.num_workers=8 \
          > "${LOGS}/infer_${label}_${ds}.log" 2>&1 \
        || echo "[$(date +%H:%M:%S)] FAILED ${label}/${ds} (see ${LOGS}/infer_${label}_${ds}.log)"
    done
    echo "[$(date +%H:%M:%S)] ${label} / lambda fit (gpu ${gpu})"
    fit_lambda "${label}" "${ckpt}" "${exp}" "${gpu}"
  done
}

lane 0 l2paper_cos_orf l20recipe_hyperbolic_orf &
lane 1 l2paper_cos_simrf l20recipe_hyperbolic_simrf &
lane 2 l20recipe_positive_orf &
lane 3 l20recipe_positive_simrf &
( echo "[$(date +%H:%M:%S)] control / lambda fit (gpu 2)"; fit_lambda control_specreg_s12345 "${CONTROL_CKPT}" sngp_specreg_cifar100 2 ) &
wait

grep -H "fitted mean_field_factor" "${LOGS}"/lambda_*.log || true
echo "[$(date +%H:%M:%S)] done. Outputs: ${ROOT}/infer/${TAG}_<label>__<dataset>/"
