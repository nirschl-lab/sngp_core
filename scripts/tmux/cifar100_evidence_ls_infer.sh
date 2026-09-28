#!/bin/bash
# Evaluation for the `evidence_ls` study (scripts/tmux/cifar100_evidence_ls.sh): the 4-row
# CIFAR-100 benchmark Baseline / SNGP / SNGP+SpecReg / SNGP+SpecReg+evidence l = 7, three seeds
# each, all at `last.ckpt` (epoch 249).
#
# Two jobs:
#   1. Inference for the 3 new checkpoints on CIFAR-100 (ID), CIFAR-10 (near-OOD) and SVHN
#      (far-OOD). The 9 existing checkpoints already have predictions
#      (infer/cifar100last_<arm>__<ds> for seed 12345, infer/overnight_2026-09-20_21-38-42_s{1,2}_<arm>__<ds>),
#      so they are not re-run.
#   2. The ONE post-hoc knob per row, fit on VALIDATION for all 12 checkpoints through the same
#      code path (calibrate_checkpoint.py --dry-run: nothing written): mean_field_factor for
#      the SNGP arms and temperature for the baseline. Every metric is then reported on test by
#      scripts/metrics/cifar100_evidence_ls_report.py.
#
# `infer.save.run_name` MUST carry the dataset: setting it replaces the whole auto-derived path
# tail, including the <dataset> segment, so without it the three datasets overwrite each other.
#
# 4 GPU lanes: lanes 0-2 run one new seed each (inference, then its fit), lane 3 fits the 9
# existing checkpoints. ~30-40 min.
#
#   scripts/tmux/cifar100_evidence_ls_infer.sh evidence_ls_<stamp>
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

TAG="${1:?usage: $0 evidence_ls_<stamp>}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_${TAG}_infer"
mkdir -p "${LOGS}"

# Existing arms -- docs/checkpoints/CIFAR_CHECKPOINTS.md. The seed-12345 SpecReg last.ckpt is the
# matched arm's despite the run-directory collision recorded there.
FIRST="${ROOT}/train"
OVERNIGHT="${ROOT}/overnight/overnight_2026-09-20_21-38-42"
declare -A EXISTING=(
  [baseline_s12345]="${FIRST}/baseline_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt|baseline_cifar100"
  [sngp_s12345]="${FIRST}/sngp_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt|sngp_cifar100"
  [specreg_s12345]="${FIRST}/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt|sngp_specreg_cifar100"
)
for s in 1 2; do
  EXISTING[baseline_s${s}]="${OVERNIGHT}/s${s}_baseline/checkpoints/last.ckpt|baseline_cifar100"
  EXISTING[sngp_s${s}]="${OVERNIGHT}/s${s}_sngp/checkpoints/last.ckpt|sngp_cifar100"
  EXISTING[specreg_s${s}]="${OVERNIGHT}/s${s}_specreg/checkpoints/last.ckpt|sngp_specreg_cifar100"
done
NEW=(els_specreg_s12345 els_specreg_s1 els_specreg_s2)
DATASETS=(cifar100 cifar10 svhn)

for label in "${NEW[@]}"; do
  [[ -f "${RUNS}/${label}/checkpoints/last.ckpt" ]] || { echo "MISSING ${RUNS}/${label}/checkpoints/last.ckpt" >&2; exit 1; }
done
for key in "${!EXISTING[@]}"; do
  [[ -f "${EXISTING[$key]%%|*}" ]] || { echo "MISSING ${EXISTING[$key]%%|*}" >&2; exit 1; }
done

echo "tag:  ${TAG}"
echo "logs: ${LOGS}"

# fit <row> <ckpt> <experiment> <gpu>
fit() {
  echo "[$(date +%H:%M:%S)] $1 / val fit (gpu $4)"
  CUDA_VISIBLE_DEVICES="$4" uv run python scripts/checkpoints/calibrate_checkpoint.py \
      --ckpt "$2" --experiment "$3" --split val --dry-run \
      > "${LOGS}/fit_$1.log" 2>&1 \
    || echo "[$(date +%H:%M:%S)] FAILED val fit $1 (see ${LOGS}/fit_$1.log)"
}

gpu=0
for label in "${NEW[@]}"; do
  ckpt="${RUNS}/${label}/checkpoints/last.ckpt"
  (
    for ds in "${DATASETS[@]}"; do
      echo "[$(date +%H:%M:%S)] ${label} / ${ds} (gpu ${gpu})"
      CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/inference/infer.py \
          ckpt_path="${ckpt}" data="${ds}" fold=test \
          "infer.save.run_name=${TAG}_${label}__${ds}" \
          data.datamodule.num_workers=8 \
          > "${LOGS}/infer_${label}_${ds}.log" 2>&1 \
        || echo "[$(date +%H:%M:%S)] FAILED ${label}/${ds} (see ${LOGS}/infer_${label}_${ds}.log)"
    done
    fit "${label}" "${ckpt}" sngp_specreg_cifar100 "${gpu}"
  ) &
  gpu=$(( gpu + 1 ))
done
(
  for key in $(printf '%s\n' "${!EXISTING[@]}" | sort); do
    fit "${key}" "${EXISTING[$key]%%|*}" "${EXISTING[$key]##*|}" 3
  done
) &
wait

grep -H "^fitted " "${LOGS}"/fit_*.log || true
echo "[$(date +%H:%M:%S)] done. Predictions: ${ROOT}/infer/${TAG}_<label>__<dataset>/; fits: ${LOGS}/fit_*.log"
echo "Next: uv run python scripts/metrics/cifar100_evidence_ls_report.py --tag ${TAG} --fit-logs ${LOGS}"
