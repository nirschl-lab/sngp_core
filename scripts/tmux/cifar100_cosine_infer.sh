#!/bin/bash
# Evaluation for the CIFAR-100 cosine study (scripts/tmux/cifar100_cosine.sh): SNGP, SNGP + SpecReg
# and the Muon GP head (AdamW and SGD aux groups) on the cosine LR schedule at l = 7, seed 12345,
# all at `last.ckpt` (epoch 249). cos_muon_wd0_s12345 was stopped at epoch 152 and is not
# evaluated by default.
#
# Per checkpoint, one lane:
#   1. inference on CIFAR-100 (ID), CIFAR-10 (near-OOD) and SVHN (far-OOD), written to
#      infer/<tag>_<label>__<dataset> -- the naming scripts/metrics/cifar100_evidence_ls_report.py
#      reads;
#   2. the ONE post-hoc knob, mean_field_factor, fit on VALIDATION (calibrate_checkpoint.py
#      --dry-run: nothing written). Training pinned it at 7.5, so no metric is read before this.
#
# `infer.save.run_name` MUST carry the dataset: setting it replaces the whole auto-derived path
# tail, so without it the three datasets overwrite each other.
#
# Lanes go round-robin over GPUS (default: all four); lanes sharing a GPU run one after another.
# LABELS="label ..." evaluates a subset. ~15-20 min per lane.
#
#   scripts/tmux/cifar100_cosine_infer.sh cosine_2026-10-01_10-17-06
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

(( $# == 1 )) || { echo "usage: $0 <cosine_tag>" >&2; exit 1; }
TAG="$1"
declare -A EXPERIMENT=(
  [cos_sngp_l7_s12345]=sngp_cifar100_cosine
  [cos_specreg_l7_s12345]=sngp_specreg_cifar100_cosine
  [cos_muon_wd0.1_s12345]=sngp_muon_cifar100_cosine
  [cos_muon_wd0_s12345]=sngp_muon_cifar100_cosine
  [cos_muonsgd_wd0.1_s12345]=sngp_muon_sgd_cifar100_cosine
  [cos_muonsgd_wd0_s12345]=sngp_muon_sgd_cifar100_cosine
)
DEFAULT_LABELS="cos_sngp_l7_s12345 cos_specreg_l7_s12345 cos_muon_wd0.1_s12345"
DEFAULT_LABELS+=" cos_muonsgd_wd0.1_s12345 cos_muonsgd_wd0_s12345"
read -r -a LABEL_LIST <<< "${LABELS:-${DEFAULT_LABELS}}"
DATASETS=(cifar100 cifar10 svhn)
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
LOGS="${ROOT}/tmux_logs/cifar100_cosine_infer_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${LOGS}"

for label in "${LABEL_LIST[@]}"; do
  [[ -n "${EXPERIMENT[${label}]:-}" ]] || { echo "unknown label ${label}" >&2; exit 1; }
  ckpt="${ROOT}/overnight/${TAG}/${label}/checkpoints/last.ckpt"
  [[ -f "${ckpt}" ]] || { echo "MISSING ${ckpt}" >&2; exit 1; }
done

# lane <label> <gpu>: three inference runs, then the val fit.
lane() {
  local label="$1" gpu="$2"
  local exp="${EXPERIMENT[${label}]}" ckpt="${ROOT}/overnight/${TAG}/${label}/checkpoints/last.ckpt"
  for ds in "${DATASETS[@]}"; do
    echo "[$(date +%H:%M:%S)] ${label} / ${ds} (gpu ${gpu})"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/inference/infer.py \
        ckpt_path="${ckpt}" data="${ds}" fold=test \
        "infer.save.run_name=${TAG}_${label}__${ds}" \
        data.datamodule.num_workers=8 \
        > "${LOGS}/infer_${label}_${ds}.log" 2>&1 \
      || echo "[$(date +%H:%M:%S)] FAILED ${label}/${ds} (see ${LOGS}/infer_${label}_${ds}.log)"
  done
  echo "[$(date +%H:%M:%S)] ${label} / val fit (gpu ${gpu})"
  # fit_<tag>_<label>.log: the name cifar100_evidence_ls_report.py looks up.
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/checkpoints/calibrate_checkpoint.py \
      --ckpt "${ckpt}" --experiment "${exp}" --split val --dry-run \
      > "${LOGS}/fit_${TAG}_${label}.log" 2>&1 \
    || echo "[$(date +%H:%M:%S)] FAILED val fit ${label} (see ${LOGS}/fit_${TAG}_${label}.log)"
}

echo "logs: ${LOGS}"
for g in "${!GPU_LIST[@]}"; do
  (
    for i in "${!LABEL_LIST[@]}"; do
      (( i % ${#GPU_LIST[@]} == g )) && lane "${LABEL_LIST[$i]}" "${GPU_LIST[$g]}"
    done
    true
  ) &
done
wait

grep -H "^fitted " "${LOGS}"/fit_*.log || true
echo "[$(date +%H:%M:%S)] ALL DONE. Predictions: ${ROOT}/infer/${TAG}_<label>__<dataset>/; fits: ${LOGS}/fit_*.log"
