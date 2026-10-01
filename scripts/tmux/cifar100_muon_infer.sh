#!/bin/bash
# Evaluation for the CIFAR-100 Muon study (scripts/tmux/cifar100_muon.sh): the GP head on an
# unconstrained WRN-28-10 trained with Muon at l = 7, seed 12345, Muon wd 0 / 0.1, under the
# piecewise (`muon_<stamp>`) and/or warmup-stable-decay (`muon_wsd_<stamp>`) schedule. All at
# `last.ckpt` (epoch 249).
#
# Per checkpoint, one GPU lane:
#   1. inference on CIFAR-100 (ID), CIFAR-10 (near-OOD) and SVHN (far-OOD), written to
#      infer/<tag>_<label>__<dataset> -- the naming scripts/metrics/cifar100_evidence_ls_report.py
#      reads;
#   2. the ONE post-hoc knob, mean_field_factor, fit on VALIDATION (calibrate_checkpoint.py
#      --dry-run: nothing written). Training pinned it at 7.5, so no metric is read before this.
# The comparison rows (Baseline / SNGP / SpecReg / evidence l / online l) already have their
# predictions and val fits (cifar100_evidence_ls_infer.sh, cifar100_online_ls_infer.sh).
#
# `infer.save.run_name` MUST carry the dataset: setting it replaces the whole auto-derived path
# tail, so without it the three datasets overwrite each other.
#
# Up to 4 lanes (tags x 2 arms), one per GPU (override with GPUS="a b c d"). ~15-20 min.
#
#   scripts/tmux/cifar100_muon_infer.sh muon_2026-09-30_15-20-01 muon_wsd_2026-09-30_15-54-17
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

(( $# >= 1 )) || { echo "usage: $0 <muon_tag> [<muon_wsd_tag>]" >&2; exit 1; }
TAGS=("$@")
LABELS=(muon_wd0_s12345 muon_wd0.1_s12345)
DATASETS=(cifar100 cifar10 svhn)
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
LOGS="${ROOT}/tmux_logs/cifar100_muon_infer_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${LOGS}"

lanes=()
for tag in "${TAGS[@]}"; do
  case "${tag}" in
    muon_wsd_*) exp=sngp_muon_cifar100_wsd ;;
    muon_*)     exp=sngp_muon_cifar100 ;;
    *) echo "unknown tag ${tag} (expected muon_<stamp> or muon_wsd_<stamp>)" >&2; exit 1 ;;
  esac
  for label in "${LABELS[@]}"; do
    ckpt="${ROOT}/overnight/${tag}/${label}/checkpoints/last.ckpt"
    [[ -f "${ckpt}" ]] || { echo "MISSING ${ckpt}" >&2; exit 1; }
    lanes+=("${tag}|${label}|${exp}|${ckpt}")
  done
done
(( ${#lanes[@]} <= ${#GPU_LIST[@]} )) || { echo "${#lanes[@]} lanes but only ${#GPU_LIST[@]} GPUs" >&2; exit 1; }

echo "logs: ${LOGS}"
for i in "${!lanes[@]}"; do
  IFS='|' read -r tag label exp ckpt <<< "${lanes[$i]}"
  gpu="${GPU_LIST[$i]}"
  (
    for ds in "${DATASETS[@]}"; do
      echo "[$(date +%H:%M:%S)] ${tag}/${label} / ${ds} (gpu ${gpu})"
      CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/inference/infer.py \
          ckpt_path="${ckpt}" data="${ds}" fold=test \
          "infer.save.run_name=${tag}_${label}__${ds}" \
          data.datamodule.num_workers=8 \
          > "${LOGS}/infer_${tag}_${label}_${ds}.log" 2>&1 \
        || echo "[$(date +%H:%M:%S)] FAILED ${tag}/${label}/${ds} (see ${LOGS}/infer_${tag}_${label}_${ds}.log)"
    done
    echo "[$(date +%H:%M:%S)] ${tag}/${label} / val fit (gpu ${gpu})"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/checkpoints/calibrate_checkpoint.py \
        --ckpt "${ckpt}" --experiment "${exp}" --split val --dry-run \
        > "${LOGS}/fit_${tag}_${label}.log" 2>&1 \
      || echo "[$(date +%H:%M:%S)] FAILED val fit ${tag}/${label} (see ${LOGS}/fit_${tag}_${label}.log)"
  ) &
done
wait

grep -H "^fitted " "${LOGS}"/fit_*.log || true
echo "[$(date +%H:%M:%S)] ALL DONE. Predictions: ${ROOT}/infer/<tag>_<label>__<dataset>/; fits: ${LOGS}/fit_*.log"
