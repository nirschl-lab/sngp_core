#!/bin/bash
# Inference for the CIFAR-100 trace_logistic SpecReg arms, on the three splits the
# mean-field sweep and the OOD AUROC tables read: CIFAR-100 (ID), CIFAR-10 (near-OOD),
# SVHN (far-OOD). Companion to scripts/tmux/cifar100_trace_logistic.sh, which trained
# them and deliberately ran no inference.
#
# `last.ckpt` (epoch 249), not `best.ckpt`: `val/loss` is computed from mean-field logits,
# so the checkpoint gate lands on a different epoch here than it did for the gaussian arms
# even though the per-epoch weights are the same. Epoch 249 is the protocol's comparison
# point anyway (docs/checkpoints/CIFAR_CHECKPOINTS.md).
#
# `infer.save.run_name` MUST carry the dataset: setting it replaces the whole auto-derived
# path tail, including the <dataset> segment, so without it the three datasets overwrite
# each other. The `<tag>_<label>__<dataset>` shape is what
# scripts/metrics/cifar100_overnight_report.py::_infer_dirname reconstructs.
#
# 3 GPU lanes (one seed each), 3 datasets serially per lane, ~15 min total.
#
#   scripts/tmux/cifar100_trace_logistic_infer.sh [TAG]
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

TAG="${1:-trace_logistic_2026-09-23_17-06-47}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_trace_logistic_infer_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${LOGS}"

LABELS=(tl_specreg_s12345 tl_specreg_s1 tl_specreg_s2)
DATASETS=(cifar100 cifar10 svhn)

echo "tag:  ${TAG}"
echo "logs: ${LOGS}"

gpu=0
for label in "${LABELS[@]}"; do
  ckpt="${RUNS}/${label}/checkpoints/last.ckpt"
  [[ -f "${ckpt}" ]] || { echo "MISSING ${ckpt}" >&2; exit 1; }
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
  ) &
  gpu=$(( gpu + 1 ))
done
wait

echo "[$(date +%H:%M:%S)] done. Outputs: ${ROOT}/infer/${TAG}_<label>__<dataset>/"
