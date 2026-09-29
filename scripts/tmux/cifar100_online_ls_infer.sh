#!/bin/bash
# Evaluation for the `online_ls` study (scripts/tmux/cifar100_online_ls.sh): SpecReg with the GP
# length scale moved during training by type-II evidence, three seeds, at `last.ckpt` (epoch 249).
#
# Inference for the 3 new checkpoints on CIFAR-100 (ID), CIFAR-10 (near-OOD) and SVHN (far-OOD),
# then the ONE post-hoc knob (mean_field_factor) fit on VALIDATION (calibrate_checkpoint.py
# --dry-run: nothing written). The other 12 rows (Baseline / SNGP / SpecReg / evidence l = 7)
# already have predictions and val fits from scripts/tmux/cifar100_evidence_ls_infer.sh; pass
# both fit-log dirs to scripts/metrics/cifar100_evidence_ls_report.py. The rebuilt net takes its
# (final) l from the checkpoint's net spec, so no length-scale override is needed here.
#
# `infer.save.run_name` MUST carry the dataset, or the three datasets overwrite each other.
# One GPU lane per seed (override with GPUS="a b c"). ~30 min.
#
#   scripts/tmux/cifar100_online_ls_infer.sh online_ls_<stamp>
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

TAG="${1:?usage: $0 online_ls_<stamp>}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_${TAG}_infer"
mkdir -p "${LOGS}"

NEW=(ols_specreg_s12345 ols_specreg_s1 ols_specreg_s2)
DATASETS=(cifar100 cifar10 svhn)
for label in "${NEW[@]}"; do
  [[ -f "${RUNS}/${label}/checkpoints/last.ckpt" ]] || { echo "MISSING ${RUNS}/${label}/checkpoints/last.ckpt" >&2; exit 1; }
done

echo "tag:  ${TAG}"
echo "logs: ${LOGS}"

i=0
for label in "${NEW[@]}"; do
  gpu="${GPU_LIST[$i]}"
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
    echo "[$(date +%H:%M:%S)] ${label} / val fit (gpu ${gpu})"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/checkpoints/calibrate_checkpoint.py \
        --ckpt "${ckpt}" --experiment sngp_specreg_cifar100 --split val --dry-run \
        > "${LOGS}/fit_${label}.log" 2>&1 \
      || echo "[$(date +%H:%M:%S)] FAILED val fit ${label} (see ${LOGS}/fit_${label}.log)"
  ) &
  i=$(( i + 1 ))
done
wait

grep -H "^fitted " "${LOGS}"/fit_*.log || true
echo "[$(date +%H:%M:%S)] done. Predictions: ${ROOT}/infer/${TAG}_<label>__<dataset>/; fits: ${LOGS}/fit_*.log"
