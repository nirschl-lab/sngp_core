#!/bin/bash
# Overnight follow-up for the CIFAR-100 / WRN-28-10 benchmark: close the three gaps the
# first run left open (docs/results/CIFAR100_RESULTS.md, "What these numbers do not
# establish").
#
#   1. Seeds. Everything so far is a single draw, so no between-arm difference has an
#      error bar. Adds seeds 1 and 2 for the three healthy arms.
#   2. The `c` estimator mismatch. `spectral_norm_bound=6.0` bounds the reshaped Miyato
#      norm; the reference bounds the true conv operator norm, measured 1.46x larger on
#      this backbone. c = 4.1 is the operator-equivalent control.
#   3. The literal arm's lost `last.ckpt` (run-directory collision).
#
# 8 trainings in 2 blocks of 4 (one per GPU) at ~2.7 h each, then inference and metrics.
# ~6 h total. Every run gets an EXPLICIT hydra run dir -- the first run lost a checkpoint
# because two arms with the same `model.name` resolved to the same timestamped directory.
#
#   scripts/tmux/cifar100_overnight.sh          # detaches into tmux, prints the log path
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
LOGS="${ROOT}/tmux_logs/cifar100_overnight_${STAMP}"
TAG="overnight_${STAMP}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="cifar100_overnight_${STAMP}"
  tmux new-session -d -s "${SESSION}" "bash '$0' --run 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}'."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  echo "  Summary:  ${LOGS}/SUMMARY.md   (written at the end)"
  exit 0
fi

COMMON=(trainer.max_epochs=250 trainer.min_epochs=250 logger.wandb.log_model=false
        "logger.wandb.group=CIFAR100_${TAG}" test=True)

# run <label> <gpu> <experiment> <extra hydra overrides...>
run() {
  local label="$1" gpu="$2" exp="$3"; shift 3
  local dir="${ROOT}/overnight/${TAG}/${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu})"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/train.py \
      experiment="${exp}" "${COMMON[@]}" "$@" \
      "name=${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

echo "=== block 1/2 -- seed 1 for the three healthy arms, plus the c=4.1 control ==="
run s1_baseline        0 baseline_cifar100             seed=1 &
run s1_sngp            1 sngp_cifar100                 seed=1 &
run s1_specreg         2 sngp_specreg_cifar100         seed=1 &
run c41_sngp           3 sngp_cifar100                 model.net.spectral_norm_bound=4.1 &
wait

echo "=== block 2/2 -- seed 2 for the three healthy arms, plus the literal re-run ==="
run s2_baseline        0 baseline_cifar100             seed=2 &
run s2_sngp            1 sngp_cifar100                 seed=2 &
run s2_specreg         2 sngp_specreg_cifar100         seed=2 &
run literal_rerun      3 sngp_specreg_cifar100_literal &
wait

echo "=== inference: last.ckpt on CIFAR-100 (ID) + CIFAR-10 / SVHN (OOD) ==="
# `infer.save.run_name` must carry the dataset: setting it replaces the whole
# auto-derived path tail, including the <dataset> segment, so without it the three
# datasets overwrite each other.
gpu=0
for label in s1_baseline s1_sngp s1_specreg c41_sngp s2_baseline s2_sngp s2_specreg literal_rerun; do
  ckpt="${ROOT}/overnight/${TAG}/${label}/checkpoints/last.ckpt"
  [[ -f "${ckpt}" ]] || { echo "SKIP ${label}: no last.ckpt"; continue; }
  (
    for ds in cifar100 cifar10 svhn; do
      CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/inference/infer.py \
          ckpt_path="${ckpt}" data="${ds}" fold=test \
          "infer.save.run_name=${TAG}_${label}__${ds}" data.datamodule.num_workers=8 \
          > "${LOGS}/infer_${label}_${ds}.log" 2>&1 || echo "infer FAILED ${label}/${ds}"
    done
  ) &
  gpu=$(( (gpu + 1) % 4 ))
  (( gpu == 0 )) && wait
done
wait

echo "=== metrics ==="
uv run python scripts/metrics/cifar100_overnight_report.py \
    --tag "${TAG}" --out "${LOGS}/SUMMARY.md" > "${LOGS}/report.log" 2>&1 \
  || echo "report FAILED (see ${LOGS}/report.log)"

echo "[$(date +%H:%M:%S)] ALL DONE. Summary: ${LOGS}/SUMMARY.md"
