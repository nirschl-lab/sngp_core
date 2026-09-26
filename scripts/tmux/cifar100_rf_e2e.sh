#!/bin/bash
# Study `rf_e2e_l2`: CIFAR-100 / WRN-28-10 spectral-regularization arm trained END-TO-END with
# each random-feature map {cos, positive, hyperbolic} x coupling {orf, simrf}, under the SNGP
# paper's GP head (l = 2, kernel amplitude sigma^2 = 7.5 in the feature map, lambda = pi/8).
# Experiment: configs/experiment/sngp_specreg_cifar100_rf.yaml -- its header has the reasoning.
# Follows up the frozen-backbone head swap in docs/results/CIFAR100_RF_HEAD_SWAP_RESULTS.md.
#
# One seed (12345), 6 arms, all started at once on 4 GPUs: GPUs 0 and 1 carry two runs each
# (a WRN-28-10 at batch 128 fits twice in an L40S), GPUs 2 and 3 one each. TRAINING ONLY.
#
# --smoke trains every arm for ~3 epochs (test=False) under a separate smoke_<stamp> tree. It
# caps optimizer steps rather than lowering max_epochs, which would rescale the piecewise LR
# schedule (warmup_piecewise_lr refuses that); 1200 steps ~ 3 epochs at batch 128. Run
# it first: at l = 2 the positive / hyperbolic maps may not train at all (rho = ||x||/l ~ 5.8
# on a trained backbone), and that should cost minutes, not the full budget.
#
# Every run gets an EXPLICIT hydra run dir -- all arms share `model.name`, and the first
# benchmark lost a `last.ckpt` to a run-directory collision (docs/checkpoints/CIFAR_CHECKPOINTS.md).
#
#   scripts/tmux/cifar100_rf_e2e.sh --smoke   # ~3-epoch check of all 6 arms
#   scripts/tmux/cifar100_rf_e2e.sh           # full 250-epoch study
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="rf_e2e_l2"
MODE="full"
for arg in "$@"; do [[ "${arg}" == "--smoke" ]] && MODE="smoke"; done

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${RF_E2E_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export RF_E2E_STAMP="${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
RUN_TAG="${STAMP}"; [[ "${MODE}" == "smoke" ]] && RUN_TAG="smoke_${STAMP}"
RUNS="${ROOT}/train/${STUDY}/${RUN_TAG}"
LOGS="${ROOT}/tmux_logs/${STUDY}_${RUN_TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="${STUDY}_${RUN_TAG}"
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="--run --smoke"
  tmux new-session -d -s "${SESSION}" -e "RF_E2E_STAMP=${STAMP}" "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE})."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  echo "  Run dirs: ${RUNS}/<feature_map>_<coupling>"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=1 ++trainer.max_steps=1200 test=False
          logger.wandb.group=CIFAR100_${STUDY}_smoke)
else
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=250 test=True logger.wandb.group=CIFAR100_${STUDY}_${STAMP})
fi
COMMON=(experiment=sngp_specreg_cifar100_rf seed=12345 logger.wandb.log_model=false "${BUDGET[@]}")

# run <feature_map> <coupling> <gpu>
run() {
  local fmap="$1" coupling="$2" gpu="$3"
  local label="${fmap}_${coupling}"
  local dir="${RUNS}/${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu})"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/train.py \
      "${COMMON[@]}" "model.net.feature_map=${fmap}" "model.net.random_feature_type=${coupling}" \
      "name=${STUDY}_${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

echo "=== ${STUDY} (${MODE}) -- sngp_specreg_cifar100_rf, 3 feature maps x {orf, simrf}, seed 12345 ==="
run cos        orf    0 &
run cos        simrf  0 &
run positive   orf    1 &
run positive   simrf  1 &
run hyperbolic orf    2 &
run hyperbolic simrf  3 &
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/"
