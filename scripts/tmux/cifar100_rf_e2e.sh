#!/bin/bash
# Study `rf_e2e`: CIFAR-100 / WRN-28-10 spectral-regularization arm trained END-TO-END with
# different random-feature maps, following up the frozen-backbone head swap in
# docs/results/CIFAR100_RF_HEAD_SWAP_RESULTS.md. Two GP heads:
#   l2paper   -- configs/experiment/sngp_specreg_cifar100_rf.yaml, a literal reading of the
#                paper's equations (l = 2, sigma^2 = 7.5 on the scaled base, lambda = pi/8);
#                cos x {orf, simrf}. Off-protocol, and not the paper's head: its 7.5 is a
#                post-hoc value (see the config header). The label is kept for the run dirs.
#   l20recipe -- configs/experiment/sngp_specreg_cifar100.yaml, the reference CIFAR head
#                (l = 20, unscaled, sigma^2 = 1, lambda = 7.5); {positive, hyperbolic} x {orf, simrf}.
# Positive / hyperbolic are only run at l = 20: at l = 2 they sit at chance in a 3-epoch smoke
# under every setting tried (rho = ||x||/l stays >~ 3, and the hyperbolic backbone features blow
# up). The l = 20 cos/orf control is the existing SpecReg seed-12345 run
# (train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59, last.ckpt).
#
# One seed (12345), 6 runs, all started at once on 4 GPUs: GPUs 0 and 1 carry two runs each
# (~5 GB per WRN-28-10 at batch 128), GPUs 2 and 3 one each. TRAINING ONLY.
#
# --smoke trains every arm for ~3 epochs (test=False) under a separate smoke_<stamp> tree. It
# caps optimizer steps rather than lowering max_epochs, which would rescale the piecewise LR
# schedule (warmup_piecewise_lr refuses that); 1200 steps ~ 3 epochs at batch 128.
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

STUDY="rf_e2e"
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
  echo "  Run dirs: ${RUNS}/<head>_<feature_map>_<coupling>"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=1 ++trainer.max_steps=1200 test=False
          logger.wandb.group=CIFAR100_${STUDY}_smoke)
else
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=250 test=True logger.wandb.group=CIFAR100_${STUDY}_${STAMP})
fi
COMMON=(seed=12345 logger.wandb.log_model=false "${BUDGET[@]}")

# run <head> <feature_map> <coupling> <gpu>
run() {
  local head="$1" fmap="$2" coupling="$3" gpu="$4" exp
  case "${head}" in
    l2paper)   exp=sngp_specreg_cifar100_rf ;;
    l20recipe) exp=sngp_specreg_cifar100 ;;
    *) echo "unknown head ${head}" >&2; return 1 ;;
  esac
  local label="${head}_${fmap}_${coupling}"
  local dir="${RUNS}/${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu})"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/train.py \
      "experiment=${exp}" "${COMMON[@]}" "model.net.feature_map=${fmap}" "model.net.random_feature_type=${coupling}" \
      "name=${STUDY}_${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

echo "=== ${STUDY} (${MODE}) -- l2paper cos x {orf, simrf}; l20recipe {positive, hyperbolic} x {orf, simrf}; seed 12345 ==="
run l2paper   cos        orf    0 &
run l2paper   cos        simrf  0 &
run l20recipe positive   orf    1 &
run l20recipe positive   simrf  1 &
run l20recipe hyperbolic orf    2 &
run l20recipe hyperbolic simrf  3 &
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/"
