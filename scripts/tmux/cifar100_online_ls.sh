#!/bin/bash
# Study `online_ls`: the CIFAR-100 / WRN-28-10 spectral-regularization arm with its GP length
# scale chosen DURING training by type-II evidence, after Immer et al. 2021 (arXiv:2104.04975),
# head only -- src/callbacks/online_length_scale.py.
#
# The two-stage alternative is the evidence_ls study (scripts/tmux/cifar100_evidence_ls.sh):
# score the evidence on a frozen l = 20 backbone, then retrain at the pick (l = 7). Here the run
# starts at the recipe's l = 20 and, every 5 epochs from epoch 10 to 159, scores the Gaussian
# random-feature evidence over l * 4^[-1, 1] on 10k train images through the CURRENT backbone,
# then takes a half (geometric) step toward the argmax. From epoch 160 (the last LR decay) l is
# fixed, so the final 90 epochs and the final-epoch precision see one l. The backbone and the
# head's weights still train by CE; ridge stays 1.0 (type-II (alpha, s) only score l), and
# mean_field_factor is fit post hoc on val at evaluation time -- the protocol's one knob.
#
# The ONLY change against sngp_specreg_cifar100.yaml is the added callback, as launcher
# overrides. The final l is in each checkpoint's net spec and in W&B `ls/length_scale`.
#
# Three seeds (12345 / 1 / 2), one per GPU (override with GPUS="a b c"). TRAINING ONLY. Every
# run gets an EXPLICIT hydra run dir (all arms share `model.name`).
#
# --smoke trains seed 12345 only, for 1200 steps (~3 epochs; steps are capped, not max_epochs,
# which would rescale the piecewise LR schedule), with the first update at epoch 1 and one per
# epoch, so two updates fire. Check that l moves in driver/run logs and train/loss survives.
#
#   scripts/tmux/cifar100_online_ls.sh --smoke   # ~3-epoch check, one run
#   scripts/tmux/cifar100_online_ls.sh           # full 250-epoch study
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="online_ls"
MODE="full"
for arg in "$@"; do [[ "${arg}" == "--smoke" ]] && MODE="smoke"; done
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${ONLINE_LS_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export ONLINE_LS_STAMP="${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
TAG="${STUDY}_${STAMP}"; [[ "${MODE}" == "smoke" ]] && TAG="${STUDY}_smoke_${STAMP}"
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_${TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="cifar100_${TAG}"
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="--run --smoke"
  tmux new-session -d -s "${SESSION}" -e "ONLINE_LS_STAMP=${STAMP}" -e "GPUS=${GPU_LIST[*]}" \
      "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE})."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  echo "  Run dirs: ${RUNS}/<label>"
  exit 0
fi

CB="+callbacks.online_length_scale"
if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=1 ++trainer.max_steps=1200 test=False
          "logger.wandb.group=CIFAR100_${STUDY}_smoke" "${CB}.burnin_epochs=1" "${CB}.every_n_epochs=1")
else
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=250 test=True "logger.wandb.group=CIFAR100_${TAG}")
fi
COMMON=(experiment=sngp_specreg_cifar100
        "${CB}._target_=src.callbacks.online_length_scale.OnlineLengthScaleEvidence"
        logger.wandb.log_model=false "${BUDGET[@]}")

# run <label> <gpu> <extra hydra overrides...>
run() {
  local label="$1" gpu="$2"; shift 2
  local dir="${RUNS}/${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu})"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/train.py \
      "${COMMON[@]}" "$@" \
      "name=${STUDY}_${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

echo "=== ${STUDY} (${MODE}) -- sngp_specreg_cifar100 + OnlineLengthScaleEvidence from l = 20 ==="
if [[ "${MODE}" == "smoke" ]]; then
  run ols_specreg_s12345  "${GPU_LIST[0]}"  seed=12345
else
  run ols_specreg_s12345  "${GPU_LIST[0]}"  seed=12345 &
  run ols_specreg_s1      "${GPU_LIST[1]}"  seed=1 &
  run ols_specreg_s2      "${GPU_LIST[2]}"  seed=2 &
  wait
fi

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/"
echo "Final l per run: grep 'OnlineLengthScaleEvidence epoch' ${LOGS}/*.log | tail"
