#!/bin/bash
# Study `muon`: the CIFAR-100 / WRN-28-10 SNGP GP head on an unconstrained backbone (no spectral
# normalization, no spectral penalty) trained with Muon, at the evidence-picked l = 7 --
# configs/experiment/sngp_muon_cifar100.yaml has the reasoning. Follow-up of the Acevedo study
# (docs/results/ACEVEDO_MUON_RESULTS.md, scripts/tmux/acevedo_muon.sh).
#
# One seed (12345), two arms, one per GPU (override with GPUS="a b"), ~2.7-3 h:
#   muon_wd0_s12345     Muon weight_decay 0   -- no cap on the conv weights' spectral norm
#   muon_wd0.1_s12345   Muon weight_decay 0.1 -- soft cap ~1/wd (the experiment's default)
#
# TRAINING ONLY. Evaluation follows the benchmark protocol: last.ckpt (epoch 249), with
# mean_field_factor fit on val before any comparison (training pins it at 7.5). Every run gets an
# EXPLICIT hydra run dir because both arms share `model.name`.
#
# --smoke trains both arms for SMOKE_STEPS optimizer steps (default 1200, ~3 epochs at batch 128),
# with test=False, under a separate muon_smoke_<stamp> tree. It caps steps rather than lowering
# max_epochs, which would rescale the piecewise LR schedule (warmup_piecewise_lr refuses that).
# For reference at 1200 steps, val/acc is 0.227 for the l = 20 recipe and 0.248 for SpecReg at
# l = 7. On Acevedo, Muon wd=0 sat at chance for the first few epochs before training, so a slow
# wd=0 smoke is not by itself a failure.
#
#   scripts/tmux/cifar100_muon.sh --smoke   # ~3-epoch check of both arms
#   scripts/tmux/cifar100_muon.sh           # full 250-epoch study
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="muon"
MODE="full"
for arg in "$@"; do [[ "${arg}" == "--smoke" ]] && MODE="smoke"; done
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
SMOKE_STEPS="${SMOKE_STEPS:-1200}"

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${CIFAR_MUON_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export CIFAR_MUON_STAMP="${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
TAG="${STUDY}_${STAMP}"; [[ "${MODE}" == "smoke" ]] && TAG="${STUDY}_smoke_${STAMP}"
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_${TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one. Every variable the child reads
# goes through -e: the tmux server does not inherit this shell's environment.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="cifar100_${TAG}"
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="--run --smoke"
  tmux new-session -d -s "${SESSION}" -e "CIFAR_MUON_STAMP=${STAMP}" -e "GPUS=${GPU_LIST[*]}" \
      -e "SMOKE_STEPS=${SMOKE_STEPS}" "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE})."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  echo "  Run dirs: ${RUNS}/<label>"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=1 "++trainer.max_steps=${SMOKE_STEPS}" test=False
          "logger.wandb.group=CIFAR100_${STUDY}_smoke")
else
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=250 test=True "logger.wandb.group=CIFAR100_${TAG}")
fi
COMMON=(experiment=sngp_muon_cifar100 logger.wandb.log_model=false "${BUDGET[@]}")

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

echo "=== ${STUDY} (${MODE}) -- sngp_muon_cifar100 (l = 7, no SN), seed 12345, Muon wd 0 / 0.1 ==="
run muon_wd0_s12345    "${GPU_LIST[0]}" seed=12345 model.optimizer.weight_decay=0.0 &
run muon_wd0.1_s12345  "${GPU_LIST[1]}" seed=12345 &
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/{last,best}.ckpt"
