#!/bin/bash
# Study `muon`: the SNGP random-feature GP head on an unconstrained Acevedo ResNet-18 (no
# spectral normalization, no spectral regularization), trained with Muon -- does Muon's
# spectrally bounded update stand in for SN / SpecReg? configs/experiment/sngp_muon_acevedo.yaml
# has the reasoning. Three arms, one per GPU (override with GPUS="a b c"), all on the
# sngp_acevedo protocol (150 epochs max, val/nll_cal selection + early stopping):
#
#   muon_wd0      Muon, weight_decay 0   -- no cap on the conv weights' spectral norm
#   muon_wd0.1    Muon, weight_decay 0.1 -- soft cap ~1/wd (the experiment's default)
#   adamw_nosn    AdamW, no SN           -- control: the GP head on an unconstrained backbone
#
# TRAINING ONLY. Each run gets an explicit dir that keeps the standard
# train/<model.name>_acevedo/runs/<id> layout (the two Muon arms share `model.name`).
#
# --smoke trains every arm for SMOKE_STEPS steps (default 300: ~1.6 epochs at 187 steps/epoch,
# so one validation pass) with max_epochs kept at 150 so the cosine schedule is the real one.
# Check driver/run logs for the MuonWithAuxAdamW split line, a falling train/loss, and no
# augmentation-fallback warning.
#
#   scripts/tmux/acevedo_muon.sh --smoke   # quick check of all 3 arms
#   scripts/tmux/acevedo_muon.sh           # full study
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="muon"
MODE="full"
for arg in "$@"; do [[ "${arg}" == "--smoke" ]] && MODE="smoke"; done
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${MUON_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export MUON_STAMP="${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
TAG="${STUDY}_${STAMP}"; [[ "${MODE}" == "smoke" ]] && TAG="${STUDY}_smoke_${STAMP}"
LOGS="${ROOT}/tmux_logs/acevedo_${TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="acevedo_${TAG}"
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="--run --smoke"
  tmux new-session -d -s "${SESSION}" -e "MUON_STAMP=${STAMP}" -e "GPUS=${GPU_LIST[*]}" -e "SMOKE_STEPS=${SMOKE_STEPS:-300}" \
      "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE})."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.min_epochs=1 "++trainer.max_steps=${SMOKE_STEPS:-300}" test=False logger.wandb.log_model=false
          "logger.wandb.group=Acevedo_${STUDY}_smoke")
  RUN_ID="${STAMP}_smoke"
else
  BUDGET=(test=True logger.wandb.log_model=false "logger.wandb.group=Acevedo_${TAG}")
  RUN_ID="${STAMP}"
fi

# run <label> <gpu> <model.name> <hydra overrides...>
run() {
  local label="$1" gpu="$2" model_name="$3"; shift 3
  local dir="${ROOT}/train/${model_name}_acevedo/runs/${RUN_ID}_${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu}) -> ${dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/train.py \
      "$@" "${BUDGET[@]}" "model.name=${model_name}" \
      "name=acevedo_${STUDY}_${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

echo "=== ${STUDY} (${MODE}) -- SNGP GP head, no SN / no SpecReg, Muon vs AdamW ==="
run muon_wd0    "${GPU_LIST[0]}" sngp_muon_classifier experiment=sngp_muon_acevedo model.optimizer.weight_decay=0.0 &
run muon_wd0.1  "${GPU_LIST[1]}" sngp_muon_classifier experiment=sngp_muon_acevedo &
run adamw_nosn  "${GPU_LIST[2]}" sngp_nosn_classifier experiment=sngp_acevedo \
    model.net.use_spectral_norm=false model.net.spectral_norm_bound=null &
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${ROOT}/train/sngp_{muon,nosn}_classifier_acevedo/runs/${RUN_ID}_*/checkpoints/"
