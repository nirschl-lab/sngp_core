#!/bin/bash
# Study `cosine`: SNGP, SNGP + SpecReg and the Muon GP head on CIFAR-100 / WRN-28-10, all on the
# Acevedo LR schedule (CosineAnnealingLR to 0 over 250 epochs, per epoch, no warmup) and all at
# l = 7. Follow-up of docs/results/CIFAR100_MUON_RESULTS.md, where the piecewise-schedule Muon rows
# trail every SGD row in-distribution; the experiment headers have the reasoning.
#
# One seed (12345), three arms, one per GPU, ~2.7-3 h:
#   cos_sngp_l7_s12345      experiment=sngp_cifar100_cosine          spectral norm c = 6.0, SGD
#   cos_specreg_l7_s12345   experiment=sngp_specreg_cifar100_cosine  spectral penalty, SGD
#   cos_muon_wd0.1_s12345   experiment=sngp_muon_cifar100_cosine     no SN, Muon wd 0.1
#
# GPUs: the first three idle ones (< 1 GiB in use), unless GPUS="a b c" is given.
#
# TRAINING ONLY. Evaluation follows the benchmark protocol: last.ckpt (epoch 249), with
# mean_field_factor fit on val before any comparison (training pins it at 7.5). Every run gets an
# EXPLICIT hydra run dir.
#
# --smoke trains all arms for SMOKE_STEPS optimizer steps (default 1200, ~3 epochs at batch 128),
# with test=False, under a separate cosine_smoke_<stamp> tree. It caps steps rather than lowering
# max_epochs, which would rescale the cosine schedule.
#
#   scripts/tmux/cifar100_cosine.sh --smoke   # ~3-epoch check of all arms
#   scripts/tmux/cifar100_cosine.sh           # full 250-epoch study
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="cosine"; N_ARMS=3
MODE="full"
for arg in "$@"; do
  [[ "${arg}" == "--smoke" ]] && MODE="smoke"
done
if [[ -z "${GPUS:-}" ]]; then
  GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | awk -F', ' '$2 < 1024 {print $1}' | head -n "${N_ARMS}" | tr '\n' ' ')"
fi
read -r -a GPU_LIST <<< "${GPUS}"
if (( ${#GPU_LIST[@]} < N_ARMS )); then
  echo "Need ${N_ARMS} idle GPUs, found ${#GPU_LIST[@]} (${GPUS:-none}); set GPUS=\"a b c\"." >&2
  exit 1
fi
SMOKE_STEPS="${SMOKE_STEPS:-1200}"

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${CIFAR_COSINE_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export CIFAR_COSINE_STAMP="${STAMP}"
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
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="${CHILD_ARGS} --smoke"
  tmux new-session -d -s "${SESSION}" -e "CIFAR_COSINE_STAMP=${STAMP}" -e "GPUS=${GPU_LIST[*]}" \
      -e "SMOKE_STEPS=${SMOKE_STEPS}" "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE}) on GPUs ${GPU_LIST[*]:0:${N_ARMS}}."
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
COMMON=(logger.wandb.log_model=false seed=12345 "${BUDGET[@]}")

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

echo "=== ${STUDY} (${MODE}) -- cosine LR, l = 7, seed 12345: SNGP / SpecReg / Muon wd 0.1 ==="
run cos_sngp_l7_s12345     "${GPU_LIST[0]}" experiment=sngp_cifar100_cosine &
run cos_specreg_l7_s12345  "${GPU_LIST[1]}" experiment=sngp_specreg_cifar100_cosine &
run cos_muon_wd0.1_s12345  "${GPU_LIST[2]}" experiment=sngp_muon_cifar100_cosine &
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/{last,best}.ckpt"
