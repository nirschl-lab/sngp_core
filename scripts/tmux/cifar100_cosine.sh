#!/bin/bash
# Study `cosine`: SNGP, SNGP + SpecReg and the Muon GP head on CIFAR-100 / WRN-28-10, all on the
# Acevedo LR schedule (CosineAnnealingLR to 0 over 250 epochs, per epoch, no warmup) and all at
# l = 7. Follow-up of docs/results/CIFAR100_MUON_RESULTS.md, where the piecewise-schedule Muon rows
# trail every SGD row in-distribution; the experiment headers have the reasoning.
#
# One seed (12345), one arm per GPU, ~2.7-3 h:
#   cos_sngp_l7_s12345        experiment=sngp_cifar100_cosine           spectral norm c = 6.0, SGD
#   cos_specreg_l7_s12345     experiment=sngp_specreg_cifar100_cosine   spectral penalty, SGD
#   cos_muon_wd0.1_s12345     experiment=sngp_muon_cifar100_cosine      no SN, Muon wd 0.1, AdamW aux
#   cos_muon_wd0_s12345       experiment=sngp_muon_cifar100_cosine      no SN, Muon wd 0, AdamW aux
#                                                                       (aux keeps its 0.01)
#   cos_muonsgd_wd0.1_s12345  experiment=sngp_muon_sgd_cifar100_cosine  no SN, Muon wd 0.1, SGD aux
#   cos_muonsgd_wd0_s12345    experiment=sngp_muon_sgd_cifar100_cosine  no SN, Muon wd 0, SGD aux
#
# In cosine_2026-10-01_10-17-06, cos_muon_wd0_s12345 was stopped at epoch 152: the AdamW aux group
# let the final BN gamma and the GP output layer drift (see sngp_muon_sgd_cifar100_cosine.yaml),
# and the two muonsgd arms replaced it.
#
# ARMS="label ..." runs a subset. With CIFAR_COSINE_STAMP=<stamp> of an earlier launch it adds those
# arms to that launch's tree and W&B group (own tmux session and driver log); that is how
# cos_muon_wd0_s12345 joined cosine_2026-10-01_10-17-06 after the first three had started.
#
# GPUs: the first idle ones (< 1 GiB in use), one per arm, unless GPUS="a b ..." is given.
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
#   CIFAR_COSINE_STAMP=2026-10-01_10-17-06 ARMS=cos_muon_wd0_s12345 scripts/tmux/cifar100_cosine.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="cosine"
declare -A ARM_OVERRIDES=(
  [cos_sngp_l7_s12345]="experiment=sngp_cifar100_cosine"
  [cos_specreg_l7_s12345]="experiment=sngp_specreg_cifar100_cosine"
  [cos_muon_wd0.1_s12345]="experiment=sngp_muon_cifar100_cosine"
  [cos_muon_wd0_s12345]="experiment=sngp_muon_cifar100_cosine model.optimizer.weight_decay=0.0"
  [cos_muonsgd_wd0.1_s12345]="experiment=sngp_muon_sgd_cifar100_cosine"
  [cos_muonsgd_wd0_s12345]="experiment=sngp_muon_sgd_cifar100_cosine model.optimizer.weight_decay=0.0"
)
ALL_ARMS="cos_sngp_l7_s12345 cos_specreg_l7_s12345 cos_muon_wd0.1_s12345 cos_muon_wd0_s12345"
ALL_ARMS+=" cos_muonsgd_wd0.1_s12345 cos_muonsgd_wd0_s12345"
read -r -a ARM_LIST <<< "${ARMS:-${ALL_ARMS}}"
for label in "${ARM_LIST[@]}"; do
  [[ -n "${ARM_OVERRIDES[${label}]:-}" ]] || { echo "Unknown arm '${label}'." >&2; exit 1; }
done
N_ARMS=${#ARM_LIST[@]}

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
  echo "Need ${N_ARMS} idle GPUs, found ${#GPU_LIST[@]} (${GPUS:-none}); set GPUS=\"a b ...\"." >&2
  exit 1
fi
SMOKE_STEPS="${SMOKE_STEPS:-1200}"

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${CIFAR_COSINE_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export CIFAR_COSINE_STAMP="${STAMP}"
# A subset launch gets its own session name and driver log, so it can join an earlier stamp.
SUFFIX="${CIFAR_COSINE_SUFFIX:-}"
[[ -z "${SUFFIX}" && -n "${ARMS:-}" ]] && SUFFIX="_add_$(date +%H-%M-%S)"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
TAG="${STUDY}_${STAMP}"; [[ "${MODE}" == "smoke" ]] && TAG="${STUDY}_smoke_${STAMP}"
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_${TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one. Every variable the child reads
# goes through -e: the tmux server does not inherit this shell's environment.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="cifar100_${TAG}${SUFFIX}"
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="${CHILD_ARGS} --smoke"
  tmux new-session -d -s "${SESSION}" -e "CIFAR_COSINE_STAMP=${STAMP}" -e "GPUS=${GPU_LIST[*]}" \
      -e "SMOKE_STEPS=${SMOKE_STEPS}" -e "ARMS=${ARM_LIST[*]}" -e "CIFAR_COSINE_SUFFIX=${SUFFIX}" \
      "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver${SUFFIX}.log'"
  echo "Launched '${SESSION}' (${MODE}) on GPUs ${GPU_LIST[*]:0:${N_ARMS}}: ${ARM_LIST[*]}."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver${SUFFIX}.log"
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

echo "=== ${STUDY} (${MODE}) -- cosine LR, l = 7, seed 12345: ${ARM_LIST[*]} ==="
for i in "${!ARM_LIST[@]}"; do
  label="${ARM_LIST[$i]}"
  # shellcheck disable=SC2086  # the override string is a space-separated list of hydra args
  run "${label}" "${GPU_LIST[$i]}" ${ARM_OVERRIDES[${label}]} &
done
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/{last,best}.ckpt"
