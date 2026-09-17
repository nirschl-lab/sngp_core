#!/bin/bash
# Launch scripts/ablation/train_ablation_parallel.sh in a detached tmux session --
# that script backgrounds one `train.py -m` lane per GPU and waits for all of them,
# so wrapping the orchestrator in tmux lets you detach/reattach to the parent while
# the arms train. Same relationship as scripts/tmux/deep_ensemble.sh ->
# scripts/ensemble/train_members_parallel.sh.
#
# Usage:
#   scripts/tmux/ablation.sh <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]
#   scripts/tmux/ablation.sh sngp_acevedo model.net.spectral_norm_bound null,0.9,1.0,2.0 1 2 3
#
# Env vars (same as the orchestrator, forwarded explicitly below -- a new tmux session
# does NOT reliably inherit the calling shell's exported vars when the tmux server is
# already running, verified directly):
#   NAME_PREFIX="<str>"       -- per-value W&B run name prefix
#   WANDB_GROUP="<str>"       -- W&B group for the whole ablation
#   EXTRA_OVERRIDES="<...>"   -- extra Hydra overrides forwarded to every lane
#   DRY_RUN=1                 -- print the lane plan and exit before launching anything
#
# Attach:  tmux attach -t <session>   (session name printed below)
# Detach:  Ctrl-b d
# Tail log without attaching: tail -f <log path printed below>
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]}"
KEY="${2:?Usage: $0 <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]}"
VALUES_CSV="${3:?Usage: $0 <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]}"
shift 3
GPU_IDS=("$@")
NAME_PREFIX="${NAME_PREFIX:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

cd "$(dirname "$0")/../.."  # repo root

if [[ ! -f "configs/experiment/${EXPERIMENT}.yaml" ]]; then
  echo "No configs/experiment/${EXPERIMENT}.yaml." >&2
  exit 1
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed." >&2; exit 1; }

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example}"
: "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example}"

LEAF="${KEY##*.}"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
SESSION="ablate_${EXPERIMENT}_${LEAF}_${RUN_ID}"
LOG_DIR="${EXPERIMENTS_HOME}/${PROJECT_NAME}/tmux_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

# Only forward the env vars that were actually set -- an empty NAME_PREFIX/WANDB_GROUP
# would defeat the orchestrator's own `${VAR:-<default>}` fallbacks.
CMD="env"
[[ -n "${NAME_PREFIX}" ]] && CMD+=" NAME_PREFIX=$(printf '%q' "${NAME_PREFIX}")"
[[ -n "${WANDB_GROUP}" ]] && CMD+=" WANDB_GROUP=$(printf '%q' "${WANDB_GROUP}")"
CMD+=" EXTRA_OVERRIDES=$(printf '%q' "${EXTRA_OVERRIDES}") DRY_RUN=$(printf '%q' "${DRY_RUN}")"
CMD+=" scripts/ablation/train_ablation_parallel.sh $(printf '%q' "${EXPERIMENT}") $(printf '%q' "${KEY}") $(printf '%q' "${VALUES_CSV}")"
for gpu_id in "${GPU_IDS[@]+"${GPU_IDS[@]}"}"; do
  CMD+=" ${gpu_id}"
done
CMD+=" 2>&1 | tee ${LOG_FILE}"

tmux new-session -d -s "${SESSION}" "${CMD}"

echo "Launched ${KEY} ablation over [${VALUES_CSV}] for '${EXPERIMENT}' in tmux session '${SESSION}'."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Log:     tail -f ${LOG_FILE}"
