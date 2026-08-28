#!/bin/bash
# Launch scripts/ensemble/train_members_parallel.sh in a detached tmux session --
# that script itself backgrounds N members and waits for all of them, so wrapping the
# whole orchestrator in tmux lets you detach/reattach to the parent while members
# train. See scripts/slurm/train_ensemble_members.sh for the SLURM equivalent (no
# tmux needed there since the sbatch job itself is already detached).
#
# Usage:
#   scripts/tmux/deep_ensemble.sh <baseline_experiment> <num_estimators> [gpu_id ...]
#   scripts/tmux/deep_ensemble.sh baseline_acevedo 5
#   scripts/tmux/deep_ensemble.sh baseline_acevedo 3 0 2 3    # pin to GPUs 0,2,3
#
# Env vars (same as train_members_parallel.sh, forwarded explicitly below -- a new
# tmux session does NOT reliably inherit the calling shell's exported vars when the
# tmux server is already running, verified directly):
#   MEMBER_EXTRA_OVERRIDES="<hydra overrides>"
#   DRY_RUN=1  -- print the packing plan and exit before launching anything.
#
# Attach:  tmux attach -t <session>   (session name printed below)
# Detach:  Ctrl-b d
# Tail log without attaching: tail -f <log path printed below>
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <baseline_experiment> <num_estimators> [gpu_id ...]}"
NUM_ESTIMATORS="${2:?Usage: $0 <baseline_experiment> <num_estimators> [gpu_id ...]}"
shift 2
GPU_IDS=("$@")
MEMBER_EXTRA_OVERRIDES="${MEMBER_EXTRA_OVERRIDES:-}"
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

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
SESSION="ensemble_${EXPERIMENT}_${RUN_ID}"
LOG_DIR="${EXPERIMENTS_HOME}/${PROJECT_NAME}/tmux_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

CMD="env MEMBER_EXTRA_OVERRIDES=$(printf '%q' "${MEMBER_EXTRA_OVERRIDES}") DRY_RUN=$(printf '%q' "${DRY_RUN}")"
CMD+=" scripts/ensemble/train_members_parallel.sh ${EXPERIMENT} ${NUM_ESTIMATORS}"
for gpu_id in "${GPU_IDS[@]+"${GPU_IDS[@]}"}"; do
  CMD+=" ${gpu_id}"
done
CMD+=" 2>&1 | tee ${LOG_FILE}"

tmux new-session -d -s "${SESSION}" "${CMD}"

echo "Launched '${EXPERIMENT}' (${NUM_ESTIMATORS} members) in tmux session '${SESSION}'."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Log:     tail -f ${LOG_FILE}"
