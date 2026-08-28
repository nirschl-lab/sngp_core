#!/bin/bash
# Launch scripts/hpo/sweep.sh in a detached tmux session. The sweep itself blocks on
# `python src/train.py -m ...` while hydra-optuna-sweeper fans trials out to SLURM
# (via submitit) and polls until n_trials complete -- this just gives the machine
# kicking the sweep off a durable, reattachable session for that wait. The trials
# themselves still run on SLURM regardless; see scripts/hpo/sweep.sh.
#
# Usage:
#   scripts/tmux/hpo_sweep.sh <baseline|sngp> <tang|acevedo|wong|kather2018>
#   scripts/tmux/hpo_sweep.sh baseline tang
#
# Attach:  tmux attach -t <session>   (session name printed below)
# Detach:  Ctrl-b d
# Tail log without attaching: tail -f <log path printed below>
set -euo pipefail

FAMILY="${1:?Usage: $0 <baseline|sngp> <tang|acevedo|wong|kather2018>}"
DATASET="${2:?Usage: $0 <baseline|sngp> <tang|acevedo|wong|kather2018>}"

cd "$(dirname "$0")/../.."  # repo root

if [[ ! -f "configs/hparams_search/${FAMILY}.yaml" ]]; then
  echo "No configs/hparams_search/${FAMILY}.yaml -- family must be 'baseline' or 'sngp'." >&2
  exit 1
fi
if [[ ! -f "configs/experiment/${FAMILY}_${DATASET}.yaml" ]]; then
  echo "No configs/experiment/${FAMILY}_${DATASET}.yaml for dataset '${DATASET}'." >&2
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
SESSION="hpo_${FAMILY}_${DATASET}_${RUN_ID}"
LOG_DIR="${EXPERIMENTS_HOME}/${PROJECT_NAME}/tmux_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

tmux new-session -d -s "${SESSION}" \
  "scripts/hpo/sweep.sh ${FAMILY} ${DATASET} 2>&1 | tee ${LOG_FILE}"

echo "Launched '${FAMILY} ${DATASET}' HPO sweep in tmux session '${SESSION}'."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Log:     tail -f ${LOG_FILE}"
