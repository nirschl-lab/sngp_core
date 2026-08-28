#!/bin/bash
# Launch a single training run (any configs/experiment/*.yaml) in a detached tmux
# session -- for long training runs on a shared interactive box without SLURM (see
# scripts/slurm/train.sh for the SLURM equivalent).
#
# Usage:
#   scripts/tmux/train_experiment.sh <experiment> [extra hydra overrides...]
#   scripts/tmux/train_experiment.sh baseline_wong
#   scripts/tmux/train_experiment.sh sngp_tang trainer.max_epochs=100 seed=7
#
# Attach:  tmux attach -t <session>   (session name printed below)
# Detach:  Ctrl-b d
# Tail log without attaching: tail -f <log path printed below>
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment> [extra hydra overrides...]}"
shift
EXTRA_OVERRIDES=("$@")

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
SESSION="train_${EXPERIMENT}_${RUN_ID}"
LOG_DIR="${EXPERIMENTS_HOME}/${PROJECT_NAME}/tmux_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

# Build one shell command string for the tmux pane. Each extra override is
# individually %q-quoted so overrides containing spaces/quotes (e.g.
# `data.datamodule.institution=ucdavis`) survive being flattened into one string.
CMD="uv run python src/train.py experiment=${EXPERIMENT}"
for override in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
  CMD+=" $(printf '%q' "${override}")"
done
CMD+=" 2>&1 | tee ${LOG_FILE}"

tmux new-session -d -s "${SESSION}" "${CMD}"

echo "Launched '${EXPERIMENT}' in tmux session '${SESSION}'."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Log:     tail -f ${LOG_FILE}"
