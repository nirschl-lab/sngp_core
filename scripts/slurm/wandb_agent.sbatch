#!/bin/bash
#SBATCH --job-name=wandb_agent
#SBATCH --output=slurm_logs/wandb_agent_%A_%a.out
#SBATCH --error=slurm_logs/wandb_agent_%A_%a.err
#SBATCH --time=03:00:00            # per array task (1 trial/task by default); sweep.py can override with --time
#SBATCH --partition=shared
#SBATCH --gres=gpu:nvidia_l40s:1   # typed GPU: comparable wall-clock across trials (as the old hpo/sweep.sh did)
#SBATCH --cpus-per-task=8          # == data.datamodule.num_workers in configs/hparams_search/<family>.yaml
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# --array=0-N%P is passed on the sbatch command line by scripts/hpo/sweep.py
# (N = trials/count - 1, P = --parallel), so it is not hard-coded here.
#
# One W&B sweep agent per array task. The agent pulls `count` trials from the sweep
# controller and runs each as an ordinary `src/train.py` single run (the `command:` in
# configs/hparams_search/wandb/<family>.yaml). Surplus tasks past the sweep's run_cap
# exit immediately.
set -euo pipefail

SWEEP_PATH="${1:?Usage: sbatch --array=0-N%P $0 <entity/project/sweep_id> [count]}"
COUNT="${2:-1}"

cd "$(dirname "$0")/../.."  # repo root: the sweep command runs `src/train.py` relative to it
mkdir -p slurm_logs

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example}"
: "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example}"

# The agent writes wandb/sweep-<id>/config-<run>.yaml under $WANDB_DIR (cwd if unset) --
# keep that off the repo root. The run files themselves go to WandbLogger's save_dir.
export WANDB_DIR="${EXPERIMENTS_HOME}/${PROJECT_NAME}/wandb_agent"
mkdir -p "${WANDB_DIR}"

echo "agent task ${SLURM_ARRAY_TASK_ID:-?} of job ${SLURM_ARRAY_JOB_ID:-?} on $(hostname) at $(date)"
echo "sweep ${SWEEP_PATH}, ${COUNT} trial(s)"

# --no-sync: scripts/hpo/sweep.py ran `uv sync --frozen` once before submitting, and
# concurrent agents must not race to sync .venv.
uv run --no-sync wandb agent --count "${COUNT}" "${SWEEP_PATH}"

echo "agent finished at $(date)"
