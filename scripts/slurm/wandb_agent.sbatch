#!/bin/bash
#SBATCH --job-name=wandb_agent
#SBATCH --output=slurm_logs/wandb_agent_%A_%a.out
#SBATCH --error=slurm_logs/wandb_agent_%A_%a.err
#SBATCH --time=03:00:00            # per array task (1 trial/task by default); sweep.py can override with --time
#SBATCH --partition=shared
#SBATCH --gres=gpu:nvidia_l40s:1   # typed GPU: comparable wall-clock across trials (as the old hpo/sweep.sh did)
#SBATCH --cpus-per-task=8          # == data.datamodule.num_workers in configs/hparams_search/<family>.yaml
#SBATCH --mem=32G                  # cpus/mem are ONE agent's share: sweep.py scales both by --agents-per-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# --array=0-N%P is passed on the sbatch command line by scripts/hpo/sweep.py
# (N = trials/count - 1, P = --parallel), so it is not hard-coded here.
#
# One W&B sweep agent per array task. The agent pulls `count` trials from the sweep
# controller and runs each as an ordinary `src/train.py` single run (the `command:` in
# configs/hparams_search/wandb/<family>.yaml). Surplus tasks past the sweep's run_cap
# exit immediately.
#
# AGENTS (3rd arg, default 1) concurrent agents share the task's single GPU -- for trials
# that are data-loader bound and leave most of an L40S idle. Each is its own `wandb agent`
# process, so trials stay isolated (own W&B run, own sweeps/<id>/<ts>_<run id> dir).
set -euo pipefail

SWEEP_PATH="${1:?Usage: sbatch --array=0-N%P $0 <entity/project/sweep_id> [count] [agents]}"
COUNT="${2:-1}"
AGENTS="${3:-1}"

# Repo root: the sweep command runs `src/train.py` relative to it. NOT derivable from
# $0 here -- sbatch copies this script to a node-local spool dir, so under SLURM $0 is
# /var/spool/slurmd/job<id>/slurm_script and "$(dirname "$0")/../.." is /var/spool.
# sbatch starts the job in the submission directory and exports it as SLURM_SUBMIT_DIR;
# the $0 fallback is for running this script directly, outside SLURM.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
[[ -f .project-root ]] || { echo "not the repo root: ${REPO_ROOT} -- sbatch this script from the repo root" >&2; exit 1; }
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

# One BLAS / OpenMP thread per process. albumentations' HEStain does numpy linear algebra in
# every DataLoader worker, and this numpy's OpenBLAS defaults to up to 64 threads per process:
# 4 trials x 8 workers pinned all 192 cores and left the GPUs at 0% (2026-10-04 pilot). With
# the cap a resnet18 Wong trial runs ~11 it/s at batch 128.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "agent task ${SLURM_ARRAY_TASK_ID:-?} of job ${SLURM_ARRAY_JOB_ID:-?} on $(hostname) at $(date)"
echo "sweep ${SWEEP_PATH}, ${AGENTS} agent(s) x ${COUNT} trial(s)"

# --no-sync: scripts/hpo/sweep.py ran `uv sync --frozen` once before submitting, and
# concurrent agents must not race to sync .venv.
pids=()
for ((i = 0; i < AGENTS; i++)); do
  uv run --no-sync wandb agent --count "${COUNT}" "${SWEEP_PATH}" &
  pids+=("$!")
  sleep 5 # stagger: concurrent HF dataset loads / CUDA init
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done

echo "agents finished at $(date) (exit ${status})"
exit "${status}"
