#!/bin/bash
#SBATCH --job-name=ensemble_members
#SBATCH --output=slurm_logs/ensemble_members_%j.out
#SBATCH --error=slurm_logs/ensemble_members_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Train N Deep Ensemble members as independent, round-robin-packed parallel runs
# (scripts/ensemble/train_members_parallel.sh) inside a single SLURM allocation --
# the SLURM-only-access counterpart to running that script directly over SSH.
# Unlike scripts/slurm/train.sh (one hardcoded experiment/GPU), this script is
# generic: <baseline_experiment>/<num_estimators> are positional args, mirroring
# scripts/hpo/sweep.sh's <family> <dataset> pattern.
#
# The #SBATCH directives above are parsed literally by sbatch before this script
# runs -- they can NOT reference the positional args below. The defaults request
# this project's usual 4-GPU box, with cpus/mem scaled 1:1 from
# scripts/slurm/train.sh's single-GPU convention (8 cpus + 32G per GPU). Override
# on the sbatch command line if your situation differs (fewer GPUs, a
# multi-tenant node, a shorter budget) -- CLI flags take precedence over the
# #SBATCH defaults below, the same pattern scripts/hpo/sweep.sh uses for
# hydra.launcher.gres=... overrides:
#   sbatch --gres=gpu:2 --cpus-per-task=16 --mem=64G \
#     scripts/slurm/train_ensemble_members.sh baseline_acevedo 6
#
# Usage:
#   sbatch scripts/slurm/train_ensemble_members.sh <baseline_experiment> <num_estimators>
#   sbatch scripts/slurm/train_ensemble_members.sh baseline_acevedo 5
#
# Optional: forward extra per-member Hydra overrides (e.g. to shrink dataloader
# worker counts so K packed members/GPU don't oversubscribe this job's CPUs --
# see train_members_parallel.sh's MEMBER_EXTRA_OVERRIDES). sbatch's default
# --export=ALL already propagates exported shell vars into the job:
#   export MEMBER_EXTRA_OVERRIDES="data.datamodule.num_workers=8"
#   sbatch scripts/slurm/train_ensemble_members.sh baseline_acevedo 6
set -euo pipefail

EXPERIMENT="${1:?Usage: sbatch $0 <baseline_experiment> <num_estimators>}"
NUM_ESTIMATORS="${2:?Usage: sbatch $0 <baseline_experiment> <num_estimators>}"

cd "$(dirname "$0")/../.."  # repo root
mkdir -p slurm_logs

echo "Starting job ${SLURM_JOB_ID:-<no-slurm>} on $(date)"
echo "Running on node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs visible to this job: $(nvidia-smi -L | wc -l) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"

# No GPU ids passed -- train_members_parallel.sh auto-detects and round-robin
# packs members across exactly the GPUs SLURM allocated to this job (it prefers
# an already-set CUDA_VISIBLE_DEVICES, which is what SLURM's gres/gpu plugin
# sets for this job's allocation, falling back to `nvidia-smi -L` only if unset).
scripts/ensemble/train_members_parallel.sh "${EXPERIMENT}" "${NUM_ESTIMATORS}"

echo "Job completed on $(date)"
