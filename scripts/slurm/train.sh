#!/bin/bash
#SBATCH --job-name=wong_drop0.2
#SBATCH --output=slurm_logs/wong_drop0.2_%j.out
#SBATCH --error=slurm_logs/wong_drop0.2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu0"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Exit immediately if a command exits with a non-zero status
set -e

# Load any required modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.9

# --------- Script ---------
# Create logs directory if it doesn't exist
mkdir -p slurm_logs

echo "Starting job $SLURM_JOB_ID on $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Run the training (remove CUDA_VISIBLE_DEVICES since SLURM handles GPU allocation)
uv run src/train.py experiment=mcdropout_wong

echo "Job completed on $(date)"