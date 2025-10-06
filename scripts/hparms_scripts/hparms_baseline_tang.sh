#!/bin/bash
set -euo pipefail

uv run python src/train.py -m \
  hparams_search=baseline_tang \
  experiment=exp_baseline_tang \
  hydra/launcher=submitit_slurm \
  hydra.sweeper.n_jobs=2 \
  hydra.launcher.partition=shared \
  hydra.launcher.gres="gpu:nvidia_l40s:1" \
  hydra.launcher.gpus_per_node=null \
  hydra.launcher.nodes=1 \
  hydra.launcher.cpus_per_task=8 \
  hydra.launcher.mem_gb=32 \
  hydra.launcher.timeout_min=120 \
  hydra.launcher.name=optuna_baseline_tang
