#!/bin/bash
# Launch one hyperparameter-search sweep (W&B Sweeps + SLURM agents).
#
# Thin wrapper kept for the documented entry point; the logic lives in
# scripts/hpo/sweep.py (see its --help). Examples:
#   scripts/hpo/sweep.sh baseline tang
#   scripts/hpo/sweep.sh sngp     acevedo --trials 48 --parallel 8
#   scripts/hpo/sweep.sh baseline tang --trials 2 --parallel 2 --no-register \
#       --override trainer.max_epochs=2 --override +trainer.limit_train_batches=0.05   # pilot
#
# <family> selects configs/hparams_search/wandb/<family>.yaml + configs/hparams_search/<family>.yaml
# (baseline | sngp); <dataset> selects configs/experiment/<family>_<dataset>.yaml. Deep
# Ensemble is deliberately not swept -- it inherits the tuned baseline config
# (docs/HPO_GUIDE.md).
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
exec uv run scripts/hpo/sweep.py "$@"
