#!/bin/bash
# Launch one hyperparameter-search sweep on SLURM via hydra-optuna-sweeper.
#
# Usage:
#   scripts/hpo/sweep.sh <family> <dataset>
#   scripts/hpo/sweep.sh baseline tang
#   scripts/hpo/sweep.sh sngp     acevedo
#
# <family> selects configs/hparams_search/<family>.yaml (baseline | sngp).
# <dataset> selects configs/experiment/<family>_<dataset>.yaml, so it must be one of
# the datasets that experiment file exists for: tang, acevedo, wong, kather2018,
# wong_ucdavis, wong_upitt, wong_utsouthwestern.
#
# See docs/HPO_GUIDE.md for the full protocol (search space, budget, selection
# metric). Deep Ensemble is deliberately not swept here -- it inherits the tuned
# baseline config (see docs/HPO_GUIDE.md).
set -euo pipefail

FAMILY="${1:?Usage: $0 <baseline|sngp> <tang|acevedo|wong|kather2018|wong_ucdavis|wong_upitt|wong_utsouthwestern>}"
DATASET="${2:?Usage: $0 <baseline|sngp> <tang|acevedo|wong|kather2018|wong_ucdavis|wong_upitt|wong_utsouthwestern>}"
EXPERIMENT="${FAMILY}_${DATASET}"

cd "$(dirname "$0")/../.."  # repo root

if [[ ! -f "configs/hparams_search/${FAMILY}.yaml" ]]; then
  echo "No configs/hparams_search/${FAMILY}.yaml -- family must be 'baseline' or 'sngp'." >&2
  exit 1
fi
if [[ ! -f "configs/experiment/${EXPERIMENT}.yaml" ]]; then
  echo "No configs/experiment/${EXPERIMENT}.yaml for dataset '${DATASET}'." >&2
  exit 1
fi

# .env carries EXPERIMENTS_HOME/PROJECT_NAME (see env_example). train.py loads it
# itself via rootutils, but this script needs the values in the shell too, to
# pre-create the Optuna sqlite storage directory below -- sqlite does not create
# missing parent directories on its own, it just fails with "unable to open database
# file".
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example}"
: "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example}"

mkdir -p "${EXPERIMENTS_HOME}/${PROJECT_NAME}/optuna"

# Only override what's genuinely different from configs/hydra/launcher/submitit_slurm.yaml's
# defaults for an HPO sweep specifically: a typed GPU (all trials should run on the
# same model for comparable wall-clock timing) and a short timeout (proxy-budget
# trials are far shorter than the 24h default sized for a full production run).
uv run python src/train.py -m \
  hparams_search="${FAMILY}" \
  experiment="${EXPERIMENT}" \
  hydra/launcher=submitit_slurm \
  hydra.launcher.gres="gpu:nvidia_l40s:1" \
  hydra.launcher.timeout_min=180 \
  hydra.launcher.name="optuna_${EXPERIMENT}"
