#!/bin/bash
# Stage the CIFAR-100 benchmark's inference CSVs into the per-method / one-CSV-per-dataset
# layout that src/paper_helpers/ood_metrics/runner.py expects.
#
# infer.py writes  <infer>/cifar100_<arm>__<dataset>/predictions.csv
# the runner wants  csv/ood_metrics/cifar100_<arm>/<dataset>.csv
#
# Symlinks, not copies: these CSVs are ~10-26k rows each and the inference tree is the
# source of truth. Re-runnable; existing links are replaced.
#
#   scripts/metrics/stage_cifar100_ood.sh
#   uv run python -m src.paper_helpers.ood_metrics.cifar100
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example}"
: "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example}"

INFER_ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}/infer"
STAGING="csv/ood_metrics"

ARMS=(baseline sngp specreg specreg_literal)
DATASETS=(cifar100 cifar10 svhn)

missing=0
for arm in "${ARMS[@]}"; do
  dest="${STAGING}/cifar100_${arm}"
  mkdir -p "${dest}"
  for ds in "${DATASETS[@]}"; do
    src="${INFER_ROOT}/cifar100_${arm}__${ds}/predictions.csv"
    if [[ ! -f "${src}" ]]; then
      echo "MISSING ${src}" >&2
      missing=1
      continue
    fi
    ln -sfn "${src}" "${dest}/${ds}.csv"
  done
done

if (( missing )); then
  echo "Some prediction CSVs are missing -- run inference first (docs/models/CIFAR100_BENCHMARK.md)." >&2
  exit 1
fi

echo "Staged ${#ARMS[@]} methods x ${#DATASETS[@]} datasets under ${STAGING}/"
