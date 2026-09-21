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
# Two checkpoint selections, staged under distinct prefixes. `cifar100last_*` (epoch
# 249) is the one the arms are compared at; `cifar100_*` (best.ckpt) is kept for the
# reference-only table. The literal arm has no last.ckpt -- it was lost to the
# run-directory collision -- so a missing dir there is expected, not an error.
PREFIXES=(cifar100 cifar100last)

missing=0
staged=0
for prefix in "${PREFIXES[@]}"; do
  for arm in "${ARMS[@]}"; do
    dest="${STAGING}/${prefix}_${arm}"
    for ds in "${DATASETS[@]}"; do
      src="${INFER_ROOT}/${prefix}_${arm}__${ds}/predictions.csv"
      if [[ ! -f "${src}" ]]; then
        if [[ "${prefix}" == "cifar100last" && "${arm}" == "specreg_literal" ]]; then
          continue  # known-absent, see above
        fi
        echo "MISSING ${src}" >&2
        missing=1
        continue
      fi
      mkdir -p "${dest}"
      ln -sfn "${src}" "${dest}/${ds}.csv"
      staged=$((staged + 1))
    done
  done
done

if (( missing )); then
  echo "Some prediction CSVs are missing -- run inference first (docs/models/CIFAR100_BENCHMARK.md)." >&2
  exit 1
fi

echo "Staged ${staged} CSVs under ${STAGING}/"
