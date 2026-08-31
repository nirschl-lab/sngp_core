#!/bin/bash
# Run src/inference/infer.py against every dataset for a single checkpoint.
#
# Usage:
#   scripts/inference/run_all_datasets.sh /absolute/path/to/model.ckpt
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <ckpt_path>" >&2
    exit 1
fi

CKPT_PATH="$1"

DATASETS=(
    acevedo
    wong
    tang
    kather2018
    kather2016
    jung
    nirschl2018
)

for DATA in "${DATASETS[@]}"; do
    echo "=== Running inference on ${DATA} ==="
    uv run src/inference/infer.py \
        ckpt_path="${CKPT_PATH}" \
        data="${DATA}"
done
