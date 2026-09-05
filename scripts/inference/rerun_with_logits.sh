#!/bin/bash
# Re-run every manifest-tracked inference run (configs/runs/infer_manifest.yaml) so
# predictions.csv gains class_logits/raw_logits -- and, if requested, member_logits
# and a seeded MC-Dropout pass.
#
# derive_default_run_name (src/inference/infer.py) is deterministic from ckpt path +
# flags, so a plain re-run overwrites the existing predictions.csv/metrics.json for
# every run *in place*. This script snapshots the tree first, then runs a safety
# gate on one deterministic checkpoint before anything is overwritten: it re-runs
# that one checkpoint into a scratch directory and requires its metrics.json to come
# back byte-identical to the existing one -- since none of the class_logits/
# raw_logits/run.json changes touch any computed metric, that's a real end-to-end
# check of the whole write-path refactor, not just a formality.
#
# Usage:
#   scripts/inference/rerun_with_logits.sh                    # snapshot + gate + print commands (default: no backfill runs)
#   EXECUTE=1 scripts/inference/rerun_with_logits.sh          # snapshot + gate + actually run the backfill
#   GATE_LABEL=sngp_wong scripts/inference/rerun_with_logits.sh
#   LABELS="baseline_acevedo sngp_acevedo" EXECUTE=1 scripts/inference/rerun_with_logits.sh
#   SKIP_GATE=1 EXECUTE=1 scripts/inference/rerun_with_logits.sh   # only if you already ran the gate once
#
# Caveat: the gate only proves reproducibility for deterministic checkpoints
# (baseline without MC-Dropout, SNGP, Deep Ensemble). mc_* entries are stochastic
# forward passes; configs/infer.yaml's `seed` is honored by run_inference as of this
# change, but is null by default and this script does not set it, so re-running an
# mc_* entry will legitimately produce different predictions/metrics than what's
# currently published for it -- expected, not a gate failure, see
# docs/KNOWN_ISSUES.md.
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example}"
: "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example}"

LOG_DIR="${EXPERIMENTS_HOME%/}/${PROJECT_NAME}"
INFER_ROOT="${LOG_DIR}/infer"

GATE_LABEL="${GATE_LABEL:-baseline_acevedo}"
SKIP_GATE="${SKIP_GATE:-0}"
EXECUTE="${EXECUTE:-0}"
LABELS="${LABELS:-}"

echo "=== Step 1/3: snapshotting ${INFER_ROOT} ==="
if [[ -d "${INFER_ROOT}" ]]; then
  SNAPSHOT_DIR="${INFER_ROOT%/}_backup_$(date +%Y-%m-%d_%H-%M-%S)"
  cp -a "${INFER_ROOT}" "${SNAPSHOT_DIR}"
  echo "Snapshot written to ${SNAPSHOT_DIR}"
else
  echo "No existing ${INFER_ROOT} -- nothing to snapshot, nothing at risk."
fi

if [[ "${SKIP_GATE}" == "1" ]]; then
  echo "=== Step 2/3: SKIP_GATE=1 -- skipping the safety gate ==="
else
  echo "=== Step 2/3: safety gate -- re-running '${GATE_LABEL}' into a scratch dir and diffing metrics.json ==="

  read -r GATE_CKPT GATE_RUN_DIR GATE_ID_DATASET <<EOF
$(uv run python -c "
from src.metrics.manifest import DEFAULT_MANIFEST_PATH, load_manifest, resolve_default_infer_root
m = load_manifest(DEFAULT_MANIFEST_PATH, infer_root=resolve_default_infer_root())
run = m.get('${GATE_LABEL}')
print(run.ckpt, run.run_dir, run.id_dataset)
")
EOF

  SCRATCH_DIR="$(mktemp -d)"
  uv run src/inference/infer.py \
    ckpt_path="${LOG_DIR}/${GATE_CKPT}" \
    data="${GATE_ID_DATASET}" \
    save_path="${SCRATCH_DIR}" \
    infer.save.run_name=gate_check

  REAL_METRICS="${INFER_ROOT}/${GATE_RUN_DIR}/${GATE_ID_DATASET}/metrics.json"
  SCRATCH_METRICS="${SCRATCH_DIR}/gate_check/metrics.json"

  if diff -q "${REAL_METRICS}" "${SCRATCH_METRICS}" > /dev/null 2>&1; then
    echo "Gate PASSED: metrics.json is byte-identical between the existing run and a fresh re-run."
  else
    echo "Gate FAILED: metrics.json differs between the existing run and a fresh re-run." >&2
    echo "--- diff (existing vs fresh) ---" >&2
    diff "${REAL_METRICS}" "${SCRATCH_METRICS}" >&2 || true
    echo "Not proceeding -- nothing under ${INFER_ROOT} was modified; the snapshot above is untouched." >&2
    exit 1
  fi
  rm -rf "${SCRATCH_DIR}"
fi

echo "=== Step 3/3: backfill commands ==="
COMMANDS="$(uv run python -c "
from src.metrics.manifest import DEFAULT_MANIFEST_PATH, load_manifest, resolve_default_infer_root
m = load_manifest(DEFAULT_MANIFEST_PATH, infer_root=resolve_default_infer_root())
labels = '${LABELS}'.split() or None
runs = [m.get(label) for label in labels] if labels else list(m.runs)
for run in runs:
    ckpt = '${LOG_DIR}/' + run.ckpt
    mc_flag = ' infer.runtime.use_mc_dropout=true' if run.uses_mc_dropout else ''
    for dataset in (*run.eval_datasets, *(('artifact_image_classifier',) if run.paired_streams else ())):
        print(f'uv run src/inference/infer.py ckpt_path={ckpt} data={dataset}{mc_flag}')
")"

if [[ "${EXECUTE}" == "1" ]]; then
  echo "EXECUTE=1 -- running the backfill now. This overwrites predictions.csv/metrics.json/run.json in place for every run below."
  while IFS= read -r cmd; do
    echo "+ ${cmd}"
    eval "${cmd}"
  done <<< "${COMMANDS}"
else
  echo "Dry run (default) -- nothing executed. Re-run with EXECUTE=1 to actually run these:"
  echo "${COMMANDS}"
fi
