#!/bin/bash
# Train Deep Ensemble members as independent, fully parallel runs across multiple
# GPUs, instead of DeepEnsembleLitModule's sequential single-run cycling (see
# src/models/deep_ensemble_lit_module.py -- "sequential" is the only train_strategy
# implemented there: one member trains at a time, N x per-member-epochs total).
#
# Each member here is trained as its own ordinary baseline run (any
# configs/experiment/baseline_*.yaml), pinned to one GPU via CUDA_VISIBLE_DEVICES and
# given a distinct `seed` for member diversity (mirrors DeepEnsemble's own per-member
# seeding convention -- seed=i for member i, see
# DeepEnsemble._reset_parameters in src/models/ensemble/deep_ensemble_model.py).
# Once all members finish, combine their checkpoints into one DeepEnsemble checkpoint:
#   uv run scripts/ensemble/assemble_ensemble_checkpoint.py \
#       --members-dir <dir printed below> --out <path/to/ensemble.ckpt>
# The result satisfies the same checkpoint contract (src/checkpointing/) as one
# produced by an actual DeepEnsembleLitModule run, so src/eval.py and
# src/inference/infer.py need no changes to use it.
#
# Round-robin packing: pass fewer GPU ids than NUM_ESTIMATORS to pack multiple
# members per GPU (member i -> GPU_IDS[i % len(GPU_IDS)]) -- a small resnet18
# member doesn't need a whole GPU, so e.g. 6 members packed onto 2 GPUs is fine:
#   scripts/ensemble/train_members_parallel.sh baseline_acevedo 6 0 1
# With no GPU ids: uses CUDA_VISIBLE_DEVICES if already set in this shell (e.g. a
# SLURM allocation, see scripts/slurm/train_ensemble_members.sh), else auto-detects
# via `nvidia-smi -L`.
#
# Env vars:
#   MEMBER_EXTRA_OVERRIDES="<hydra overrides>"  -- forwarded to every member's
#     train.py call, e.g. to shrink per-member dataloader workers when packing
#     (configs/data/*.yaml commonly default num_workers=32 per job, so K packed
#     members multiply that by K on one node):
#       MEMBER_EXTRA_OVERRIDES="data.datamodule.num_workers=8" \
#         scripts/ensemble/train_members_parallel.sh baseline_acevedo 6 0 1
#   DRY_RUN=1  -- print the packing plan and exit before launching anything.
#
# Usage:
#   scripts/ensemble/train_members_parallel.sh <baseline_experiment> <num_estimators> [gpu_id ...]
#   scripts/ensemble/train_members_parallel.sh baseline_acevedo 5
#   scripts/ensemble/train_members_parallel.sh baseline_acevedo 3 0 2 3    # pin to GPUs 0,2,3
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <baseline_experiment> <num_estimators> [gpu_id ...]}"
NUM_ESTIMATORS="${2:?Usage: $0 <baseline_experiment> <num_estimators> [gpu_id ...]}"
shift 2
GPU_IDS=("$@")
MEMBER_EXTRA_OVERRIDES="${MEMBER_EXTRA_OVERRIDES:-}"

cd "$(dirname "$0")/../.."  # repo root

if [[ ! -f "configs/experiment/${EXPERIMENT}.yaml" ]]; then
  echo "No configs/experiment/${EXPERIMENT}.yaml." >&2
  exit 1
fi

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Respect an already-set restriction rather than re-discovering the whole
    # node's GPUs and ignoring it -- this is exactly what SLURM's gres/gpu plugin
    # sets for a job's allocation (see scripts/slurm/train_ensemble_members.sh).
    # Whether the values are physical GPU indices or cgroup-narrowed local ones
    # depends on the cluster, but either way they're the only tokens valid in
    # this shell, so we just forward them verbatim below rather than re-deriving.
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    echo "No GPU ids given; using the ${#GPU_IDS[@]} GPU(s) already selected by CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
  elif command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L | wc -l)" -gt 0 ]]; then
    NUM_GPUS="$(nvidia-smi -L | wc -l)"
    for ((i = 0; i < NUM_GPUS; i++)); do GPU_IDS+=("$i"); done
    echo "No GPU ids given; auto-detected ${NUM_GPUS} GPU(s) via 'nvidia-smi -L'."
  else
    echo "No GPU ids given, CUDA_VISIBLE_DEVICES is unset, and 'nvidia-smi -L' found no GPUs; pass GPU ids explicitly, e.g.: $0 ${EXPERIMENT} ${NUM_ESTIMATORS} 0 1 2 3" >&2
    exit 1
  fi
fi

# Round-robin members across however many GPU ids we ended up with -- may be
# fewer than NUM_ESTIMATORS (multiple members packed per GPU) or more (some GPU
# ids idle). Computed once and reused by both the printout below and the launch
# loop further down, so the two can never diverge.
MEMBER_GPU=()
for ((i = 0; i < NUM_ESTIMATORS; i++)); do
  MEMBER_GPU+=("${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}")
done

echo "Packing plan (${NUM_ESTIMATORS} member(s) across ${#GPU_IDS[@]} GPU id(s), round-robin):"
GPU_ORDER=()
declare -A GPU_PLAN
for ((i = 0; i < NUM_ESTIMATORS; i++)); do
  GPU_ID="${MEMBER_GPU[$i]}"
  if [[ -z "${GPU_PLAN[${GPU_ID}]+set}" ]]; then
    GPU_ORDER+=("${GPU_ID}")
    GPU_PLAN[${GPU_ID}]=""
  fi
  GPU_PLAN[${GPU_ID}]+="${GPU_PLAN[${GPU_ID}]:+, }member ${i}"
done
for GPU_ID in "${GPU_ORDER[@]}"; do
  echo "  GPU ${GPU_ID}: ${GPU_PLAN[${GPU_ID}]}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 -- exiting before launching anything."
  exit 0
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example}"
: "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example}"

# Same <model.name>_<data.name> key train/eval use for their own output directories
# (see docs/OUTPUT_LAYOUT.md) -- resolved from the experiment config itself so this
# stays correct if the experiment's model/data overrides ever change, rather than
# reusing $EXPERIMENT (the experiment config's filename) directly.
MODEL_DATASET_KEY="$(uv run python -c "
import hydra
with hydra.initialize(version_base='1.3', config_path='configs'):
    cfg = hydra.compose(config_name='train.yaml', overrides=['experiment=${EXPERIMENT}'])
    print(f'{cfg.model.name}_{cfg.data.name}')
")"

# Nested under train/<model>_<dataset>/ -- the same root ordinary single-run training
# (train/<model>_<dataset>/runs/<run_id>/) uses -- as a sibling "ensemble_members/"
# batch kind, rather than a separate top-level tree. See docs/OUTPUT_LAYOUT.md §6.
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
MEMBERS_ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}/train/${MODEL_DATASET_KEY}/ensemble_members/${RUN_ID}"
mkdir -p "${MEMBERS_ROOT}"

PIDS=()
for ((i = 0; i < NUM_ESTIMATORS; i++)); do
  MEMBER_DIR="${MEMBERS_ROOT}/member_${i}"
  GPU_ID="${MEMBER_GPU[$i]}"
  echo "Launching member ${i}/${NUM_ESTIMATORS} on GPU ${GPU_ID} -> ${MEMBER_DIR}"
  # shellcheck disable=SC2086  # MEMBER_EXTRA_OVERRIDES is intentionally word-split
  # into zero or more separate Hydra overrides, same as any Hydra CLI call.
  CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python src/train.py \
    experiment="${EXPERIMENT}" \
    seed="${i}" \
    name="${EXPERIMENT}_member${i}" \
    hydra.run.dir="${MEMBER_DIR}" \
    ${MEMBER_EXTRA_OVERRIDES} \
    > "${MEMBERS_ROOT}/member_${i}.log" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for ((i = 0; i < NUM_ESTIMATORS; i++)); do
  if ! wait "${PIDS[$i]}"; then
    echo "Member ${i} FAILED -- see ${MEMBERS_ROOT}/member_${i}.log" >&2
    FAILED=1
  fi
done

if [[ "${FAILED}" -eq 1 ]]; then
  echo "One or more members failed; not all checkpoints under ${MEMBERS_ROOT} are usable." >&2
  exit 1
fi

echo ""
echo "All ${NUM_ESTIMATORS} members finished. Assemble with:"
echo "  uv run scripts/ensemble/assemble_ensemble_checkpoint.py --members-dir ${MEMBERS_ROOT} --out ${MEMBERS_ROOT}/ensemble.ckpt"
