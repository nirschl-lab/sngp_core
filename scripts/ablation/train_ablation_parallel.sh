#!/bin/bash
# Ablate ONE Hydra key over a list of values, running the arms in parallel across
# several GPUs -- the multi-GPU counterpart of a plain
# `uv run python src/train.py -m <key>=v1,v2,...`, which Hydra's default (basic)
# launcher would run strictly one after another on a single device.
#
# Each GPU gets its own lane: one `train.py -m` process pinned to that device via
# CUDA_VISIBLE_DEVICES, sweeping the subset of values round-robined to it. Lanes run
# concurrently; arms within a lane run sequentially, so exactly one training job is
# resident per GPU at a time. All lanes share one `hydra.sweep.dir`, so the whole
# ablation lands in a single multiruns/<run_id>_<leaf>_ablation/ tree
# (docs/OUTPUT_LAYOUT.md §1) rather than one tree per GPU.
#
# `hydra.sweep.subdir` is overridden to <leaf>_<value> (e.g. spectral_norm_bound_4.0/)
# instead of the default ${hydra.job.num}. Two reasons: job numbers restart at 0 in
# every lane, so the default would collide three separate arms onto .../0/; and a
# value-named directory is readable without opening .hydra/overrides.yaml. The
# separator is "_", not "=", because Hydra's override grammar rejects an unquoted "="
# inside an override's value. A null value yields "..._None" -- Python's None
# stringified, i.e. for spectral_norm_bound the stock hard-normalization arm.
#
# This is a Hydra multirun, NOT a W&B sweep: no search algorithm, no objective, no
# entry in docs/MASTER_SWEEPS.md (that file is for scripts/hpo/sweep.py). Arms here
# DO write checkpoints, unlike HPO trials -- see docs/HPO_GUIDE.md for that contrast.
#
# Env vars:
#   NAME_PREFIX="<str>"       -- `name` becomes <NAME_PREFIX><value>, and `name` is what
#                                configs/logger/wandb.yaml uses as the W&B run name.
#                                Default "<experiment>_<leaf>_". Without a per-value
#                                name every arm shows up on W&B under the experiment
#                                config's single `name:` and is indistinguishable.
#   WANDB_GROUP="<str>"       -- W&B group for the whole ablation. Default "<leaf>_ablation".
#   EXTRA_OVERRIDES="<...>"   -- extra Hydra overrides forwarded to every lane, e.g.
#                                "data.datamodule.num_workers=8" when packing many
#                                lanes onto one node (configs/data/*.yaml default to
#                                32 workers per job, multiplied by the lane count).
#   DRY_RUN=1                 -- print the lane plan and the exact commands, then exit.
#
# Usage:
#   scripts/ablation/train_ablation_parallel.sh <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]
#   scripts/ablation/train_ablation_parallel.sh sngp_acevedo model.net.spectral_norm_bound null,1.0,2.0 1 2
#   DRY_RUN=1 scripts/ablation/train_ablation_parallel.sh sngp_acevedo model.net.spectral_norm_bound 1.0,6.0 0
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]}"
KEY="${2:?Usage: $0 <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]}"
VALUES_CSV="${3:?Usage: $0 <experiment> <hydra.key> <v1,v2,...> [gpu_id ...]}"
shift 3
GPU_IDS=("$@")
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

cd "$(dirname "$0")/../.."  # repo root

if [[ ! -f "configs/experiment/${EXPERIMENT}.yaml" ]]; then
  echo "No configs/experiment/${EXPERIMENT}.yaml." >&2
  exit 1
fi

IFS=',' read -ra VALUES <<< "${VALUES_CSV}"
if [[ ${#VALUES[@]} -eq 0 ]]; then
  echo "No values to ablate over." >&2
  exit 1
fi

# Last dotted segment of the key -- used for the default run-name prefix, the W&B
# group, and the sweep directory's suffix.
LEAF="${KEY##*.}"
NAME_PREFIX="${NAME_PREFIX:-${EXPERIMENT}_${LEAF}_}"
WANDB_GROUP="${WANDB_GROUP:-${LEAF}_ablation}"

# Same GPU-discovery fallback as scripts/ensemble/train_members_parallel.sh: an
# explicit list wins, else honour a restriction this shell already carries (a SLURM
# allocation sets exactly this), else take the whole node.
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    echo "No GPU ids given; using the ${#GPU_IDS[@]} GPU(s) already selected by CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
  elif command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L | wc -l)" -gt 0 ]]; then
    NUM_GPUS="$(nvidia-smi -L | wc -l)"
    for ((i = 0; i < NUM_GPUS; i++)); do GPU_IDS+=("$i"); done
    echo "No GPU ids given; auto-detected ${NUM_GPUS} GPU(s) via 'nvidia-smi -L'."
  else
    echo "No GPU ids given, CUDA_VISIBLE_DEVICES is unset, and 'nvidia-smi -L' found no GPUs; pass GPU ids explicitly, e.g.: $0 ${EXPERIMENT} ${KEY} ${VALUES_CSV} 0 1" >&2
    exit 1
  fi
fi

# Round-robin values across GPUs, then collapse each GPU's values back into one
# comma-separated list -- that list is a single Hydra multirun sweep for that lane.
LANE_GPUS=()
declare -A LANE_VALUES
for ((i = 0; i < ${#VALUES[@]}; i++)); do
  GPU_ID="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}"
  if [[ -z "${LANE_VALUES[${GPU_ID}]+set}" ]]; then
    LANE_GPUS+=("${GPU_ID}")
    LANE_VALUES[${GPU_ID}]=""
  fi
  LANE_VALUES[${GPU_ID}]+="${LANE_VALUES[${GPU_ID}]:+,}${VALUES[$i]}"
done

echo "Ablating ${KEY} over ${#VALUES[@]} value(s) across ${#LANE_GPUS[@]} GPU lane(s):"
for GPU_ID in "${LANE_GPUS[@]}"; do
  echo "  GPU ${GPU_ID}: ${LANE_VALUES[${GPU_ID}]}"
done

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example}"
: "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example}"

# Resolved from the experiment config rather than assumed from $EXPERIMENT, so the
# tree stays correct if the experiment's model/data overrides ever change -- same
# <model.name>_<data.name> key train/eval use (docs/OUTPUT_LAYOUT.md).
MODEL_DATASET_KEY="$(uv run python -c "
import hydra
with hydra.initialize(version_base='1.3', config_path='configs'):
    cfg = hydra.compose(config_name='train.yaml', overrides=['experiment=${EXPERIMENT}'])
    print(f'{cfg.model.name}_{cfg.data.name}')
")"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
SWEEP_DIR="${EXPERIMENTS_HOME}/${PROJECT_NAME}/train/${MODEL_DATASET_KEY}/multiruns/${RUN_ID}_${LEAF}_ablation"

# \${...} keeps bash from expanding these -- Hydra resolves them per job, against the
# composed config, giving one directory and one W&B run name per ablated value.
SUBDIR_OVERRIDE="hydra.sweep.subdir=${LEAF}_\${${KEY}}"
NAME_OVERRIDE="name=${NAME_PREFIX}\${${KEY}}"
# The inner '...' are literal characters Hydra's override grammar requires around a
# resolver call on the CLI (the commas would otherwise read as override separators);
# resolved by src/utils/resolvers.py's tags_with. Same idiom as
# scripts/ensemble/train_members_parallel.sh.
TAGS_OVERRIDE="logger.wandb.tags='\${tags_with:\${tags},\${data.datamodule.institution},ablation,${LEAF}}'"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "Sweep dir: ${SWEEP_DIR}"
  for GPU_ID in "${LANE_GPUS[@]}"; do
    echo "  CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python src/train.py -m experiment=${EXPERIMENT} ${KEY}=${LANE_VALUES[${GPU_ID}]} ${NAME_OVERRIDE} ${TAGS_OVERRIDE} logger.wandb.group=${WANDB_GROUP} logger.wandb.job_type=ablation hydra.sweep.dir=${SWEEP_DIR} ${SUBDIR_OVERRIDE} ${EXTRA_OVERRIDES}"
  done
  echo "DRY_RUN=1 -- exiting before launching anything."
  exit 0
fi

mkdir -p "${SWEEP_DIR}"
echo ""
echo "Sweep dir: ${SWEEP_DIR}"

PIDS=()
for GPU_ID in "${LANE_GPUS[@]}"; do
  echo "Launching lane on GPU ${GPU_ID} (${KEY}=${LANE_VALUES[${GPU_ID}]}) -> ${SWEEP_DIR}/lane_gpu${GPU_ID}.log"
  # shellcheck disable=SC2086  # EXTRA_OVERRIDES is intentionally word-split into zero
  # or more separate Hydra overrides, same as any Hydra CLI call.
  CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python src/train.py -m \
    experiment="${EXPERIMENT}" \
    "${KEY}=${LANE_VALUES[${GPU_ID}]}" \
    "${NAME_OVERRIDE}" \
    "${TAGS_OVERRIDE}" \
    logger.wandb.group="${WANDB_GROUP}" \
    logger.wandb.job_type="ablation" \
    hydra.sweep.dir="${SWEEP_DIR}" \
    "${SUBDIR_OVERRIDE}" \
    ${EXTRA_OVERRIDES} \
    > "${SWEEP_DIR}/lane_gpu${GPU_ID}.log" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for ((i = 0; i < ${#LANE_GPUS[@]}; i++)); do
  if ! wait "${PIDS[$i]}"; then
    echo "Lane on GPU ${LANE_GPUS[$i]} FAILED -- see ${SWEEP_DIR}/lane_gpu${LANE_GPUS[$i]}.log" >&2
    FAILED=1
  fi
done

echo ""
if [[ "${FAILED}" -eq 1 ]]; then
  echo "One or more lanes failed; the ablation under ${SWEEP_DIR} is incomplete." >&2
  exit 1
fi

echo "All ${#VALUES[@]} arm(s) finished. Checkpoints:"
echo "  ${SWEEP_DIR}/${LEAF}_<value>/checkpoints/best.ckpt"
echo "Record the ones you keep in docs/MASTER_CHECKPONT_PATHS.md."
