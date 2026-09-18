#!/bin/bash
# Run the whole evaluation suite for ONE checkpoint -- in-distribution + cross-dataset OOD
# (the 7 datasets of run_all_datasets.sh) and the 11 artifact-simulation arms of
# run_artifact_axes.sh -- as independent src/inference/infer.py jobs spread round-robin
# over several GPU lanes, instead of 18 sequential runs.
#
# Output layout (docs/OUTPUT_LAYOUT.md, docs/DATASETS.md "Saving sweep results"):
#   <save_path>/<run_name_prefix>/<dataset>                              -- 7 datasets
#   <save_path>/<run_name_prefix>/acevedo_artifact/real_baseline         -- stream=real
#   <save_path>/<run_name_prefix>/acevedo_artifact/config/count_{1..5}   -- stream=artifact
#   <save_path>/<run_name_prefix>/acevedo_artifact/procedural/severity_{1..5}
#   <save_path>/<run_name_prefix>/lane_logs/lane_gpu<id>.log             -- one log per lane
# so `calculate_ood_metrics.py --run-dir <save_path>/<run_name_prefix>` and the artifact
# sidecar YAMLs read the same tree a sequential sweep would have produced. Pass the
# checkpoint's auto-derived name (`<netname_dataset>/<ckpt_run_id>`, see
# src/inference/infer.py::derive_default_run_name) as <run_name_prefix> to stay
# byte-compatible with runs written without an explicit run_name.
#
# Every job is independent (the artifact arms each write their own directory; the real
# stream is inferred exactly once, in real_baseline), so lanes need no ordering. Jobs are
# queued artifact arms first, then datasets, so each lane gets a mix of slow compositing
# arms and fast plain-inference runs.
#
# GPU lanes: positional ids after <run_name_prefix>; else an inherited CUDA_VISIBLE_DEVICES;
# else every GPU nvidia-smi lists. Same discovery order as
# scripts/ablation/train_ablation_parallel.sh.
#
# Env vars:
#   DATASETS="acevedo jung ..."  -- dataset configs to run (default: the 7 below); "" = none
#   LEVELS="1 2 3 4 5"           -- count/severity levels for the artifact arms; "" = no arms
#   AXES="config procedural"     -- which artifact axes to sweep (default both); e.g.
#                                   AXES=procedural runs real_baseline + severity_N only
#   ARTIFACT_LEAF=acevedo_artifact -- subfolder for the artifact arms
#   SAVE_PATH=<dir>              -- inference root (default ${EXPERIMENTS_HOME}/${PROJECT_NAME}/infer)
#   FORCE=1                      -- re-run jobs that already have a metrics.json (default: skip)
#   DRY_RUN=1                    -- print the lane plan and exit
#
# Usage:
#   scripts/inference/run_eval_suite_parallel.sh <ckpt_path> <run_name_prefix> [gpu_id ...] [-- hydra overrides...]
#   scripts/inference/run_eval_suite_parallel.sh /abs/best.ckpt sngp_classifier_acevedo/<id> 0 1 2 3 \
#       -- data.datamodule.num_workers=8
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ckpt_path> <run_name_prefix> [gpu_id ...] [-- hydra overrides...]" >&2
    exit 1
fi

CKPT_PATH="$1"
RUN_NAME_PREFIX="$2"
shift 2

GPU_IDS=()
OVERRIDES=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        OVERRIDES=("$@")
        break
    fi
    GPU_IDS+=("$1")
    shift
done

cd "$(dirname "$0")/../.."  # repo root

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "No checkpoint at ${CKPT_PATH}." >&2
    exit 1
fi

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -z "${SAVE_PATH:-}" ]]; then
    : "${EXPERIMENTS_HOME:?Set EXPERIMENTS_HOME in .env -- see env_example, or pass SAVE_PATH}"
    : "${PROJECT_NAME:?Set PROJECT_NAME in .env -- see env_example, or pass SAVE_PATH}"
    SAVE_PATH="${EXPERIMENTS_HOME%/}/${PROJECT_NAME}/infer"
fi

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    else
        mapfile -t GPU_IDS < <(nvidia-smi -L 2>/dev/null | sed -n 's/^GPU \([0-9]\+\):.*/\1/p')
    fi
fi
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "No GPU ids given, none in CUDA_VISIBLE_DEVICES, none found by nvidia-smi." >&2
    exit 1
fi

read -ra DATASET_LIST <<< "${DATASETS-acevedo jung kather2016 kather2018 nirschl2018 tang wong}"
read -ra LEVEL_LIST <<< "${LEVELS-1 2 3 4 5}"
read -ra AXIS_LIST <<< "${AXES-config procedural}"
ARTIFACT_LEAF="${ARTIFACT_LEAF:-acevedo_artifact}"
for AXIS in "${AXIS_LIST[@]+"${AXIS_LIST[@]}"}"; do
    if [[ "${AXIS}" != "config" && "${AXIS}" != "procedural" ]]; then
        echo "AXES may only contain 'config' and/or 'procedural', got '${AXIS}'." >&2
        exit 1
    fi
done

# One job per line: "<run_name leaf>|<infer.py overrides...>". Artifact arms mirror
# scripts/inference/run_artifact_axes.sh exactly (same policies, same stream narrowing).
JOBS=()
if [[ ${#LEVEL_LIST[@]} -gt 0 && ${#AXIS_LIST[@]} -gt 0 ]]; then
    JOBS+=("${ARTIFACT_LEAF}/real_baseline|data=artifact_image_classifier data.datamodule.artifact_config_path=none data.datamodule.artifact_procedural_config=none infer.save.streams=[real]")
    if [[ " ${AXIS_LIST[*]} " == *" config "* ]]; then
        for LEVEL in "${LEVEL_LIST[@]}"; do
            JOBS+=("${ARTIFACT_LEAF}/config/count_${LEVEL}|data=artifact_image_classifier data.datamodule.artifact_config_path=artifact_balanced data.datamodule.artifact_procedural_config=none data.datamodule.artifact_count=${LEVEL} infer.save.streams=[artifact]")
        done
    fi
    if [[ " ${AXIS_LIST[*]} " == *" procedural "* ]]; then
        for LEVEL in "${LEVEL_LIST[@]}"; do
            JOBS+=("${ARTIFACT_LEAF}/procedural/severity_${LEVEL}|data=artifact_image_classifier data.datamodule.artifact_config_path=none data.datamodule.artifact_procedural_config=procedural_ood data.datamodule.artifact_severity=${LEVEL} infer.save.streams=[artifact]")
        done
    fi
fi
for DATA in "${DATASET_LIST[@]}"; do
    JOBS+=("${DATA}|data=${DATA}")
done
if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "Nothing to run: DATASETS is empty and LEVELS/AXES select no artifact arm." >&2
    exit 1
fi

# Round-robin the queue over the lanes.
N_LANES=${#GPU_IDS[@]}
declare -a LANE_JOBS
for ((i = 0; i < N_LANES; i++)); do LANE_JOBS[$i]=""; done
for ((j = 0; j < ${#JOBS[@]}; j++)); do
    lane=$((j % N_LANES))
    LANE_JOBS[$lane]+="${JOBS[$j]}"$'\n'
done

ROOT_OUT="${SAVE_PATH}/${RUN_NAME_PREFIX}"
LOG_DIR="${ROOT_OUT}/lane_logs"

echo "Checkpoint: ${CKPT_PATH}"
echo "Output root: ${ROOT_OUT}"
echo "${#JOBS[@]} job(s) over ${N_LANES} lane(s): GPUs ${GPU_IDS[*]}"
[[ ${#OVERRIDES[@]} -gt 0 ]] && echo "Extra overrides: ${OVERRIDES[*]}"
for ((i = 0; i < N_LANES; i++)); do
    echo "--- lane gpu${GPU_IDS[$i]}:"
    while IFS= read -r JOB; do
        [[ -z "${JOB}" ]] && continue
        LEAF="${JOB%%|*}"
        if [[ "${FORCE:-0}" != "1" && -f "${ROOT_OUT}/${LEAF}/metrics.json" ]]; then
            echo "    ${LEAF}  (skip: metrics.json exists; FORCE=1 to re-run)"
        else
            echo "    ${LEAF}"
        fi
    done <<< "${LANE_JOBS[$i]}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo ""
    echo "DRY_RUN=1 -- exiting before launching anything."
    exit 0
fi

mkdir -p "${LOG_DIR}"

run_lane() {
    local GPU_ID="$1"
    local JOB_LIST="$2"
    local LOG_FILE="${LOG_DIR}/lane_gpu${GPU_ID}.log"
    local FAILED=0
    {
        echo "[lane gpu${GPU_ID}] start $(date -Is)"
        while IFS= read -r JOB; do
            [[ -z "${JOB}" ]] && continue
            local LEAF="${JOB%%|*}"
            local JOB_OVERRIDES="${JOB#*|}"
            local RUN_NAME="${RUN_NAME_PREFIX}/${LEAF}"
            if [[ "${FORCE:-0}" != "1" && -f "${ROOT_OUT}/${LEAF}/metrics.json" ]]; then
                echo "[lane gpu${GPU_ID}] skip ${LEAF} (metrics.json exists)"
                continue
            fi
            echo "[lane gpu${GPU_ID}] === ${LEAF} === $(date -Is)"
            # shellcheck disable=SC2086  # JOB_OVERRIDES is intentionally word-split into
            # separate Hydra overrides, same as any Hydra CLI call.
            if CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run src/inference/infer.py \
                ckpt_path="${CKPT_PATH}" \
                fold=test \
                ${JOB_OVERRIDES} \
                save_path="${SAVE_PATH}" \
                infer.save.run_name="${RUN_NAME}" \
                "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; then
                echo "[lane gpu${GPU_ID}] done ${LEAF} $(date -Is)"
            else
                echo "[lane gpu${GPU_ID}] FAILED ${LEAF} $(date -Is)"
                FAILED=1
            fi
        done <<< "${JOB_LIST}"
        echo "[lane gpu${GPU_ID}] end $(date -Is) failed=${FAILED}"
    } > "${LOG_FILE}" 2>&1
    return "${FAILED}"
}

PIDS=()
for ((i = 0; i < N_LANES; i++)); do
    run_lane "${GPU_IDS[$i]}" "${LANE_JOBS[$i]}" &
    PIDS+=("$!")
    echo "Launched lane gpu${GPU_IDS[$i]} (pid $!) -> ${LOG_DIR}/lane_gpu${GPU_IDS[$i]}.log"
done

STATUS=0
for ((i = 0; i < N_LANES; i++)); do
    if ! wait "${PIDS[$i]}"; then
        echo "Lane gpu${GPU_IDS[$i]} had at least one failed job -- see ${LOG_DIR}/lane_gpu${GPU_IDS[$i]}.log" >&2
        STATUS=1
    fi
done

echo ""
if [[ ${STATUS} -eq 0 ]]; then
    echo "All lanes finished. Results under: ${ROOT_OUT}/"
else
    echo "Finished with failures. Results under: ${ROOT_OUT}/ (re-run the same command: finished jobs are skipped)" >&2
fi
echo "Record the sweep in docs/MASTER_INFER_RESULTS_PATH.md."
exit "${STATUS}"
