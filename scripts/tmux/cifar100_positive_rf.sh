#!/bin/bash
# Study `positive_rf`: the CIFAR-100 / WRN-28-10 spectral-regularization arm trained end-to-end with
# POSITIVE random features, at a given length scale, for several couplings and seeds.
#
# Follows up two pages:
#   * docs/results/CIFAR100_RF_E2E_RESULTS.md -- positive/{orf, simrf} at l = 20, seed 12345 only,
#     where positive/simrf posted the best far-OOD numbers but the simrf-orf gap flips sign
#     between feature maps, i.e. looks like seed noise. Block A adds seeds 1 / 2.
#   * docs/results/CIFAR100_EVIDENCE_LS_RESULTS.md -- cos/orf retrained at its type-II-evidence
#     length scale (l = 7). Block B is the fair counterpart: positive features retrained at THEIR
#     OWN evidence l*, picked the same way (the positive map scored on backbones trained with a
#     positive l = 20 head), by scripts/metrics/cifar100_length_scale_evidence.py --feature-map positive.
#
# Against sngp_specreg_cifar100.yaml only these change: `model.net.feature_map=positive`,
# `model.net.random_feature_type=<coupling>`, `model.net.length_scale=<l>`. Everything else is the
# protocol: kernel_amplitude 1.0, unscaled features, ridge 1.0, gaussian likelihood, mean_field_factor
# fit post hoc on val at evaluation time (the ONE post-hoc knob), comparison at `last.ckpt`.
#
# Runs are assigned round-robin over --gpus, so listing a GPU twice packs two runs on it (~5 GB each
# at batch 128; an L40S has 46 GB). Every run gets an EXPLICIT hydra run dir: all arms share
# `model.name`, and the first benchmark lost a `last.ckpt` to a run-directory collision
# (docs/checkpoints/CIFAR_CHECKPOINTS.md). TRAINING ONLY; evaluation is
# scripts/tmux/cifar100_positive_rf_infer.sh.
#
# --smoke trains every run for ~3 epochs (test=False) under a separate smoke tag. It caps optimizer
# steps rather than lowering max_epochs, which would rescale the piecewise LR schedule; 1200 steps
# ~ 3 epochs at batch 128. For reference, positive/{orf, simrf} at l = 20 reach val/acc 0.20-0.22
# there, and at l = 2 they sat at chance.
#
#   scripts/tmux/cifar100_positive_rf.sh --length-scale 20 --seeds "1 2" --gpus "0 1 2 3"     # block A
#   scripts/tmux/cifar100_positive_rf.sh --length-scale <l*> --seeds 12345 --gpus "0 1" --smoke
#   scripts/tmux/cifar100_positive_rf.sh --length-scale <l*> --seeds "12345 1 2" --gpus "0 1 2 3 0 1"  # block B
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="positive_rf"
LENGTH_SCALE=""
COUPLINGS="orf simrf"
SEEDS="12345 1 2"
GPUS="0 1 2 3"
MODE="full"
CHILD=0
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --length-scale) LENGTH_SCALE="$2"; shift 2 ;;
    --couplings)    COUPLINGS="$2"; shift 2 ;;
    --seeds)        SEEDS="$2"; shift 2 ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --smoke)        MODE="smoke"; shift ;;
    --run)          CHILD=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
done
[[ -n "${LENGTH_SCALE}" ]] || { echo "--length-scale is required" >&2; exit 1; }
# `7.0` and `7` must give the same labels and tag.
L_LABEL="$(printf '%g' "${LENGTH_SCALE}")"

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${POSITIVE_RF_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export POSITIVE_RF_STAMP="${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
TAG="${STUDY}_l${L_LABEL}_${STAMP}"; [[ "${MODE}" == "smoke" ]] && TAG="${STUDY}_l${L_LABEL}_smoke_${STAMP}"
# Under overnight/ with `<tag>_<label>__<dataset>` infer naming, as the evidence_ls arms.
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_${TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${CHILD}" == 0 ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="cifar100_${TAG}"
  printf -v QUOTED '%q ' "${ARGS[@]}"
  tmux new-session -d -s "${SESSION}" -e "POSITIVE_RF_STAMP=${STAMP}" "bash '$0' --run ${QUOTED} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE})."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  echo "  Run dirs: ${RUNS}/pos_<coupling>_l${L_LABEL}_s<seed>"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=1 ++trainer.max_steps=1200 test=False
          "logger.wandb.group=CIFAR100_${STUDY}_smoke")
else
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=250 test=True "logger.wandb.group=CIFAR100_${TAG}")
fi
COMMON=(experiment=sngp_specreg_cifar100 model.net.feature_map=positive "model.net.length_scale=${LENGTH_SCALE}"
        logger.wandb.log_model=false "${BUDGET[@]}")

# run <label> <gpu> <extra hydra overrides...>
run() {
  local label="$1" gpu="$2"; shift 2
  local dir="${RUNS}/${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu})"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/train.py \
      "${COMMON[@]}" "$@" \
      "name=${STUDY}_${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

read -r -a GPU_LIST <<< "${GPUS}"
echo "=== ${STUDY} (${MODE}) -- positive features @ length_scale=${LENGTH_SCALE}; couplings [${COUPLINGS}]; seeds [${SEEDS}]; gpus [${GPUS}] ==="
i=0
for coupling in ${COUPLINGS}; do
  for seed in ${SEEDS}; do
    run "pos_${coupling}_l${L_LABEL}_s${seed}" "${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}" \
        "model.net.random_feature_type=${coupling}" "seed=${seed}" &
    i=$(( i + 1 ))
  done
done
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/"
echo "Next: record them in docs/checkpoints/CIFAR_CHECKPOINTS.md, then scripts/tmux/cifar100_positive_rf_infer.sh"
