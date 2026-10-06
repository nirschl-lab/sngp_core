#!/bin/bash
# ADRC study, final runs: the one cell kept per arm from the Wong grid sweeps (docs/MASTER_SWEEPS.md,
# adrc_wong_*), plus the deterministic SGD baseline, each over 5 seeds at the full 150-epoch budget.
# configs/experiment/sngp_wong_sgd.yaml carries the shared recipe.
#
#   sngp       sngp_wong_sgd          c = 1,  sigma^2 = 1           (sweep #1, val/nll 0.0447)
#   bnsn       sngp_bnsn_wong_sgd     c = 8 (BN cap tied), sigma^2 = 1  (#1, 0.0431)
#   specreg    sngp_specreg_wong_sgd  lambda = 0.003, sigma^2 = 1   (#2, 0.0478; #1 differs only in sigma^2 = 2)
#   muon       sngp_muon_wong_sgd     Muon wd = 0.01, sigma^2 = 1   (#1, 0.0257)
#   baseline   baseline_wong_sgd      linear head, SGD + Nesterov
#
# Differences from the sweep trials:
#   * early stopping OFF and min_epochs = max_epochs = 150 -- the sweeps' patience-8 stop killed
#     about half of every grid in the high-LR part of the cosine schedule.
#   * GP arms use the binary_logistic Laplace weight (model.net.likelihood). It only changes the
#     precision accumulation, so the trained weights for a seed are the same as with gaussian,
#     but val/nll at pi/8 -- and hence the best.ckpt epoch -- can differ. last.ckpt is kept too.
#   * W&B: group adrc_final_<stamp>; tags = each experiment's own + <ds>_adrc_5seed, final, and
#     binary_logistic on the GP arms.
#
# INSTITUTION=<id> (e.g. ucdavis) trains the same 25 runs, same hyperparameters, on that
# institution's rows only (data.datamodule.institution). <ds> becomes wong_<id> in the run dirs
# (train/<model.name>_wong_<id>/), tmux session/log names and the 5seed tag, and the W&B group /
# run names become adrc_final_<id>_*. The institution id itself is also added as a tag by the
# experiments' tags_with expression. Unset = full Wong, as before.
#
# All 25 runs start at once, round-robin over GPUS (default all four: 7/6/6/6 per GPU). Training
# is data-loader (CPU) bound, so each run gets NUM_WORKERS workers (default 6: 25 x 7 = 175 of
# 192 cores) and BLAS/OMP threads are capped at 1 (see scripts/slurm/wandb_agent.sbatch for why).
#
# --smoke trains seed 12345 of each arm for SMOKE_STEPS steps (default 450: past one validation at
# 383 steps/epoch), max_epochs kept at 150 so the cosine schedule is the real one.
#
#   scripts/tmux/adrc_wong_final.sh --smoke   # 5 runs, quick check
#   scripts/tmux/adrc_wong_final.sh           # 25 runs
#   INSTITUTION=ucdavis NUM_WORKERS=4 scripts/tmux/adrc_wong_final.sh   # 25 runs, UC Davis only
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

INSTITUTION="${INSTITUTION:-}"
NUM_WORKERS="${NUM_WORKERS:-6}"
DS="wong${INSTITUTION:+_${INSTITUTION}}"
STUDY="adrc_final${INSTITUTION:+_${INSTITUTION}}"
MODE="full"
for arg in "$@"; do [[ "${arg}" == "--smoke" ]] && MODE="smoke"; done
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
SEEDS=(12345 1 2 3 4)

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${ADRC_FINAL_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export ADRC_FINAL_STAMP="${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
TAG="${STUDY}_${STAMP}"; [[ "${MODE}" == "smoke" ]] && TAG="${STUDY}_smoke_${STAMP}"
LOGS="${ROOT}/tmux_logs/${DS}_${TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  uv sync --frozen  # once, so the 25 concurrent `uv run --no-sync` never race on the venv
  SESSION="${DS}_${TAG}"
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="--run --smoke"
  tmux new-session -d -s "${SESSION}" -e "ADRC_FINAL_STAMP=${STAMP}" -e "GPUS=${GPU_LIST[*]}" \
      -e "SMOKE_STEPS=${SMOKE_STEPS:-450}" -e "INSTITUTION=${INSTITUTION}" -e "NUM_WORKERS=${NUM_WORKERS}" \
      "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE})."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  exit 0
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COMMON=(callbacks.early_stopping=null "data.datamodule.num_workers=${NUM_WORKERS}" logger.wandb.log_model=false
        "logger.wandb.group=${TAG}")
[[ -n "${INSTITUTION}" ]] && COMMON+=("data.datamodule.institution=${INSTITUTION}")
if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.min_epochs=1 trainer.max_epochs=150 "++trainer.max_steps=${SMOKE_STEPS:-450}" test=False)
  RUN_ID="${STAMP}_smoke"
  SEEDS=(12345)
else
  BUDGET=(trainer.min_epochs=150 trainer.max_epochs=150 test=True)
  RUN_ID="${STAMP}"
fi
# Appends to each experiment's own tags via the tags_with resolver. The \$ keeps bash from expanding
# it, and the inner single quotes are required: Hydra's override grammar rejects an unquoted
# nested interpolation ("extraneous input '}'").
GP_TAGS="logger.wandb.tags='\${tags_with:\${tags},\${data.datamodule.institution},${DS}_adrc_5seed,final,binary_logistic}'"
BASE_TAGS="logger.wandb.tags='\${tags_with:\${tags},\${data.datamodule.institution},${DS}_adrc_5seed,final}'"
GP=("${GP_TAGS}" model.net.likelihood=binary_logistic)

# run <label> <gpu> <model.name> <hydra overrides...>
run() {
  local label="$1" gpu="$2" model_name="$3"; shift 3
  local dir="${ROOT}/train/${model_name}_${DS}/runs/${RUN_ID}_${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu}) -> ${dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python src/train.py \
      "$@" "${COMMON[@]}" "${BUDGET[@]}" \
      "name=${STUDY}_${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

echo "=== ADRC ${DS} final (${MODE}): 5 arms x seeds ${SEEDS[*]} on GPUs ${GPU_LIST[*]} ==="
i=0
launch() {  # launch <label> <model.name> <overrides...>  -- next GPU round-robin, staggered start
  local gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  i=$((i + 1))
  run "$1" "${gpu}" "${@:2}" &
  sleep 5  # don't hit the HF dataset cache with 25 simultaneous loads
}
for s in "${SEEDS[@]}"; do
  launch "sngp_s${s}"     sngp_sgd_classifier         experiment=sngp_wong_sgd seed="${s}" "${GP[@]}" \
      model.net.spectral_norm_bound=1 model.net.kernel_amplitude=1
  launch "muon_s${s}"     sngp_muon_sgd_classifier    experiment=sngp_muon_wong_sgd seed="${s}" "${GP[@]}" \
      model.optimizer.weight_decay=0.01 model.net.kernel_amplitude=1
  launch "bnsn_s${s}"     sngp_bnsn_sgd_classifier    experiment=sngp_bnsn_wong_sgd seed="${s}" "${GP[@]}" \
      model.net.spectral_norm_bound=8 model.net.kernel_amplitude=1
  launch "specreg_s${s}"  sngp_specreg_sgd_classifier experiment=sngp_specreg_wong_sgd seed="${s}" "${GP[@]}" \
      model.spec_reg_coef=0.003 model.net.kernel_amplitude=1
  launch "baseline_s${s}" baseline_sgd_classifier     experiment=baseline_wong_sgd seed="${s}" "${BASE_TAGS}"
done
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${ROOT}/train/*_sgd_classifier_${DS}/runs/${RUN_ID}_*/checkpoints/"
