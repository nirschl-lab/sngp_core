#!/bin/bash
# CIFAR-100 / WRN-28-10 spectral-regularization arm, retrained with the `trace_logistic`
# Laplace weight (w = 1 - ||p||^2) instead of the reference's unit `gaussian` weight.
#
# `likelihood` feeds ONLY the GP precision accumulator -- the loss sees raw logits and no
# variance is computed in train mode -- so these runs train the same model as the existing
# `*_specreg` arms at the same seed. What changes is the eval-time predictive variance, and
# so the mean-field-corrected logits that `val/loss`, `trainer.test()` and inference use.
# Three seeds because the existing SpecReg rows in docs/results/CIFAR100_RESULTS.md carry
# three, and the comparison needs the same error bar.
#
# TRAINING ONLY -- no inference, no metrics report. Unlike the gaussian arms,
# `mean_field_factor` is pinned at 7.5 and a logistic weight inflates the variance by ~50x
# on a converged CIFAR-100 classifier, so numbers read at the pinned factor are not a fair
# comparison. Evaluation must start with scripts/metrics/cifar100_mean_field_sweep.py
# (free: offline re-scoring from the prediction CSVs), and must compare at `last.ckpt`
# (epoch 249) -- `val/loss` is computed from mean-field logits, so `best.ckpt` will land on
# a different epoch than the gaussian runs even though per-epoch weights are identical.
#
# 3 trainings in 1 block (one per GPU) at ~2.7 h each. Every run gets an EXPLICIT hydra run
# dir: both spectral-reg arms share `model.name`, and the first benchmark lost a
# `last.ckpt` to a run-directory collision (docs/checkpoints/CIFAR_CHECKPOINTS.md).
#
#   scripts/tmux/cifar100_trace_logistic.sh     # detaches into tmux, prints the log path
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
LOGS="${ROOT}/tmux_logs/cifar100_trace_logistic_${STAMP}"
TAG="trace_logistic_${STAMP}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="cifar100_trace_logistic_${STAMP}"
  tmux new-session -d -s "${SESSION}" "bash '$0' --run 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}'."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  echo "  Run dirs: ${ROOT}/overnight/${TAG}/<label>"
  exit 0
fi

# Run dirs live under overnight/ (rather than a new tree) so scripts/metrics/
# cifar100_overnight_report.py only needs new ARMS label entries to pick them up later.
COMMON=(trainer.max_epochs=250 trainer.min_epochs=250 logger.wandb.log_model=false
        "logger.wandb.group=CIFAR100_${TAG}" model.net.likelihood=trace_logistic test=True)

# run <label> <gpu> <experiment> <extra hydra overrides...>
run() {
  local label="$1" gpu="$2" exp="$3"; shift 3
  local dir="${ROOT}/overnight/${TAG}/${label}"
  mkdir -p "${dir}"
  echo "[$(date +%H:%M:%S)] start ${label} (gpu ${gpu})"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python src/train.py \
      experiment="${exp}" "${COMMON[@]}" "$@" \
      "name=${label}" "hydra.run.dir=${dir}" \
      > "${LOGS}/${label}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE ${label}" \
    || echo "[$(date +%H:%M:%S)] FAILED ${label} (see ${LOGS}/${label}.log)"
}

echo "=== block 1/1 -- sngp_specreg_cifar100 @ likelihood=trace_logistic, three seeds ==="
run tl_specreg_s12345  0  sngp_specreg_cifar100  seed=12345 &
run tl_specreg_s1      1  sngp_specreg_cifar100  seed=1 &
run tl_specreg_s2      2  sngp_specreg_cifar100  seed=2 &
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${ROOT}/overnight/${TAG}/<label>/checkpoints/"
echo "Next: record them in docs/checkpoints/CIFAR_CHECKPOINTS.md, then evaluate --"
echo "  mean-field sweep FIRST (the pinned factor 7.5 over-shrinks a logistic weight)."
