#!/bin/bash
# Study `evidence_ls`: the CIFAR-100 / WRN-28-10 spectral-regularization arm retrained at the
# length scale that type-II GP evidence picks, l = 7, with nothing else changed.
#
# Where l = 7 comes from: docs/results/CIFAR100_LENGTH_SCALE_EVIDENCE_RESULTS.md. Type-II
# evidence of the random-feature Bayesian linear model on the frozen SpecReg (l = 20) backbones
# picks l = 7 at the recipe's rff_dim 1024, on all 3 seeds and on train and held-out val alike.
# It is chosen from in-distribution data only. Post hoc, that l lifts the GP variance's SVHN
# AUROC from 0.41 to ~0.63; this study asks what training end-to-end at it does.
#
# ONE knob changes against sngp_specreg_cifar100.yaml: `model.net.length_scale=7.0`. It is a
# launcher override rather than a new experiment config so the diff stays exactly that one
# override. Everything else is the protocol: kernel_amplitude 1.0, unscaled features, ridge
# 1.0, gaussian likelihood, mean_field_factor fit post hoc on val (the ONE post-hoc knob) at
# evaluation time, and comparison at `last.ckpt` (epoch 249).
#
# Three seeds (12345 / 1 / 2), matching the existing SNGP and SpecReg rows in
# docs/results/CIFAR100_RESULTS.md, one per GPU, ~2.7 h. TRAINING ONLY; evaluation is
# scripts/tmux/cifar100_evidence_ls_infer.sh. Every run gets an EXPLICIT hydra run dir: all arms
# share `model.name`, and the first benchmark lost a `last.ckpt` to a run-directory collision
# (docs/checkpoints/CIFAR_CHECKPOINTS.md).
#
# --smoke trains every arm for ~3 epochs (test=False) under a separate smoke_<stamp> tree. It
# caps optimizer steps rather than lowering max_epochs, which would rescale the piecewise LR
# schedule (warmup_piecewise_lr refuses that); 1200 steps ~ 3 epochs at batch 128. For
# reference, the l = 20 recipe reaches val/acc ~0.227 at that point, and l = 1 collapsed
# training to ~0.06 in the original 6-epoch probe (docs/models/CIFAR100_BENCHMARK.md).
#
#   scripts/tmux/cifar100_evidence_ls.sh --smoke   # ~3-epoch check of all 3 arms
#   scripts/tmux/cifar100_evidence_ls.sh           # full 250-epoch study
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${EXPERIMENTS_HOME:?}" "${PROJECT_NAME:?}"

STUDY="evidence_ls"
LENGTH_SCALE="7.0"
MODE="full"
for arg in "$@"; do [[ "${arg}" == "--smoke" ]] && MODE="smoke"; done

# Passed down to the re-exec'd child so the parent and child agree on one stamp.
STAMP="${EVIDENCE_LS_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EVIDENCE_LS_STAMP="${STAMP}"
ROOT="${EXPERIMENTS_HOME}/${PROJECT_NAME}"
TAG="${STUDY}_${STAMP}"; [[ "${MODE}" == "smoke" ]] && TAG="${STUDY}_smoke_${STAMP}"
# Under overnight/ like the trace_logistic arms, so the `<tag>_<label>__<dataset>` infer
# naming of scripts/metrics/cifar100_overnight_report.py::_infer_dirname carries over.
RUNS="${ROOT}/overnight/${TAG}"
LOGS="${ROOT}/tmux_logs/cifar100_${TAG}"
mkdir -p "${LOGS}"

# Re-exec into a detached tmux session unless already inside one.
if [[ "${1:-}" != "--run" ]]; then
  command -v tmux >/dev/null || { echo "tmux is not installed." >&2; exit 1; }
  SESSION="cifar100_${TAG}"
  CHILD_ARGS="--run"; [[ "${MODE}" == "smoke" ]] && CHILD_ARGS="--run --smoke"
  tmux new-session -d -s "${SESSION}" -e "EVIDENCE_LS_STAMP=${STAMP}" "bash '$0' ${CHILD_ARGS} 2>&1 | tee '${LOGS}/driver.log'"
  echo "Launched '${SESSION}' (${MODE})."
  echo "  Attach:   tmux attach -t ${SESSION}"
  echo "  Driver:   tail -f ${LOGS}/driver.log"
  echo "  Run dirs: ${RUNS}/<label>"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=1 ++trainer.max_steps=1200 test=False
          "logger.wandb.group=CIFAR100_${STUDY}_smoke")
else
  BUDGET=(trainer.max_epochs=250 trainer.min_epochs=250 test=True "logger.wandb.group=CIFAR100_${TAG}")
fi
COMMON=(experiment=sngp_specreg_cifar100 "model.net.length_scale=${LENGTH_SCALE}"
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

echo "=== ${STUDY} (${MODE}) -- sngp_specreg_cifar100 @ length_scale=${LENGTH_SCALE}, three seeds ==="
run els_specreg_s12345  0  seed=12345 &
run els_specreg_s1      1  seed=1 &
run els_specreg_s2      2  seed=2 &
wait

echo "[$(date +%H:%M:%S)] ALL DONE. Checkpoints: ${RUNS}/<label>/checkpoints/"
echo "Next: record them in docs/checkpoints/CIFAR_CHECKPOINTS.md, then"
echo "  scripts/tmux/cifar100_evidence_ls_infer.sh ${TAG}"
