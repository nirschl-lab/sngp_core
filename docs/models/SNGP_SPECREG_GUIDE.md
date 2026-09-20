# SNGP + spectral regularization (rep-spectral) — workflow

Pilot variant of SNGP in which the backbone's Lipschitz constant is controlled by a
**loss penalty** on the top singular values (Yang, Zavatone-Veth & Pehlevan 2024,
[arXiv 2405.17181](https://arxiv.org/abs/2405.17181)) instead of spectral
normalization. What it is and how it is built:
[SUPPORTED_MODELS.md#sngp-with-spectral-regularization-rep-spectral](../SUPPORTED_MODELS.md#sngp-with-spectral-regularization-rep-spectral).
This page is the operational side: how to train, what to watch, what to change.

Two settings exist. This page covers the **Acevedo pilot**. The CIFAR-100 /
WideResNet-28-10 arms — `sngp_specreg_cifar100` (matched against a reproduced SNGP run)
and `sngp_specreg_cifar100_literal` (this paper's own burn-in/no-weight-decay recipe) —
are on [CIFAR100_BENCHMARK.md](CIFAR100_BENCHMARK.md). That is where the "is it actually
better than spectral normalization" question gets answered; this page is where the method
was first tried.

Status: **research pilot**, one dataset (Acevedo). It is deliberately *off* several
protocol defaults that `docs/HPO_GUIDE.md` fixes for the fair cross-family comparison;
every deviation is listed in `configs/experiment/sngp_specreg_acevedo.yaml` and mirrored
in `tests/test_configs.py::_SELECTION_OVERRIDES`. Do not report its numbers next to the
protocol runs without saying so.

## The recipe (paper-literal, Acevedo pilot)

| Knob | Value | Where | Paper (ResNet18/CIFAR-10) |
|---|---|---|---|
| Penalty | `CE + 0.01 · Σ σ_max²` over backbone convs | `spec_reg_coef` (model config) | γ = 0.01 |
| Cadence | every 24 optimizer steps | `spec_reg_every_n_steps` (model config) | `--reg-freq-update 24` |
| Burn-in | 50 of 100 epochs unregularized | `spec_reg_burnin_epochs` (experiment) | 160 of 200 |
| σ estimator | true conv-operator norm, 1 power iteration/step | `spec_reg_conv_mode: operator`, `spec_reg_n_power_iterations: 1` | exact FFT eig, "N=1 is enough" |
| Spectral norm | off (`use_spectral_norm: false`) | model config | n/a (plain ResNet) |
| Optimizer | AdamW, lr 1e-3, **weight_decay 0** | experiment | SGD 0.01, wd 1e-4 |
| Scheduler | CosineAnnealingLR, T_max 100 | experiment | none |
| Early stopping | **none** (fixed budget) | `callbacks.early_stopping: null` | fixed budget |
| Selection | `val/nll` (raw), epochs ≥ 50 only | `ModelCheckpointFromEpoch(start_epoch=50)` | last epoch |

Why some of these differ from the paper: AdamW/cosine are the project's protocol
optimizer and scheduler; weight decay is 0 so the spectral penalty is the *only*
weight regularizer under test.

## 1. Train

```bash
# detached tmux session; prints the session name and log path
scripts/tmux/train_experiment.sh sngp_specreg_acevedo test=True

# or in the foreground
uv run src/train.py experiment=sngp_specreg_acevedo test=True
```

Outputs land under `train/sngp_specreg_classifier_acevedo/runs/<timestamp>/`
(`docs/OUTPUT_LAYOUT.md`), W&B group `SpectralReg`, tags `sngp_specreg, sngp,
rep-spectral, acevedo, resnet18, training`.

Quick checks before a long run:

```bash
# regularized code path on GPU in one step (burn-in and cadence forced off). The `+` is
# needed because fast_dev_run / limit_*_batches are not keys in configs/trainer/default.yaml;
# W&B refuses offline=true together with the experiment's log_model=True, hence both flags.
uv run src/train.py experiment=sngp_specreg_acevedo +trainer.fast_dev_run=true \
    logger.wandb.offline=true logger.wandb.log_model=false \
    model.spec_reg_burnin_epochs=0 model.spec_reg_every_n_steps=1
# burn-in -> regularized switch and the checkpoint gate, in 3 tiny epochs
uv run src/train.py experiment=sngp_specreg_acevedo trainer.max_epochs=3 trainer.min_epochs=3 \
    +trainer.limit_train_batches=10 +trainer.limit_val_batches=5 model.spec_reg_burnin_epochs=1 \
    model.spec_reg_every_n_steps=1 logger.wandb.offline=true logger.wandb.log_model=false
```

## 2. What to watch in W&B

- `train/sigma_max`, `train/sigma_mean`, `train/spec_reg` (raw Σσ²) — logged from
  epoch 0, gradient-free during burn-in. Expect them to *grow* through the burn-in and
  come down once `train/spec_reg_active` flips to 1 at epoch 50. If they do not move
  in the regularized half, the penalty is too weak for this optimizer (see §4).
- `train/ce` vs `train/loss` — the gap is the weighted penalty.
- `val/nll` (the selection metric here) and `val/nll_cal` (protocol metric, logged for
  reference). Expect a `val/nll` bump at epoch 50 when the penalty switches on.
- Sanity: an untrained resnet18 at 224 px has Σσ² ≈ 130 under the operator norm
  (≈ 70 under the reshape estimate) — `tests/models/test_spectral_reg.py` pins the order
  of magnitude.
- Reference trajectory from the first pilot (2026-09-18, W&B run `5bnnbxdb`): with no
  weight decay the burn-in let σ grow freely (BatchNorm makes conv scale free) — Σσ² ≈ 92k,
  σ_max ≈ 160, σ_mean ≈ 51 by epoch 49, versus ≈ 130 at init. At the onset (epoch 50) the
  epoch-mean `train/loss` jumped to ≈ 19 (the penalty lands on 1 step in 24), CE stayed
  ≈ 0.22, `val/acc` dipped 0.95 → 0.92; by epoch 53 Σσ² was ≈ 11k, σ_max ≈ 76, `val/acc`
  0.94, `val/nll` 0.16. So AdamW does absorb the paper's sparse-cadence penalty without
  diverging, and σ moves fast once it is on.

## 3. Calibrate and run inference

Same tools as SNGP proper (the checkpoint is an ordinary SN-free SNGP checkpoint,
`net_spec.name == "sngp_classifier"`):

```bash
uv run scripts/checkpoints/calibrate_checkpoint.py \
    --ckpt <run>/checkpoints/best.ckpt --experiment sngp_specreg_acevedo --split val
uv run src/inference/infer.py ckpt_path=<run>/checkpoints/best.calibrated.ckpt \
    data=acevedo fold=test save_path=<out>/sngp_specreg/acevedo
```

`best.ckpt` is guaranteed to be a regularized-phase epoch; `last.ckpt` is epoch 99.
Record both in `docs/MASTER_CHECKPONT_PATHS.md`. The whole ID / OOD / artifact suite
runs in one go over the idle GPUs with
`scripts/inference/run_eval_suite_parallel.sh <ckpt> sngp_specreg_classifier_acevedo/<run_id> 0 1 2 3 -- data.datamodule.num_workers=8`
(see the `inference` skill's `references/commands.md`). Results of the first pilot:
[../results/ACEVEDO_SPECREG_RESULTS.md](../results/ACEVEDO_SPECREG_RESULTS.md). When comparing against the protocol
SNGP (`sngp_acevedo`, selected on `val/nll_cal`, lr 0.00916, wd 3.7e-5, c = 4), compare
**post-hoc-calibrated** numbers from `calibrate_checkpoint.py --split test --report-only`
for both, and remember three things besides the regularizer differ (lr, weight decay,
selection metric).

## 4. If it does not behave

- **σ barely shrinks after epoch 50.** The paper's every-24-steps amortization with
  γ = 0.01 was designed for SGD; AdamW normalizes the (sparse) penalty gradient by its
  second-moment estimate, and the regularized half also runs on the decaying half of the
  cosine schedule. First knobs, in order: `model.spec_reg_every_n_steps=1
  model.spec_reg_coef=4e-4` (same time-averaged strength, dense), then a larger `coef`,
  then a scheduler that restarts at the burn-in boundary
  (`CosineAnnealingWarmRestarts(T_0=50)`) or a constant lr.
- **Loss spikes at epoch 50.** Expected to a degree (the penalty is ~0.01 · Σσ² of a
  50-epoch-unregularized backbone). If training destabilizes, lower `coef` or ramp it
  (not implemented; add a linear ramp over the first regularized epochs).
- **`val/nll` best epoch is 50–52.** The gate works but the regularizer hurts NLL; look
  at the OOD / artifact metrics before concluding — that is the trade the paper reports.
- **Comparing estimators.** `model.spec_reg_conv_mode=reshape` regularizes the same
  quantity `spectral_norm_bound` bounds; its σ values are ~1.5× smaller at init, so
  rescale `coef` accordingly.
