# CIFAR-100 / WideResNet-28-10 — SNGP vs. spectral regularization

Reproduction of the benchmark SNGP was published on (Liu et al. 2022,
[arXiv 2205.00403](https://arxiv.org/abs/2205.00403)), so that the spectral-regularization
variant can be judged against a *reproduced* SNGP number rather than only against the
Acevedo pilot ([ACEVEDO_SPECREG_RESULTS.md](../results/ACEVEDO_SPECREG_RESULTS.md)).

This page is the operational side. What the method is:
[SUPPORTED_MODELS.md](../SUPPORTED_MODELS.md); the biomedical-protocol workflow:
[SNGP_GUIDE.md](SNGP_GUIDE.md) and [SNGP_SPECREG_GUIDE.md](SNGP_SPECREG_GUIDE.md).

Status: **off-protocol by design.** These four experiments deliberately break several
things `docs/HPO_GUIDE.md` fixes for the biomedical cross-family comparison (optimizer,
schedule, selection metric, GP constants). Do not report their numbers next to the
protocol runs without saying so.

## The four arms

| Experiment | What it is | Differs by |
|---|---|---|
| `baseline_cifar100` | Deterministic WRN-28-10 | The credibility anchor |
| `sngp_cifar100` | SNGP, reference recipe | Bounded spectral norm on every conv |
| `sngp_specreg_cifar100` | Spectral reg, **matched** | Spectral *penalty* instead — everything else identical to the SNGP arm |
| `sngp_specreg_cifar100_literal` | Spectral reg, **rep-spectral-literal** | 200/250 burn-in, weight decay 0 |

The two spectral-reg arms answer different questions. The matched arm asks *is spectral
regularization better than spectral normalization, all else equal* — it is the one the
comparison rests on. The literal arm asks *does the published rep-spectral recipe
(arXiv 2405.17181) transfer to CIFAR-100*. Expect the literal arm to lose on accuracy:
200 epochs with no weight decay overfits a 36M-parameter WRN, and by the time its penalty
switches on the LR has decayed to `0.04 × 0.2³`. That gap is a finding, not a defect.

## Reference recipe

Cross-checked against the paper and the official
[`baselines/cifar/sngp.py`](https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/sngp.py)
/ [`wide_resnet_sngp.py`](https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/wide_resnet_sngp.py).

| Knob | Value | Where it lives here |
|---|---|---|
| Backbone | WRN-28-10, 640-d features, 36.5M params | `wide_resnet28_10` in `src/models/backbones.py` |
| Optimizer | SGD, Nesterov, momentum 0.9 | experiment config |
| Peak LR | **0.04** at batch 128 | experiment config |
| Schedule | 1 warmup epoch, ×0.2 at 75 / 150 / 200 | `warmup_piecewise_lr` |
| Epochs / batch | 250 / 128 | experiment config |
| Weight decay | **6e-4** | experiment config |
| Spectral norm | bound 6.0, 1 power iteration, convs only | `spectral_norm_bound` |
| GP head | `rff_dim` 1024, `length_scale` **20.0**, `ridge_penalty` 1.0, `normalize_input` false, `cov_momentum` -1.0, ORF, no output bias, `scale_random_features` **false** | experiment config |
| Mean-field factor | 7.5 | `mean_field_factor` |
| Dropout | filter-wise 0.1 | baked into `wide_resnet28_10` |
| Augmentation | zero-pad 4 → random crop 32 → hflip | `configs/img_augmentations/cifar32.yaml` |

### Three numbers that are easy to get wrong

- **LR is 0.04, not 0.08.** 0.08 is the CIFAR-10 value; `sngp.py`'s docstring says *"When
  running this script on CIFAR-100, set `base_learning_rate=0.04` and
  `gp_mean_field_factor=7.5`."* The reference also scales LR as `base_lr · batch/128`, so
  batch 128 @ 0.04 and batch 256 @ 0.08 are the same recipe.
- **Weight decay is 6e-4, not 3e-4.** `l2=3e-4` is a Keras `l2·Σw²` penalty added to the
  loss, whose gradient is `2·l2·w`; PyTorch's `weight_decay` adds `wd·w`. Equivalent is 2×.
- **Milestones 75/150/200, not 60/120/160.** The reference fixes `[60, 120, 160]` against a
  200-epoch budget and rescales: `epoch * train_epochs // 200`. `warmup_piecewise_lr`
  takes `T_max` and applies that same formula, so it is correct at any budget (a 20-epoch
  pilot decays at 6/12/16).

### Deviations from the reference, and why

| Deviation | Reason |
|---|---|
| 45k train / 5k val, stratified | The reference trains on all 50k and reports on test, with no validation split. This project selects checkpoints on a validation metric, and that split must come out of `train` or selection leaks into the reported numbers. Costs ~0.3–0.5% accuracy against published. |
| Selection on `val/loss`, fixed budget, no early stopping | The reference does no calibration-aware selection; `val/nll_cal` is still logged, just never acted on. Identical across all four arms. |
| No post-hoc calibration pass | `mean_field_factor` is pinned at the reference's 7.5 rather than fitted by `calibrate_checkpoint.py`. `val/mean_field_factor_fit` is logged, so you can see what a fitted σ would have given. |
| Warmup steps per epoch, not per step | `LitModuleBase.configure_optimizers` hardcodes `interval: "epoch"`. 1 of 250 epochs; immaterial. |

### `scale_random_features` and the GP length scale — found by piloting, not by reading

The first 20-epoch pilot had SNGP at **val/acc 0.173 against the baseline's 0.587**, on the
same backbone and optimizer. It was not failing to generalize, it was failing to *fit*
(train/acc 0.153), which pointed at the GP head rather than at the recipe.

Two things were wrong, and only one of them is visible in `sngp.py`'s flag list.

**1. `scale_random_features`.** The feature map here is `phi = sqrt(2/m)·cos(xW + b)`, and
`sqrt(2/1024) ≈ 0.044` multiplies the gradient that reaches the backbone. edward2 makes
this optional and the reference CIFAR baseline passes `scale_random_features=False`, with
the comment: *"When using GP layer as the output layer of a neural network, it is
recommended to turn this scaling off to prevent it from changing the learning rate to the
hidden layers."* Under this project's AdamW protocol the factor is invisible — Adam
renormalizes per parameter — which is why the biomedical SNGP runs never showed it. Under
plain SGD at a fixed LR it trains the backbone roughly 22x too slowly. `SNGPClassifier`
now takes `scale_random_features` (default `True`, so every existing checkpoint and the
biomedical protocol are untouched); the CIFAR arms set it `False`.

**2. The length scale is 20.0, not 1.0.** `gp_scale=1.0` is only half the reference's
parameterization. `wide_resnet_sngp.py` also passes
`OrthogonalRandomFeatures(stddev=0.05)`, and edward2 composes the two —
`gp_inputs = inputs * gp_input_scale` (with `gp_input_scale = 1/sqrt(gp_kernel_scale) = 1`)
and then a kernel whose entries have stddev 0.05. This repo folds both into one
`length_scale`, where `W = standard_normal / length_scale`, so the reference's effective
value is `1 / 0.05 = 20.0`. Reading `gp_scale` alone gives 1.0, at which the cosine
pre-activations have std ≈ 5 and the random features decorrelate (0.033 correlation
across samples) — the kernel degenerates and the head becomes noise.

Both are now set in the CIFAR experiment configs. Measured at 6 epochs, against a
deterministic control at val/acc 0.345:

| GP head config | val/acc |
|---|---|
| `scale_random_features` off + `length_scale` 20 | **0.327** |
| scaling off, `length_scale` 1.0 | 0.063 |
| scaling off, LayerNorm + `length_scale` 1.4142 | 0.010 |
| `length_scale` 20, scaling left on | 0.045 |

and re-confirmed at 20 epochs: SNGP val/acc **0.599** against the baseline's 0.595, versus
0.173 before the fix.

The lesson for the next reproduction: an edward2 GP layer's effective length scale is
`1 / (gp_input_scale · initializer stddev)`, and neither factor alone is the answer.
`scale_random_features` is a live hazard for **any** SGD-trained SNGP in this repo, not
just CIFAR — under AdamW it is invisible, so nothing else here has ever exposed it.

### Sizing `spec_reg_coef`

γ = 0.01 at cadence 24 is the rep-spectral paper's value, tuned for a ResNet18 on
CIFAR-10 under SGD 0.01, so it was not safe to assume it transfers to a 28-conv WRN at
lr 0.04. The 20-epoch pilot says it does. Against γ = 0.0025: Σσ² 227 vs 314, σ_max 4.06
vs 4.88 — measurably tighter — with CE (1.250 vs 1.238) and train accuracy (0.634 vs
0.632) unchanged. Kept at 0.01.

For scale: an untrained backbone is at Σσ² ≈ 290, and the SNGP arm's `spectral_norm_bound`
of 6.0 corresponds to an operator-norm σ_max of ~8.8 (the 1.46× below), so at σ_max 4.06
the penalty is constraining the spectrum *harder* than the spectral-norm arm is.

### `spectral_norm_bound = 6.0` is the reference's number but not the same quantity

`BoundedSpectralNorm` constrains the **reshaped** `[out, in·k·k]` Miyato norm; the
reference's `SpectralNormalizationConv2D` constrains the **true conv operator norm**.
Measured on an untrained WRN-28-10 at 32px (28 conv layers, 50 power iterations):

| Estimator | Σσ² | σ_max | σ_mean |
|---|---|---|---|
| operator | 290.0 | 6.19 | 3.09 |
| reshape | 154.1 | 6.19 | 2.18 |

Ratio operator/reshape: **mean 1.46**, range 0.99–2.09 — consistent with the ~1.5× that
[SNGP_SPECREG_GUIDE.md](SNGP_SPECREG_GUIDE.md) reports for ResNet18 at 224px.

So bounding the reshape norm at 6.0 is a **weaker** constraint than the reference's
operator-norm 6.0; the operator-equivalent bound is roughly `6 / 1.46 ≈ 4.1`. Worth an
`model.net.spectral_norm_bound=4.0` arm if the SNGP row underperforms its published
accuracy.

## Running it

Four L40S, one arm per GPU:

```bash
scripts/tmux/train_experiment.sh baseline_cifar100 test=True
scripts/tmux/train_experiment.sh sngp_cifar100 test=True
scripts/tmux/train_experiment.sh sngp_specreg_cifar100 test=True
scripts/tmux/train_experiment.sh sngp_specreg_cifar100_literal test=True
```

Outputs land under `train/<model.name>_cifar100/runs/<timestamp>/`
([OUTPUT_LAYOUT.md](../OUTPUT_LAYOUT.md)), W&B group `CIFAR100`. Record checkpoints in
[../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md).

### The `trace_logistic` arm

An override-only arm, like the `c = 4.1` control: `sngp_specreg_cifar100` with
`model.net.likelihood=trace_logistic`, which replaces the GP head's unit Laplace weight
with `1 - ||p||²` (the trace of the multinomial Hessian — see
[SNGP_GUIDE.md](SNGP_GUIDE.md#the-laplace-weight-likelihood)). Three seeds, one block:

```bash
scripts/tmux/cifar100_trace_logistic.sh     # 3 x 250 epochs, ~2.7 h, training only
```

Two things to keep straight when reading it against the `gaussian` SpecReg rows:

- **The trained model is the same.** `likelihood` feeds only the precision accumulator; it
  never reaches the loss. So accuracy and every raw-logit metric should land within
  seed-level noise of the existing SpecReg arm (0.8048 ± 0.0036). A large gap means the
  change leaked into training, not that the likelihood helped.
- **The pinned `mean_field_factor = 7.5` is wrong for this arm.** On a converged CIFAR-100
  classifier `w = 1 - ||p||² ≈ 0.02`, so the accumulator shrinks ~50×, the predictive
  variance rises by about as much, and `1 + 7.5·var` goes from ~1.15 to ~8.5 — a ~3× logit
  shrink. Run
  [`scripts/metrics/cifar100_mean_field_sweep.py`](../../scripts/metrics/cifar100_mean_field_sweep.py)
  before comparing anything; it is free, offline re-scoring from the prediction CSVs.
  Compare at `last.ckpt` as usual — `val/loss` reads mean-field logits, so `best.ckpt` will
  select a different epoch here than in the gaussian runs despite identical per-epoch
  weights.

### Pilot first

`spec_reg_coef` is **not yet calibrated for this backbone**. γ = 0.01 at cadence 24 is the
rep-spectral paper's value, tuned for a ResNet18 on CIFAR-10 under SGD 0.01; WRN-28-10 has
28 conv layers and trains at lr 0.04. Run 20 epochs of each spec-reg arm first and watch
`train/sigma_max` / `train/spec_reg` / `train/ce`:

```bash
uv run src/train.py experiment=sngp_specreg_cifar100 \
    trainer.max_epochs=20 trainer.min_epochs=20 logger.wandb.offline=true logger.wandb.log_model=false
```

Reference point: an untrained WRN-28-10 at 32px has Σσ² ≈ 290 under the operator norm
(σ_max ≈ 6.2), so `0.01 · Σσ²` starts around 2.9 against a CE of ~4.6. If σ does not move
in the regularized phase, follow the escalation order in
[SNGP_SPECREG_GUIDE.md §4](SNGP_SPECREG_GUIDE.md).

### Inference and OOD

No calibration step — run inference straight off the checkpoint. Verify with
`read_meta(<ckpt>)` first (`net_spec.arch` must be `wide_resnet28_10`).

```bash
uv run src/inference/infer.py ckpt_path=<run>/checkpoints/best.ckpt \
    data=cifar100 fold=test save_path=<out>/sngp/cifar100
# OOD: same checkpoint, the two OOD data configs
uv run src/inference/infer.py ckpt_path=<run>/checkpoints/best.ckpt \
    data=svhn fold=test save_path=<out>/sngp/cifar100_ood_svhn
uv run src/inference/infer.py ckpt_path=<run>/checkpoints/best.ckpt \
    data=cifar10 fold=test save_path=<out>/sngp/cifar100_ood_cifar10
```

Repeat for `last.ckpt` (epoch 249 — the reference's reporting point). Record output dirs
in [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md). Cross-dataset OOD
AUROC and smooth-ECE come from `src/metrics/`; figures from `src/visualization/`.

CIFAR-10 and SVHN use the same `cifar32` augmentation preset as CIFAR-100 on purpose: an
OOD input has to be preprocessed exactly the way the in-distribution data was, or the
measured OOD score partly reflects a preprocessing shift rather than the model's
uncertainty.

## Results

[../results/CIFAR100_RESULTS.md](../results/CIFAR100_RESULTS.md) — full tables for both
checkpoints, in-distribution and OOD.

Headline, three seeds, epoch 249 (`last.ckpt`, the reference's reporting point). AUROC
is MSP over the full test splits — the paper's own protocol; the Dempster-Shafer
companion table is in the results page.

| Arm | seeds | Acc | NLL | smECE | MSP AUROC C-10 | MSP AUROC SVHN |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3 | 0.8061 ± 0.0027 | 0.8074 ± 0.0096 | 0.0779 ± 0.0038 | 0.8056 ± 0.0024 | 0.7235 ± 0.0378 |
| SNGP (`c = 6.0`) | 3 | 0.8025 ± 0.0045 | 0.7912 ± 0.0073 | 0.0685 ± 0.0042 | 0.8070 ± 0.0033 | 0.7483 ± 0.0057 |
| SpecReg (matched) | 3 | 0.8048 ± 0.0036 | **0.7739 ± 0.0107** | **0.0609 ± 0.0013** | **0.8111 ± 0.0014** | **0.7857 ± 0.0080** |

Accuracy and near-OOD reproduce — baseline 0.806 / SNGP 0.803 against published
~0.798/~0.791, and near-OOD MSP 0.806/0.807 against 0.795/0.798. **Far-OOD does not**:
MSP SVHN lands 7–10 pts below the published 0.799/0.846, on 45k training images against
50k and 3 seeds against 10. Read the between-arm gaps on that axis, not the level.

Spectral regularization beats SNGP on **far-OOD (+0.037 MSP, +0.033 DS), NLL (−0.017)
and calibration (−0.008)**, sign-consistent across all three seeds under both scores. It
does **not** beat it on accuracy or near-OOD — both mixed in sign across seeds.
`c = 4.1`, the operator-norm-equivalent control for the 1.46× estimator gap below, came
out slightly worse than `c = 6.0`, so the bound was not the limiting factor.

**Do not compare the arms at `best.ckpt`.** `val/loss` selects epoch 75 / 76 / 246 for
baseline / SNGP / spectral-reg respectively — CE validation loss degrades after the first
LR drop while accuracy keeps climbing, and the spectral penalty suppresses exactly that
degradation. The apparent +2.6 pt accuracy win for spectral regularization at `best.ckpt`
is mostly that epoch gap and vanishes at equal epoch.

## What is new in the repo for this

| Path | What |
|---|---|
| `src/models/backbones.py` | `wide_resnet28_10` + `WideResNet` / `_WideBasicBlock`, vendored |
| `src/data/benchmark_image_datamodule.py` | Adapts stock HF benchmarks to the project schema |
| `src/models/components/schedulers.py` | `warmup_piecewise_lr` |
| `configs/img_augmentations/cifar32.yaml` | Native-32px CIFAR augmentation |
| `configs/data/{cifar100,cifar10,svhn}.yaml` | The three datasets |
| `configs/experiment/{baseline,sngp,sngp_specreg,sngp_specreg_..._literal}_cifar100.yaml` | The four arms |

`timm` is *not* involved: it has no CIFAR WideResNet (only the ImageNet bottleneck
`wide_resnet50_2`/`wide_resnet101_2`), and neither does torchvision. `torch_uncertainty`
ships a `wideresnet28x10`, but `src/models/backbones.py` is inlined verbatim into the HF
`trust_remote_code` bundle by `scripts/hf/export_to_hub.py`, so a third-party import there
would follow it into every exported model. Hence the vendored copy.

No `.env` changes are needed: `HF_DATASETS_CACHE` already points at the shared cache, and
`pretrained: false` means nothing is downloaded from torchvision.
