# Supported Models

This is a reference for what the framework can currently build — architectures,
backbones, and their compatibility constraints. For *why* the framework is structured
this way (shared contracts, registry, checkpoint identity), see
[DEVELOPMENT.md](DEVELOPMENT.md#model-methodology). For which trained checkpoints have
actually been published, see the root [README.md](../README.md#-available-models).

## Model families

| Family | Config | LightningModule | Notes |
|---|---|---|---|
| **Baseline** | `configs/model/baseline_classifier.yaml` | `BaselineLitModule` | Deterministic classifier. Optional MC-Dropout at inference (`use_mc=true`, `mc_passes=N`) for a cheap uncertainty estimate without retraining. Post-hoc calibration knob: `temperature` (`logits / T`), fit on validation NLL by `scripts/checkpoints/calibrate_checkpoint.py` and stored in the checkpoint's `net_spec`. |
| **SNGP** | `configs/model/sngp_classifier.yaml` | `SNGPLitModule` | Spectral-normalized backbone + a random-feature Gaussian Process head. Produces a predictive `variance` alongside `logits`. Constants fixed to Liu et al. (2022) Table 9; `spectral_norm_bound` (the paper's `c`, eq. 15) is the swept knob and `mean_field_factor` (the paper's kernel amplitude sigma) the post-hoc one. **resnet backbones only** — see [ViT compatibility](#sngp--vit-compatibility) below. |
| **SNGP + spectral regularization** | `configs/model/sngp_specreg_classifier.yaml` | `SNGPSpectralRegLitModule` | Same net as SNGP (`sngp_classifier` registry key, `use_spectral_norm: false`) but the backbone's Lipschitz constant is controlled by a **loss term**, `CE + coef · Σ σ_max²`, over the backbone's Conv2d/Linear layers (rep-spectral, Yang et al. 2024) instead of weight rescaling. GP head excluded from the penalty. Pilot / research variant — see [below](#sngp-with-spectral-regularization-rep-spectral) and [docs/models/SNGP_SPECREG_GUIDE.md](models/SNGP_SPECREG_GUIDE.md). |
| **Deep Ensemble** | `configs/model/deep_ensemble_classifier.yaml` | `DeepEnsembleLitModule` | Wraps `num_estimators` independently-initialized baseline members (any registered net as the member architecture). Uncertainty from member disagreement (`variance`, `entropy`, or `mutual_info`). Ensemble-level post-hoc `temperature` on the pooled logits. See [docs/models/DEEP_ENSEMBLES_GUIDE.md](models/DEEP_ENSEMBLES_GUIDE.md) for training-schedule and tuning details. |

Every family returns the same `ModelOutput` shape (`src/models/outputs.py`) and registers
into the same `NET_REGISTRY` (`src/models/registry.py`) — an ensemble member is just
another net built from a `base_model_spec`, so e.g. an ensemble of SNGP members is
possible without new code, only a config that points `base_model_spec.name` at
`sngp_classifier`'s registry key.

### Comparing uncertainty across these families

`ModelOutput.variance` is a **different physical quantity** in each: a GP latent
variance for SNGP, a mean per-class logit *variance* for a Deep Ensemble, a mean
per-class logit *standard deviation* for MC-Dropout, and `None` for a plain Baseline.
It reaches the prediction CSVs as one `uncertainty` column, so they are not
interchangeable and ranking families by that column is meaningless.

For cross-family comparison use the `predictive_entropy` / `confidence_margin` /
`dempster_shafer` columns, written on every row for every family, or
`mutual_information` between two runs that both have member stacks. Each row's
`uncertainty_kind` records which quantity its `uncertainty` is. One implementation
behind all of them: [src/metrics/uncertainty.py](../src/metrics/uncertainty.py); full
table in
[.claude/skills/metrics/references/csv_schema.md](../.claude/skills/metrics/references/csv_schema.md).

## Backbones

Built in exactly one place, `src/models/backbones.py`:

| `arch` | Kind | Pretrained weights |
|---|---|---|
| `resnet18` | resnet | ImageNet1K_V1 |
| `resnet34` | resnet | ImageNet1K_V1 |
| `resnet50` | resnet | ImageNet1K_V2 |
| `vit_b_16` | vit | ImageNet1K_V1 |
| `vit_b_32` | vit | ImageNet1K_V1 |
| `vit_l_16` | vit | ImageNet1K_V1 |
| `vit_l_32` | vit | ImageNet1K_V1 |
| `vit_h_14` | vit | ImageNet1K_V1 |

Select via `model.net.arch=<name>` (baseline/SNGP) or
`model.net.base_model_spec.arch=<name>` (deep ensemble).

## SNGP × ViT compatibility

SNGP recursively wraps every `Conv2d`/`Linear` in the backbone with spectral
normalization. That wrapping isn't validated against ViT internals
(`LayerNorm`/attention), so ViT architectures are rejected at construction time —
`SPECTRAL_NORM_COMPATIBLE` in `src/models/backbones.py` enumerates the allowed set
(currently all `resnet*` entries). Passing a `vit_*` arch to `sngp_classifier` raises
immediately rather than silently training something unvalidated.

The wrapping is bounded, not hard: with `spectral_norm_bound: c` (the paper's eq. 15,
`BoundedSpectralNorm` in `src/models/components/spectral_norm.py`) a weight is rescaled
to spectral norm `c` only when its estimate exceeds `c`, otherwise left alone -- exactly
edward2's `SpectralNormalization(norm_multiplier=c)`. `null` is the stock
`torch.nn.utils.spectral_norm` (always sigma = 1), kept so checkpoints written before the
bound existed rebuild identically. Both use torch's reshaped-matrix estimate of a conv
kernel's spectral norm, not the conv operator norm, which is one reason `c` is a swept
hyperparameter rather than a derived constant.

`apply_spectral_norm` also runs `DEFAULT_SN_WARMUP_ITERATIONS` power iterations at
construction. `torch.nn.utils.spectral_norm` only advances its power iteration on
*train-mode* forwards, so without this a freshly built net divides each weight by
`u^T W v` for random unit `u`, `v` — an estimate that under-shoots badly, and whose
error compounds across layers. Unwarmed, an untrained spectral-normed resnet50 emits
`nan` in eval mode and resnet18 about `1e30`. Training fixes this within a few steps on
its own, so the warmup matters for anything that evaluates a net *before* training it:
Lightning's `num_sanity_val_steps` pass, and any construct-then-`eval()` call.

## SNGP with spectral regularization (rep-spectral)

Yang, Zavatone-Veth & Pehlevan, *Spectral regularization for adversarially-robust
representation learning* ([arXiv 2405.17181](https://arxiv.org/abs/2405.17181),
[code](https://github.com/Pehlevan-Group/rep-spectral)) bound the representation map's
Jacobian not by rescaling weights but by penalizing them, eq. (5):

```
loss = CE + gamma * sum_{l < L} sigma_max^2(W_l)
```

summed over the *representation* layers only — the readout `W_L` is deliberately left
out. `SNGPSpectralRegLitModule` (`src/models/sngp_specreg_lit_module.py`) is that
recipe wrapped around the SNGP net: the penalty runs over every `Conv2d`/`Linear` of
`net.backbone`, the GP head (LayerNorm, fixed random-feature projection, `beta`
classifier) is never touched, and the net is built with `use_spectral_norm: false` so
**no weight is rescaled anywhere** — the LightningModule refuses a spectral-normalized
backbone, the two mechanisms are never combined. Everything else is `SNGPLitModule`
(per-epoch precision reset, train-mode-only precision accumulation, plain CE; validation
and test are pure CE so `val/nll` stays comparable across families).

`SpectralRegularizer` (`src/models/components/spectral_reg.py`) does the measuring. For
a conv it estimates, by default, the spectral norm of the **linearized convolution**
(the paper's `K~`, App. B.1) with a `conv2d`/`conv_transpose2d` power iteration on the
layer's real input shape — exact for zero padding and any stride, the same operator
edward2's `SpectralNormalizationConv2D` normalizes. `spec_reg_conv_mode: reshape`
switches to Miyato's `[out, in·k·k]` kernel-matrix estimate, the quantity
`spectral_norm_bound` is expressed in; it under-shoots the operator norm (by up to the
kernel size), so logged `train/sigma_*` values and `c` are **not** numerically
comparable. The gradient is the paper's closed form `2 sigma u v^T` (u, v detached);
the power-iteration vectors are non-persistent, so checkpoints of this variant are plain
SN-free SNGP checkpoints and load through `src/checkpointing/io.py` unchanged.

Knobs (`configs/model/sngp_specreg_classifier.yaml`, paper-literal for its ResNet18
recipe): `spec_reg_coef` (γ, 0.01), `spec_reg_every_n_steps` (penalty and power
iteration every N optimizer steps; 24), `spec_reg_burnin_epochs` (epochs trained
without the penalty first — a budget knob set per experiment; the paper uses 80 % of
the run), `spec_reg_n_power_iterations` (1), `spec_reg_conv_mode`,
`spec_reg_warmup_iterations`. During burn-in the penalty is still evaluated (without
gradient) on the same cadence so `train/sigma_max` / `train/sigma_mean` /
`train/spec_reg` show how far the unregularized backbone drifts before regularization
starts; `train/spec_reg_active` is 0/1.

Because a burn-in trains two different models in one run, pair it with
`ModelCheckpointFromEpoch(start_epoch=<burn-in>)`
(`src/callbacks/model_checkpoint_from_epoch.py`): `last.ckpt` rolls from epoch 0, but
`best.ckpt` is ranked only over regularized epochs. Caveats shared with spectral
normalization: BatchNorm's affine scale and the residual `1 + prod sigma` are controlled
by neither method, so the Lipschitz bound is loose in the same way for both.

## Adding a backbone or net family

- A new **backbone option** for existing families: add an entry to `BACKBONES` in
  `src/models/backbones.py`.
- A new **net class/family**: implement `forward()` returning `ModelOutput`, add a
  `.spec` property, and register with `@register_net("...")` — see
  [../.claude/rules/hard-contracts.md](../.claude/rules/hard-contracts.md) for the
  full contract every net must satisfy.
- A new **training strategy** around an existing net (different loss, multi-stage
  training): add a new `LitModuleBase` subclass instead — that's a training-loop
  change, not a new net. See
  [DEVELOPMENT.md#model-methodology](DEVELOPMENT.md#model-methodology).
