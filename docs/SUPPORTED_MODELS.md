# Supported Models

This is a reference for what the framework can currently build — architectures,
backbones, and their compatibility constraints. For *why* the framework is structured
this way (shared contracts, registry, checkpoint identity), see
[DEVELOPMENT.md](DEVELOPMENT.md#model-methodology). For which trained checkpoints have
actually been published, see the root [README.md](../README.md#-available-models).

## Model families

| Family | Config | LightningModule | Notes |
|---|---|---|---|
| **Baseline** | `configs/model/baseline_classifier.yaml` | `BaselineLitModule` | Deterministic classifier. Optional MC-Dropout at inference (`use_mc=true`, `mc_passes=N`) for a cheap uncertainty estimate without retraining. |
| **SNGP** | `configs/model/sngp_classifier.yaml` | `SNGPLitModule` | Spectral-normalized backbone + a random-feature Gaussian Process head. Produces a predictive `variance` alongside `logits`. **resnet backbones only** — see [ViT compatibility](#sngp--vit-compatibility) below. |
| **Deep Ensemble** | `configs/model/deep_ensemble_classifier.yaml` | `DeepEnsembleLitModule` | Wraps `num_estimators` independently-initialized baseline members (any registered net as the member architecture). Uncertainty from member disagreement (`variance`, `entropy`, or `mutual_info`). See [docs/models/DEEP_ENSEMBLES_GUIDE.md](models/DEEP_ENSEMBLES_GUIDE.md) for training-schedule and tuning details. |

All three return the same `ModelOutput` shape (`src/models/outputs.py`) and register
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

`apply_spectral_norm` also runs `DEFAULT_SN_WARMUP_ITERATIONS` power iterations at
construction. `torch.nn.utils.spectral_norm` only advances its power iteration on
*train-mode* forwards, so without this a freshly built net divides each weight by
`u^T W v` for random unit `u`, `v` — an estimate that under-shoots badly, and whose
error compounds across layers. Unwarmed, an untrained spectral-normed resnet50 emits
`nan` in eval mode and resnet18 about `1e30`. Training fixes this within a few steps on
its own, so the warmup matters for anything that evaluates a net *before* training it:
Lightning's `num_sanity_val_steps` pass, and any construct-then-`eval()` call.

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
