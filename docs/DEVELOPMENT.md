# Development Guide

This is the architecture and methodology reference for developing in this repo — how
training, models, inference, and Hugging Face publishing fit together, and where to
extend each. For *what* datasets and model architectures are currently supported, see
[docs/DATASETS.md](DATASETS.md) and [docs/SUPPORTED_MODELS.md](SUPPORTED_MODELS.md) —
kept separate so they can change without touching this document's structure.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Architecture Overview](#architecture-overview)
- [Model Methodology](#model-methodology)
- [Configuration System](#configuration-system)
- [Training](#training)
- [Checkpointing](#checkpointing)
- [Evaluation vs. Inference](#evaluation-vs-inference)
- [Publishing to Hugging Face](#publishing-to-hugging-face)
- [Metrics & Visualization](#metrics--visualization)
- [Testing](#testing)
- [Extending the Framework](#extending-the-framework)
- [Contributing](#contributing)

---

## Environment Setup

This project uses **uv** exclusively — never bare `python`/`pip install`.
`requirements.txt`/`environment.yaml` exist for legacy reasons only; `uv.lock` /
`pyproject.toml` are authoritative.

```bash
curl -Ls https://astral.sh/uv/install.sh | sh   # install uv, if needed
uv sync                                          # install locked dependencies
```

Copy `env_example` to `.env` and set `PROJECT_ROOT` — `configs/paths/default.yaml`
resolves `root_dir` from it. `log_dir`, `feature_cache_dir`, `data_cache_dir` in that
same file point at absolute cluster paths under `/data1/...`; they're
machine-specific, not portable, and not controlled by `.env`.

W&B and HF Hub auth are handled by their own CLIs, not `.env` variables:

```bash
wandb login
huggingface-cli login
```

---

## Architecture Overview

```
src/
  models/
    backbones.py            build_backbone(arch, pretrained) -- the ONE place resnet/vit get constructed
    outputs.py               ModelOutput -- every net's forward() return type
    registry.py               NET_REGISTRY / register_net / build_net -- net identity as data, not code
    components/spectral_norm.py   spectral-norm wrapping + the SNGP/ViT compatibility guard
    baseline/baseline_models.py   BaselineClassifier (+ MC-Dropout)
    sngp/sngp_classifier.py       SNGPClassifier (spectral-normed backbone + RFF-GP head)
    ensemble/deep_ensemble_model.py  DeepEnsemble net
    lit_module_base.py            shared LightningModule: lean train/val loop only, torchmetrics, checkpoint hooks
    <family>_lit_module.py        thin per-family subclasses (BaselineLitModule / SNGPLitModule /
                                   DeepEnsembleLitModule) -- override forward()/_predict_forward() only
  callbacks/
    test_artifacts_callback.py    ALL test-time analysis: per-class metrics, calibration, uncertainty,
                                   prediction CSV, diagnostic figures -- not in the LightningModule
  checkpointing/
    spec.py     CheckpointMeta / FORMAT_VERSION -- the plain-data checkpoint metadata contract
    io.py       read_meta / load_net / load_lit_module -- the ONLY sanctioned way to read a checkpoint
    legacy.py   pre-v2 checkpoint support, used only by scripts/checkpoints/migrate_checkpoints.py
    resolve.py  wandb-artifact:// URI resolution
  inference/
    infer.py           the one canonical Hydra inference entrypoint (handles artifact mode internally)
    predict_image.py   thin single/batch-image helper for scripts (not a parallel API)
  metrics/      offline/research metrics (AUROC-OOD, smooth-ECE, Dempster-Shafer) driven by prediction CSVs
  visualization/  reusable plotting, used both live (test loop) and offline (paper figures)
  paper_helpers/  per-paper/per-dataset analysis scripts consuming those CSVs
configs/        Hydra config tree (see Configuration System)
scripts/        cluster/HF-export/utility scripts (not part of the importable package)
notebooks/      exploration; notebooks/archive/ is frozen -- don't import from it or extend it
```

### Hard contracts

A small number of invariants hold across every model family and every checkpoint in
the project — every net returns a `ModelOutput`, registers via `@register_net`, keeps
non-primitive objects out of `save_hyperparameters()`, and every checkpoint read goes
through `src/checkpointing/io.py`. See
[../.claude/rules/hard-contracts.md](../.claude/rules/hard-contracts.md) for the exact
list and rationale — new code should be checked against it before anything else.

---

## Model Methodology

Three model families share the same backbone factory
(`src/models/backbones.py::build_backbone`) and the same `ModelOutput`/registry/
checkpoint contracts above — they differ only in the *head* on top of the backbone and
the *training strategy* around it:

- **Baseline** (`src/models/baseline/baseline_models.py`) — a plain classification
  head. Optional MC-Dropout at inference time trades zero retraining cost for a rough
  uncertainty estimate.
- **SNGP** (`src/models/sngp/sngp_classifier.py`) — wraps the backbone in spectral
  normalization (`src/models/components/spectral_norm.py`) and adds a random-feature
  Gaussian Process head, trading a more involved forward pass for a principled
  predictive variance. Backbone choice is restricted at construction time (resnet
  only) — see [docs/SUPPORTED_MODELS.md](SUPPORTED_MODELS.md#sngp--vit-compatibility)
  for why. Follows Liu et al. 2020 and the `edward2` reference implementation; the
  details that are easy to get wrong are spelled out in
  [#sngp-precision-matrix-and-mean-field](#sngp-precision-matrix-and-mean-field).
- **Deep Ensemble** (net in `src/models/ensemble/`, training strategy in
  `src/models/deep_ensemble_lit_module.py`) — trains `N` independently-initialized
  baseline-family members and derives uncertainty from their disagreement, trading
  `N`x training/inference cost for an uncertainty estimate that needs no architectural
  change to the underlying net.

Each family's training loop is a thin `LitModuleBase` subclass
(`src/models/<family>_lit_module.py`) that overrides only `forward`/`_predict_forward`
(and `model_step` only if the family's training loss genuinely isn't plain CE) — the
shared, lean train/val loop and checkpoint hooks live once in
`src/models/lit_module_base.py` (`LitModuleBase`) and don't get re-implemented per
family. Rich test-time analysis (per-class metrics, calibration, uncertainty, CSV/
figure export) lives entirely in
`src/callbacks/test_artifacts_callback.py::TestArtifactsCallback`, not in the
LightningModule.

See [docs/SUPPORTED_MODELS.md](SUPPORTED_MODELS.md) for the concrete backbone list and
compatibility matrix, including how to add a new backbone, net, or training strategy.

### SNGP: precision matrix and mean field

Four properties of `RandomFeatureGaussianProcess` that are easy to get wrong, and are
each covered by a test in `tests/models/sngp/`:

1. **Mean-field is applied at inference only.** In train mode `forward` returns the raw
   logits and `variance is None`; the correction `logits / sqrt(1 + lambda * var)`
   happens in eval mode only. `lambda` (`mean_field_factor`) is a **tunable** knob, not a
   constant: the reference uses 1.0 for ImageNet and 20.0 for CIFAR, and the original
   paper does not use mean field at all (it Monte-Carlo averages 10 samples). `pi/8` is
   the textbook probit value but is nearly inert at realistic dataset sizes. Tune it
   post-hoc with `scripts/checkpoints/tune_sngp_mean_field.py` -- it is inference-only,
   so it needs no retraining and cannot move accuracy or macro-F1 at all.
   Putting the correction in the *training loss* instead makes the CE objective depend on
   the covariance state, and turns it into a detached per-example gradient reweighting
   that *down-weights* uncertain examples. This project did exactly that before the
   `sngp-corrections` work; see the note below.
2. **The precision matrix is reset every epoch.** `SNGPLitModule.on_train_epoch_start`
   calls `reset_precision()`, so at any epoch boundary `precision_accum` holds exactly
   one pass over the training set — which is what the paper's single post-training sum
   means operationally, and what the reference implementation does. It also keeps
   `save_top_k=1` checkpointing consistent: the epoch that gets checkpointed carries its
   own complete precision matrix.
3. **The inverse is cached, not recomputed per batch.** `_cov_stale` marks the cached
   `covariance` dirty on every precision update; `_ensure_covariance()` recomputes lazily
   on the next eval-mode forward. Lazily, rather than in `on_train_epoch_end`, because
   Lightning runs the validation loop *before* that hook. `_cov_stale` is non-persistent,
   so a reloaded checkpoint recomputes from `precision_accum` and self-heals.
4. **Spectral norm needs warming up.** `torch.nn.utils.spectral_norm` only advances its
   power iteration on train-mode forwards, so a freshly built net divides by an estimate
   taken from random `u`/`v`. The error compounds across layers — an unwarmed
   spectral-normed resnet50 emits `nan` in eval mode, resnet18 about `1e30`.
   `apply_spectral_norm` therefore runs `DEFAULT_SN_WARMUP_ITERATIONS` power iterations
   at construction.

**Pre-correction checkpoints do not load.** The old head stored `cov_ema`/`num_updates`
instead of `precision_accum`/`covariance`, and its weights were trained against
mean-field-corrected logits, so there is no meaningful migration. `load_state_dict`
raises a message pointing at the `sngp-pre-correction` tag (equivalently the `isbi2026`
branch), which is where the ISBI 2026 paper's checkpoints remain reproducible.

---

## Configuration System

Three separate Hydra config roots, each composing the same underlying config groups
(`data/`, `model/`, `callbacks/`, `trainer/`, `paths/`, ...):

| Root | Purpose |
|---|---|
| `configs/train.yaml` | Composes `data`, `model`, `callbacks`, `logger`, `trainer`, `paths`, `extras`, optional `experiment` override, `hparams_search`, `debug`. |
| `configs/eval.yaml` | Same shape, test-only; also resolves `wandb-artifact://` checkpoint URIs (`src/checkpointing/resolve.py`). |
| `configs/infer.yaml` | Composes `data`, `infer/runtime` (device/batch-size/MC-dropout knobs), `infer/metrics`, `infer/save`. Deliberately has **no** model-architecture config group — the checkpoint is authoritative for that (see [Checkpointing](#checkpointing)), so inference never needs to be told what architecture to build. |

The preferred way to define a run is an `experiment/*.yaml` override — it patches
`data`, `model`, `trainer`, hyperparameters, and logger group/name in one file, rather
than a long CLI override string:

```bash
uv run src/train.py experiment=baseline_tang
```

Per-dataset variation is config-only (`dataset_name` + `num_classes`), never
per-dataset Python — see [docs/DATASETS.md](DATASETS.md) for the dataset list and how
to add one.

`task_name` (and therefore where a run's output directory lands under `$EXPERIMENTS_HOME`)
is derived from `model.name`/`data.name` — see [docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md)
for the full directory layout and how training/eval/inference outputs relate to each other.

---

## Training

```bash
uv run src/train.py experiment=sngp_wong
```

or compose the pieces directly instead of using an experiment preset:

```bash
uv run src/train.py \
  model=sngp_classifier \
  data=acevedo \
  data.datamodule.dataset_name=nirschl-lab/tang_et_al_2019 \
  data.datamodule.num_classes=4 \
  trainer.max_epochs=150 \
  model.optimizer.lr=2e-4 \
  logger=wandb
```

Resume from a checkpoint by passing `ckpt_path`:

```bash
uv run src/train.py experiment=sngp_wong ckpt_path=logs/train/sngp_classifier_wong/runs/<run>/checkpoints/last.ckpt
```

Deep Ensembles use one extra axis — a sequential training schedule across members
(`model.train_strategy=sequential`, the default) so `trainer.max_epochs` is divided
across `model.num_estimators` members rather than each seeing the full epoch count.
See [docs/DEEP_ENSEMBLES_GUIDE.md](DEEP_ENSEMBLES_GUIDE.md) for the schedule math and
tuning notes.

---

## Checkpointing

Every checkpoint carries a `checkpoint["sngp_core"]` metadata block
(`src/checkpointing/spec.py::CheckpointMeta`) alongside Lightning's own
`hyper_parameters`/`state_dict` — pure primitives (`format_version`, `lit_module`
dotted path, `net_spec`, `num_classes`, `idx_to_class`, `dataset_name`). This is what
lets `read_meta()` (`src/checkpointing/io.py`) tell you a checkpoint's architecture
without unpickling a live `net` object, and lets inference (§ below) skip needing a
model-architecture config entirely.

```python
from src.checkpointing.io import read_meta, load_net, load_lit_module

meta = read_meta(ckpt_path)          # architecture/format, no unpickling
net = load_net(ckpt_path)             # just the net, for inference
lit_model = load_lit_module(ckpt_path)  # full LightningModule (net + hparams)
```

Checkpoints below `FORMAT_VERSION` (`src/checkpointing/spec.py`) are refused outright,
not silently degraded — migrate once:

```bash
uv run scripts/checkpoints/migrate_checkpoints.py --in <glob> --out <dir>
```

There is no dual-format reading anywhere in production code; `src/checkpointing/legacy.py`
exists only to support that one migration script.

---

## Evaluation vs. Inference

Two distinct entrypoints, not interchangeable:

- **`src/eval.py`** — Hydra-composed, mirrors `train.yaml`'s shape (needs `model=` and
  `data=` groups). Test-set evaluation against the training-loop's own metrics/logger
  setup, and the only place `wandb-artifact://` checkpoint URIs get resolved.
- **`src/inference/infer.py`** — the **one** canonical inference entrypoint for
  everything else: single/batch prediction, checkpoint sweeps, and artifact
  (paired real/simulated) inference, driven by `configs/infer.yaml`. It reads the
  architecture straight from the checkpoint (see Checkpointing), so a call only ever
  needs a checkpoint path and a dataset:

```bash
uv run src/inference/infer.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  data=acevedo \
  save_path=/absolute/path/to/output
```

`save_path` is optional — left unset, it defaults to a dataset-scoped location under
`$EXPERIMENTS_HOME` with a run folder auto-derived from the checkpoint itself (see
[docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md)).

Artifact-paired inference is triggered by pointing `data` at
`artifact_image_classifier` — no separate script; the artifact code path is selected
automatically when the dataloader returns paired artifact batches. Full flag
reference, including guided/multi-checkpoint sweeps, is in
[docs/INFERENCE_GUIDE.md](INFERENCE_GUIDE.md). **Don't add a second inference
script** — every request in scope is expressible as an `infer.py` invocation.

---

## Publishing to Hugging Face

One export path, driven off a checkpoint's own `net_spec` (nothing to retype by hand):

| Script | Format | Use when |
|---|---|---|
| `scripts/hf/export_to_hub.py` | Self-contained `trust_remote_code` bundle (`modeling_<family>.py` assembled from `src/models/backbones.py` + `outputs.py` + the net class + a `transformers.PreTrainedModel` wrapper). | You want `AutoModel.from_pretrained(..., trust_remote_code=True)` compatibility for external consumers. |

```bash
uv run scripts/hf/export_to_hub.py --ckpt path/to/model.ckpt --out hf_export/my-model \
  --push-to-hub org/my-model --hf-token $HF_TOKEN
```

`export_to_hub.py`'s generated module has **zero `src.*` imports** — it's assembled by
concatenating the source of the dependency-light net modules
(`src/models/backbones.py` is deliberately kept free of `hydra`/`lightning`/other
`src.*` imports for exactly this reason). Edit those source files, not the generated
`modeling_*.py`; re-run the export script to regenerate it.

`tests/hf/test_export_bundle.py` verifies the full round trip — train one real step,
save, export, and check `AutoModel.from_pretrained(trust_remote_code=True)` reproduces
identical outputs. This caught a real `transformers`/`spectral_norm` interaction bug
during development; treat that test as load-bearing, not a formality.

---

## Metrics & Visualization

Two-tier split:

- **Online/per-step metrics** — `torchmetrics` objects inside `LitModuleBase`
  (accuracy, precision/recall/F1, macro-AUPRC, ECE, NLL), automatic during
  training/eval, logged via `self.log(...)`. `val/auprc_best` is the hyperparameter
  -search selection metric (see [docs/HPO_GUIDE.md](HPO_GUIDE.md)) — `val/nll`/
  `val/ece` are read-only calibration diagnostics, never a training-time selection
  target.
- **Offline/research metrics** — `src/metrics/` (cross-dataset OOD-AUROC, smooth-ECE,
  Dempster-Shafer uncertainty), driven by prediction CSVs from inference/eval runs --
  see [docs/METRICS_GUIDE.md](METRICS_GUIDE.md).

Publication figures go through `src/visualization/style.py` for consistent styling
rather than one-off plotting code -- see [docs/VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md).

---

## Testing

Four tiers, matched to what's being changed — not blanket coverage:

1. **Unit** (always run, no network/GPU): pure functions in `src/metrics/*`,
   `build_backbone`/registry/spec round-trips, checkpoint-hparams-are-JSON.
2. **Config smoke** (`tests/test_configs.py`): each config subtree composes and
   instantiates without error, checked per-group so failures are attributable.
3. **Integration** (`@pytest.mark.slow`): real (truncated) Hydra-composed training
   loops, plus full checkpoint/HF-export round trips.
4. **Untested by policy**: notebooks, `src/paper_helpers/**` plotting, cluster `.sh`
   scripts, W&B interaction — a smoke test (import + one call on synthetic data) is
   enough when touched.

```bash
make test        # fast tests, no network/GPU, no @pytest.mark.slow
make test-full     # full suite including slow/integration tests
```

Current known flaky tests are tracked in [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md) —
check there before adding a test for something that already has a documented caveat.
For which command to run for a given code change, see
[../.claude/rules/testing.md](../.claude/rules/testing.md).

---

## Extending the Framework

| Task | Use |
|---|---|
| New dataset | Copy an `configs/experiment/baseline_<name>.yaml`, change `dataset_name`/`num_classes` — see [docs/DATASETS.md](DATASETS.md). |
| New backbone / net family | Add an entry to `BACKBONES`, or a new registered net class — see [docs/SUPPORTED_MODELS.md#adding-a-backbone-or-net-family](SUPPORTED_MODELS.md#adding-a-backbone-or-net-family). |
| New training strategy (different loss, multi-stage training) | A new `LitModuleBase` subclass — see [Model Methodology](#model-methodology). |
| Run inference / checkpoint sweeps / artifact inference | `src/inference/infer.py` — see [docs/INFERENCE_GUIDE.md](INFERENCE_GUIDE.md). |
| Tune hyperparameters (fair cross-model comparison) | `scripts/hpo/sweep.sh <baseline\|sngp> <dataset>` — see [docs/HPO_GUIDE.md](HPO_GUIDE.md). |
| Offline/research metrics | `src/metrics/` — see [Metrics & Visualization](#metrics--visualization). |
| Publication figures | `src/visualization/` — see [Metrics & Visualization](#metrics--visualization). |
| Publish a trained model to HF Hub | See [Publishing to Hugging Face](#publishing-to-hugging-face). |

---

## Contributing

```bash
git checkout -b feature-name
# make changes
make format     # pre-commit run -a
make test        # before opening a PR
make test-full     # before considering the work done
```

Known out-of-scope items and pre-existing test flakiness (not to be re-litigated per
change) are tracked in [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md).
