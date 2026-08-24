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
  models/            net architectures + Lightning training strategies (see below)
  checkpointing/      the plain-data checkpoint contract (see Checkpointing)
  inference/           the one canonical inference entrypoint
  metrics/            offline/research metrics (AUROC-OOD, smooth-ECE, Dempster-Shafer)
  visualization/       reusable plotting (live during training + offline for figures)
  paper_helpers/       per-paper/per-dataset analysis scripts consuming metrics CSVs
  data/                datamodules (dataset-agnostic, config-driven)
  train.py / eval.py    Hydra entrypoints for training / test-set evaluation
configs/               Hydra config tree (see Configuration System)
scripts/                cluster jobs, HF export/upload, checkpoint migration
```

The full directory-level map lives in `CLAUDE.md` §3 — this document is the narrative
version: *why* it's shaped this way, not a file listing.

### Hard contracts

A small number of invariants hold across every model family and every checkpoint in
the project. New code should be checked against these before anything else:

- **Every net's `forward()` returns a `ModelOutput`**
  (`src/models/outputs.py`) — `.logits` for loss/argmax, `.variance` for uncertainty,
  `.raw_logits`/`.features`/`.member_logits` where relevant. Never a bare tensor,
  never a family-specific tuple. This is what lets `LitModuleBase` and
  `src/inference/infer.py` stay ignorant of which model family they're driving.
- **Every net has a `.spec` property** (plain JSON-serializable dict: registry `name`
  + constructor kwargs) and registers into `NET_REGISTRY` via `@register_net(...)`
  (`src/models/registry.py`). Architecture identity is data, not a Python import path
  — a checkpoint's `net_spec` is enough to rebuild the exact net that produced it.
- **`LitModuleBase.save_hyperparameters()` never pickles `net`/`optimizer`/
  `scheduler`** — those aren't JSON-serializable, and pickling a
  live object into hparams is what used to make checkpoints depend on the exact code
  structure they were saved with (unpickling required the original class's import
  path to still resolve). `tests/checkpointing/test_hparams_are_primitive.py` is a
  permanent regression guard for this.
- **All checkpoint I/O goes through `src/checkpointing/io.py`** — never raw
  `torch.load` + manual state-dict surgery, never a bare
  `LightningModule.load_from_checkpoint()` scattered across scripts.

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
  for why.
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
compatibility matrix. To add a new backbone, net, or training strategy, use the
`add-model` / `add-lightning-module` skills — see
[Extending the Framework](#extending-the-framework).

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
uv run src/train.py experiment=sngp_wong ckpt_path=logs/train/runs/<run>/checkpoints/last.ckpt
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

Artifact-paired inference is triggered by pointing `data` at
`artifact_image_classifier` — no separate script; the artifact code path is selected
automatically when the dataloader returns paired artifact batches. Full flag reference
in [docs/INFERENCE_GUIDE.md](INFERENCE_GUIDE.md), or use the `inference` skill for
guided/multi-checkpoint runs. **Don't add a second inference script** — every request
in scope is expressible as an `infer.py` invocation.

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
  Dempster-Shafer uncertainty), driven by prediction CSVs from inference/eval runs.
  Use the `metrics` skill.

Publication figures go through `src/visualization/style.py` for consistent styling;
use the `visualizations` skill rather than one-off plotting code.

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

Full tier definitions and current known flaky tests are in `CLAUDE.md` §7 — check
there before adding a test for something that already has a documented caveat.

---

## Extending the Framework

| Task | Use |
|---|---|
| New dataset | Copy an `configs/experiment/baseline_<name>.yaml`, change `dataset_name`/`num_classes` — see [docs/DATASETS.md](DATASETS.md). |
| New backbone / net family | `add-model` skill. |
| New training strategy (different loss, multi-stage training) | `add-lightning-module` skill. |
| Run inference / checkpoint sweeps / artifact inference | `inference` skill, or `src/inference/infer.py` directly per [docs/INFERENCE_GUIDE.md](INFERENCE_GUIDE.md). |
| Tune hyperparameters (fair cross-model comparison) | `scripts/hpo/sweep.sh <baseline\|sngp> <dataset>` — see [docs/HPO_GUIDE.md](HPO_GUIDE.md). |
| Offline/research metrics | `metrics` skill. |
| Publication figures | `visualizations` skill. |
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
change) are tracked in `CLAUDE.md` §6.
