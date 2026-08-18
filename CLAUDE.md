# CLAUDE.md

> **Status: v0, written at the start of an architecture migration.** Sections marked
> `⚠️ target (being implemented)` describe contracts that do not fully hold in the code yet — they are
> the brief this refactor is working toward. Once the refactor lands, this file gets rewritten in
> present tense (v1). See "Known issues / roadmap" for what's tracked and not yet done.

## 1. What this project is

Uncertainty-aware histopathology image classification research (`sngp-core`), built on PyTorch Lightning
+ Hydra (fork of `ashleve/lightning-hydra-template`). Two core model families — deterministic classifiers
(with optional MC-Dropout) and SNGP (Spectral-Normalized Neural Gaussian Process) — plus a Deep Ensembles
family, all sharing backbones `resnet18/34/50`, `vit_b_16/32`, `vit_l_16/32`, `vit_h_14`. **SNGP excludes
ViT backbones** — spectral normalization is not compatible with ViT internals.

Models are trained/evaluated against 7 HuggingFace datasets under the `nirschl-lab` org, all with the
same schema, differing only in class count and content:

| Dataset | HF id |
|---|---|
| Tang et al. 2019 | `nirschl-lab/tang_et_al_2019` |
| Wong et al. 2022 | `nirschl-lab/wong_et_al_2022` |
| Jung et al. 2022 | `nirschl-lab/jung_et_al_2022` |
| Nirschl et al. 2018 | `nirschl-lab/nirschl_et_al_2018` |
| Kather et al. 2016 | `nirschl-lab/kather_et_al_2016` |
| Kather et al. 2018 | `nirschl-lab/kather_et_al_2018` |
| Acevedo et al. 2020 | `nirschl-lab/acevedo_et_al_2020` |

Only 4 of the 7 currently have `configs/experiment/*.yaml` presets (acevedo, kather2018, tang, wong);
the rest are run via `configs/data/image_classifier.yaml` overrides.

The paper (ISBI 2026, arXiv:2602.02370) is already accepted; ongoing work is evaluating robustness to
simulated imaging artifacts (via the external `histo-artifact-sim` package) and publishing trained
models to the HF Hub. Outputs of this project are: trained checkpoints, HF Hub model repos, and
metrics/figures for publications.

## 2. Environment & commands

**uv only.** Never run bare `python` or `pip install` — always `uv run <script>` / `uv sync`.
`requirements.txt` / `environment.yaml` exist for legacy reasons; `uv.lock` + `pyproject.toml` are
authoritative.

Requires a `.env` (see `env_example`) and a `PROJECT_ROOT` environment variable — `configs/paths/default.yaml`
resolves `root_dir` from it. Note `log_dir`, `feature_cache_dir`, and `data_cache_dir` currently point at
absolute cluster paths under `/data1/...` — these are machine-specific, not portable.

```bash
uv sync                                          # install deps
uv run src/train.py experiment=baseline_tang     # train (Hydra-driven; see §4 for config system)
uv run src/eval.py ckpt_path=<path> ...          # evaluate a checkpoint
uv run src/inference/infer.py ckpt_path=<path> data=<dataset config>   # the single inference entrypoint, see §5
make test                                        # fast tests (not slow)
make test-full                                   # full test suite including @pytest.mark.slow
make format                                      # pre-commit run -a
```

## 3. Architecture map

```
src/
  models/
    baseline/           net class: deterministic classifier (+ MC-Dropout)
    sngp/                net class: SNGP classifier (spectral-normed backbone + RFF-GP head)
    ensemble/            net class: Deep Ensemble (wraps N baseline members)
    lit_module_base.py   shared LightningModule: train/val/test loop, torchmetrics, CSV/plot logging
    *_lit_module.py      thin per-family subclasses (model_step / forward overrides only)
  data/                  HF-dataset-backed LightningDataModules (classification + artifact-paired variants)
  inference/infer.py     the one canonical inference entrypoint (Hydra-driven), see docs/INFERENCE_GUIDE.md
  metrics/                offline/research metrics (AUROC-OOD, smooth-ECE, Dempster-Shafer) — standalone,
                          driven by prediction CSVs written at test time, not integrated with torchmetrics
  visualization/          reusable plotting used both live (test loop) and offline (paper figures)
  paper_helpers/          per-paper/per-dataset analysis scripts consuming those CSVs
configs/                  Hydra config tree — see §4
scripts/                  one-off/cluster/HF-export/utility scripts (not part of the importable package)
notebooks/                exploration; notebooks/archive/ holds frozen, superseded work — do not import from it
```

### ⚠️ Target contracts (being implemented)

These are not all true of the code yet — they are the direction this refactor is taking, so treat any
code you touch as needing to move toward them, and check new code against them:

- Every net's `forward` returns a single `ModelOutput` dataclass (`logits`, `raw_logits`, `variance`,
  `features`, `member_logits`), never a bare tensor or a bespoke tuple. Callers use `.logits` for
  loss/argmax and `.variance` for uncertainty — no per-model-family branching in callers.
- Every net exposes a `spec` property of **plain, JSON-serializable data** (name/arch/num_classes/...)
  and is registered under a string key in a central registry (`src/models/registry.py`). Backbone
  construction (resnet/vit) lives in exactly one place, `src/models/backbones.py` — no per-family
  copies.
- `LitModuleBase.save_hyperparameters(...)` must `ignore=` any non-primitive constructor argument
  (`net`, `optimizer`, `scheduler`, `calibration_cfg`, ...). **`checkpoint["hyper_parameters"]` must
  always be JSON-serializable.** This is the fix for the historical "checkpoints expect the same code
  structure they were saved with" problem: today `save_hyperparameters()` is called with no `ignore=`
  and pickles the live `net` object, which breaks the moment a model class moves or is renamed (this
  is why `src/models/torch_vision_base.py` currently exists as a compatibility shim).
- All checkpoint loading goes through one canonical loader (`src/checkpointing/io.py` once it lands) —
  never raw `torch.load` + manual state-dict surgery, never bare `LightningModule.load_from_checkpoint`
  scattered across scripts. Today there are 3-4 independent, differently-fragile ways checkpoints get
  loaded across `train.py`, `eval.py`, `infer.py`, and `scripts/`.

## 4. Config system

Three separate Hydra config roots, all under `configs/`:

- `train.yaml` — composes `data`, `model`, `callbacks`, `logger`, `trainer`, `paths`, `extras`, optional
  `experiment` override, `hparams_search`, `debug`.
- `eval.yaml` — same shape as `train.yaml` but test-only; also resolves `wandb-artifact://` checkpoint URIs.
- `infer.yaml` — the inference entrypoint's root config; currently composes model config through a
  **separate, hand-maintained `configs/infer/model/*.yaml` namespace** cross-referenced by filename
  string rather than Hydra's defaults list (this indirection is a refactor target, see roadmap).

Preferred way to define a run: an `experiment/*.yaml` override (patches `data`, `model`, `trainer`,
hyperparameters, logger group/name) rather than long CLI override strings. Per-dataset variation is
**config-only** — `dataset_name` + `num_classes` — never per-dataset Python code; keep it that way when
adding a dataset (see §5).

Config groups: `data/`, `model/`, `callbacks/`, `logger/`, `trainer/`, `paths/`, `experiment/`,
`hparams_search/`, `debug/`, `img_augmentations/`, `infer/`, `artifact/` (external
`histo-artifact-sim` sampling profiles).

## 5. Common workflows

- **Add a dataset**: copy an existing `configs/experiment/baseline_<name>.yaml`, change `dataset_name`
  and `num_classes`. No new Python code — the datamodule schema is uniform across all 7 HF datasets.
- **Add a model / backbone**: use the `add-model` skill.
- **Add a new training strategy (Lightning module)**: use the `add-lightning-module` skill.
- **Run inference**: use the `inference` skill, or `uv run src/inference/infer.py` directly per
  `docs/INFERENCE_GUIDE.md`. There is deliberately **one** inference entrypoint — don't add a second.
- **Compute offline/research metrics** (AUROC-OOD, calibration, artifact-quantification): use the
  `metrics` skill.
- **Produce a publication figure**: use the `visualizations` skill.
- **Publish a trained model to HF Hub**: see `scripts/migrate_to_hf.py` / `scripts/upload_to_hf.py` (this
  path is being unified — see roadmap).

## 6. Known issues / roadmap

Being addressed in this migration (see the plan file for the active session, or ask what stage we're on):
- Checkpoint `hyper_parameters` pickling a live `net` object (root cause of code-structure coupling).
- Backbone-construction code duplicated across ~4 files (`baseline_models.py`, `sngp_classifier.py`,
  HF-export copies).
- `net.forward()` return-type asymmetry (tensor vs. tuple) forcing per-family branching in consumers.
- `configs/infer/model/*.yaml` as a parallel, string-cross-referenced namespace to `configs/model/*.yaml`.
- Duplicated OOD/AUROC analysis code between `src/paper_helpers/ood_metrics/` and (now-removed)
  `notebooks/metrics/`.
- `examples/README.md` previously described an inference API (`ImagePreprocessor`, `ImageInference`,
  `src/inference/core.py`) that no longer exists in the codebase — being reconciled.

Explicitly **out of scope** for this migration (tracked here so it isn't re-litigated per session):
- Consolidating the ~14 generated/output directories at repo root (`artifacts/`, `csv/`, `figures/`,
  `logs/`, `wandb/`, etc.) under one output root — high churn, touches every cluster script, no
  correctness benefit; do as its own PR with a compat period.
- The two `ClassificationImageDataModule` classes (in `src/data/classification_image_datamodule.py` and
  `src/data/artifact_image_datamodule.py`) share a class name, disambiguated only by Hydra `_target_`
  string. `classification_image_datamodule.py`'s `setup()` also has a real bug: when `stage=None` with a
  trainer attached, neither the train/val nor test/predict branch fires. Both need a careful, separately
  tested pass — noted here, not yet fixed.
- Re-enabling CI (`.github/workflows/*.yaml.disabled`) — worth doing once the test suite is fast/green.
- `histo-artifact-sim`'s own roadmap (tissue-aware placement, multi-instance coverage, stain-space
  effects, elastic deformation, a real-vs-simulated validation report) — see `configs/artifact/README.md`.
  This is an external package; not this repo's scope.

`notebooks/archive/` holds superseded exploration (old HF-checkpoint-migration iterations, an old
white-blood-cell side project) — frozen, don't import from it or extend it.

## 7. Testing strategy

Four tiers — match new tests to the tier of what you're changing, don't aim for blanket coverage:

1. **Unit (always run, no network/GPU)**: pure functions — `src/metrics/*`, backbone/registry
   construction at `pretrained=False`, checkpoint-hparams-are-JSON assertions. This is already the
   strongest-tested part of the repo (`tests/metrics/`) — treat it as the model to follow.
2. **Config smoke**: `tests/test_configs.py` composes + instantiates each config subtree individually
   (deliberately per-group, not all-at-once, so failures are attributable to one file) and checks it
   constructs without error.
3. **Integration (`@pytest.mark.slow`)**: `tests/test_train.py` / `test_eval.py` / `test_sweeps.py` run
   real (truncated) Hydra-composed training loops. `tests/test_datamodules.py` shows the pattern for
   testing HF-dataset-dependent code without network access — monkeypatch `datasets.load_dataset` with
   an in-memory fake dataset; reuse that fixture rather than re-deriving it.
4. **Untested by policy** (stated explicitly, not a gap to feel guilty about): notebooks,
   `src/paper_helpers/**` plotting scripts, cluster `.sh` scripts, W&B interaction. When touched, a
   smoke test (import + one call on tiny synthetic data) is enough.

Run `make test` (fast) during normal iteration; `make test-full` before considering work done.

## 8. Conventions

- Logging: `loguru` / the project's `RankedLogger` in Lightning code, not bare `print`.
- Type hints on public functions.
- No new top-level output directories at repo root without discussion — the existing sprawl
  (`artifacts/`, `csv/`, `figures/`, `plots/`, ...) is already a known problem (§6).
- Figures go through `src/visualization/style.py` for consistent styling across paper figures.
- Never edit or import from `notebooks/archive/`.
- Absolute paths belong in Hydra configs (`configs/paths/`), not hardcoded in source.
