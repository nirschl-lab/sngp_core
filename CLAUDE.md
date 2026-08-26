# CLAUDE.md

## 1. What this project is

Uncertainty-aware histopathology image classification research (`sngp-core`), built on
PyTorch Lightning + Hydra (fork of `ashleve/lightning-hydra-template`). Three model
families share backbones `resnet18/34/50`, `vit_b_16/32`, `vit_l_16/32`, `vit_h_14`:

- **Baseline** — deterministic classifier, optional MC-Dropout at inference.
- **SNGP** (Spectral-Normalized Neural Gaussian Process) — resnet-only; ViT is
  rejected at construction time (`SPECTRAL_NORM_COMPATIBLE` in
  `src/models/backbones.py`) because spectral-norm wrapping isn't validated against
  ViT internals.
- **Deep Ensemble** — wraps N independently-initialized baseline members.

Trained/evaluated against 7 HuggingFace datasets under the `nirschl-lab` org, all with
the same schema, differing only in class count and content:

| Dataset | HF id | Classes |
|---|---|---|
| Acevedo et al. 2020 | `nirschl-lab/acevedo_et_al_2020` | 8 |
| Tang et al. 2019 | `nirschl-lab/tang_et_al_2019` | 4 |
| Wong et al. 2022 | `nirschl-lab/wong_et_al_2022` | 4 |
| Kather et al. 2018 | `nirschl-lab/kather_et_al_2018` | 9 |
| Jung et al. 2022 | `nirschl-lab/jung_et_al_2022` | 5 |
| Nirschl et al. 2018 | `nirschl-lab/nirschl_et_al_2018` | 2 |
| Kather et al. 2016 | `nirschl-lab/kather_et_al_2016` | 8 |

Each dataset has its own `configs/data/<dataset>.yaml` (`acevedo`, `tang`, `wong`,
`kather2018`, `kather2016`, `jung`, `nirschl2018`) carrying `dataset_name`/
`num_classes`/`class_to_idx`. Only the first four have `configs/experiment/*.yaml`
presets today; the rest are used via `data=<dataset>` CLI composition (see §5).

The paper (ISBI 2026, arXiv:2602.02370) is accepted; ongoing work evaluates
robustness to simulated imaging artifacts (external `histo-artifact-sim` package) and
publishes trained models to the HF Hub. Outputs of this project: trained checkpoints,
HF Hub model repos, and metrics/figures for publications.

## 2. Environment & commands

**uv only.** Never run bare `python` or `pip install` — always `uv run <script>` /
`uv sync`. `requirements.txt`/`environment.yaml` exist for legacy reasons;
`uv.lock`/`pyproject.toml` are authoritative.

Requires `.env` (see `env_example`) and a `PROJECT_ROOT` env var —
`configs/paths/default.yaml` resolves `root_dir` from it. `log_dir`,
`feature_cache_dir`, `data_cache_dir` point at absolute cluster paths under
`/data1/...` — machine-specific, not portable.

```bash
uv sync                                              # install deps
uv run src/train.py experiment=baseline_tang         # train (see §4 for config system)
uv run src/eval.py ckpt_path=<path> ...               # evaluate a checkpoint
uv run src/inference/infer.py ckpt_path=<path> data=<dataset config>   # inference, see §5
uv run scripts/checkpoints/migrate_checkpoints.py --in <glob> --out <dir>   # migrate old checkpoints
uv run scripts/hf/export_to_hub.py --ckpt <path> --out <dir>          # export to HF Hub bundle
make test        # fast tests (not slow)
make test-full    # full suite including @pytest.mark.slow
make format       # pre-commit run -a
```

## 3. Architecture map

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
configs/        Hydra config tree, see §4
scripts/        cluster/HF-export/utility scripts (not part of the importable package)
notebooks/      exploration; notebooks/archive/ is frozen -- don't import from it or extend it
```

### Hard contracts

Every net and every checkpoint in this project satisfies these; check new code
against them:

- **Every net's `forward()` returns a `ModelOutput`** (`src/models/outputs.py`) —
  `.logits` for loss/argmax, `.variance` for uncertainty (SNGP predictive variance /
  ensemble disagreement), `.raw_logits`/`.features`/`.member_logits` where relevant.
  Never a bare tensor, never a bespoke tuple.
- **Every net has a `.spec` property** (plain JSON-serializable dict: registry `name` +
  ctor kwargs) and is registered in `NET_REGISTRY` via `@register_net("...")`. Backbone
  construction lives in exactly one place, `src/models/backbones.py`.
- **`LitModuleBase.save_hyperparameters()` ignores `net`/`optimizer`/`scheduler`**
  — none of those are JSON-serializable. `checkpoint["hyper_parameters"]`
  must always be JSON-serializable; `tests/checkpointing/test_hparams_are_primitive.py`
  is a permanent regression guard for this. This is the fix for the historical
  "checkpoints expect the same code structure they were saved with" problem: the old
  code pickled the live `net` object into hparams, so unpickling required the
  *original* class's import path to still resolve.
- **All checkpoint I/O goes through `src/checkpointing/io.py`** — never raw
  `torch.load` + manual state-dict surgery, never a bare
  `LightningModule.load_from_checkpoint()` scattered across scripts. `read_meta()`
  tells you a checkpoint's architecture/format without unpickling anything.
- **Checkpoints below `FORMAT_VERSION` are refused**, not silently degraded. Migrate
  once with `scripts/checkpoints/migrate_checkpoints.py`; there is no dual-format
  reading in production code.

## 4. Config system

Three separate Hydra config roots under `configs/`:

- **`train.yaml`** — composes `data`, `model`, `callbacks`, `logger`, `trainer`,
  `paths`, `extras`, optional `experiment` override, `hparams_search`, `debug`.
- **`eval.yaml`** — same shape, test-only; also resolves `wandb-artifact://` checkpoint
  URIs (`src/checkpointing/resolve.py`).
- **`infer.yaml`** — composes `data`, `infer/runtime` (device/batch-size/MC-dropout
  knobs only), `infer/metrics`, `infer/save`. There is **no** model-architecture group
  here — the checkpoint is authoritative for that (§3), so inference never needs to be
  told what architecture to build.

Preferred way to define a run: an `experiment/*.yaml` override (patches `data`,
`model`, `trainer`, hyperparameters, logger group/name) rather than long CLI override
strings. Per-dataset variation is **config-only** — `dataset_name` + `num_classes` —
never per-dataset Python; keep it that way when adding a dataset (§5).

Config groups: `data/`, `model/`, `callbacks/`, `logger/`, `trainer/`, `paths/`,
`experiment/`, `hparams_search/`, `debug/`, `img_augmentations/`, `infer/`,
`artifact/` (external `histo-artifact-sim` sampling profiles).

## 5. Common workflows

- **Add a dataset**: add a `configs/data/<dataset>.yaml` (copy an existing one, e.g.
  `configs/data/tang.yaml`, and change `dataset_name`/`num_classes`/`class_to_idx`/`name`
  — the last is a short, path-safe identifier distinct from `dataset_name`, used for
  output-directory naming, see `docs/OUTPUT_LAYOUT.md`) — that's the single source of
  truth for dataset identity now, not something set inline per experiment file.
  Reference it from an experiment file via `override /data: <dataset>` (or compose ad
  hoc on the CLI, `data=<dataset> model=baseline_classifier`, for a dataset with no
  experiment preset yet). No new Python — the datamodule schema is uniform across all
  7 HF datasets.
- **Add a new DataModule variant** (a new way of loading/pairing/filtering samples,
  not just a new dataset with the existing schema — e.g. real+simulated image
  pairing): use the `add-datamodule` skill.
- **Add a model / backbone**: use the `add-model` skill.
- **Add a new training strategy (Lightning module)**: use the `add-lightning-module`
  skill.
- **Run inference**: use the `inference` skill, or `uv run src/inference/infer.py`
  directly per `docs/INFERENCE_GUIDE.md`. There is deliberately **one** inference
  entrypoint — don't add a second.
- **Compute offline/research metrics** (AUROC-OOD, calibration, artifact
  quantification): use the `metrics` skill.
- **Tune hyperparameters** for a fair cross-model comparison: `scripts/hpo/sweep.sh
  <baseline|sngp> <dataset>` (Hydra + Optuna, `configs/hparams_search/`). Selects on
  `val/auprc_best`, never on calibration/uncertainty metrics — see
  `docs/HPO_GUIDE.md` for the full protocol and rationale.
- **Produce a publication figure**: use the `visualizations` skill.
- **Publish a trained model to HF Hub**: `scripts/hf/export_to_hub.py` (self-contained
  `trust_remote_code` bundle, `AutoModel.from_pretrained` compatible).

## 6. Known issues / roadmap

**Explicitly out of scope** for the current migration (tracked here so it isn't
re-litigated per session):
- Consolidating the ~14 generated/output directories at repo root (`artifacts/`,
  `csv/`, `figures/`, `logs/`, `wandb/`, etc.) under one output root — high churn,
  touches every cluster script, no correctness benefit; do as its own PR with a compat
  period.
- The two `ClassificationImageDataModule` classes (in
  `src/data/classification_image_datamodule.py` and
  `src/data/artifact_image_datamodule.py`) share a class name, disambiguated only by
  Hydra `_target_` string. `classification_image_datamodule.py`'s `setup()` also has a
  real bug: when `stage=None` with a trainer attached, neither the train/val nor
  test/predict branch fires. Both need a careful, separately tested pass.
- Re-enabling CI (`.github/workflows/*.yaml.disabled`) — worth doing once the test
  suite is fast/green.
- `histo-artifact-sim`'s own roadmap (tissue-aware placement, multi-instance coverage,
  stain-space effects, elastic deformation, a real-vs-simulated validation report) —
  see `configs/artifact/README.md`. External package, not this repo's scope.

**Known test flakiness** (pre-existing, not caused by this migration, verified against
an untouched baseline): `tests/metrics/test_smooth_ece.py` has order-dependent
failures when run as part of the full suite but passes cleanly in isolation — some
global state leaks between tests. Worth a real fix, not yet done.
`tests/test_train.py::test_train_resume` asserts checkpoint filenames
(`epoch_000.ckpt`) from an older `ModelCheckpoint` config pattern that no longer
matches `configs/callbacks/model_checkpoint.yaml`/`default.yaml` (which now save
`best.ckpt`/`last.ckpt`, not one file per epoch) — needs a product decision about the
intended checkpoint-retention behavior before fixing, not just a test update.

`notebooks/archive/` holds superseded exploration (old HF-checkpoint-migration
iterations, an old white-blood-cell side project) — frozen, don't import from it or
extend it.

## 7. Testing strategy

Four tiers — match new tests to the tier of what you're changing, don't aim for
blanket coverage:

1. **Unit (always run, no network/GPU)**: pure functions — `src/metrics/*`,
   `build_backbone`/registry/spec round-trips (`tests/models/`), checkpoint-hparams-
   are-JSON (`tests/checkpointing/test_hparams_are_primitive.py`).
2. **Config smoke**: `tests/test_configs.py` composes + instantiates each config
   subtree individually (deliberately per-group, not all-at-once, so failures are
   attributable to one file) and checks it constructs without error.
3. **Integration (`@pytest.mark.slow`)**: `tests/test_train.py`/`test_eval.py`/
   `test_sweeps.py` run real (truncated) Hydra-composed training loops.
   `tests/checkpointing/test_roundtrip.py` and `tests/hf/test_export_bundle.py` train
   one real step, save, and verify every load path (including
   `AutoModel.from_pretrained(trust_remote_code=True)` for the HF bundle) reproduces
   identical outputs — this is how a real transformers/spectral_norm interaction bug
   was caught during Stage 4; an import-only check would have missed it. Use the
   in-memory fake-dataset pattern in `tests/checkpointing/test_roundtrip.py`'s
   `_FakeClassificationDataset` (same idea as `tests/test_datamodules.py`'s
   monkeypatched `datasets.load_dataset`) to avoid network access.
4. **Untested by policy** (stated explicitly, not a gap to feel guilty about):
   notebooks, `src/paper_helpers/**` plotting, cluster `.sh` scripts, W&B interaction.
   When touched, a smoke test (import + one call on tiny synthetic data) is enough.

### Scoped test selection (don't run the whole suite for a local change)

`make test`/`make test-full` run every test regardless of what changed. For normal
iteration, run only the tests mapped to the paths you touched — not the full suite:

| Changed path | Run |
|---|---|
| `src/metrics/**` | `pytest tests/metrics/` |
| `src/models/backbones.py`, `src/models/components/spectral_norm.py` | `pytest tests/models/test_backbone_factory.py tests/models/baseline/test_backbones.py tests/models/sngp/test_spectral_norm.py` |
| `src/models/baseline/**`, `src/models/baseline_lit_module.py` | `pytest tests/models/baseline/ tests/models/test_output_contract.py` |
| `src/models/sngp/**`, `src/models/sngp_lit_module.py` | `pytest tests/models/sngp/` |
| `src/models/ensemble/**`, `src/models/deep_ensemble_lit_module.py` | `pytest tests/checkpointing/test_ensemble_assembly.py tests/models/test_output_contract.py` |
| `src/checkpointing/legacy.py`, `src/checkpointing/resolve.py` | `pytest tests/checkpointing/` |
| `src/data/classification_image_datamodule.py`, `artifact_image_datamodule.py`, `mnist_datamodule.py` | `pytest tests/test_datamodules.py` |
| `src/callbacks/**` | `pytest tests/callbacks/` |
| `src/inference/**` | `pytest tests/test_infer.py` |
| `src/train.py` | `pytest tests/test_train.py` |
| `src/eval.py` | `pytest tests/test_eval.py` |
| `src/paper_helpers/**` | `pytest tests/paper_helpers/` |
| `src/visualization/**` | tier 4 — no dedicated suite; smoke-test manually |
| `scripts/hf/export_to_hub.py` | `pytest tests/hf/` |
| `configs/data/*.yaml` | `pytest tests/test_configs.py::TestDatasetConfigDrift tests/test_datamodules.py` |
| `configs/model/*.yaml` | `pytest tests/test_configs.py::TestModelConfigs` |
| `configs/experiment/*.yaml` | `pytest tests/test_configs.py::TestExperimentClassFreqConsistency` |
| `configs/hparams_search/**` | `pytest tests/test_sweeps.py` |
| any other `configs/**` | `pytest tests/test_configs.py` |

**Core files — no safe scoped subset, run `make test` instead:** these are the
single-source-of-truth contracts from §3's architecture map, load-bearing across
every model family and dataset, so a scoped subset gives false confidence —
`src/models/lit_module_base.py`, `src/models/outputs.py`, `src/models/registry.py`,
`src/checkpointing/spec.py`, `src/checkpointing/io.py`,
`src/data/base_image_datamodule.py`, `src/utils/**`, `tests/conftest.py`,
`configs/paths/**`, `configs/trainer/**`.

Rules:
1. Map each changed file to its row (or to "core"); if a change spans multiple rows,
   union their commands rather than escalating to the full suite.
2. A path not in this table (new module/top-level dir) defaults to `make test` —
   don't guess a narrower scope the table doesn't cover.
3. Scoped runs keep the same fast/slow split as `make test`/`make test-full`
   (`pytest <paths> -k "not slow"` by default); include the slow tests in that same
   scope only when the change specifically touches integration/round-trip behavior
   there (e.g. checkpoint save/load logic → also run
   `tests/checkpointing/test_roundtrip.py`'s slow cases).
4. Scoped runs are for fast in-loop feedback, not a substitute for the real gate:
   still run `make test` (fast) before considering a change ready, and `make
   test-full` before considering the work genuinely done.

## 8. Conventions

- Logging: `loguru` / the project's `RankedLogger` in Lightning code, not bare
  `print`.
- Type hints on public functions.
- No new top-level output directories at repo root without discussion (§6).
- Figures go through `src/visualization/style.py` for consistent styling.
- Never edit or import from `notebooks/archive/`.
- Absolute paths belong in Hydra configs (`configs/paths/`), not hardcoded in source.
- Every new net registers via `@register_net(...)` and returns `ModelOutput`; every
  new LightningModule keeps `net`/`optimizer`/`scheduler`/anything non-primitive out
  of `save_hyperparameters()`. See the `add-model`/`add-lightning-module` skills.
- After making code/config/doc changes for a task, stage and commit them (no need to
  ask first). Stage only the specific files touched for that task, by name — this
  repo's working tree routinely carries unrelated pre-existing modifications/deletions
  (scratch notebooks, WIP scripts, etc.) that must not be swept in with `git add -A`/`.`.
