# Known Issues & Roadmap

Tracked here so they aren't re-litigated on every change.

## Explicitly out of scope (for now)

- **Consolidating output directories.** ~14 generated/output directories sit at repo
  root (`artifacts/`, `csv/`, `figures/`, `logs/`, `wandb/`, etc.). Unifying them under
  one output root is high-churn (touches every cluster script) with no correctness
  benefit on its own — worth doing as its own PR with a compatibility period, not
  bundled into unrelated work.
- **Duplicate `ClassificationImageDataModule` class name.** Two classes share this
  name — `src/data/classification_image_datamodule.py` and
  `src/data/artifact_image_datamodule.py` — disambiguated only by Hydra `_target_`
  string. `classification_image_datamodule.py`'s `setup()` also has a real bug: when
  `stage=None` with a trainer attached, neither the train/val nor test/predict branch
  fires. Both need a careful, separately tested pass.
- **Re-enabling CI** (`.github/workflows/*.yaml.disabled`) — worth doing once the test
  suite is fast/green.
- **`histo-artifact-sim`'s own roadmap** (tissue-aware placement, multi-instance
  coverage, stain-space effects, elastic deformation, a real-vs-simulated validation
  report) — see that package's own README. External package, not this repo's scope.

## Stale data on disk

- **The `uncertainty` column in every pre-existing `mc_*` inference run is wrong.**
  `BaselineClassifier.mc_predict` returns a per-class std (`[B, C]`), but the old record
  builder flattened it and took the first `B` values, so each row got another sample's
  first-class std. Affects `infer/mc_baseline_classifier_{acevedo,wong}/**/predictions.csv`
  written before the fix in `src/inference/records.py::_reduce_uncertainty` (mean over
  classes, matching `predict_image.py`). No published number moves — nothing reads that
  column; `csv/ood_metrics/*.csv` scores on max-softmax and entropy. Re-running those
  checkpoints regenerates the column correctly.
- **Every pre-existing `mc_*` inference run is unseeded and therefore not
  reproducible.** `configs/infer.yaml`'s `seed` field was declared but never applied
  before `run_inference` started calling `set_random_seed` (see
  [docs/INFERENCE_GUIDE.md](../docs/INFERENCE_GUIDE.md)). Deterministic checkpoints
  (baseline without MC-Dropout, SNGP, Deep Ensemble) are unaffected -- this only
  matters for MC-Dropout's stochastic forward passes. Re-running an `mc_*` checkpoint
  with `seed` set will not reproduce the exact numbers already published for it, since
  those were never seeded to begin with.

- **Every checkpoint and sweep from before the fair-comparison protocol is
  off-protocol.** They were trained with `ClassBalancedFocalLoss` (tuned `cb_beta` /
  `focal_gamma` per family), stock hard spectral normalization (no `spectral_norm_bound`),
  swept `rff_dim` / `length_scale`, and selected on `val/auprc_best`; the SNGP ones from
  before the `sngp-corrections` branch also carry the pre-correction head. They still
  load (additive spec keys default to the old behaviour) and the ISBI 2026 numbers remain
  reproducible from them, but they are not comparable to runs under the current protocol
  (plain CE, calibrated-val-NLL selection, one post-hoc knob per family --
  [docs/HPO_GUIDE.md](HPO_GUIDE.md)). The Optuna sqlite studies under
  `${EXPERIMENTS_HOME}/${PROJECT_NAME}/optuna/*.db` are historical only; nothing reads
  them any more.

## Known test flakiness

Pre-existing, not caused by any particular change, verified against an untouched
baseline:

- `tests/test_train.py::test_train_resume` asserts checkpoint filenames
  (`epoch_000.ckpt`) from an older `ModelCheckpoint` config pattern that no longer
  matches `configs/callbacks/model_checkpoint.yaml`/`default.yaml` (which now save
  `best.ckpt`/`last.ckpt`, not one file per epoch) — needs a product decision about
  the intended checkpoint-retention behavior before fixing, not just a test update.
- `tests/test_eval.py::test_train_eval` fails with `KeyError: 'test/acc_final'`. That
  metric is logged by `TestArtifactsCallback` (`src/callbacks/test_artifacts_callback.py`),
  so the test depends on a callback the eval path does not always run.

## Frozen / archived

`notebooks/archive/` holds superseded exploration (old HF-checkpoint-migration
iterations, an old white-blood-cell side project) — frozen, don't import from it or
extend it.
