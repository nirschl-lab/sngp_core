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
  report) — see `configs/artifact/README.md`. External package, not this repo's scope.

## Known test flakiness

Pre-existing, not caused by any particular change, verified against an untouched
baseline:

- `tests/metrics/test_smooth_ece.py` has order-dependent failures when run as part of
  the full suite but passes cleanly in isolation — some global state leaks between
  tests. Worth a real fix, not yet done.
- `tests/test_train.py::test_train_resume` asserts checkpoint filenames
  (`epoch_000.ckpt`) from an older `ModelCheckpoint` config pattern that no longer
  matches `configs/callbacks/model_checkpoint.yaml`/`default.yaml` (which now save
  `best.ckpt`/`last.ckpt`, not one file per epoch) — needs a product decision about
  the intended checkpoint-retention behavior before fixing, not just a test update.

## Frozen / archived

`notebooks/archive/` holds superseded exploration (old HF-checkpoint-migration
iterations, an old white-blood-cell side project) — frozen, don't import from it or
extend it.
