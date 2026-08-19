---
name: add-datamodule
description: Add or modify an image DataModule for this project -- a new way of loading, pairing, filtering, or constructing samples (e.g. real+simulated image pairs, custom sampling). Use whenever the dataset construction/loading logic itself needs to differ, or when extending/debugging BaseImageDataModule, ClassificationImageDataModule, or ArtifactImageDataModule (batch-size-per-device DDP scaling, class_to_idx validation, setup() stage dispatch, transform handling). Do NOT use this for adding a plain new HuggingFace dataset with the existing schema -- that's config-only via configs/data/<name>.yaml per CLAUDE.md, no code needed.
---

# Add or modify a DataModule

Scope: a new `LightningDataModule` subclassing `BaseImageDataModule` — used when *how
samples get loaded or constructed* differs (a new pairing strategy, custom
filtering/sampling, a different data source), not just a new dataset with the existing
schema. If you only need a new dataset (same 7-dataset HF schema, just different
`dataset_name`/`num_classes`/`class_to_idx`), that's a config-only change — add
`configs/data/<name>.yaml` and stop; no Python needed (see CLAUDE.md §5).

## Ask first

1. Does the *loading/construction logic* actually need to differ, or is this really
   just a new dataset config? (Most "add a dataset" requests are the latter.)
2. What differs from plain `HFDataset` — a new sample pairing (like
   `ArtifactHFDataset`'s real+simulated pairs), filtering, multi-source combination?
3. Does the new behavior apply to train/val too, or only at test time (post-hoc
   evaluation/inference)? This determines whether you override just `_make_dataset` or
   also `_make_test_dataset`.
4. Any new constructor params, and do they need validation at construction time (the
   way `class_to_idx` is validated against `num_classes`)?

## Architecture (read before writing)

- `src/data/base_image_datamodule.py` — `BaseImageDataModule`, the shared base every
  DataModule in this project subclasses. Owns: `__init__` (`dataset_name`,
  `num_classes`, `class_to_idx`, `batch_size`, `num_workers`, `pin_memory`,
  `train/val/test_augmentations`, `test_all_folds`, `sample_rate`), transform
  resolution (`_default_transform` — falls back to ImageNet `ToTensor`+`Normalize`
  with a logged warning if no augmentation config is passed), `setup()`'s stage
  dispatch, `sample_rate` subsampling, `test_all_folds` → `ConcatDataset`, class-mapping
  validation (`_record_class_mapping`), all three dataloaders, and the
  `teardown`/`state_dict`/`load_state_dict` stubs. A subclass should almost never need
  to override any of this directly — see the hook contract below.
- `src/data/components/hf_dataset.py` — `HFDataset` (wraps one HuggingFace `datasets`
  split) and `_apply_transform` (dispatches on `albumentations.A.Compose` vs
  `torchvision.transforms.Compose`). This is the one piece of dataset-construction
  logic actually shared below the DataModule level — reuse it rather than writing a
  third dataset wrapper.
- `src/data/classification_image_datamodule.py` — the minimal reference subclass:
  overrides only `_make_dataset`.
- `src/data/artifact_image_datamodule.py` — the fuller reference subclass: overrides
  both hooks to show the "different behavior at test time only" pattern
  (`ArtifactHFDataset` pairs real+simulated images, but only when building the test
  dataset — train/val always stay plain `HFDataset`), plus lazy construction of an
  external pipeline (`self.artifact_pipeline` is built on first use, not in `__init__`).
- `tests/test_datamodules.py` — the test patterns to follow, including how to
  monkeypatch `datasets.load_dataset` without hitting the network (see Steps below).

## The hook contract

A subclass implements only these two methods — everything else is inherited:

```python
def _make_dataset(self, hf_split, fold: str, transform) -> Dataset:
    """Build a Dataset for one HF split. Used for train/val, and for test/predict
    unless _make_test_dataset is overridden below."""
    raise NotImplementedError

def _make_test_dataset(self, hf_split, fold: str, transform) -> Dataset:
    """Optional override for test/predict-only construction. Defaults to
    _make_dataset — only override when test-stage behavior genuinely differs
    from train/val (e.g. artifact simulation, which never touches train/val)."""
    return self._make_dataset(hf_split, fold, transform)
```

`setup(stage=...)` resolves the effective stage (from the explicit argument, or from
`self.trainer.state.stage` when a trainer is attached and `stage` wasn't passed) and
routes to `_setup_fit` (builds `data_train`/`data_val`, calling `_make_dataset`) or
`_setup_test` (builds `data_test`, calling `_make_test_dataset`, and records the
class-to-index mapping). You should not need to touch `setup()` itself.

## Best practices / hard rules

- **Batch size**: never read the configured `batch_size` directly inside a dataloader
  method. Always use `self.batch_size_per_device`, which the base class's
  `_scale_batch_size_for_devices()` sets by dividing the configured *global*
  `batch_size` by `self.trainer.world_size`. This keeps `batch_size` in a config
  meaning the same thing (effective global batch size) regardless of how many GPUs a
  run happens to use, and raises a clear `RuntimeError` if `batch_size` isn't evenly
  divisible by `world_size` instead of letting DDP silently produce uneven per-rank
  batches. It's a no-op (`self.trainer is None`) when the DataModule is driven
  standalone, e.g. by `src/inference/infer.py` without an attached `Trainer` — you get
  this for free from the base class; don't reimplement it in a subclass.
- **`class_to_idx`**: if your subclass accepts it, just forward it to
  `super().__init__(class_to_idx=class_to_idx, ...)`. The base class validates
  `len(class_to_idx) == num_classes` at construction, and cross-checks it against the
  dataset's own `classes_to_idx` field at test-stage `setup()` (raising a clear
  `AssertionError` on mismatch). Don't build a second, parallel mapping mechanism.
- **Train/val vs. test are genuinely separate concerns.** `src/train.py` calls
  `trainer.test()` only *after* `trainer.fit()` completes (not interleaved), and
  `src/inference/infer.py` drives `setup(stage="test")`/`test_dataloader()` as a fully
  separate, post-hoc entrypoint. Keep `train_dataloader`/`val_dataloader` as the "live
  during training" contract and `test_dataloader` as strictly post-hoc — if your new
  variant needs different behavior at test time, that's exactly what
  `_make_test_dataset` is for; don't try to make test-time behavior available inside
  the fit loop.
- **Load lazily.** `_load_raw_dataset()` calls `datasets.load_dataset(self.dataset_name)`
  once per `setup()` call, only for the branch that's actually running. Don't eagerly
  load data a given `setup()` call won't use — this was a real inefficiency in the
  pre-refactor artifact DataModule that got fixed when it moved onto the shared base.
- **Transforms may be either `torchvision.transforms.Compose` or
  `albumentations.Compose`** (see `configs/img_augmentations/*.yaml` —
  `light_augmentations.yaml` is the live default using `A.Compose`, `torchvision.yaml`
  is a switchable alternative). Any new dataset-construction code must apply
  transforms through `_apply_transform` (`src/data/components/hf_dataset.py`) rather
  than assuming one type.
- **`save_hyperparameters(logger=False)` is called exactly once**, in
  `BaseImageDataModule.__init__`. Lightning's hparam capture inspects the local
  variables of whichever frame calls it, so calling it again in a subclass `__init__`
  would silently capture the wrong argument set. Subclass-only constructor params
  (e.g. `ArtifactImageDataModule`'s `artifact_csv_path`) should stay as plain
  `self.foo = foo` attributes rather than being routed through `self.hparams` — nothing
  in this codebase reads subclass-only params via `self.hparams.*`.
- This is a *data*-pipeline contract, not the *model* contract — don't confuse it with
  `ModelOutput`/`NET_REGISTRY` (see the `add-model` skill), which is a separate part of
  the codebase.

## Steps

1. Confirm this really needs new Python (see "Ask first" #1) — if not, stop and just
   add a `configs/data/<name>.yaml`.
2. Create `src/data/<name>_datamodule.py`:
   ```python
   from src.data.base_image_datamodule import BaseImageDataModule
   from src.data.components.hf_dataset import HFDataset

   class MyImageDataModule(BaseImageDataModule):
       def _make_dataset(self, hf_split, fold, transform):
           return HFDataset(hf_split, transform=transform, fold=fold)

       # only if test-time construction differs from train/val:
       def _make_test_dataset(self, hf_split, fold, transform):
           ...
   ```
   If you're adding constructor params beyond the base class's, forward the shared
   ones to `super().__init__(...)` and keep the new ones as plain attributes (see
   `ArtifactImageDataModule.__init__` for the pattern).
3. Add `configs/data/<name>.yaml` with `_target_: src.data.<name>_datamodule.MyImageDataModule`
   and the constructor kwargs (see `configs/data/artifact_image_classifier.yaml` for a
   subclass with extra params, or any of the 7 plain `configs/data/*.yaml` for the
   minimal case).
4. Add tests in `tests/test_datamodules.py`, following the existing pattern:
   monkeypatch `"src.data.base_image_datamodule.datasets.load_dataset"` (the patch
   target is the **base module**, not your subclass's module — that's where the actual
   `datasets.load_dataset` call lives) with a small in-memory fake dataset, no real
   network/HF access. Cover: `setup(stage="fit")` builds the expected dataset type for
   train/val, `setup(stage="test")` builds the expected type for test, and — if you
   added `class_to_idx` — the mismatch-raises-`AssertionError` case.

## Hard checklist

- [ ] Only `_make_dataset`/`_make_test_dataset` overridden — no reimplementation of
      `setup`, the dataloaders, or batch-size scaling.
- [ ] New constructor params forward the shared ones to `super().__init__(...)`; no
      second call to `save_hyperparameters()`.
- [ ] Test-only behavior (if any) lives in `_make_test_dataset`, not baked into
      `_make_dataset` with a stage check.
- [ ] Transform application goes through `_apply_transform`, not a new
      albumentations/torchvision branch.
- [ ] `configs/data/<name>.yaml` added with the correct `_target_`.
- [ ] Tests added in `tests/test_datamodules.py` using the monkeypatch pattern (no
      network access required to run them).

## Verify

```bash
uv run pytest tests/test_datamodules.py -v
uv run pytest tests/test_configs.py -v   # instantiates every configs/data/*.yaml
```

If you have cluster data access, also run a manual end-to-end smoke check —
`setup(stage="fit")` → `train_dataloader()`/`val_dataloader()`, then
`setup(stage="test")` → `test_dataloader()` — against real data (see the `__main__`
block in `src/data/classification_image_datamodule.py` for the pattern).
