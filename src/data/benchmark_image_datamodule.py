"""DataModule for stock HuggingFace vision benchmarks (CIFAR-100/10, SVHN).

The seven `nirschl-lab` datasets share one schema — `image` / `label` / `image_id`, a
`classes_to_idx` string on every test row, and train/validation/test splits — which is
why adding one of *those* is config-only. Public benchmarks do not share it:

| | `uoft-cs/cifar100` | `uoft-cs/cifar10` | `ufldl-stanford/svhn` |
|---|---|---|---|
| image column | `img` | `img` | `image` |
| label column | `fine_label` (+ `coarse_label`) | `label` | `label` |
| `image_id` | absent | absent | absent |
| `classes_to_idx` | absent | absent | absent |
| splits | train/test | train/test | train/test/extra |

So `ClassificationImageDataModule` fails three ways on them: `_setup_fit` needs
`data["validation"]`, `HFDataset.__getitem__` needs `image_id`, and
`_record_class_mapping` needs `classes_to_idx`. This subclass closes exactly those
gaps and changes nothing else — the batch contract downstream is still the same
`(image_ids, images, labels, fold)` 4-tuple.

Used by `configs/data/cifar100.yaml` (the CIFAR-100 / WRN-28-10 SNGP benchmark) and by
`configs/data/cifar10.yaml` / `configs/data/svhn.yaml`, which exist only as OOD sets for
cross-dataset AUROC. See `docs/models/CIFAR100_BENCHMARK.md`.
"""
from typing import Any, Dict, Optional, Sequence

import datasets

from src.data.classification_image_datamodule import ClassificationImageDataModule

_USED_SPLITS = ("train", "validation", "test")


class BenchmarkImageDataModule(ClassificationImageDataModule):
    """Adapts a stock HF vision benchmark to this project's split/column schema.

    :param dataset_config_name: HF dataset config (SVHN needs ``"cropped_digits"``).
    :param image_column: Source column holding the image, renamed to ``image``.
    :param label_column: Source column holding the class index, renamed to ``label``.
    :param drop_columns: Columns to discard (CIFAR-100's ``coarse_label``).
    :param val_split_size: Rows to hold out of ``train`` as ``validation``, stratified
        by label. Ignored when the dataset already ships a ``validation`` split.
    :param val_split_seed: Seed for that carve — fixed, so the split is identical across
        every run and every model family being compared.
    """

    def __init__(
        self,
        *,
        dataset_config_name: Optional[str] = None,
        image_column: str = "img",
        label_column: str = "label",
        drop_columns: Sequence[str] = (),
        val_split_size: int = 5000,
        val_split_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Subclass-only params stay plain attributes: `save_hyperparameters` is called
        # exactly once, in BaseImageDataModule.__init__, and inspects *that* frame.
        self.dataset_config_name = dataset_config_name
        self.image_column = image_column
        self.label_column = label_column
        self.drop_columns = tuple(drop_columns)
        self.val_split_size = val_split_size
        self.val_split_seed = val_split_seed

    def _load_raw_dataset(self) -> Dict[str, datasets.Dataset]:
        """Load the benchmark and return splits in this project's schema.

        Overriding this (rather than `_make_dataset`) is the point: the shape mismatch is
        in the *raw data*, so fixing it here leaves `setup()`, the dataloaders, batch-size
        scaling and `HFDataset` all working unmodified.
        """
        raw = datasets.load_dataset(self.dataset_name, self.dataset_config_name)

        splits = {}
        for name, split in raw.items():
            # `setup()` only ever reads these three. SVHN also ships a 531k-row `extra`
            # split, which there is no reason to rename, index and cache.
            if name not in _USED_SPLITS:
                continue
            if self.image_column != "image":
                split = split.rename_column(self.image_column, "image")
            if self.label_column != "label":
                split = split.rename_column(self.label_column, "label")
            drop = [c for c in self.drop_columns if c in split.column_names]
            if drop:
                split = split.remove_columns(drop)
            splits[name] = split

        if "validation" not in splits:
            splits = self._carve_validation_split(splits)

        # `add_column` appends to the Arrow table without touching the `image` column;
        # a `.map()` here would decode and re-encode every image instead.
        return {
            name: split.add_column("image_id", [f"{name}-{i}" for i in range(len(split))])
            for name, split in splits.items()
        }

    def _carve_validation_split(
        self, splits: Dict[str, datasets.Dataset]
    ) -> Dict[str, datasets.Dataset]:
        """Hold `val_split_size` rows out of `train`, stratified by label.

        The reference CIFAR recipes train on all 50k and report on the test set, with no
        validation split at all. This project selects checkpoints on a validation metric,
        so a split has to come from somewhere — and it has to come out of `train`, never
        out of `test`, or selection leaks into the reported numbers.
        """
        if "train" not in splits:
            raise ValueError(
                f"{self.dataset_name!r} has no 'train' split to carve a validation split "
                f"from (found {sorted(splits)})."
            )
        if not 0 < self.val_split_size < len(splits["train"]):
            raise ValueError(
                f"val_split_size must be in (0, {len(splits['train'])}) for "
                f"{self.dataset_name!r}, got {self.val_split_size}."
            )

        carved = splits["train"].train_test_split(
            test_size=self.val_split_size,
            seed=self.val_split_seed,
            stratify_by_column="label",
        )
        return {**splits, "train": carved["train"], "validation": carved["test"]}

    def _record_class_mapping(self, test_split) -> None:
        """Cross-check the configured `class_to_idx` against the split's `ClassLabel`.

        The base class reads a `classes_to_idx` string column off test row 0, which these
        datasets do not carry. Rather than materializing that column (a ~2 KB JSON string
        duplicated across every test row), validate against the `ClassLabel` feature — the
        dataset's own authoritative mapping, and free to read. Same guarantee the base
        class provides: a mis-specified `class_to_idx` in the YAML fails loudly at
        `setup(stage="test")` instead of silently mislabelling every prediction.
        """
        if self.trainer is None:
            return

        label_feature = test_split.features["label"]
        names = getattr(label_feature, "names", None)
        if names is None:
            raise ValueError(
                f"The 'label' column of {self.dataset_name!r} is a "
                f"{type(label_feature).__name__}, not a ClassLabel, so its class names "
                "cannot be read."
            )
        dataset_classes = {name: idx for idx, name in enumerate(names)}

        self.trainer.test_classes_to_idx = dataset_classes
        self.trainer.test_idx_to_classes = {idx: cls for cls, idx in dataset_classes.items()}

        if self.class_to_idx is not None:
            assert self.class_to_idx == dataset_classes, (
                "configured class_to_idx does not match the dataset's own ClassLabel names: "
                f"{self.class_to_idx} != {dataset_classes}"
            )
        else:
            self.class_to_idx = dataset_classes
