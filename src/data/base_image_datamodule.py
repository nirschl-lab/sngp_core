import ast
from typing import Any, Dict, Optional

import albumentations as A
import datasets
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import transforms

from src.utils import RankedLogger
from loguru import logger

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BaseImageDataModule(LightningDataModule):
    """Shared `LightningDataModule` behavior for the HuggingFace-backed image datasets.

    Subclasses only need to implement `_make_dataset` (used to build the train/val
    datasets, and the test dataset unless `_make_test_dataset` is overridden).
    """

    def __init__(
        self,
        dataset_name: str,
        num_classes: int = 8,
        class_to_idx: Optional[Dict[str, int]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_augmentations: Optional[transforms.Compose | A.Compose] = None,
        val_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_all_folds: Optional[bool] = False,
        sample_rate: int = 0,
        institution: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.dataset_name = dataset_name
        self.num_classes = num_classes
        if class_to_idx is not None:
            assert len(class_to_idx) == num_classes, (
                f"class_to_idx has {len(class_to_idx)} entries but num_classes={num_classes}: {class_to_idx}"
            )
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.test_all_folds = test_all_folds
        self.institution = institution

        self.train_transform = train_augmentations or self._default_transform("train")
        self.val_transform = val_augmentations or self._default_transform("val")
        self.test_transform = test_augmentations or self._default_transform("test")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.log_ = RankedLogger(__name__, rank_zero_only=True)

    @staticmethod
    def _default_transform(name: str) -> transforms.Compose:
        logger.warning(f"No {name} augmentations provided, using default ToTensor + Normalize.")
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _scale_batch_size_for_devices(self) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number "
                    f"of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def _load_raw_dataset(self):
        return datasets.load_dataset(self.dataset_name)

    # ---- hooks subclasses implement ----
    def _make_dataset(self, hf_split, fold: str, transform) -> Dataset:
        raise NotImplementedError

    def _make_test_dataset(self, hf_split, fold: str, transform) -> Dataset:
        return self._make_dataset(hf_split, fold, transform)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or
            `"predict"`. Defaults to ``None``, in which case the trainer's own state
            (if a trainer is attached) is used to resolve it.
        """
        self._scale_batch_size_for_devices()

        resolved_stage = stage or getattr(getattr(self.trainer, "state", None), "stage", None)
        data = self._load_raw_dataset()
        if self.institution is not None:
            data = self._filter_by_institution(data)

        if resolved_stage in {"test", "predict"}:
            self._setup_test(data)
        else:
            self._setup_fit(data)

    def _filter_by_institution(self, data) -> Dict[str, Any]:
        filtered = {}
        for split_name, split_ds in data.items():
            if "institution" not in split_ds.column_names:
                raise ValueError(
                    f"institution={self.institution!r} was requested but the {split_name!r} split "
                    f"of {self.dataset_name!r} has no 'institution' column."
                )
            filtered_split = split_ds.filter(lambda row: row["institution"] == self.institution)
            if len(filtered_split) == 0:
                raise ValueError(
                    f"institution={self.institution!r} matched zero rows in the {split_name!r} split "
                    f"of {self.dataset_name!r}."
                )
            filtered[split_name] = filtered_split
        return filtered

    def _setup_fit(self, data) -> None:
        # Sample images from train and val for faster training.
        if self.sample_rate > 0:
            train_split = data["train"].shuffle(seed=42).select(range(self.sample_rate))
            val_split = data["validation"].shuffle(seed=42).select(range(self.sample_rate // 2))
        else:
            train_split = data["train"]
            val_split = data["validation"]

        self.data_train = self._make_dataset(train_split, "train", self.train_transform)
        self.data_val = self._make_dataset(val_split, "validation", self.val_transform)

    def _setup_test(self, data) -> None:
        if self.test_all_folds:
            self.log_.info("Testing on all folds - train, val and test")
            parts = [
                self._make_test_dataset(data["train"], "train", self.test_transform),
                self._make_test_dataset(data["validation"], "validation", self.test_transform),
                self._make_test_dataset(data["test"], "test", self.test_transform),
            ]
            self.data_test = ConcatDataset(parts)
        else:
            self.data_test = self._make_test_dataset(data["test"], "test", self.test_transform)

        self._record_class_mapping(data["test"])

    def _record_class_mapping(self, test_split) -> None:
        if self.trainer is None:
            return

        dataset_classes = ast.literal_eval(test_split[0]["classes_to_idx"])
        self.trainer.test_classes_to_idx = dataset_classes
        self.trainer.test_idx_to_classes = {idx: cls for cls, idx in dataset_classes.items()}

        if self.class_to_idx is not None:
            assert self.class_to_idx == dataset_classes, (
                "configured class_to_idx does not match the dataset's own mapping: "
                f"{self.class_to_idx} != {dataset_classes}"
            )
        else:
            self.class_to_idx = dataset_classes

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        self.log_.info(f"Training samples: {len(self.data_train)}")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        self.log_.info(f"Validation samples: {len(self.data_val)}")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        if self.data_test is None:
            raise RuntimeError(
                "No test dataset was built. Call setup(stage='test') before creating a test dataloader."
            )
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given
        datamodule `state_dict()`.
        """
        pass
