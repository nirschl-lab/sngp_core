from typing import Dict, Optional

import albumentations as A
from torchvision.transforms import transforms

from src.data.base_image_datamodule import BaseImageDataModule
from src.data.components.hf_dataset import HFDataset


class ClassificationImageDataModule(BaseImageDataModule):
    """`LightningDataModule` for the HuggingFace image classification datasets."""

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
        super().__init__(
            dataset_name=dataset_name,
            num_classes=num_classes,
            class_to_idx=class_to_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            test_all_folds=test_all_folds,
            sample_rate=sample_rate,
            institution=institution,
        )

    def _make_dataset(self, hf_split, fold: str, transform) -> HFDataset:
        return HFDataset(hf_split, transform=transform, fold=fold)


if __name__ == "__main__":
    import yaml

    yaml_file_path = "configs/paths/default.yaml"

    try:
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        print("YAML data loaded successfully:")
    except FileNotFoundError:
        print(f"Error: The file '{yaml_file_path}' was not found.")

    data_dir = data["data_cache_dir"]
    dm = ClassificationImageDataModule(data_dir, test_all_folds=True)

    dm.setup(stage="fit")
    print("Train loader:")
    for batch in dm.train_dataloader():
        image_id, X, y, fold = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break

    print("Val loader:")
    for batch in dm.val_dataloader():
        image_id, X, y, fold = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break

    dm.setup(stage="test")
    print("Test loader:")
    for batch in dm.test_dataloader():
        image_id, X, y, fold = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break
