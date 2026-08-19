from pathlib import Path
from typing import Dict, Optional

import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from loguru import logger

from src.data.base_image_datamodule import BaseImageDataModule
from src.data.components.hf_dataset import HFDataset, _apply_transform

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_CSV = REPO_ROOT / "data" / "artifact" / "artifacts.csv"
DEFAULT_ARTIFACT_CONFIG = REPO_ROOT / "configs" / "artifact" / "balanced.yaml"


def build_artifact_pipeline(
    csv_path: str = DEFAULT_ARTIFACT_CSV,
    config_path: str = DEFAULT_ARTIFACT_CONFIG,
    seed: int = 42,
):
    try:
        from histo_artifacts import ArtifactPipeline
    except ImportError as exc:
        raise ImportError(
            "histo_artifacts is required for artifact simulation. Install it before using this module."
        ) from exc

    return ArtifactPipeline.from_files(str(csv_path), str(config_path), seed=seed)


class ArtifactHFDataset(Dataset):
    """Pairs each real image with an artifact-simulated version from `simulator`."""

    def __init__(self, hf_dataset, simulator, transform=None, fold=None):
        self.dataset = hf_dataset
        self.simulator = simulator
        self.transform = transform
        self.fold = fold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        image_id = item.get("image_id", idx)

        real_image = _apply_transform(image, self.transform)
        try:
            simulated = self.simulator(image)
            artifact_simulated_image = _apply_transform(simulated["image"], self.transform)
        except ValueError as exc:
            if "Artifact mask is empty" not in str(exc):
                raise
            logger.warning("Using the real image for dataset item {} because artifact mask was empty.", idx)
            artifact_simulated_image = real_image

        return {
            "image_id": image_id,
            "real_image": real_image,
            "artifact_simulated_image": artifact_simulated_image,
            "target": label,
            "fold": self.fold,
        }


class ArtifactImageDataModule(BaseImageDataModule):
    """`LightningDataModule` that pairs real images with artifact-simulated versions
    for the test/predict stage. Train/val always use plain (unsimulated) images.
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
        artifact_csv_path: str = DEFAULT_ARTIFACT_CSV,
        artifact_config_path: str = DEFAULT_ARTIFACT_CONFIG,
        artifact_seed: int = 42,
        simulate_artifacts_for_test: bool = True,
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
        )
        self.artifact_csv_path = artifact_csv_path
        self.artifact_config_path = artifact_config_path
        self.artifact_seed = artifact_seed
        self.simulate_artifacts_for_test = simulate_artifacts_for_test
        self.artifact_pipeline = None

    def _make_dataset(self, hf_split, fold: str, transform) -> HFDataset:
        # Train/val never see artifact simulation, regardless of simulate_artifacts_for_test.
        return HFDataset(hf_split, transform=transform, fold=fold)

    def _make_test_dataset(self, hf_split, fold: str, transform) -> Dataset:
        if not self.simulate_artifacts_for_test:
            return HFDataset(hf_split, transform=transform, fold=fold)

        if self.artifact_pipeline is None:
            self.artifact_pipeline = build_artifact_pipeline(
                csv_path=self.artifact_csv_path,
                config_path=self.artifact_config_path,
                seed=self.artifact_seed,
            )

        return ArtifactHFDataset(
            hf_split, simulator=self.artifact_pipeline, transform=transform, fold=fold
        )
