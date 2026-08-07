import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from datasets import load_dataset
from torchvision.transforms import transforms
import albumentations as A
from PIL import Image
from src.utils import RankedLogger
import pdb
import hydra
import datasets
import ast
import numpy as np
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_CSV = REPO_ROOT / "data" / "artifact" / "artifacts.csv"
DEFAULT_ARTIFACT_CONFIG = REPO_ROOT / "configs" / "artifact" / "balanced.yaml"
DEFAULT_OUTPUT_PATH = "/data1/maheswararao/artifact_sim/dataloader"


def _normalize_fold_name(fold: str) -> str:
    fold = fold.strip().lower()
    if fold in {"val", "validation"}:
        return "validation"
    if fold in {"train", "test", "validation"}:
        return fold
    raise ValueError(f"Unsupported fold '{fold}'. Expected train, validation, or test.")


def _apply_transform(image, transform):
    if transform is None:
        return image

    if isinstance(transform, A.Compose):
        image_np = np.array(image)
        transformed = transform(image=image_np)
        return transformed["image"]

    return transform(image)


def _to_pil_image(image):
    if isinstance(image, Image.Image):
        return image

    if torch.is_tensor(image):
        array = image.detach().cpu()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = array.permute(1, 2, 0)
        array = array.numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array.squeeze(-1)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)

    if array.dtype != np.uint8:
        if array.size and array.min() >= 0 and array.max() <= 1:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    return Image.fromarray(array)


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


def _build_dataset_for_split(hf_split, fold: str, transform, simulator=None):
    if simulator is None:
        return HFDataset(hf_split, transform=transform, fold=fold)
    return ArtifactHFDataset(hf_split, simulator=simulator, transform=transform, fold=fold)


def save_simulated_artifacts(
    num_images: int,
    output_path: str = DEFAULT_OUTPUT_PATH,
    dataset_name: Optional[str] = None,
    csv_path: str = DEFAULT_ARTIFACT_CSV,
    config_path: str = DEFAULT_ARTIFACT_CONFIG,
    seed: int = 42,
) -> None:
    if dataset_name is None:
        yaml_file_path = REPO_ROOT / "configs" / "paths" / "default.yaml"
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        dataset_name = data["data_cache_dir"]

    simulator = build_artifact_pipeline(csv_path=csv_path, config_path=config_path, seed=seed)
    dataset = datasets.load_dataset(dataset_name)["test"]

    output_dir = Path(output_path)
    real_dir = output_dir / "real"
    artifact_dir = output_dir / "artifact"
    metadata_dir = output_dir / "metadata"
    for directory in (real_dir, artifact_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    saved_images = 0
    for index in range(len(dataset)):
        if saved_images >= num_images:
            break

        item = dataset[index]
        real_image = item["image"]
        artifact_applied = True
        simulated = simulator(real_image)
        try:
            simulated = simulator(real_image)
        except ValueError as exc:
            if "Artifact mask is empty" in str(exc):
                logger.warning(
                    "Artifact mask missing/empty for image {}. Saving original image as artifact fallback.",
                    index,
                )
                simulated = {"image": real_image, "metadata": {}}
                artifact_applied = False
            else:
                raise
        image_id = str(item.get("image_id", index)).replace("/", "_")

        _to_pil_image(real_image).save(real_dir / f"{image_id}.png")
        _to_pil_image(simulated["image"]).save(artifact_dir / f"{image_id}.png")

        metadata = dict(simulated.get("metadata", {}))
        metadata["image_id"] = image_id
        metadata["target"] = item.get("label")
        metadata["artifact_applied"] = artifact_applied
        with open(metadata_dir / f"{image_id}.json", "w") as file:
            json.dump(metadata, file, indent=2, default=str)

        saved_images += 1

    print(f"Saved {saved_images} simulated samples to {output_dir}")

# Custom Dataset wrapper
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, fold=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.fold = fold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            image = item['image']  # PIL Image
            label = item['label']
            mask = item.get('mask', None)  # Optional mask
            if mask is not None:
                mask = torch.tensor(mask, dtype=torch.float32)

            image_id = item['image_id']
        except TypeError as e:
            pdb.set_trace()
            raise
        if self.transform:
            image = _apply_transform(image, self.transform)

        return image_id, image, label, self.fold


class ArtifactHFDataset(Dataset):
    def __init__(self, hf_dataset, simulator, transform=None, fold=None):
        self.dataset = hf_dataset
        self.simulator = simulator
        self.transform = transform
        self.fold = fold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            image = item["image"]
            label = item["label"]
            image_id = item.get("image_id", idx)
        except TypeError:
            pdb.set_trace()
            raise

        real_image = _apply_transform(image, self.transform)
        try:
            simulated = self.simulator(image)
            artifact_simulated_image = _apply_transform(simulated["image"], self.transform)
        except ValueError as exc:
            if "Artifact mask is empty" not in str(exc):
                raise
            logger.warning("Using the real image for dataset item {} because artifact mask was empty.", idx)
            simulated = {"metadata": {}}
            artifact_simulated_image = real_image

        return {
            "image_id": image_id,
            "real_image": real_image,
            "artifact_simulated_image": artifact_simulated_image,
            "target": label,
            "fold": self.fold,
        }
    
class ClassificationImageDataModule(LightningDataModule):
    """`LightningDataModule` for the Acevedo Image dataset.

    """

    def __init__(
        self,
        dataset_name: str,
        num_classes: int = 8,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_augmentations: Optional[transforms.Compose | A.Compose] = None,  # New argument
        val_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_all_folds: Optional[bool] = False,
        sample_rate: int = 0,
        dry_run_test_fold: str = "test",
        artifact_csv_path: str = DEFAULT_ARTIFACT_CSV,
        artifact_config_path: str = DEFAULT_ARTIFACT_CONFIG,
        artifact_seed: int = 42,
        simulate_artifacts_for_test: bool = True,
    ) -> None:
        """Initialize AcevedoImageDataModule.

        :param data_dir: The data directory.
        :param batch_size: The batch size.
        :param num_workers: The number of workers.
        :param pin_memory: Whether to pin memory.
        :param augmentations: Transformations for training (from config).
        :param test_folds: None -> tests on only test fold, 'all' -> tests on train test and val
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.dry_run_test_fold = _normalize_fold_name(dry_run_test_fold)
        self.artifact_csv_path = artifact_csv_path
        self.artifact_config_path = artifact_config_path
        self.artifact_seed = artifact_seed
        self.simulate_artifacts_for_test = simulate_artifacts_for_test
        self.artifact_pipeline = None

        if train_augmentations:
            # It's a config, instantiate it
            self.train_transform = train_augmentations
        else:
            # Default fallback
            logger.warning("No train augmentations provided, using default ToTensor + Normalize.")
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # val transforms
        if val_augmentations:
                # It's a config, instantiate it
                self.val_transform = val_augmentations
        else:
            # Default fallback
            logger.warning("No train augmentations provided, using default ToTensor + Normalize.")
            self.val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Same logic for eval_transforms
        if test_augmentations:
                # It's a config, instantiate it
                self.test_transform = test_augmentations
        else:
            # Default fallback
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.log_ = RankedLogger(__name__, rank_zero_only=True)

        self.test_all_folds = test_all_folds


    def setup(self, stage: Optional[str] = None, dry_run=False) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        # pdb.set_trace()
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.simulate_artifacts_for_test and self.artifact_pipeline is None:
            self.artifact_pipeline = build_artifact_pipeline(
                csv_path=self.artifact_csv_path,
                config_path=self.artifact_config_path,
                seed=self.artifact_seed,
            )

        data = datasets.load_dataset(self.dataset_name)
        simulator = self.artifact_pipeline if self.simulate_artifacts_for_test else None

        if dry_run:
            self.data_train = _build_dataset_for_split(data["train"], "train", self.test_transform, simulator)
            self.data_val = _build_dataset_for_split(data["validation"], "validation", self.test_transform, simulator)
            self.data_test = _build_dataset_for_split(data[self.dry_run_test_fold], self.dry_run_test_fold, self.test_transform, simulator)

        # set these indices for plotting
        elif stage == "test" or (self.trainer and self.trainer.state.stage == "test"):
            if self.test_all_folds:
                self.log_.info('Testing on all folds - train, val and test')
                eval_train = _build_dataset_for_split(data['train'], 'train', self.test_transform, simulator)
                eval_val = _build_dataset_for_split(data['validation'], 'validation', self.test_transform, simulator)
                eval_test = _build_dataset_for_split(data['test'], 'test', self.test_transform, simulator)

                self.data_test = ConcatDataset([eval_train, eval_val, eval_test])
            else:
                self.data_test = _build_dataset_for_split(data['test'], 'test', self.test_transform, simulator)
            # pdb.set_trace()
            if not dry_run and self.trainer is not None:
                data_test_ = data['test']
                self.trainer.test_classes_to_idx = ast.literal_eval(data_test_[0]['classes_to_idx'])
                self.trainer.test_idx_to_classes = {idx:cls for cls,idx in self.trainer.test_classes_to_idx.items()}

        else:
            #sample images from train and val for faster training
            if self.sample_rate > 0:
                data_train_ = data['train'].shuffle(seed=42).select(range(self.sample_rate))
                data_val_ = data["validation"].shuffle(seed=42).select(range(self.sample_rate//2))
            else:
                data_train_ = data['train']
                data_val_ = data["validation"]
            
            self.data_train = HFDataset(data_train_, transform=self.train_transform, fold='train')
            self.data_val = HFDataset(data_val_, transform=self.val_transform, fold='val')

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # self.log_.info('------------------->< * * ><----------train loader called---')
        self.log_.info(f'Training samples: {len(self.data_train)}')
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # self.log_.info('------------------->< * * ><----------val loader called---')
        self.log_.info(f'Validation samples: {len(self.data_val)}')
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.

        """
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

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save simulated artifact samples to disk.")
    parser.add_argument("--num-images", '-n', type=int, help="Number of simulated samples to save.")
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Directory where the simulated outputs will be written. Default: {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()

    save_simulated_artifacts(num_images=args.num_images, output_path=args.output_path)
    