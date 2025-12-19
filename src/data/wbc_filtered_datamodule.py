from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from datasets import load_dataset
from torchvision.transforms import transforms
import albumentations as A
import yaml
from src.utils import RankedLogger
import pdb
import hydra
import datasets
import ast
import numpy as np
from loguru import logger
import pandas as pd
from PIL import Image
import os

class WBCDataset(Dataset):
    def __init__(self, csv_file, image_dir, classes_to_idx, transform=None, fold=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            fold (str, optional): Fold identifier (train/val/test).
        """
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.classes_to_idx = classes_to_idx
        self.transform = transform
        self.fold = fold
        
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Handle the case where idx might be a tensor (from Subset)
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
            
        img_name = self.annotations.iloc[idx]['ID']
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label if available
        if 'labels' in self.annotations.columns and pd.notna(self.annotations.iloc[idx]['labels']):
            label_name = self.annotations.iloc[idx]['labels']
            label = self.classes_to_idx[label_name]
        else:
            # For test set or missing labels
            label = -1
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations expects numpy arrays
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return img_name, image, label, self.fold

class WBCTrainDatasetWBC(Dataset):
    def __init__(self, csv_file, image_root_dir, classes_to_idx, transform=None, fold=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            fold (str, optional): Fold identifier (train/val/test).
        """
        self.annotations = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.classes_to_idx = classes_to_idx
        self.transform = transform
        self.fold = fold
        
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Handle the case where idx might be a tensor (from Subset)
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
            
        img_name = self.annotations.iloc[idx]['ID']
        phase = self.annotations.iloc[idx]['phase']
        if phase == 'phase2':
            img_path = os.path.join(self.image_root_dir, phase, 'train', img_name)
        else: #phase1
            img_path = os.path.join(self.image_root_dir, phase, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label if available
        assert 'labels' in self.annotations.columns, 'Labels column is missing in annotations'
        assert pd.notna(self.annotations.iloc[idx]['labels']), f"Label is missing for index {idx}"
        label_name = self.annotations.iloc[idx]['labels']
        label = self.classes_to_idx[label_name]
    
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations expects numpy arrays
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return img_name, image, label, self.fold


class WBCClassificationDataModule(LightningDataModule):
    """LightningDataModule for the WBC Phase2 classification dataset.
    
    This datamodule handles CSV-based WBC data with image files in separate directories.
    Designed specifically for the WBC-bench-2026 dataset structure.
    """

    def __init__(
        self,
        data_dir: str = '/data1/shared/data/wbc-bench-2026/',
        num_classes: int = 8,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 32,
        pin_memory: bool = False,
        class_indices: Dict[str, int] = None,
        train_augmentations: Optional[transforms.Compose | A.Compose] = None,
        val_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_augmentations: Optional[transforms.Compose | A.Compose] = None,
        sample_rate: int = 0,
    ) -> None:
        """Initialize WBCImageDataModule.

        :param data_dir: The data directory containing CSV files and phase2 folder.
        :param batch_size: The batch size.
        :param num_workers: The number of workers.
        :param pin_memory: Whether to pin memory.
        :param train_augmentations: Transformations for training.
        :param val_augmentations: Transformations for validation.
        :param test_augmentations: Transformations for testing.
        :param sample_rate: Number of samples to use for faster training (0 for all).
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.sample_rate = sample_rate

        # Set up transforms
        if train_augmentations:
            self.train_transform = train_augmentations
        else:
            logger.warning("No train augmentations provided, using default Resize + ToTensor + Normalize.")
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to consistent dimensions
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if val_augmentations:
            self.val_transform = val_augmentations
        else:
            logger.warning("No val augmentations provided, using default Resize + ToTensor + Normalize.")
            self.val_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to consistent dimensions
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if test_augmentations:
            self.test_transform = test_augmentations
        else:
            self.test_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to consistent dimensions
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_batch_size_per_device = train_batch_size
        self.val_batch_size_per_device = val_batch_size
        self.test_batch_size_per_device = test_batch_size

        self.log_ = RankedLogger(__name__, rank_zero_only=True)

        # Store class mappings
        self.classes_to_idx = class_indices
        self.idx_to_classes = {v: k for k, v in class_indices.items()} if class_indices else None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        
        # Divide batch size by the number of devices
        if self.trainer is not None:
            if self.hparams.train_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.train_batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.train_batch_size_per_device = self.hparams.train_batch_size // self.trainer.world_size
            self.val_batch_size_per_device = self.hparams.val_batch_size // self.trainer.world_size
            self.test_batch_size_per_device = self.hparams.test_batch_size // self.trainer.world_size

        if stage == "predict" or self.trainer.state.stage == "predict":
            val_csv = os.path.join(self.data_dir, 'phase2_eval.csv')
            val_img_dir = os.path.join(self.data_dir, 'phase2', 'eval')
            # For testing, load the test dataset
            self.data_val = WBCDataset(
                csv_file=val_csv,
                image_dir=val_img_dir,
                classes_to_idx=self.classes_to_idx,
                transform=self.val_transform,
                fold='validation'
            )           

            # phase1_train_csv = os.path.join(self.data_dir, 'phase1_label.csv')
            # phase1_train_img_dir = os.path.join(self.data_dir, 'phase1')
            
            # phase2_train_csv = os.path.join(self.data_dir, 'phase2_train.csv')
            # phase2_train_img_dir = os.path.join(self.data_dir, 'phase2', 'train')

            # self.phase1_data_train = WBCDataset(
            #     csv_file=phase1_train_csv,
            #     image_dir=phase1_train_img_dir,
            #     classes_to_idx=self.classes_to_idx,
            #     transform=self.train_transform,
            #     fold='train'
            # )

            # self.phase2_data_train = WBCDataset(
            #     csv_file=phase2_train_csv,
            #     image_dir=phase2_train_img_dir,
            #     classes_to_idx=self.classes_to_idx,
            #     transform=self.train_transform,
            #     fold='train'
            # )

            # self.data_val = ConcatDataset([self.phase1_data_train, self.phase2_data_train]) 
        
        elif stage == 'test' or self.trainer.state.stage == "test":
            test_csv = os.path.join(self.data_dir, 'phase2_test.csv')
            test_img_dir = os.path.join(self.data_dir, 'phase2', 'test')
            # For testing, load the test dataset
            self.data_test = WBCDataset(
                csv_file=test_csv,
                image_dir=test_img_dir,
                classes_to_idx=self.classes_to_idx,
                transform=self.test_transform,
                fold='validation'
            )


        else: # train and val
            # For prediction, load the test dataset (or specify a different dataset for prediction)
            filtered_train_csv = os.path.join(self.data_dir, 'non_corrupt_train.csv')
            train_root_dir = self.data_dir
         
            val_csv = os.path.join(self.data_dir, 'phase2_eval.csv')
            val_img_dir = os.path.join(self.data_dir, 'phase2', 'eval')

            test_csv = os.path.join(self.data_dir, 'phase2_test.csv')
            test_img_dir = os.path.join(self.data_dir, 'phase2', 'test')
            
            self.dist1_data_train = WBCTrainDatasetWBC(
                csv_file=filtered_train_csv,
                image_root_dir=train_root_dir,
                classes_to_idx=self.classes_to_idx,
                transform=self.train_transform,
                fold='train'
            )

            self.data_train = ConcatDataset([self.dist1_data_train])

            self.data_val = WBCDataset(
                csv_file=val_csv,
                image_dir=val_img_dir,
                classes_to_idx=self.classes_to_idx,
                transform=self.val_transform,
                fold='validation'
            )

            
            # For testing, load the test dataset
            self.data_test = WBCDataset(
                csv_file=test_csv,
                image_dir=test_img_dir,
                classes_to_idx=self.classes_to_idx,
                transform=self.test_transform,
                fold='test'
            )

            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        self.log_.info(f'Training samples: {len(self.data_train)}')
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        self.log_.info(f'Validation samples: {len(self.data_val)}')
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.
        
        By default, uses the test dataset for prediction.
        For custom prediction datasets, override this method.
        """
        
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after training/testing."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        pass


if __name__ == "__main__":
    # Add project root to path for testing
    import sys
    sys.path.append('/home/wisc/maheswararao/code/lightning-hydra-template')
    
    # Test the WBC datamodule
    print("Testing WBC Phase2 Classification DataModule...")
    
    # Initialize the WBC datamodule
    wbc_dm = WBCClassificationDataModule(
        data_dir='/data1/shared/data/wbc-bench-2026/',
        batch_size=4,
        num_workers=0,
        sample_rate=10  # Use small sample for testing
    )
    
    # Mock trainer for testing
    class MockTrainer:
        def __init__(self):
            self.world_size = 1
            self.state = MockState()
    
    class MockState:
        def __init__(self):
            self.stage = "fit"
    
    wbc_dm.trainer = MockTrainer()
    wbc_dm.setup()
    
    print(f"Classes to idx: {wbc_dm.classes_to_idx}")
    print(f"Number of classes: {len(wbc_dm.classes_to_idx) if wbc_dm.classes_to_idx else 0}")
    
    print("\nTrain loader:")
    for batch in wbc_dm.train_dataloader():
        image_id, X, y, fold = batch
        print(f"Image IDs: {image_id}")
        print(f"X shape: {X.shape}, y shape: {y.shape}, fold: {fold}")
        break
    
    print("\nVal loader:")
    for batch in wbc_dm.val_dataloader():
        image_id, X, y, fold = batch
        print(f"Image IDs: {image_id}")
        print(f"X shape: {X.shape}, y shape: {y.shape}, fold: {fold}")
        break
    