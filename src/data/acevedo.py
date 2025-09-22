from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.fileio.hdf.readers import load_dataset

class AcevedoDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            dataset: str = 'acevedo_et_al_2020',
            features: str = 'dinov2_vit_S14',
            feature_cache: str | Path = '/data1/shared/cache/feature_cache/'
            ):
        super().__init__()

        self.dataset = dataset 
        self.features = features
        self.feature_cache = feature_cache

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.batch_size_per_device = batch_size
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.data_train = load_dataset(self.dataset, self.features, split = 'train', feature_cache = self.feature_cache)
        self.data_val = load_dataset(self.dataset, self.features, split = 'validation', feature_cache = self.feature_cache)
        self.data_test = load_dataset(self.dataset, self.features, split = 'test', feature_cache = self.feature_cache)
    
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        X = self.data_train['']
        y = torch.tensor(self.data_train["label"].values, dtype=torch.long)
        
        return DataLoader(torch.utils.data.TensorDataset(X, y), self.batch_size_per_device, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        X = torch.tensor(self.data_val[self.data_val.columns[8:]].to_numpy(), dtype=torch.float32)
        y = torch.tensor(self.data_val["label"].values, dtype=torch.long)
        
        return DataLoader(torch.utils.data.TensorDataset(X, y), self.batch_size_per_device, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        X_train = torch.tensor(self.data_test[self.data_test.columns[8:]].to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(self.data_test["label"].values, dtype=torch.long)
        
        return DataLoader(torch.utils.data.TensorDataset(X_train, y_train), self.batch_size_per_device, shuffle=False)

if __name__ == "__main__":
    # Example usage and test for AcevedoDataModule
    dm = AcevedoDataModule(
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        dataset="acevedo_et_al_2020",
        features="dinov2_vit_S14",
        feature_cache="/data1/shared/cache/feature_cache/"
    )
    dm.setup()
    print("Train loader:")
    for batch in dm.train_dataloader():
        X, y = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break
    print("Val loader:")
    for batch in dm.val_dataloader():
        X, y = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break
    print("Test loader:")
    for batch in dm.test_dataloader():
        X, y = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break