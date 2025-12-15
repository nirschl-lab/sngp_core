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
            # Apply transformations
            if isinstance(self.transform, A.Compose):
                # Albumentations expects numpy arrays
                image_np = np.array(image)
                transformed = self.transform(image=image_np, mask=mask)
                if mask is not None:
                    raise NotImplementedError("Mask augmentation is not yet returned")
                    #image, mask = transformed['image'], transformed['mask']
                else:
                    image = transformed['image']
            else:
                image = self.transform(image)

        return image_id, image, label, self.fold
    
class ClassificationImageDataModule(LightningDataModule):
    """`LightningDataModule` for the Acevedo Image dataset.

    """

    def __init__(
        self,
        dataset_name: str,
        num_classes: int = 8,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 32,
        pin_memory: bool = False,
        class_indices: Dict[str, int] = None,
        train_augmentations: Optional[transforms.Compose | A.Compose] = None,  # New argument
        val_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_all_folds: Optional[bool] = False,
        sample_rate: int = 0,
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

        self.train_batch_size_per_device = train_batch_size
        self.val_batch_size_per_device = val_batch_size
        self.test_batch_size_per_device = test_batch_size

        self.log_ = RankedLogger(__name__, rank_zero_only=True)

        self.test_all_folds = test_all_folds

        # Store class mappings
        self.classes_to_idx = class_indices
        self.idx_to_classes = {v: k for k, v in class_indices.items()} if class_indices else None


    def setup(self, stage: Optional[str] = None) -> None:
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
            if self.hparams.train_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.train_batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.train_batch_size_per_device = self.hparams.train_batch_size // self.trainer.world_size
            self.val_batch_size_per_device = self.hparams.val_batch_size // self.trainer.world_size
            self.test_batch_size_per_device = self.hparams.test_batch_size // self.trainer.world_size

        if stage == "predict" or self.trainer.state.stage == "predict":
            self.log_.info('Testing on all folds - train, val and test')
            data_train_ = datasets.load_dataset(self.dataset_name)['train']
            data_val_   = datasets.load_dataset(self.dataset_name)['validation']
            data_test_  = datasets.load_dataset(self.dataset_name)['test']
            eval_train = HFDataset(data_train_, transform=self.test_transform, fold='train')
            eval_val   = HFDataset(data_val_, transform=self.test_transform, fold='val')
            eval_test  = HFDataset(data_test_, transform=self.test_transform, fold='test')

            self.data_test = ConcatDataset([eval_train, eval_val, eval_test])
        
        
        # set these indices for plotting
        elif self.trainer.state.stage == "test":
            if self.test_all_folds:
                self.log_.info('Testing on all folds - train, val and test')
                data_train_ = datasets.load_dataset(self.dataset_name)['train']
                data_val_   = datasets.load_dataset(self.dataset_name)['validation']
                data_test_  = datasets.load_dataset(self.dataset_name)['test']
                eval_train = HFDataset(data_train_, transform=self.test_transform, fold='train')
                eval_val   = HFDataset(data_val_, transform=self.test_transform, fold='val')
                eval_test  = HFDataset(data_test_, transform=self.test_transform, fold='test')

                self.data_test = ConcatDataset([eval_train, eval_val, eval_test])
            else:
                data_test_  = datasets.load_dataset(self.dataset_name)['test']
                self.data_test = HFDataset(data_test_, transform=self.test_transform, fold='test')

            self.trainer.test_classes_to_idx = ast.literal_eval(data_test_[0]['classes_to_idx'])
            self.trainer.test_idx_to_classes = {idx:cls for cls,idx in self.trainer.test_classes_to_idx.items()}

        else:
            data = datasets.load_dataset(self.dataset_name)

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
            batch_size=self.train_batch_size_per_device,
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
            batch_size=self.val_batch_size_per_device,
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
            dataset=self.data_test,
            batch_size=self.test_batch_size_per_device,
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
    import yaml
    yaml_file_path = 'configs/paths/default.yaml'

    try:
        with open(yaml_file_path, 'r') as file:
            # Load the YAML data using yaml.safe_load() for security
            data = yaml.safe_load(file)
        print("YAML data loaded successfully:")

    except FileNotFoundError:
        print(f"Error: The file '{yaml_file_path}' was not found.")

    data_dir = data['data_cache_dir']
    dm = ClassificationImageDataModule(
        data_dir
    )
    dm.setup()
    print("Train loader:")
    for batch in dm.train_dataloader():
        X, y = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break
    
    print("Test loader:")
    for batch in dm.test_dataloader():
        X, y = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break
        
    
    print("Val loader:")
    for batch in dm.val_dataloader():
        X, y = batch
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break
    