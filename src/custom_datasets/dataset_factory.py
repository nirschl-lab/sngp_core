#!/usr/bin/env python3
"""dataset_factory.py in src/sngp_core/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/factory.py
"""
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Union

from src.custom_datasets.base_dataset import BaseDataset
from src.custom_datasets.dataset_registry import DatasetType


def create_dataset(
    dataset_name: str,
    root: Union[str, Path],
    split: str = "train",
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    dry_run: bool = False,
) -> BaseDataset:
    """Create a dataset given its corresponding DatasetType enum value.

    Args:
        dataset_name (str): An enum value representing the dataset to be created.
        root (str): The root directory where the dataset is stored or should be downloaded.
        split (bool): If True, the dataset represents the training set; otherwise, it's the test set.
        transform (Optional[Callable[[Image.Image], torch.Tensor]]):
            An optional transform function to be applied to the dataset images.
            It takes a PIL image as input and returns a PyTorch tensor as output.

    Returns:
        SADataset: An instance of the dataset specified by the DatasetType enum value.

    """
    dataset_type = DatasetType[dataset_name]
    return dataset_type.value(
        root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        dry_run=dry_run,
    )
