#!/usr/bin/env python3
"""BaseSADataset class in src/custom_datasets."""

from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import albumentations
import numpy as np
import PIL
from loguru import logger
from torchvision import transforms

from src.conf.pydantic_validators import Split
from src.custom_datasets.base_dataset import BaseDataset
from src.custom_datasets.utils import create_samples_targets
from src.fileio.text.readers import json_loader


# from argusdp.processing.data_utils import find_image_json_pairs


class SADataset(BaseDataset):
    """A dataset class for loading paired images and json files.

    Args:
        root (str): Root directory containing subfolders train, validation, and test.
        split (str, optional): One of ``train``, ``validation``, ``test``.
        loader (callable, optional): A function to load a sample given its path.
        transform (callable, optional): A function/transform that  takes in an image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _url: str = "url"
    _base_folder: str = ""
    _zip_metadata: dict = {
        "filename": Path(_url).name,
        "md5": "42",
        "url": _url,
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: Split = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        loader: Optional[Union[str, Callable]] = None,
        random_seed: Optional[int] = 8675309,
        config_file: Optional[Union[str, Path]] = None,
        target_key: Optional[str] = "all",
        **kwargs: Any,  # balance_classes, stratify
    ):
        """Initialize the dataset class."""
        super().__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            random_seed=random_seed,
            config_file=config_file,
            **kwargs,
        )

        # set target key
        self.target_key = target_key

        # create dataset
        self.samples, self.targets = self.make_dataset(
            image_ext=image_ext,
            recursive=True,
            force=False,
            # dry_run=kwargs.get("dry_run"),
        )
        self._classes = list(self.config.classes_to_idx.keys())

    def __getitem__(  # noqa: C901 # TODO refactor to reduce complexity
        self, index: int
    ) -> Dict[str, Any]:
        """Load an image and json pair and return the image and target."""
        # load image and json
        img_file, json_file = self.samples[index]
        image = self.loader(img_file)
        json_data = json_loader(json_file)
        data_dict = json_data.get("custom_metadata").copy()
        data_dict["filepath"] = img_file.as_posix()
        data_dict["instances"] = json_data.get("instances")
        data_dict["tags"] = json_data.get("tags")

        # check label and label_name match
        label_name = data_dict.get("label_name", "")
        label = self.classes_to_idx.get(label_name, -1)
        if label == -1:
            logger.warning(f"Label '{label_name}' not found in classes_to_idx.")
            logger.warning(f"Classes: {self.classes_to_idx}")
            logger.warning(f"Setting class {label_name} to {label}")
            self.classes_to_idx[label_name] = label

        # override base class label, since the image-level task/label may be
        # different from the instance-level task (e.g., image-level cancer classification
        # benign/malignant vs. instance-level nuclei segmentation)
        data_dict["label"] = label
        data_dict["label_name"] = label_name

        # get target
        target = self.targets[index]  # Is this needed?
        if self.target_key == "all":
            target = data_dict.copy()
        elif data_dict.get(self.target_key):
            target = data_dict.get(self.target_key)
        else:
            logger.error(f"No data for column {self.target_key}.")
            raise RuntimeError

        # transform image: expects torchvision or albumentations Compose
        if isinstance(self.transform, transforms.Compose):
            if not isinstance(image, PIL.Image.Image) and isinstance(image, np.ndarray):
                try:
                    image = PIL.Image.fromarray(image)
                except Exception as e:
                    logger.error(f"Error converting image to PIL: {e}")

            image = self.transform(image)
        elif isinstance(self.transform, albumentations.Compose) and self.target_key in {
            "mask",
            "bbox",
            "polygon",
        }:
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            augmentations = self.transform(image=image, mask=target)
            image = augmentations["image"]
            target = augmentations["mask"]

        # transform target
        if self.target_transform is not None and self.target_key in {
            "mask",
            "bbox",
            "polygon",
        }:
            target = self.target_transform(target)

        # convert image to np.array, if not already
        image = np.array(image) if isinstance(image, PIL.Image.Image) else image
        data_dict["image"] = image

        #
        if self.target_key in {"mask", "bbox", "polygon"}:
            data_dict[self.target_key] = target

        return data_dict

    def make_dataset(
        self,
        image_ext: str = "png",
        recursive: bool = True,
        force=False,
        **kwargs: Optional[dict],
    ) -> tuple[list[Any], list[Any]]:
        """Generates a list of samples of a form (path_to_sample, path_to_json)."""
        data_root = self._root
        jsonl_filepath = None
        if self.split != "all":
            data_root = self._root  # do not need to add split
            jsonl_filepath = self._root.joinpath(f"{self.split}.jsonl")

        # find image-json pairs or read from jsonl (if available)

        # create samples and targets
        return create_samples_targets(
            data_root,
            self.split,
            image_ext,
            recursive,
            jsonl_filepath,
            force,
            self.target_key,
        )
