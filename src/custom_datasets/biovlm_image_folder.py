#!/usr/bin/env python3
"""base_imagefolder_dataset in src/bcv/datasets."""


from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from loguru import logger
from torchvision.datasets.folder import ImageFolder

from src.custom_datasets.base_dataset import BaseDataset
from src.custom_datasets.sa_json_dataclass import SACustomMetadata
from src.fileio.text.readers import get_metadata_file
from src.processing.data_utils import is_image_folder_format


class BioVLMImageFolder(BaseDataset):
    """Base class for loading images from a folder structure."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Union[str, Callable]] = None,
        random_seed: Optional[int] = 8675309,
        **kwargs: Any,  # balance_classes, stratify, image_ext
    ):
        """Initialize the BioVLMImageFolder class.

    This class inherits from :class:`src.custom_datasets.base_dataset`
        """
        if not is_image_folder_format(root):
            error_msg = f"The directory is not in BioVLMImageFolder format:\n\t{root}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        metadata_file = get_metadata_file(root, metadata_format="yaml")

        # initialize superclass BaseDataset from sngp_core.custom_datasets.base_dataset
        super().__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            random_seed=random_seed,
            config_file=metadata_file,
            **kwargs,
        )

        # inherit from ImageFolder
        self.dataset = ImageFolder(
            root=Path(self.root).joinpath("images").as_posix(),
            transform=self.transform,
            target_transform=self.target_transform,
            loader=self.loader,
        )
        self._classes = self.dataset.classes
        # imagefolder index differs from config._classes_to_idx
        self.dataset.ifolder_idx_to_class = {
            v: k for k, v in self.dataset.class_to_idx.items()
        }

        # check if classes_to_idx is provided in config file:
        if self.config.classes_to_idx:
            self._classes_to_idx = self.config.classes_to_idx
        else:
            # use the default classes_to_idx from ImageFolder
            self._classes_to_idx = self.dataset.class_to_idx
            self.config.classes_to_idx = self._classes_to_idx

        #
        self.idx_to_classes = {v: k for k, v in self._classes_to_idx.items()}

        # separate images with "_mask" into separate list
        mask_images = []
        image_files = []
        for image_file in self.dataset.samples:
            if "_mask" in image_file[0]:
                mask_images.append(image_file)
            else:
                image_files.append(image_file)

        # check same number of images and masks
        # okay to have no masks, but if masks are present, they must match
        if mask_images and len(image_files) != len(mask_images):
            logger.error(f"Images: {len(image_files)} !=Masks: {len(mask_images)}")
            raise ValueError("Number of images != number of masks")

        # convert samples to List[Tuple[str, Dict[str, Any]]]
        samples = []
        targets = []
        for idx, sample in enumerate(image_files):
            # sample[0] is the image path, sample[1] is the target
            label_name = self.dataset.ifolder_idx_to_class[sample[1]]
            label = self._classes_to_idx[label_name]
            target_dict = {
                "label": label,
                "label_name": label_name,
                "split": None,
                "mask": None,
            }
            if mask_images:
                mask = mask_images[idx]
                target_dict["mask"] = mask[0]

            samples.append((sample[0], target_dict))  # sample[0] is a str
            targets.append(label)  # target is a List[int]

        self.samples = samples
        self.targets = targets

        # set metadata from  class attr from DatasetConfig
        # only use non "_" prefixed attributes
        self.custom_metadata = {}
        self.label_metadata = {}
        if self.config.__dict__:
            self.custom_metadata = {
                key: value
                for key, value in self.config.__dict__.items()
                if not key.startswith("_")
            }

        # get label specific metadata from self.custom_metadata
        self.label_metadata = self.custom_metadata.pop("label_metadata", {})
        if not self.label_metadata:
            logger.warning("No label metadata found in config file.")

        # include all fields in SACustomMetadata and label_metadata
        metadata_fields = list(
            set(list(SACustomMetadata.model_fields) + ["label_metadata"])
        )
        self.custom_metadata = {
            k: v for k, v in self.custom_metadata.items() if k in metadata_fields
        }
        # update dataset_name if not provided
        if not self.custom_metadata.get("dataset_name"):
            self.custom_metadata["dataset_name"] = (
                self.config.dataset_name or self._root.name
            )

        # stratify and split
        self.samples, self.targets = self._create_train_test_split(
            self.split,
            self.samples,
            self.targets,
            # stratify=self.targets,
            random_seed=self.random_seed,
            train_val_test=self._train_val_test,
        )

    def __getitem__(self, index: int) -> Dict:
        """Get the image and metadata at the specified index."""
        # samples and targets created by ImageFolder
        image_filepath, target_dict = self.samples[index]
        image_filepath = Path(image_filepath)
        target = target_dict["label"]
        label_name = self.idx_to_classes.get(target)

        if label_name != image_filepath.parent.stem:
            logger.warning(
                f"Label mismatch at index {index}. Using self.idx_to_classes."
            )
            logger.warning(
                f"Expected {target_dict.get('label_name')} ({target}), got {label_name} ({target})."
            )
            raise RuntimeError(f"Label mismatch at index {index}.")

        # load image and mask
        image = self.loader(image_filepath)
        mask = target_dict.get("mask")
        if mask:
            mask = self.loader(mask)
            mask = np.array(mask)
            # convert to 2d
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]

            # check image and mask have same dimensions
            if np.array(image).shape[:2] != mask.shape:
                logger.error(
                    f"Image: {np.array(image).shape[:2] } != Mask: {mask.shape}"
                )
                raise ValueError("Image and mask have different dimensions.")

        # transform image: expects torchvision or albumentations Compose
        if self._transform_format == "tv":
            image = self.transform(image)
        elif self._transform_format == "alb":
            augmentations = self.transform(image=image, mask=target)
            image = augmentations["image"]
            target = augmentations["mask"]

        # transform target
        if self.target_transform is not None:
            target = self.target_transform(target)

        # create output dictionary
        data_dict = {}
        data_dict.update(self.custom_metadata)
        if self.label_metadata:
            # update data_dict with label specific metadata
            try:
                data_dict |= self.label_metadata.get(label_name, {})
            except TypeError as e:
                logger.error(f"Error: {e}")
                logger.error(f"Label metadata: {self.label_metadata}")
                logger.error(f"Label name: {label_name}")
                logger.error(f"Label: {target}")
                raise e
        else:
            logger.warning("No label metadata found in config file.")

        return {
            "filename": image_filepath.name,
            "image": image,
            "label": target,
            "label_name": label_name,
            "original_filename": image_filepath.name,
            "split": target_dict["split"],
            "mask": mask,
            **data_dict,
        }

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)
