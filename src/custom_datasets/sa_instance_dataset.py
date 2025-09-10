#!/usr/bin/env python3
"""BaseSADataset class in src/custom_datasets."""
import pprint
from collections import Counter
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
from src.fileio.text.readers import json_loader
from src.processing.data_utils import find_image_json_pairs
from src.processing.image.imutils import bbox_dist_to_border
from src.processing.image.imutils import center_bbox
from src.processing.image.imutils import crop_to_bbox


class SAInstanceDataset(BaseDataset):
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
            instance_key=None,  # temp hack to use entire image as target
            dry_run=kwargs.get("dry_run"),
        )
        self._classes = list(self.config.classes_to_idx.keys())

    def __getitem__(  # noqa: C901 # TODO refactor to reduce complexity
        self, index: int
    ) -> Dict[str, Any]:
        """Load an image and json pair and return the image and target."""
        # load sample
        img_file, json_file, image_id, inst_id, label_name, bbox = self.samples[index]
        image = self.loader(img_file)
        image = crop_to_bbox(np.array(image), bbox)
        json_data = json_loader(json_file)
        data_dict = json_data.get("custom_metadata").copy()
        data_dict["filepath"] = img_file.as_posix()
        data_dict["image_id"] = image_id
        data_dict["instance_id"] = inst_id
        data_dict["tags"] = json_data.get("tags")

        # check label and label_name match
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
        instance_key: Optional[str] = "bbox",
        border_size: int = 0,
        bbox_size: int = 256,
        exclude_classes: Optional[list[str]] = None,
        dry_run: Optional[bool] = False,
    ) -> tuple[list[Any], list[Any]]:
        """Generates a list of samples of a form (path_to_sample, path_to_json)."""
        data_root = self._root
        jsonl_filepath = None
        if self.split != "all":
            # TODO: check this - the jsonl filepath should now always include the split in relative filepath
            # data_root = self._root.joinpath(self.split)
            jsonl_filepath = self._root.joinpath(f"{self.split}.jsonl")

        # find image-json pairs or read from jsonl (if available)
        json_files, image_files = find_image_json_pairs(
            data_root,
            ext=image_ext,
            recursive=recursive,
            jsonl_filepath=jsonl_filepath,
            force=force,
        )
        image_files = [data_root.joinpath(f) for f in image_files]
        json_files = [data_root.joinpath(f) for f in json_files]

        # for each json file, read json and get instances
        samples = []
        targets = []
        for idx, (image_file, json_file) in enumerate(
            zip(image_files, json_files, strict=True)
        ):
            if dry_run and idx > 10:
                break

            # load json file
            json_data = json_loader(json_file)
            image_id = json_data["custom_metadata"]["image_id"]
            instances = json_data.get("instances")
            for instance in instances:
                instance_id = instance.get("id")
                instance_class = instance.get("className")
                if exclude_classes and instance_class in exclude_classes:
                    continue
                elif self.classes_to_idx.get(instance_class) is None:
                    logger.warning(
                        f"Class '{instance_class}' not found in classes_to_idx."
                    )
                    new_label = max(self.classes_to_idx.values()) + 1
                    logger.info(f"Adding class '{instance_class}': {new_label}")
                    self.classes_to_idx[instance_class] = new_label
                    logger.info(
                        f"Updated classes_to_idx: {pprint.pformat(self.classes_to_idx)}"
                    )
                    # continue

                if instance_key == "bbox":
                    bbox = instance.get("points")
                    bbox = center_bbox(bbox, bbox_size)
                    image_shape = (
                        json_data["metadata"]["height"],
                        json_data["metadata"]["width"],
                    )
                    # check if coords are too close to the border
                    safe_dist = bbox_dist_to_border(bbox, image_shape) > border_size

                    if bbox and safe_dist:
                        samples.append(
                            (
                                image_file,
                                json_file,
                                image_id,
                                instance_id,
                                instance_class,
                                bbox,
                            )
                        )
                        targets.append(instance_class)
                elif instance_key == "polygon":
                    raise NotImplementedError
                elif instance_key == "centroid":
                    raise NotImplementedError
                else:
                    logger.error(f"Invalid instance key: {instance_key}")
                    raise ValueError(f"Invalid instance key: {instance_key}")

        #
        logger.info(f"Found {len(samples)} instances for split '{self.split}'")
        logger.info(f"Classes: {dict(Counter(targets))}")
        targets = [self.classes_to_idx.get(t) for t in targets]

        return samples, targets
