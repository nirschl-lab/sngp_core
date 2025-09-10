#!/usr/bin/env python3
"""base_dataset in src/argusdp/custom_datasets."""
import pprint
import random
from collections import Counter
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import albumentations
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import calculate_md5
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

from src import MODULE_ROOT
from src.conf import VERSIONS
from src.conf.base_metadata import BioVLMDatasetMetadata
from src.fileio.image.readers import cv2_loader
from src.fileio.text.readers import get_metadata_file
from src.fileio.text.readers import load_metadata_file
from src.fileio.text.readers import yaml_loader
from src.processing.text import convert_liststr_to_str
from src.processing.text.harmonize_text import validate_split
from src.processing.text.harmonize_text import validate_split_list
from src.utils.random_seed import set_random_seed


class BaseDataset(VisionDataset):  # pylint: disable=too-few-public-methods
    """BaseDataset class.

    Args:
        root (str): Root directory of dataset where ``<dataset_name>`` folder exists
            or will be saved to if download is set to True.
        split (str, optional): One of ``train``, ``validation``, ``hold-out``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _cv2_loader: Callable = cv2_loader
    _pil_loader: Callable = default_loader
    _classes_to_idx: Dict[str, int] = {}

    def __init__(
        self,
        root: Union[str, Path],
        loader: Union[str, Callable] = default_loader,
        random_seed: Optional[int] = 8675309,
        # loader_type: str = "VisionDataset",
        split: str = "train",
        target_transform: Optional[Union[Callable, list]] = None,
        train_val_test: Optional[Tuple[float]] = (0.7, 0.1, 0.2),
        transform: Optional[Union[Callable, list]] = None,
        config_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,  # balance_classes, stratify
    ) -> None:
        """Initialize BaseDataset class."""
        self._root = self._check_root(root)
        self._base_folder = Path(root).stem.lower().replace(" ", "_")

        # initialize superclass VisionDataset
        super().__init__(
            self._root.as_posix(),
            transform=transform,
            target_transform=target_transform,
        )

        # set dataset config
        config_file = config_file or get_metadata_file(
            self.root, metadata_format="json"
        )
        self._config_file = config_file
        self.config = self._load_config_from_file(config_file)
        self._base_folder = self.config.dataset_slug

        # # set classes_to_idx
        # if self.config._dummy_classes_to_idx:
        #     # use dummy classes_to_idx with options to override in child class
        #     self._classes_to_idx = self.config.classes_to_idx
        if isinstance(self.config.classes_to_idx, str):
            # if classes_to_idx is a string, evaluate to convert to dict
            classes_to_idx = eval(self.config.classes_to_idx)
            if not isinstance(classes_to_idx, dict):
                raise ValueError(
                    f"Invalid classes_to_idx: {self.config.classes_to_idx}"
                )

            self.config.classes_to_idx = classes_to_idx  # update config
            self._classes_to_idx = self.config.classes_to_idx  # update class
            self._classes = list(self.config.classes_to_idx.keys())
        elif isinstance(self.config.classes_to_idx, dict):
            self._classes_to_idx = self.config.classes_to_idx
            self._classes = list(self._classes_to_idx.keys())
        else:
            raise ValueError(f"Invalid classes_to_idx: {self.config.classes_to_idx}")

        # sort _classes_to_idx by value
        self._classes_to_idx = dict(
            sorted(self._classes_to_idx.items(), key=lambda item: item[1])
        )

        # set image loader function
        self._set_loader(loader or default_loader)

        # set random seed, if provided
        self._seed_set = False
        self.random_seed = random_seed
        if self.random_seed is not None:
            self._set_random_seed(random_seed)

        # set proportion for train/val/test splits
        self._train_val_test = train_val_test

        # ensure consistent split name
        self.split = self._set_split(split)

        # ensure transforms are callable
        self._transform_mean_std = (None, None)
        if self.transform is not None and not callable(self.transform):
            raise TypeError(f"Transform must be callable, not {type(self.transform)}.")
        elif isinstance(self.transform, transforms.Compose):
            self._transform_format = "tv"
            xforms = self.transform.transforms
            xforms = xforms if isinstance(xforms, list) else [xforms]
            norm_xform = [x for x in xforms if "normalize" in str(x).lower()]
            if norm_xform and len(norm_xform) == 1:
                self._transform_mean_std = (norm_xform[0].mean, norm_xform[0].std)
        elif isinstance(self.transform, albumentations.Compose):
            self._transform_format = "alb"
            xforms = list(self.transform)
            norm_xform = [elem for elem in xforms if "normalize" in str(elem).lower()]
            if norm_xform and len(norm_xform) == 1:
                self._transform_mean_std = (norm_xform[0].mean, norm_xform[0].std)
        elif self.transform is None:
            self._transform_format = None
        else:
            error_msg = f"Expected transform to be callable, not {self.transform} ({type(self.transform)})."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.target_transform is not None and not callable(self.target_transform):
            raise TypeError(
                f"Target_transform must be callable, not {type(self.target_transform)}."
            )

        self.samples = None
        self.targets = None

    def _check_root(self, root: Union[str, Path]) -> Path:
        """Check if root directory is valid."""
        if Path(root).is_dir() and Path(root).exists():
            root = Path(root).resolve()
        elif Path(root).is_dir() and not Path(root).exists():
            raise NotADirectoryError(f"Root directory {root} does not exist.")
        elif Path(root).is_file():
            raise NotADirectoryError(f"Expected directory, not file: {root}")
        else:
            logger.error(f"Invalid root directory: {root}")
            raise NotADirectoryError(f"Invalid root directory: {root}")

        return root

    def __len__(self) -> int:
        """Length."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load an image and return tuple of Image and label."""
        raise NotImplementedError("getitem method not implemented.")

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return (
            f"{self.__class__.__name__}"  # noqa: B907
            f"({self.root},"
            f"split='{self.split}',"
            f"transform={self.transform},"
            f"target_transform={self.target_transform},"
            f"loader='{self.loader.__name__}',"
            f"random_seed={self.random_seed})"
        )

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        classes_to_idx_formatted = pprint.pformat(
            self.classes_to_idx, indent=8, sort_dicts=False
        )
        classes_to_idx_formatted = classes_to_idx_formatted.replace("{", "{\n ")
        classes_to_idx_formatted = classes_to_idx_formatted.replace("}", "\n}")

        base_str = (
            f"Dataset {self.name.replace('_',' ').capitalize()}\n"
            f"\tRoot:\t\t\t{Path(self.root)}\n"
            f"\tDomain (sub):\t{self.domain} ({self.subdomain})\n"
            f"\tModality (sub):\t{self.modality} ({self.submodality})\n"
            # f"\tTasks:\t\t{self.supported_tasks}\n"
            f"\tSplit:\t\t\t{self.split.capitalize()}\n"
            f"\tSamples:\t\t{len(self)} \n"
            f"\tImage shape:\t{self.image_size}\n"
            f"\tNum classes:\t{self.num_classes}"
            f"\t{classes_to_idx_formatted}\n"
        )

        additional_str = ""
        for key in ["Microns per pixel", "DOI", "PMID", "Version", "Version comment"]:
            safe_key = key.lower().replace(" ", "_")
            if not getattr(self, safe_key):
                continue
            elif safe_key == "pmid" and isinstance(getattr(self, safe_key), int):
                value = f"https://pubmed.ncbi.nlm.nih.gov/{getattr(self, safe_key)}/"
            elif safe_key == "microns_per_pixel":
                key = "Pixel size (um)"
                value = getattr(self, safe_key)
            else:
                value = getattr(self, safe_key)

            # add tab spacing for alignment
            num_tabs = len(safe_key) // 3 + int(len(safe_key) < 3)
            sep = "\t" * max((4 - num_tabs), 0)
            additional_str += f"\t{key}:{sep}{value}\n"

        license_str = f"\tLicense:\t\t{self.license.replace(' ', '-')}\n"

        return base_str + additional_str + license_str

    def _load_config_from_file(
        self, config_file: Union[str, Path]
    ) -> BioVLMDatasetMetadata:
        """Load dataset configuration from file."""
        _config_file = Path(config_file)
        if not _config_file:
            logger.error(f"Config file not found: {_config_file}")
            raise FileNotFoundError(f"Config file not found: {_config_file}")
        elif _config_file := get_metadata_file(self.root):
            config_dict = load_metadata_file(self.root)
        else:
            logger.error(f"Invalid config file type: {self._config_file.suffix}")
            raise ValueError(f"Invalid config file type: {self._config_file.suffix}")

        self._config_file = _config_file
        # remove "_" prefixed attributes
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}
        return BioVLMDatasetMetadata(**config_dict)

    @property
    def bibtex(self) -> str:
        """Return the bibtex for the primary citation."""
        return self.config.bibtex

    @property
    def classes(self) -> List[str]:
        """Return the classes to index mapping."""
        return self._classes

    @property
    def classes_to_idx(self) -> Dict[str, int]:
        """Return the classes to index mapping."""
        return self._classes_to_idx

    @property
    def description(self) -> str:
        """Return brief description of the dataset."""
        return self.config.description

    @property
    def doi(self) -> Optional[str]:
        """Return DOI."""
        if isinstance(self.config.doi, str):
            return self.config.doi.strip()
        return self.config.doi

    @property
    def domain(self) -> Optional[str]:
        """Return domain of the dataset."""
        if isinstance(self.config.domain, list):
            return ", ".join(self.config.domain)
        return self.config.domain

    @property
    def image_mean_std(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return the mean and standard deviation of the dataset."""
        return self.config.image_mean_std

    @property
    def image_size(self) -> Tuple[int, int, int]:
        """Return the image size."""
        if isinstance(self.config.image_size, list):
            return tuple(self.config.image_size)
        return self.config.image_size

    @property
    def keywords(self) -> Optional[str]:
        """Return keywords of the dataset."""
        if isinstance(self.config.keywords, list):
            return ", ".join(self.config.keywords)

        return self.config.keywords

    @property
    def label_task(self) -> List[str]:
        """Return the label task."""
        return self.config.label_task

    @property
    def language(self) -> Optional[str]:
        """Return dataset language."""
        if isinstance(self.config.language, list):
            return ", ".join(self.config.language)

        return self.config.language

    @property
    def license(self) -> Optional[str]:
        """Return dataset license."""
        if isinstance(self.config.license, list):
            license = convert_liststr_to_str(self.config.license)
            return ", ".join(license)

        return self.config.license

    @property
    def loader(self) -> Callable:
        """Return image loader."""
        return self._loader

    @property
    def microns_per_pixel(self) -> Optional[float]:
        """Return the microns per pixel."""
        return self.config.microns_per_pixel

    @property
    def modality(self) -> Optional[str]:
        """Return dataset modality."""
        if isinstance(self.config.modality, list):
            return ", ".join(self.config.modality)
        return self.config.modality

    @property
    def name(self) -> str:
        """Return dataset name."""
        return self.config.dataset_slug or self._base_folder

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.classes)

    @property
    def pmid(self) -> Optional[str]:
        """Return PubMed identifier for primary citation."""
        if isinstance(self.config.pmid, str):
            return self.config.pmid.strip()
        elif isinstance(self.config.pmid, int):
            return str(self.config.pmid)

    @property
    def stain(self) -> Optional[str]:
        """Return stain used for cell biology and pathology custom_datasets (None if not applicable)."""
        if isinstance(self.config.stain, list):
            return ", ".join(self.config.stain)
        return self.config.stain

    @property
    def subdomain(self) -> Optional[str]:
        """Return dataset subdomain."""
        if isinstance(self.config.subdomain, list):
            return ", ".join(self.config.subdomain)
        return self.config.subdomain

    @property
    def submodality(self) -> Optional[str]:
        """Return dataset submodality."""
        if isinstance(self.config.submodality, list):
            return ", ".join(self.config.submodality)
        return self.config.submodality

    @property
    def supported_tasks(self) -> Optional[str]:
        """Return the task for the dataset."""
        return self._task if hasattr(self, "_task") else None

    @property
    def version(self) -> str:
        """Return dataset version."""
        return self.config.version

    @property
    def version_comment(self) -> str:
        """Return dataset version."""
        return self._get_version_comment()

    def _get_version_comment(self) -> str:
        """Return dataset version comment."""
        if self.config.version_comment:
            return self.config.version_comment

        versions_file = Path(__file__).parents[1].joinpath("conf", "versions.yaml")
        if versions_file.exists():
            versions_dict = yaml_loader(versions_file)
            return versions_dict.get(self.version, "No version comment found.")
        else:
            logger.warning(f"Versions file not found: {versions_file}")
            return "No version comment found."

    def _balance_classes(
        self,
        df: pd.DataFrame,
        sampler_method: str = "rand_over",
        sampling_strategy: str = "auto",
        random_seed: Optional[int] = 8675309,
    ) -> None:
        """Balance classes."""
        raise NotImplementedError("balance_classes method not implemented.")

    def _check_data_exists(self, base_folder=None) -> bool:
        """Check if data exists."""
        data_dir = Path(self.root).joinpath(base_folder or self._base_folder)
        if not data_dir.exists():
            data_dir = Path(self.root).parent.joinpath(self._base_folder)
            if data_dir.exists():
                logger.info(f"Folder {self._base_folder} found in {data_dir.parent}.")
                logger.info(f"Updating root to {data_dir}.")
                self.root = data_dir.parent
            else:
                logger.info(f"Folder {self._base_folder} not found in {self.root}.")

        return data_dir.exists()

    def _check_integrity(self) -> bool:
        """Check integrity of dataset."""
        # for zip or each csv file, check if the md5 hash matches
        data_dir = Path(self.root)
        if "md5" in self._zip_metadata.keys():
            archive_list = [data_dir.joinpath(self._zip_metadata["filename"])]
            md5_list = [data_dir.joinpath(self._zip_metadata["md5"])]
            url_list = [
                (
                    data_dir.joinpath(self._zip_metadata["url"])
                    if self._zip_metadata["url"]
                    else None
                )
            ]
            # else check if "md5" is in a sub dictionary of _zip_metadata
        elif any("md5" in d.keys() for d in self._zip_metadata.values()):
            archive_list, md5_list, url_list = [], [], []
            for archive in self._zip_metadata.values():
                archive_list.append(Path(self.root).joinpath(archive["filename"]))
                md5_list.append(archive["md5"])
                url_list.append(archive["url"])

        # set base datadir and output var result
        dataset_folder = Path(self.root).joinpath(self._base_folder)
        result = False

        # check csv files if base_folder exists
        if dataset_folder.exists():
            # check if all csv files exist and have correct md5 hash
            # TODO update to accept **self._folder_dirhash
            # for filename, md5 in {**self._csv_list}.items():
            #     fpath = Path(self.root).joinpath(self._base_folder, filename)
            #     if not self._check_md5(fpath, md5):
            #         logger.error(f"Data {fpath.name} not found or corrupted: {fpath}")
            #         return False

            result = True
        elif all(elem.exists() for elem in archive_list):
            # check if all zip files have correct md5 hash
            for archive, md5, url in zip(archive_list, md5_list, url_list, strict=True):
                if self._check_md5(archive, md5):
                    logger.info(f"Archive found and checksum verified {archive.name}.")
                    self._start_download(url=url, filename=archive, md5=md5)
                    result = True
                else:
                    logger.error(
                        f"Data {archive.name} not found or corrupted: {archive}"
                    )
                    return False

        if not self._print_check_integrity and result:
            logger.info("Files already downloaded and verified")
            self._print_check_integrity = True

        return result

    def _check_md5(self, filepath: Path, md5: str) -> bool:
        """Get md5 hash of file or directory."""
        valid_file = False
        if filepath.is_dir() and self._get_md5_dir(filepath) == md5:
            valid_file = True
        elif filepath.is_file() and check_integrity(filepath, md5):
            valid_file = True

        return valid_file

    def download(
        self, url: str, filename: str, md5: str, extracted_folder: str = None
    ) -> None:
        """Download the dataset."""
        if extracted_folder is None:
            extracted_folder = self.root

        if self._check_integrity():
            return

        download_msg = (
            f"Downloading {Path(url).name} from {Path(url).parent} to {self.root}"
        )
        logger.info(download_msg)
        self._start_download(
            url=url,
            filename=filename,
            md5=md5,
            extracted_folder=extracted_folder,
        )

    def _start_download(
        self, url: str, filename: str, md5: str, extracted_folder: str
    ) -> None:
        """Start download."""
        download_and_extract_archive(
            url,
            self.root,
            extract_root=self.root,
            filename=filename,
            md5=md5,
        )
        native_extracted_folder = Path(self.root).joinpath(
            self._zip_metadata["extracted_folder"]
        )
        if native_extracted_folder.exists():
            logger.info(
                f"Renaming folder {self._zip_metadata['extracted_folder']}"
                f" to {self._base_folder}"
            )
            native_extracted_folder.rename(
                native_extracted_folder.with_name(self._base_folder)
            )

    def _get_md5_dir(self, filepath: Path) -> str:
        """Get md5 hash of directory."""
        raise NotImplementedError("get_md5_dir method not implemented.")

    def _get_md5_file(self, filepath: Path) -> str:
        """Get md5 hash of file."""
        return calculate_md5(filepath)

    def get_random_images(
        self,
        num_images: int = 1,
        # groupby: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get random images sampled from each class."""
        if random_seed is not None:
            self._set_random_seed(random_seed)

        # create dataframe from samples and target_dict
        samples_labels = [
            (s[0], t) for s, t in zip(self.samples, self.targets, strict=True)
        ]
        df = pd.DataFrame(samples_labels, columns=["filepath", "label"])
        if random_seed is None:
            # shuffle dataframe for more diversity
            df = df.sample(frac=1, random_state=self.random_seed)

        # sample num_images from each group
        sample_df = df.groupby("label").apply(
            lambda x: x.sample(n=num_images, random_state=self.random_seed)
        )
        idx_list = [idx[1] for idx in sample_df.index]
        # add image and label to output_dict
        output_dict = []
        for idx in idx_list:
            sample_dict = self.__getitem__(idx)
            output_dict += [
                {
                    "image": sample_dict.get("image"),
                    "filename": sample_dict.get("filename"),
                    "label_name": sample_dict.get("label_name"),
                    "image_id": sample_dict.get("image_id"),
                }
            ]

        return output_dict

    def _set_loader(self, loader: Union[str, Callable] = default_loader) -> None:
        """Set loader."""
        if isinstance(loader, str):
            self._loader = self._set_loader_from_str(loader.lower())
        elif callable(loader):
            self._loader = loader
        else:
            raise ValueError(f"Loader {loader} not supported.")

    def _set_loader_from_str(self, loader: str) -> Callable:
        """Set loader from string."""
        if loader in {"pil", "pillow", "default"}:
            return default_loader
        elif loader in {"cv2", "opencv"}:
            return cv2_loader
        else:
            raise ValueError(f"Loader {loader} not supported.")

    def _set_split(self, split: Union[str, list]) -> Union[str, list]:
        """Ensure consistent split name."""
        if isinstance(split, str):
            return validate_split(split)
        elif isinstance(split, list):
            return validate_split_list(split)
        else:
            raise ValueError(f"Invalid split type: {type(split)}")

    def _set_random_seed(self, random_seed: Optional[int] = None) -> None:
        """Set random seed.

        pytorch-lightning.seed_everything is not used to not interfere with
        random seed for ptl trainer, datamodule, etc.
        """
        if random_seed is not None and not self._seed_set:
            self._seed_set = set_random_seed(random_seed)

    def make_dataset(self) -> None:
        """Make dataset."""
        raise NotImplementedError("make_dataset method not implemented.")

    def to_pandas(
        self, fields: Optional[List[str]] = None, ndigits: Optional[int] = 4
    ) -> pd.DataFrame:
        """Collect information about the dataset into a pandas dataframe.

        :param fields: List of fields to include in the dataframe.
        :return: Pandas DataFrame.
        """
        raise NotImplementedError("to_pandas method not implemented.")

    def class_counts(self, sort: Optional[str] = None) -> Dict[str, int]:
        """Enumerates and returns the number of elements for each class.

        :return: Dictionary of class counts.
        """
        class_counts = Counter(self.targets)
        class_counts = {self.classes[k]: v for k, v in class_counts.items()}

        if sort == "alpha":
            # sort by class name
            return dict(sorted(class_counts.items()))
        elif sort == "num":
            # sort by class count
            return dict(sorted(class_counts.items(), key=lambda item: item[1]))
        else:
            # sort by order in classes_to_idx
            return {k: class_counts.get(k) for k in self.classes_to_idx.keys()}

    def class_ratios(
        self,
        ndigits: Optional[int] = 3,
        relative: bool = False,
        sort: Optional[str] = None,  # default sort by order in classes_to_idx
    ) -> Dict[str, float]:
        """Returns the ratio of each class count divided the total samples (default) or max class count)

        :return: Dictionary of class ratios.
        """
        class_counts = self.class_counts()

        # Find the denominator
        denominator = max(class_counts.values()) if relative else len(self.targets)
        logger.debug(f"Class ratios relative to total samples: {denominator}")

        # Calculate ratios
        # relative to total samples (default) or max class count
        class_ratios = {
            k: round(v or 0 / denominator, ndigits=ndigits)
            for k, v in class_counts.items()
        }

        if sort == "alpha":
            # sort by class name
            return dict(sorted(class_ratios.items()))
        elif sort == "num":
            # sort by class count
            return dict(sorted(class_ratios.items(), key=lambda item: item[1]))
        else:
            # sort by order in classes_to_idx
            return {k: class_ratios.get(k) for k in self.classes_to_idx.keys()}

    def _create_train_test_split(
        self,
        split: str,
        samples: List[Tuple[str, Dict[str, Any]]],
        targets: List[int],
        random_seed: Optional[int] = None,
        train_val_test: Optional[Tuple[float]] = (0.7, 0.1, 0.2),
    ) -> List[Tuple[str, int]]:
        """Create train/dev/text split and set self.samples."""
        # set random seed
        self._set_random_seed(random_seed)

        # split list of tuple into two lists
        img_idx = range(len(samples))

        # split into train/dev/test
        train_size = int(len(samples) * train_val_test[0])
        val_size = int(len(samples) * train_val_test[1])
        test_size = len(samples) - train_size - val_size
        self._train_val_test = (train_size, val_size, test_size)

        #
        _filenames, labels = zip(*samples, strict=True)
        if isinstance(labels, tuple):
            labels = [elem.get("label") for elem in labels]

        all_train_idx, test_idx, all_train_y, _ = train_test_split(
            img_idx,
            labels,
            stratify=labels,
            test_size=test_size,
            random_state=self.random_seed,
            shuffle=True,
        )

        train_idx, val_idx, _, _ = train_test_split(
            all_train_idx,
            all_train_y,
            test_size=val_size,
            random_state=self.random_seed,
        )

        # Assign test dataset for use in dataloader(s)
        if split in {"train", None}:
            samples = [samples[idx] for idx in train_idx]
            targets = [targets[idx] for idx in train_idx]
        elif split == "validation":
            samples = [samples[idx] for idx in val_idx]
            targets = [targets[idx] for idx in val_idx]
        elif split in "train+val":
            samples = [samples[idx] for idx in train_idx + val_idx]
            targets = [targets[idx] for idx in train_idx + val_idx]
        elif split == "test":
            samples = [samples[idx] for idx in test_idx]
            targets = [targets[idx] for idx in test_idx]
        elif split == "all":
            samples = samples
            targets = targets
            # update split in_place in samples and targets
            for idx in train_idx:
                samples[idx][1]["split"] = "train"
                # targets[idx]["split"] = "train"

            for idx in val_idx:
                samples[idx][1]["split"] = "validation"
                # targets[idx]["split"] = "validation"

            for idx in test_idx:
                samples[idx][1]["split"] = "test"
                # targets[idx]["split"] = "test"

        else:
            raise ValueError(f"Split {split} not found.")

        # paired shuffle samples and targets
        samples_targets = list(zip(samples, targets))
        random.shuffle(samples_targets)
        samples, targets = zip(*samples_targets)

        # check all target_dict["label"] and target are the same
        for idx, (sample, target) in enumerate(zip(samples, targets)):
            if sample[1]["label"] != target:
                logger.error(
                    f"Sample {idx} label mismatch: {sample[1]['label']} != {target}"
                )

        # log info
        logger.info(
            f"Creating split '{split}' ({len(samples)}/{len(labels)}) total samples"
        )
        logger.info(
            f"Train: {len(train_idx)} samples, Validation: {len(val_idx)} samples, Test: {len(test_idx)} samples"
        )

        return samples, targets
