#!/usr/bin/env python3
"""writers.py in src/argusdp/fileio/hdf."""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from typing import Union

import h5py
import numpy as np
from loguru import logger
from numpy._typing import NDArray

from src.custom_datasets.base_dataset import BaseDataset
from src.fileio.backup.create_backup import create_backup
from src.processing.data_utils import timestamp


def hdf_writer(
    vector_file: Union[Path, str],
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    image_ids: NDArray[np.str_],
    instance_ids: NDArray[np.str_],
    image_md5s: NDArray[np.str_],
    image_files: NDArray[np.str_],
    split: str,
    dataset: Optional[BaseDataset] = None,
) -> None:
    """Save the extracted features to an HDF file.

    Args:
        features (NDArray[np.float32]): Extracted features as a NumPy array.
        labels (NDArray[np.int64]): Image labels as a NumPy array.
        image_ids (NDArray[np.str_]): Image IDs as a NumPy array.
        instance_ids (NDArray[np.str_]): Instance IDs as a NumPy array.
        image_md5s (NDArray[np.str_]): Image MD5s as a NumPy array.
        vector_file (str): Path to the HDF file to save the features.
        split (str): Split name, either 'train' or 'test'.
    """
    if features.shape[0] != labels.shape[0] or features.shape[0] != image_ids.shape[0]:
        raise ValueError("Features and labels must have the same number of samples.")

    # check if all features zero
    if np.allclose(features, 0.0):
        raise ValueError("All features are zero.")

    # ensure features are finite
    if not np.all(np.isfinite(features)):
        non_finite_vals = np.where(~np.isfinite(features))
        logger.error(
            f"Features contain non-finite values at indices: {non_finite_vals[0][:5]}"
        )
        logger.error(f"Non-finite values: {features[non_finite_vals][:5]}")
        raise ValueError()

    # convert image_ids to List[str]
    image_ids_str = [str(image_id) for image_id in image_ids]
    instance_ids_str = [str(instance_id) for instance_id in instance_ids]
    image_md5s_str = [str(md5) for md5 in image_md5s]
    image_files_str = [str(f) for f in image_files]

    #
    empty_float32: NDArray[np.float32] = np.full(
        (features.shape[0], 1), np.nan, dtype=np.float32
    )
    empty_int32: NDArray[np.int32] = np.full((features.shape[0], 1), -1, dtype=np.int32)
    empty_bool: NDArray[np.bool_] = np.zeros((features.shape[0], 1), dtype=np.bool_)

    # create backup copy in /tmp
    vector_file = Path(vector_file)
    backup_dir = create_backup(vector_file)

    try:
        with h5py.File(vector_file, "a") as fh:
            # create metadata attributes
            if dataset is not None:
                fh.attrs["dataset_repr"] = dataset.__repr__()
                fh.attrs["dataset_str"] = dataset.__str__()
                fh.attrs["random_seed"] = dataset.random_seed
                # save the dataset transform mean and std
                mean_std = dataset._transform_mean_std
                normalize_mean = mean_std[0] if mean_std[0] is not None else None
                normalize_std = mean_std[1] if mean_std[1] is not None else None
                fh.attrs["normalize_mean_std"] = (normalize_mean, normalize_std)
                fh.attrs["transform"] = str(dataset.transform)
                config = dataset.config.model_dump(exclude_unset=True)
                for key, value in config.items():
                    if value is None:
                        continue
                    elif isinstance(value, (str, int, float, bool)):
                        fh.attrs[key] = value
                    else:
                        fh.attrs[key] = str(value)
            else:
                fh.attrs["dataset"] = vector_file.parent.stem
                fh.attrs["num_classes"] = np.unique(labels).shape[0]

            fh.attrs["model"] = vector_file.stem
            fh.attrs["num_features"] = features.shape[1]
            fh.attrs["split"] = split
            fh.attrs["last_updated"] = timestamp()
            # create datasets
            fh.create_dataset(f"{split}/features", data=features)
            fh.create_dataset(f"{split}/labels", data=labels)
            fh.create_dataset(f"{split}/image_ids", data=image_ids_str)
            fh.create_dataset(f"{split}/instance_ids", data=instance_ids_str)
            fh.create_dataset(f"{split}/image_md5s", data=image_ids_str)
            fh.create_dataset(f"{split}/image_files", data=image_files_str)

            if f"{split}/active_learning" not in fh:
                fh.create_dataset(f"{split}/active_learning/selected", data=empty_bool)
                fh.create_dataset(f"{split}/active_learning/scores", data=empty_float32)
                fh.create_dataset(f"{split}/active_learning/labels", data=empty_int32)
    except Exception as e:
        logger.error(f"Error while writing features to HDF5 file: {e}")
        if backup_dir is not None and Path(backup_dir.name).exists():
            backup_file = Path(backup_dir.name).joinpath(vector_file.name)
            logger.info(f"Restoring backup file: {backup_file}")
            shutil.copy(backup_file, vector_file)
        else:
            logger.info(f"Deleting incomplete file: {vector_file}")
            Path(vector_file).unlink()

    logger.info(f"Saved features to {vector_file}")
    if backup_dir is not None:
        logger.debug(f"Cleanup: {backup_dir.name}") if os.getenv("DEBUG") else None
        backup_dir.cleanup()
