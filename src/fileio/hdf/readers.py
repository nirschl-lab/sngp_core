#!/usr/bin/env python3
"""readers.py in src/sngp_core/fileio/hdf."""

import ast
import os
import pprint
from pathlib import Path
from typing import Optional
from typing import Union

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from numpy._typing import NDArray
from pydantic import BaseModel
from rapidfuzz import fuzz
from rapidfuzz import process

from src import DATA_ROOT
from src import setup_cache_dirs
from src.conf.base_hdf import BioVLMHDF
from src.conf.base_hdf import HDFMetadata
from src.conf.base_hdf import HDFSplit
from src.conf.pydantic_validators import ListStr
from src.custom_datasets.dataset_registry import DatasetType
from src.fileio.text import is_empty_file
from src.fileio.text import valid_file_ext
from src.models.model_registry import ModelType
from src.processing.text.harmonize_text import validate_split
from src.processing.text.harmonize_text import validate_split_list


def hdf_loader(
    filepath: Union[str, Path, os.PathLike], split: Optional[Union[str, ListStr]] = None
) -> BioVLMHDF:
    """Load HDF file and return its contents as a DataFrame.

    Args:
        filepath (Union[str, Path]): The path to the HDF file.

    Returns:
        pd.DataFrame: The contents of the HDF file as a DataFrame.

    Raises:
        ValueError: If file_path is None or empty.
        ValueError: If file_path has an invalid file type.
        FileNotFoundError: If file_path does not exist.

    Examples:
        >>> hdf_loader("data.h5")
        key
        0  value
    """
    filepath = Path(filepath)
    split = validate_split_list(split)

    # check filepath
    if not valid_file_ext(filepath, {".h5", ".hdf", ".hdf5"}):
        logger.error(f"Invalid file type: {filepath.suffix}")
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if is_empty_file(filepath):
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"File is empty: {filepath}")

    # check if split exists in HDF file
    with h5py.File(filepath, "r") as fh:
        for s in split:
            if s not in fh.keys():
                logger.error(f"Split '{s}' not found in HDF file: {filepath}")
                raise ValueError(f"Split '{s}' not found in HDF file: {filepath}")

    # load dataset metadata
    with h5py.File(filepath, "r") as fh:
        logger.debug(f"Available splits: {list(fh.keys())}")
        metadata = {attr: fh.attrs[attr] for attr in fh.attrs}

    for k in ["classes_to_idx", "image_mean_std"]:
        metadata[k] = eval(metadata[k]) if k in metadata else None

    # create BioVLMHDF object
    metadata = HDFMetadata(**metadata)

    # load data
    output = {}
    for s in split:
        logger.debug(f"Loading {filepath.stem} {s}") if os.getenv("DEBUG") else ""
        with h5py.File(filepath, "r") as fh:
            feats = fh[s]["features"][()].astype(np.float32)
            labels = fh[s]["labels"][()].astype(np.int32)
            img_ids = fh[s]["image_ids"][()].astype(np.str_)
            inst_ids = fh[s]["instance_ids"][()].astype(np.str_)
            img_files = fh[s]["image_files"][()].astype(np.str_)
            al_selected = fh[s]["active_learning/selected"][()].astype(np.bool_)
            al_scores = fh[s]["active_learning/scores"][()].astype(np.float32)
            al_labels = fh[s]["active_learning/labels"][()].astype(np.int32)

        # convert img_files from ndarray[np.str_] to List[str]
        img_ids = [ast.literal_eval(item)[0] for item in img_ids]
        inst_ids = [ast.literal_eval(item)[0] for item in inst_ids]
        img_files = [ast.literal_eval(item)[0] for item in img_files]

        # create HDFSplit object
        split_dict = {
            "features": feats,
            "labels": labels.squeeze(),
            "image_ids": img_ids,
            "instance_ids": inst_ids,
            "image_files": img_files,
            "al_selected": al_selected.squeeze(),
            "al_scores": al_scores.squeeze(),
            "al_labels": al_labels.squeeze(),
        }
        output[s] = HDFSplit(**split_dict)
        logger.debug(f"Loaded {len(feats)} samples from {filepath} {s}.")

    return BioVLMHDF(metadata=metadata, **output)


def load_hdf_as_df(
    filepath: Union[str, Path, os.PathLike], split: Optional[Union[str, ListStr]] = None
) -> pd.DataFrame:
    hdf = hdf_loader(filepath, split)
    metadata = hdf.metadata.model_dump()

    # split must be list
    if isinstance(split, str):
        split = [split]

    df_list = []
    for s in split:
        hdf_split = getattr(hdf, s)
        if hdf_split is None:
            logger.warning(
                f"Split {s} not found in HDF file: {filepath}"
                f"\nAvailable splits: {list(hdf.keys())}"
            )
            continue

        feats_df = pd.DataFrame(hdf_split.features)
        split_list = [s for _ in range(len(hdf_split.image_files))]
        try:
            info_df = pd.DataFrame(
                {
                    "filepath": hdf_split.image_files,
                    "split": split_list,
                    "image_id": hdf_split.image_ids,
                    "instance_id": hdf_split.instance_ids,
                    "label": hdf_split.labels,
                    "al_selected": hdf_split.al_selected,
                    "al_score": hdf_split.al_scores,
                    "al_label": hdf_split.al_labels,
                }
            )
            df = pd.concat([info_df, feats_df], axis=1)
            df_list.append(df)
        except ValueError as e:
            logger.error(f"Error loading split {s} from {filepath}: {e}")
            raise ValueError(f"Error loading split {s} from {filepath}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    df["split"] = df["split"].astype(
        pd.CategoricalDtype(categories={"train", "validation", "test"})
    )
    if df.empty:
        logger.error(f"Empty DataFrame loaded from {filepath}")
        raise ValueError(f"Empty DataFrame loaded from {filepath}")

    # check for nan in cols except those starting with "al_"
    al_cols = [col for col in df.columns if str(col).startswith("al_")]
    if df.drop(columns=al_cols).isnull().values.any():
        cols_w_nan = df.columns[df.isnull().any()].tolist()
        logger.error(f"DataFrame contains NaN values in columns: {cols_w_nan}")
        raise ValueError(f"DataFrame contains NaN values in columns: {cols_w_nan}")

    logger.info(f"Loaded {len(df)} samples from {filepath}.")
    dataset_str = metadata.pop("dataset_str", "").replace(r"\n", "")
    dataset_repr = metadata.pop("dataset_repr", "").replace(r"\n", "")
    (
        logger.info(f"\n{pprint.pformat(dataset_str)}\n")
        if dataset_str and os.getenv("DEBUG")
        else ""
    )
    logger.info(f"\n{dataset_repr}\n") if dataset_str and os.getenv("DEBUG") else ""
    # (
    #     logger.debug(f"\n{pprint.pformat(metadata, sort_dicts=False)}")
    #     if os.getenv("DEBUG")
    #     else ""
    # )

    # convert int col names to str
    df.columns = [str(col) for col in df.columns]

    return df


def load_dataset(
    dataset_name: Union[str, Path],
    model_name: str,
    split: Union[str, ListStr],
    feature_cache: Optional[Path],
    score_cutoff: int = 95,
) -> pd.DataFrame:
    data_root = DATA_ROOT

    split = validate_split_list(split)

    # setup cache directories
    feature_cache, model_cache = setup_cache_dirs(feature_cache)

    # print logs
    logger.add(
        Path(feature_cache).joinpath(f"{Path(dataset_name).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    if os.path.sep in str(dataset_name):
        dataset_type = Path(dataset_name).stem
    else:
        dataset_type = DatasetType[dataset_name].name

    model_type = ModelType[model_name]
    output_dir = Path(feature_cache).joinpath(f"{dataset_type}")
    vector_file = output_dir.joinpath(f"{model_type.name}.hdf")

    if not vector_file.exists():
        logger.warning(f"Unable to find exact match for vector file: {vector_file}")
        file_list = list(output_dir.glob("*.hdf"))
        result = process.extractOne(
            vector_file.name,
            [f.name for f in file_list],
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )
        if result:
            logger.info(f"Found similar file: {result[0]} ({result[1]})")
            vector_file = file_list[result[2]]
        else:
            logger.error(f"Unable to find similar vector file: {vector_file}")
            raise FileNotFoundError(f"Vector file does not exist: {vector_file}")

    if not vector_file.exists():
        logger.error(f"Unable to find vector file: {vector_file}")
        raise FileNotFoundError(f"Vector file does not exist: {vector_file}")

    # check if split in vector file
    for s in split:
        with h5py.File(vector_file, "r") as fh:
            if s not in fh.keys() and s != "all":
                logger.error(f"Split {split} not found in file: {vector_file}")
                raise KeyError(f"Split {split} not found in file: {vector_file}")

    # load dataset (can be multiple splits or all)
    if "all" in split:
        split = ["train", "validation", "test"]
    if isinstance(split, str):
        split = [split]

    # load hdf as df and concat into single df
    df_list = {}
    for s in split:
        if s not in {"train", "validation", "test"}:
            logger.error(f"Invalid split: {s}")
            raise ValueError(f"Invalid split: {s}")

        df_list[s] = load_hdf_as_df(vector_file, split=s)

    # concat df_list into single df
    df = pd.concat(df_list.values(), ignore_index=True)
    df["split"] = df["split"].astype(
        pd.CategoricalDtype(categories={"train", "validation", "test"})
    )
    logger.info(
        f"Loaded dataset {dataset_name} with shape {df.shape} from {vector_file}"
    )
    return df
