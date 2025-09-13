#!/usr/bin/env python3
"""data_utils.py in src/sngp_core/processing."""
import datetime
import hashlib
import os
import re
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from loguru import logger
from rapidfuzz import process

from src.fileio.dataframe.readers import df_loader

# from src.data import REQUIRED_COLUMNS
from src.fileio.text import is_empty_file
from src.fileio.text import is_none_or_empty
from src.fileio.text.readers import get_metadata_file
from src.fileio.text.readers import json_loader
from src.fileio.text.readers import yaml_loader


# compiled regex for case-insensitive nan, NaN, NAN, or np.nan
regex_nan = re.compile(r"nan|np.nan", re.IGNORECASE)
regex_timestamp = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|(\+|-)\d{2}:\d{2})"
)


def create_filename(
    output_dir: Path,
    idx: int,
    image_id: str,
    label_name: str,
    split: str,
    patient_id: str = "na",
) -> Path:
    # convert input args to str if not already
    if isinstance(patient_id, (int, float)):
        # no decimal points
        patient_id = str(patient_id).replace(".", "-")

    patient_id = patient_id.replace(" ", "-").replace("_", "-")
    split = split.replace(" ", "-").replace("_", "-")
    label_name = label_name.replace(" ", "-").replace("_", "-")
    output_dir = Path(output_dir).joinpath(split)

    return output_dir.joinpath(
        f"{idx:05d}_{image_id[:8]}_{patient_id}_split-{split}_{label_name}.png",
    )


def unpack_list(mixed_list):
    if isinstance(mixed_list, str):
        return mixed_list

    output_list = []
    if isinstance(mixed_list, list):
        for elem in mixed_list:
            output_list += elem if isinstance(elem, list) else [elem]

    # hack
    output_list = [
        elem for elem in output_list if elem not in {"na", "nan", "None", "none", "N/A"}
    ]
    # remove np.nan/inf
    output_list = [
        elem
        for elem in output_list
        if not isinstance(elem, float) and not pd.isnull(elem)
    ]

    return sorted(set(output_list))


def update_metadata(
    query: str,
    metadata: dict,
    options: list,
    limit: int = 1,
    min_score: float = 0.75,
) -> dict:
    result = metadata.get(query)
    if result is None:
        # create and set to nan
        metadata[query] = "nan"
    elif result not in options:
        match, score, _ = process.extract(result, options, limit=limit)[0]
        if score / 100 < min_score or isinstance(match, list):
            logger.warning(f"Low score {score / 100} for {result}")
            metadata[query] = "nan"
        else:
            logger.info(f"Matched {query} to {result}({score / 100}).")
            metadata[query] = match

    return metadata


def is_image_folder_format(input_dir: Path, ext: Optional[list] = None) -> bool:
    """Check if the input directory follows the ImageFolder format criteria.

    Args:
        input_dir: Path to the input directory to be checked.
        ext: A list of valid image file extensions (default is None).

    Returns:
        bool: True if the input directory meets the ImageFolder format criteria, False otherwise.
    """
    ext = ext or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    # check for ImageFolder format:
    # - must have subdir "images"
    # - must have at least two subdirectories
    # - each subdirectory must contain at least one image file
    image_folder = input_dir.joinpath("images")
    if not image_folder.exists():
        return False

    # check for at least two non-hidden subdirectories
    subdirs = [
        elem
        for elem in image_folder.iterdir()
        if elem.is_dir() and not elem.name.startswith(".")
    ]
    if len(subdirs) < 2:
        return False

    return not any(
        all(elem.suffix not in ext for elem in subdir.iterdir()) for subdir in subdirs
    )


def check_data_format(input_dir: Path, metadata_format: str = "yaml") -> str:
    # check for yaml, json, csv, or feather metadata file
    metadata_file = get_metadata_file(input_dir, metadata_format)

    # remove "image_id" from required columns (can be generated)
    REQUIRED_COLUMNS = ["dataset_name", "split", "label_name", "label"]
    required_cols = set(REQUIRED_COLUMNS) - {"image_id"}

    # check dataset format
    if metadata_file.suffix in {".csv", ".feather"}:
        metadata = df_loader(metadata_file)
        if missing_cols := set(required_cols) - set(metadata.columns):
            raise ValueError(f"Missing required columns: {missing_cols}")
        else:
            return "dataframe"

    if not is_image_folder_format(input_dir):
        raise ValueError("Invalid data format in input directory.")

    metadata = (
        json_loader(metadata_file)
        if metadata_file.suffix == ".json"
        else yaml_loader(metadata_file)
    )
    # check for required keys in json/yaml metadata
    # - "split", "label_name", "label" are defined at the image level
    required_cols = set(REQUIRED_COLUMNS) - {"split", "label_name", "label", "image_id"}
    if missing_keys := set(required_cols) - set(metadata.keys()):
        raise ValueError(f"Missing required keys: {missing_keys}")
    else:
        logger.info("Data format: src.custom_datasets.BioVLMImageFolder")
        return "image_folder"


def is_relative_path(path):
    """Check if a path is relative."""
    return not Path(path).is_absolute()


def _is_valid_timestamp(value: str) -> bool:
    """Check if a string is a valid ISO 8601 timestamp."""
    if value.endswith("Z"):
        # Python expects a strictly local time or an offset
        # from UTC (e.g., +00:00 or -04:00) but not 'Z'.
        # Replace 'Z' before checking if valid ISO 8601 timestamp.
        value = value.replace("Z", "+00:00")

    try:
        datetime.datetime.fromisoformat(value)
        return True
    except ValueError as e:
        logger.error(f"Invalid timestamp format: {value}\n{e}")
        return False


def timestamp(str_format: str = "%Y-%m-%dT%H:%M:%S.%f") -> str:
    """Generate a timestamp string in the specified format.

    Args:
        str_format: A string specifying the format of the timestamp (default is "%Y-%m-%dT%H:%M:%S.%f").

    Returns:
        str: A string representing the current timestamp in the specified format.
    """
    utc_tz = datetime.timezone.utc
    result = datetime.datetime.now(utc_tz).strftime(str_format)[:-3]
    result = result + "Z"  # add Zulu/UTC timezone
    # check if valid ISO 8601 timestamp
    if not _is_valid_timestamp(result):
        raise ValueError(f"Invalid timestamp format: {result}")

    return result


def find_image_json_pairs(
    input_dir: Union[Path, str],
    ext: str,
    recursive: bool,
    jsonl_filepath: Optional[Union[Path, str]] = None,
    ignore: Optional[str] = None,
    exclude_empty: bool = True,
    force: bool = False,
) -> tuple:
    """Find JSON and image pairs."""
    input_dir = Path(input_dir)
    if jsonl_filepath is None:
        jsonl_filepath = input_dir.joinpath("image_json_pairs.jsonl")

    ext = ext if ext.startswith(".") else f".{ext}"

    if jsonl_filepath.exists() and not force:
        # load jsonl file with image-json pairs
        image_json_df = pd.read_json(jsonl_filepath, orient="records", lines=True)

        # update path with input_dir if relative path, otherwise assume
        max_sample = 10 if os.getenv("DEBUG") else 100
        min_sample = int(len(image_json_df) * 0.01) or 1
        num_sample = min(min_sample, max_sample)
        if num_sample < len(image_json_df):
            logger.info(f"Sampling {num_sample} image-json pairs.")
            df_subset = image_json_df["json_file"].sample(num_sample)
        else:
            logger.info(f"Sampling {len(image_json_df)} image-json pairs.")
            df_subset = image_json_df["json_file"]

        if all(is_relative_path(f) for f in df_subset):
            image_json_df["json_file"] = (
                image_json_df["json_file"]
                .apply(lambda x: input_dir.joinpath(x))
                .tolist()
            )
            image_json_df["image_file"] = (
                image_json_df["image_file"]
                .apply(lambda x: input_dir.joinpath(x))
                .tolist()
            )

        # check subset of files exist
        df_subset = image_json_df["json_file"].sample(num_sample)
        if not all(df_subset.apply(lambda x: Path(x).exists())):
            logger.error(
                f"Expected json files not found in {input_dir}. "
                "Please rerun with force=True to regenerate image_json_pairs."
            )
            raise FileNotFoundError("Expected json files not found")

        # get json and image files
        json_files = image_json_df["json_file"].tolist()
        image_files = image_json_df["image_file"].tolist()
    else:
        # r
        # set ignore flags (defaults to any hidden files, autosave or backup files
        emacs_autosave = r"~#|#.*|#$"  # emacs autosave files
        # hidden files, autosave, backup files
        hidden_autosave_backup = r"^\..*|\~$|\.swp$"
        if ignore:
            ignore_pattern = f"{hidden_autosave_backup}|{emacs_autosave}|{ignore}"
        else:
            ignore_pattern = f"{hidden_autosave_backup}|{emacs_autosave}"

        # find json files
        annot_ext = ".json"
        json_files = sorted(
            list(input_dir.rglob("*" + annot_ext))
            if recursive
            else list(input_dir.glob("*" + annot_ext))
        )
        # ignore json file with same name as parent directory
        json_files = [elem for elem in json_files if elem.stem != elem.parent.name]

        # remove ignored files
        json_files = [
            elem for elem in json_files if not re.match(ignore_pattern, elem.name)
        ]

        # remove files that are empty (stat size == 0)
        if exclude_empty:
            json_files_clean = [elem for elem in json_files if not is_empty_file(elem)]
            if len(json_files) != len(json_files_clean):
                missing_files = set(json_files) - set(json_files_clean)
                logger.warning(
                    f"Removed {len(missing_files)} empty json files: {missing_files}"
                )
                raise FileNotFoundError(f"Empty json files found: {missing_files}")

        # error checks
        if not json_files:
            logger.error(f"No json files found in {input_dir}")
            return [], []
            # raise FileNotFoundError(f"No json files found in {input_dir}")

        # find image files assuming image is json_file.with_suffix(ext)
        image_files = [
            json_file.parent.joinpath(json_file.with_suffix(ext))
            for json_file in json_files
        ]
        # error checks
        for elem in image_files:
            if not elem.exists():
                logger.error(f"Expected image-json pairs. Image not found: {elem}")
                raise FileNotFoundError(f"Image not found: {elem}")

        if len(json_files) != len(image_files):
            logger.error(
                f"Number of json files {len(json_files)} "
                f"!= number of image files {len(image_files)}"
            )
            raise ValueError("Number of json files != number of image files")

        # save jsonl of all image-json pairs
        image_json_df = pd.DataFrame(
            {
                "json_file": [f.relative_to(input_dir).as_posix() for f in json_files],
                "image_file": [
                    f.relative_to(input_dir).as_posix() for f in image_files
                ],
            }
        )
        json_str = image_json_df.to_json(orient="records", lines=True)
        with jsonl_filepath.open("w") as f:
            f.write(json_str)

    return json_files, image_files


def match_one_to_many(
    list_1: List[str], list_2: List[str], ext: str = "png", strict: bool = False
) -> dict:
    """Find matching files between two lists of file paths.

    Args:
        list_1: The first list of file paths.
        list_2: The second list of file paths.
        ext: The file extension to match (default: "png").
        strict: Whether to perform strict matching based on the full stem
             or loose matching based on the stem prefix (default: False).

    Returns:
        A dictionary where the keys are the file names from list_1 and
        the values are lists of matching file paths from list_2.

    Examples:
        >>> list_1 = ["file1.png", "file2.png", "file3.png"]
        >>> list_2 = ["file1_1.png", "file1_2.png", "file2_1.png", "file3_1.png"]
        >>> matches = match_one_to_many(list_1, list_2)
    """
    # convert to Path objects
    list_1 = [Path(x) for x in list_1]
    list_2 = [Path(x) for x in list_2]

    # dictionary to store the results
    matches = {}

    # Iterate through each svs file to find matching png files
    for ref_file in list_1:
        # Extract the stem for each file to regex match with list_2
        prefix = ref_file.stem if strict else ref_file.stem.split("_")[0]
        ref_pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(ext)}$")

        # Find all png files that match the pattern
        matched_pngs = [elem for elem in list_2 if ref_pattern.match(elem.name)]

        # Add the matches to the dictionary
        matches[ref_file.name] = matched_pngs

    return matches


def match_lists(
    list_1: List[Union[str, Path]],
    list_2: List[Union[str, Path]],
    ext: str = "png",
    strict: bool = False,
    prefix_fields: int = 1,
    suffix_fields: Optional[int] = None,
) -> Tuple[List[Path], List[Path]]:
    """Match files between two lists of file paths based on specified prefix and suffix fields.

    Args:
        list_1: The first list of file paths.
        list_2: The second list of file paths.
        ext: The file extension to match (default: "png").
        strict: Whether to perform strict matching based on the
            full stem or loose matching based on the specified
            prefix fields (default: False).
        prefix_fields: The number of prefix fields to consider (default: 1).
        suffix_fields: The number of suffix fields to consider (default: None).

    Returns:
        A tuple containing two lists: list_1_out and list_2_out.
        list_1_out contains the matched file paths from list_1,
        and list_2_out contains the corresponding matched file
        paths from list_2.

    Raises:
        ValueError: If the lengths of list_1_out and list_2_out are not equal.
    """
    # convert to Path objects
    list_1 = [Path(x) for x in list_1]
    list_2 = [Path(x) for x in list_2]

    # dictionary to store the results
    list_1_out = []
    list_2_out = []

    # Iterate through each svs file to find matching png files
    for ref_file in list_1:
        # Extract the stem for each file to regex match with list_2
        prefix = (
            ref_file.stem
            if strict
            else "_".join(ref_file.stem.split("_")[:prefix_fields])
        )
        suffix = (
            "_".join(ref_file.stem.split("_")[-suffix_fields:]) if suffix_fields else ""
        )
        # hack replace y with x|y
        suffix = suffix.replace("y", "(x|y)")
        ref_pattern = re.compile(f"^{re.escape(prefix)}.*{suffix}.*{re.escape(ext)}$")

        # Find all png files that match the pattern
        matched_pngs = [elem for elem in list_2 if ref_pattern.match(elem.name)]

        if len(matched_pngs) == 1:
            list_1_out.append(ref_file)
            list_2_out.append(matched_pngs[0])
        elif len(matched_pngs) > 1:
            logger.warning(
                f"Multiple matches found for {ref_file.name}: {len(matched_pngs)}"
            )

    if len(list_1_out) != len(list_2_out):
        error_msg = (
            f"Length of list_1_out ({len(list_1_out)}) "
            f"and list_2_out ({len(list_2_out)}) must be equal."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return list_1_out, list_2_out


def replace_str(
    v: str,
    pattern: str,
    replace_val: str,
    exact_match: bool = True,
    replace_substr: bool = False,
) -> str:  # sourcery skip: remove-redundant-if
    """Replaces a pattern with a replacement value in a string.

    Args:
        v: The input string to process.
        pattern: The pattern to replace.
        replace_val: The value to replace the pattern with.
        exact_match: If True, replace the entire value if pattern == v.
            If False, replace the entire value if pattern is found in v.

    Returns:
        The string with the pattern replaced by the replacement value.

    Examples:
        >>> replace_str("Hello, world!", "world", "Python")
        'Hello, Python!'
    """
    if exact_match:
        return replace_val if pattern == v else v

    # partial match
    if (
        pattern in {"nan", "NaN", "NAN", "np.nan"} and not replace_substr
    ):  # sourcery skip: remove-redundant-if
        # replace full string if replace_substr is False
        return replace_val if regex_nan.search(v) else v
    elif pattern in {"nan", "NaN", "NAN", "np.nan"} and replace_substr:
        # replace substring if replace_substr is True
        return regex_nan.sub(replace_val, v)
    elif replace_substr:
        # string replace with substring if replace_substr is True
        return v.replace(pattern, replace_val)
    else:
        # string replace with full string if replace_substr is False
        return replace_val if pattern.lower() in v.lower() else v


def replace_str_nested_dict(
    x: dict,
    to_replace: str,
    value: Union[str, int, float],
    exact_match: bool = True,
    replace_substr: bool = False,
) -> dict:
    """Recursively replaces 'pattern' with 'replace_val' in a nested dictionary.

    Args:
        x: The nested dictionary to process.
        to_replace: The string to replace.
        value: The value to replace pattern with.
        exact_match: If True, replace entire value only if pattern == v.
        replace_substr: If True, replace substring of value that matches pattern.

    Returns:
        The dictionary with 'pattern' values replaced by 'replace_val'.

    Examples:
        >>> data = {'a': 'nan', 'b': {'c': 'nan'}}
        >>> replace_str_nested_dict(data, to_replace='nan', value=np.nan)
        {'a': np.nan, 'b': {'c': np.nan}}
    """
    if not isinstance(to_replace, str):
        raise TypeError(f"pattern must be a string, not {type(to_replace)}")

    value = value or None
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, str):
                x[k] = replace_str(v, to_replace, value, exact_match, replace_substr)
            elif isinstance(v, dict):
                x[k] = replace_str_nested_dict(
                    v, to_replace, value, exact_match, replace_substr
                )
            elif isinstance(v, list):
                x[k] = [
                    replace_str(elem, to_replace, value, exact_match, replace_substr)
                    for elem in v
                ]
    return x


def _compute_checksum_md5(
    file_path: Union[Path, str], chunk_size: int = 1024 * 1024
) -> str:
    """Compute the MD5 checksum of a file.

    Args:
        file_path (Union[Path, str]): The path to the file.
        chunk_size (int, optional): Size of chunks to read. Defaults to 1024 * 1024.

    Returns:
        str: The MD5 checksum of the file.

    Examples:
        >>> file_path = "data.txt"
        >>> _compute_checksum_md5(file_path)
        '5eb63bbbe01eeed093cb22bb8f5acdc3'
    """
    hash_md5 = hashlib.md5(usedforsecurity=False)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _compute_checksum_sha1(
    file_path: Union[Path, str], chunk_size: int = 1024 * 1024
) -> str:
    """Compute the SHA-1 checksum of a file.

    Args:
        file_path (Union[Path, str]): The path to the file.
        chunk_size (int, optional): Size of chunks to read. Defaults to 1024 * 1024.

    Returns:
        str: The SHA-1 checksum of the file.

    Examples:
        >>> file_path = "data.txt"
        >>> _compute_checksum_sha1(file_path)
        '2ef7bde608ce5404e97d5f042f95f89f1c232871'
    """
    hash_sha1 = hashlib.sha1(usedforsecurity=False)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha1.update(chunk)
    return hash_sha1.hexdigest()


def _compute_checksum_sha256(
    file_path: Union[Path, str], chunk_size: int = 1024 * 1024
) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        file_path (Union[Path, str]): The path to the file.
        chunk_size (int, optional): Size of chunks to read. Defaults to 1024 * 1024.

    Returns:
        str: The SHA-256 checksum of the file.

    Examples:
        >>> file_path = "data.txt"
        >>> _compute_checksum_sha256(file_path)
        '3c9b8e5b230f9b8e5b230f9b8e5b230f9b8e5b230f9b8e5b230f9b8...'
    """
    hash_sha256 = hashlib.sha256(usedforsecurity=False)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def _compute_checksum_sha512(
    file_path: Union[Path, str], chunk_size: int = 1024 * 1024
) -> str:
    """Compute the SHA-512 checksum of a file.

    Args:
        file_path (Union[Path, str]): The path to the file.
        chunk_size (int, optional): Size of chunks to read. Defaults to 1024 * 1024.

    Returns:
        str: The SHA-512 checksum of the file.

    Examples:
        >>> file_path = "data.txt"
        >>> _compute_checksum_sha512(file_path)
        'c8b5b6a7d8e9f0c1d2e3f4c5d6e7f8c9d0e1f2c3d4e5f6c7d8e9f0...'
    """
    hash_sha512 = hashlib.sha512(usedforsecurity=False)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha512.update(chunk)
    return hash_sha512.hexdigest()


def compute_checksum(
    filepath: Union[Path, str], method: str = "md5", chunk_size: int = 1024 * 1024
) -> str:
    """Compute the checksum of a file using the specified method.

    Args:
        filepath (Union[Path, str]): The path to the file.
        method (str, optional): The checksum method to use. Defaults to "md5".
        chunk_size (int, optional): Size of chunks to read. Defaults to 1024 * 1024.

    Returns:
        str: The checksum of the file.

    Raises:
        ValueError: If an invalid checksum method is provided.
        ValueError: If filepath is None or empty.
        ValueError: If the file is empty.

    Examples:
        >>> filepath = "data.txt"
        >>> compute_checksum(filepath, method="sha256")
        '3c9b8e5b230f9b8e5b230f9b8e5b230f9b8e5b230f9b8e5b230f9b...'
    """
    if method not in {"md5", "sha1", "sha256", "sha512"}:
        raise ValueError(f"Invalid checksum method: {method}")

    if is_none_or_empty(filepath):
        raise ValueError("file_path cannot be None or empty")

    if is_empty_file(filepath):
        raise ValueError(f"File is empty: {filepath}")

    filepath = Path(filepath)
    if method.lower() == "md5":
        return _compute_checksum_md5(filepath, chunk_size=chunk_size)
    elif method.lower() == "sha1":
        return _compute_checksum_sha1(filepath, chunk_size=chunk_size)
    elif method.lower() == "sha256":
        return _compute_checksum_sha256(filepath, chunk_size=chunk_size)
    elif method.lower() == "sha512":
        return _compute_checksum_sha512(filepath, chunk_size=chunk_size)
    else:
        raise ValueError(f"Invalid checksum method: {method}")
