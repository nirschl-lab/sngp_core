#!/usr/bin/env python3
"""save_pickle.py in src/sngp_core/fileio/binary."""
from pathlib import Path
from typing import Any
from typing import Union

from loguru import logger

from argusdp.fileio.text import is_none_or_empty


try:
    import dill as pickle
except ModuleNotFoundError:
    logger.warning("dill not found. Using pickle instead.")
    import pickle


def pickle_writer(obj: Any, filepath: Union[str, Path]) -> None:
    """Save an object to a pickle file.

    Args:
        obj: The object to save.
        filepath (str): The path to the pickle file.

    Returns:
        None

    Raises:
        ValueError: If file_path is None or empty.
        ValueError: If file_path has an invalid file type.
    """
    filepath = Path(filepath)
    if is_none_or_empty(obj):
        logger.error(f"Expected obj to be not None or empty. Actual: {type(obj)}")
        raise ValueError(f"Expected obj to be not None or empty. Actual: {type(obj)}")

    if is_none_or_empty(filepath):
        logger.error(
            f"Expected filepath to be not None or empty. Actual: {type(filepath)}"
        )
        raise ValueError(
            f"Expected filepath to be not None or empty. Actual: {type(filepath)}"
        )

    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        logger.error(f"Error saving pickle file: {filepath}")
        raise e
