#!/usr/bin/env python3
"""readers.py in src/sngp_core/fileio/binary."""


from loguru import logger

from argusdp.fileio.text import is_empty_file


try:
    import dill as pickle
except ModuleNotFoundError:
    logger.warning("dill not found. Using pickle instead.")
    import pickle


def pickle_reader(filepath: str) -> None:
    """Load an object from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        Any: The object loaded from the pickle file.

    Raises:
        ValueError: If file_path is None or empty.
        ValueError: If file_path has an invalid file type.
    """
    if is_empty_file(filepath):
        logger.error(
            f"Expected filepath to be not None or empty. Actual: {type(filepath)}"
        )
        raise ValueError(
            f"Expected filepath to be not None or empty. Actual: {type(filepath)}"
        )

    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file: {filepath}")
        raise e
