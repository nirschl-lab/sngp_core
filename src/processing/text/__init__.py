#!/usr/bin/env python3
"""__init__.py in src/biovlmdata/processing/text."""


import re
from typing import Union

from loguru import logger


# from argusdp.data import VALID_SPLIT
VALID_SPLIT = {
    "test": ["hold-out", "hold_out", "test"],
    "train": ["train", "training"],
    "validation": ["validation", "dev"],
}


def str2num(x: str) -> Union[int, float, str]:
    """Convert string to number, or return string unchanged."""
    try:
        if x.isdigit() or re.match(r"^-?(\d+)$", x):
            return int(x)
        elif re.match(r"^-?\d+(?:\.\d+)$", x):
            return float(x)
        else:
            return x
    except Exception as e:
        logger.warning(f"Error converting {x} to number: {e}")
        return x


def update_split(split: str) -> str:
    """Update split name."""
    split = split.replace("_", "-")
    if split.lower() in VALID_SPLIT["train"]:
        return "train"
    elif split.lower() in VALID_SPLIT["test"]:
        return "test"
    elif split.lower() in VALID_SPLIT["validation"]:
        return "validation"
    elif split.lower() == "all":
        return "all"
    else:
        # unpack list of valid splits
        valid_splits = [item for sublist in VALID_SPLIT.values() for item in sublist]
        raise ValueError(f"Invalid split: {split}. Expected one of {valid_splits}.")


def convert_liststr_to_str(x: Union[str, list]) -> str:
    """Convert list as string to string."""
    if not isinstance(x, list):
        return x.strip()

    # remove None, null, and empty strings
    x = [str(i).strip() for i in x if i not in [None, "None", "null", "na", "n/a", ""]]
    return ", ".join(x)
