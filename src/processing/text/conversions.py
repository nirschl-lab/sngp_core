#!/usr/bin/env python3
"""conversions.py in src/sngp_core/processing/text."""

import re
from typing import Union

from loguru import logger


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
