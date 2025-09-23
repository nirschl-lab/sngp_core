#!/usr/bin/env python3
"""fuzzy_search.py in src/sngp_core/models."""

from typing import Optional
from typing import Union

from loguru import logger
from rapidfuzz import fuzz
from rapidfuzz import process

from argusdp import VERBOSE
from argusdp.models import ModelType


def find_match(
    name: str,
    choices: Union[dict, list],
    score_cutoff: Union[int, float] = 99,
    verbose: Optional[bool] = None,
) -> str:
    """Find closest match to the specified name in the list of choices.

    Args:
        name (str): The name to search for.
        choices (Union[dict, list]): The list of choices to search in.
        score_cutoff (Union[int, float], optional): The score cutoff for the match. Defaults to 95.

    Returns:
        ModelType: The model with the specified name.

    Raises:
        ValueError: If the model with the specified name is not found.
    """
    verbose = verbose or VERBOSE
    if name in choices:
        return name

    logger.warning(f"{name} not found. ") if verbose else None
    logger.debug(f"Searching for closest match in {choices}") if verbose else None
    result = process.extractOne(
        name, choices=choices, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff
    )
    if result is None:
        logger.warning(f"{name} not found. Expected one of: {choices}")
        return None

    logger.info(f"Match to {result[0]} ({result[1]})") if verbose else None
    return result[0]
