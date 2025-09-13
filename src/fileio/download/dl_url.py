#!/usr/bin/env python3
"""dl_url.py in src/sngp_core/fileio/download."""

from typing import Optional

import requests
from loguru import logger
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

from src.fileio.download import MAX_ATTEMPTS
from src.fileio.download import MAX_WAIT
from src.fileio.download import MIN_WAIT
from src.fileio.download.dl_utils import get_useragent


@retry(
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_random_exponential(multiplier=1, min=MIN_WAIT, max=MAX_WAIT),
)
def download_url(url: str, headers: Optional[str] = None) -> tuple[int, bytes]:
    """Downloads a file from the given URL using requests.

    Args:
        url: The URL of the file to download.

    Returns:
        The content of the downloaded file as bytes.
    Raises:
        Any exceptions encountered during the download process.
    """
    fake = Faker()
    if headers is None:
        headers = {}

    headers["User-Agent"] = fake.user_agent()

    # create session
    session = requests.Session()

    try:
        response = session.get(url, headers=headers)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.warning(f"Retrying: {url}")
        raise e

    response.raise_for_status()
    return response.status_code, response.content
