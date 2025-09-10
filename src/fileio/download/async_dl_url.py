#!/usr/bin/env python3
"""async_dl_url.py in src/argusdp/fileio/download."""

import asyncio
from hashlib import md5
from pathlib import Path

import aiofile
import aiohttp
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientResponseError
from aiohttp.client_exceptions import ServerDisconnectedError
from loguru import logger
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from src.fileio.download.dl_utils import get_useragent


async def download_file(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if "content-disposition" in response.headers:
                header = response.headers["content-disposition"]
                filename = header.split("filename=")[1]
            else:
                filename = url.split("/")[-1]
            with open(filename, mode="wb") as file:
                while True:
                    chunk = await response.content.read()
                    if not chunk:
                        break
                    file.write(chunk)
                print(f"Downloaded file {filename}")


async def main():
    tasks = [download_file(url) for url in urls]
    await asyncio.gather(*tasks)


asyncio.run(main())

#
# async def download_file(session: ClientSession, url: str, headers: dict) -> bytes:
#     async with session.get(url, headers=headers) as response:
#         response.raise_for_status()
#         return await response.read()
#
#
# async def save_file(file_path: Path, data: bytes, md5_hash: str):
#     """Save the file to disk."""
#     if md5_hash == "d41d8cd98f00b204e9800998ecf8427e":
#         logger.error(f"Empty MD5 hash for {file_path}")
#         return
#
#     async with aiofile.async_open(file_path, "wb") as f:
#         await f.write(data)
#     return md5_hash
#
#
# @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=10, max=60))
# async def resilient_download_file(
#     session: ClientSession, url: str, headers: dict
# ) -> bytes:
#     try:
#         return await download_file(session, url, headers)
#     except (ClientResponseError, ServerDisconnectedError) as e:
#         logger.warning(f"Retrying due to error: {e}")
#         raise e
#
#
# async def async_download_url(
#     session: ClientSession,
#     filename: str,
#     url: str,
#     status: bool,
#     image_md5: str,
#     path: Path,
#     semaphore: asyncio.Semaphore = asyncio.Semaphore(10),
# ) -> tuple[bool, str, str]:
#     """Downloads a file from a given URL and saves it to a specified path.
#
#     Args:
#         session (ClientSession): aiohttp ClientSession for the download.
#         filename (str): Name of the file to download.
#         url (str): URL of the file.
#         status (bool): Status flag.
#         image_md5 (str): MD5 hash of the image.
#         path (Path): Destination path.
#
#     Returns:
#         tuple: Download status, any error message, and md5 hash.
#     """
#     if status and image_md5:
#         logger.debug(f"{filename} already exists.")
#         return True, "", image_md5
#
#     file_path = path.joinpath(filename)
#     if not file_path.parent.exists():
#         file_path.parent.mkdir(parents=True, exist_ok=True)
#
#     if file_path.exists():
#         logger.debug(f"{filename} already exists.")
#         return True, "", image_md5
#
#     async with semaphore:
#         headers = get_useragent()
#         try:
#             data = await resilient_download_file(session, url, headers)
#             md5_hash = md5(data).hexdigest()
#             await save_file(file_path, data, md5_hash)
#             return True, "", md5_hash
#         except Exception as e:
#             logger.error(f"url: {url}, filename: {filename}")
#             logger.error(f"HTTP Error: {e}")
#             return False, "HTTPError", None
