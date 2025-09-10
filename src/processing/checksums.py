#!/usr/bin/env python3
"""checksums.py in src/biovlmdata/processing."""

import hashlib
from pathlib import Path
from typing import Union

from argusdp.fileio.text import is_empty_file
from argusdp.fileio.text import is_none_or_empty


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
        raise ValueError("filepath cannot be None or empty")

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
