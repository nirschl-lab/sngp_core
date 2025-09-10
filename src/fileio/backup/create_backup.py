#!/usr/bin/env python3
"""create_backup.py in src/argusdp/fileio/backup."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from typing import Union

from loguru import logger

from src.fileio.text import is_none_or_empty


def create_backup(
    filepath: Union[Path, str], size: Optional[int] = None
) -> TemporaryDirectory:
    """Create a backup of a file, if it exists and is not empty."""
    filepath = Path(filepath)
    if is_none_or_empty(filepath) or not filepath.exists():
        return None

    # exit early if file is too large
    if size is not None and filepath.stat().st_size > size:
        logger.warning(f"File is too large to backup: {filepath}")
        return None

    temp_dir = TemporaryDirectory()
    backup_file = Path(temp_dir.name).joinpath(filepath.name)
    logger.debug(f"Backing up {filepath.name} to {backup_file}")
    shutil.copy(filepath, backup_file)
    return temp_dir
