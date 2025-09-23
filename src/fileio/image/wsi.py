#!/usr/bin/env python3
"""wsi.py in src/sngp_core/fileio/image."""
from pathlib import Path
from typing import Union

from loguru import logger

from src.fileio.text import is_empty_file
from src.fileio.text import valid_file_ext
from src.processing.text.conversions import str2num


# try:
#     import openslide
# except ImportError:
#     openslide = None
#     logger.warning("Error importing openslide. Some functions may not work.")


def get_aperio_metadata(wsi_path: Union[Path, str]) -> dict:
    """Read Aperio metadata from svs file using openslide."""
    wsi_path = Path(wsi_path).resolve()
    valid_extensions = {".svs", ".tif", ".tiff"}
    if not valid_file_ext(wsi_path, valid_extensions):
        raise ValueError(f"Expected one of {valid_extensions}, got {wsi_path.suffix}")

    if is_empty_file(wsi_path):
        raise ValueError(f"File is empty: {wsi_path}")

    # read Aperio metadata
    aperio_metadata = openslide.OpenSlide(str(wsi_path)).properties

    # convert keys from aperio.* to * and remove aperio.*
    aperio_metadata = {
        key.split("aperio.")[-1]: value for key, value in dict(aperio_metadata).items()
    }

    # convert to dict and clean up strings
    aperio_metadata = {key: str2num(value) for key, value in aperio_metadata.items()}
    # # rename Openslide.comment to "Openslide Comment"
    aperio_metadata["Openslide comment"] = aperio_metadata.pop("openslide.comment")
    # # rename Openslide.vendor to "Openslide Vendor"
    aperio_metadata["Openslide vendor"] = aperio_metadata.pop("openslide.vendor")
    aperio_metadata["ICC applied"] = ""  # "", "Embedded", "Applied"

    return aperio_metadata
