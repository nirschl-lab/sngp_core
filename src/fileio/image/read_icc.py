#!/usr/bin/env python3
"""read_icc_profile.py in src/sngp_core/fileio/image."""
# flake8: noqa: B950
import io
from io import BytesIO
from pathlib import Path
from typing import Any
from typing import Tuple
from typing import Union

import PIL
import tiffile
from loguru import logger
from PIL import ImageCms
from PIL.ExifTags import TAGS
from PIL.ImageCms import ImageCmsProfile
from PIL.ImageCms import ImageCmsTransform
from PIL.ImageCms import Intent

from src.fileio.text import is_empty_file
from src.fileio.text import valid_file_ext


def read_icc_bytes(
    file_path: Union[str, Path], icc_tag: int = 34675, save: bool = True
) -> BytesIO:
    """Reads the ICC profile bytes from the specified file.

    Args:
        file_path: The path to the file.
        icc_tag: Exif tag index for the ICC profile tag. Defaults to standard tag 34675.
        save: Save the ICC profile to a separate file. Defaults to True.

    Returns:
        The ICC profile bytes.

    Raises:
        ValueError: If the file is empty or has an unsupported extension.
        RuntimeError: If there is an error reading the ICC profile.
    """
    if icc_tag != 34675:
        logger.warning(f"Using non-standard ICC profile tag {icc_tag}:{TAGS[icc_tag]}")

    file_path = Path(file_path).resolve()
    if is_empty_file(file_path):
        raise ValueError(f"File is empty: {file_path}")

    valid_ext = {".icc", ".svs", ".png"}
    if not valid_file_ext(file_path, valid_ext):
        error_message = f"Unsupported file extension: {file_path.suffix}. Expected one of {valid_ext}"
        logger.error(error_message)
        raise ValueError(error_message)

    #
    icc_bytes = None
    try:
        if file_path.suffix == ".icc" or file_path.with_suffix(".icc").exists():
            file_path = file_path.with_suffix(".icc")
            logger.debug(f"Reading ICC profile from {file_path.name}")
            icc_bytes = file_path.read_bytes()

        if file_path.suffix == ".png" and icc_bytes is not None:
            # read and apply ICC profile to image
            img = PIL.Image.open(file_path)
            icc_bytes = img.info.get("icc_profile", None)
            if icc_bytes is None:
                raise ValueError(f"No ICC profile found in {file_path}")

        if file_path.suffix == ".svs" and icc_bytes is not None:
            with tiffile.TiffFile(file_path) as svs_img:
                tag = svs_img.pages[0].tags[icc_tag]  # ICC profile tag
                icc_bytes = io.BytesIO(tag.value)

        if icc_bytes is None:
            error_message = f"Unable to read ICC profile from {file_path.name}"
            logger.error(error_message)
            raise RuntimeError(error_message)

    except Exception as e:
        error_msg = f"Error reading ICC profile from {file_path}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    logger.debug(f"ICC profile size: {len(icc_bytes)} bytes")
    if save and not file_path.with_suffix(".icc").exists():
        icc_file = file_path.parent.joinpath(f"{file_path.stem}.icc")
        logger.debug(f"Saving ICC profile to {icc_file.name}")
        icc_bytes.write_bytes(icc_bytes.getvalue())

    return icc_bytes


def build_icc_profile(
    icc_bytes: bytes, rendering_intent: int = Intent.PERCEPTUAL
) -> Tuple[ImageCmsProfile, ImageCmsTransform]:
    """Builds an ICC profile from the given ICC profile bytes.

    Args:
        icc_bytes: The ICC profile bytes.
        rendering_intent: Rendering intent for the transform. Defaults to Intent.PERCEPTUAL.

    Returns:
        Tuple[ImageCmsProfile, ImageCmsTransform]: The ICC profile and ICC to RGB transform.
    """
    if not isinstance(icc_bytes, io.BytesIO):
        icc_bytes = io.BytesIO(icc_bytes)

    icc_profile = ImageCms.ImageCmsProfile(icc_bytes)  # create ICC profile
    icc_profile_name = ImageCms.getProfileName(icc_profile).strip()
    logger.debug(f"Creating ICC profile: {icc_profile_name}")
    rgb_profile = ImageCms.createProfile("sRGB")  # create sRGB profile
    icc2rgb = ImageCms.buildTransformFromOpenProfiles(
        icc_profile, rgb_profile, "RGB", "RGB", renderingIntent=rendering_intent
    )
    return icc_profile, icc2rgb


def read_build_icc(
    file_path: Union[Path, str], tags_idx: int = 34675, save: bool = False
) -> tuple[None, None] | tuple[ImageCmsProfile, ImageCmsTransform]:
    """Read an ICC profile from an image file and build an ICC to RGB transform.

    Args:
        file_path: Path to the image file.
        tags_idx: Index of the ICC profile tag in the TIFF file. Defaults to 34675.
        save: Whether to save the ICC profile as a separate file. Defaults to False.

    Returns:
        Tuple containing the ICC profile as an ImageCmsProfile object and the ICC to
        RGB transform as an ImageCmsTransform object.  If the ICC profile cannot be
        read or has an incorrect size, returns (None, None).
    """
    icc_bytes = read_icc_bytes(file_path, tags_idx, save=save)
    return build_icc_profile(icc_bytes)
