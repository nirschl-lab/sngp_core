#!/usr/bin/env python3
"""apply_icc_profile.py in src/argusdp/processing/image."""

from pathlib import Path
from typing import Optional

import click
import numpy as np
from loguru import logger
from PIL import Image
from PIL import ImageCms

from src.fileio.image.read_icc import build_icc_profile
from src.fileio.image.read_icc import read_build_icc
from src.fileio.image.writers import create_png_metadata


try:
    from argusdp.fileio.image.wsi import get_aperio_metadata
except ImportError:
    logger.error(
        "Error importing get_aperio_metadata from wsi.py. Some functions may not work."
    )


def build_apply_icc(icc_bytes: bytes, img: np.ndarray) -> np.ndarray:
    """Builds an ICC profile from the given ICC profile bytes.

    Args:
        icc_bytes: The ICC profile bytes.

    Returns:
        Tuple[ImageCmsProfile, ImageCmsTransform]: ICC profile and RGB transform.
    """
    _icc_profile, icc2rgb = build_icc_profile(icc_bytes)
    return apply_icc_array(img, icc2rgb=icc2rgb)


def apply_icc_array(
    image: np.ndarray,
    icc_profile: Optional[ImageCms.ImageCmsProfile] = None,
    icc2rgb: Optional[ImageCms.ImageCmsTransform] = None,
) -> np.ndarray:
    """Applies the ICC profile to an image array.

    Args:
        image: np.ndarray: Image to apply ICC profile to.
        icc_profile: ImageCms.ImageCmsProfile: ICC profile to apply.
        icc2rgb: ImageCms.ImageCmsTransform: ICC profile transform.

    Returns:
        np.ndarray: Image with ICC profile applied.

    References:
        Application of ICC profiles to digital pathology images
        http://www.andrewjanowczyk.com/application-of-icc-profiles-to-digital-pathology-images
    """
    # read ICC profile from svs file
    if isinstance(icc2rgb, ImageCms.ImageCmsTransform):
        logger.debug("Using provided ICC ImageCmsTransform")
    elif isinstance(icc_profile, ImageCms.ImageCmsProfile):
        logger.debug("Using provided ICC ImageCmsProfile")
        rgb_profile = ImageCms.createProfile("sRGB")  # create sRGB profile
        icc2rgb = ImageCms.buildTransformFromOpenProfiles(
            icc_profile, rgb_profile, "RGB", "RGB"
        )
    else:
        logger.error("Must provide either icc_profile or icc2rgb.")
        raise ValueError("Must provide either icc_profile or icc2rgb.")

    try:
        # apply ICC profile to image
        logger.debug("Applying ICC profile")
        image = Image.fromarray(image, mode="RGB")
        output = ImageCms.applyTransform(image, icc2rgb)
        output = np.array(output)
    except Exception as e:
        logger.error(f"Error applying ICC profile: {e}")
        raise e

    return output


def apply_icc_file(
    input_file: Path,
    wsi_file: Path,
    output_dir: Path,
    scale_factor: Optional[float] = None,
    app_mag: Optional[int] = None,
    apply: bool = True,
    force: bool = False,
) -> None:
    """Apply an ICC profile to an image file.

    Args:
        input_file: Path to the input image file.
        wsi_file: Path to the whole slide image file.
        output_dir: Path to the output directory.
        scale_factor: Scale factor to adjust the image resolution. Defaults to None.
        app_mag: Apparent magnification level. Defaults to None.
        apply: Whether to apply the ICC profile to the image. Defaults to True.
        force: Whether to overwrite the output file if it already exists. Defaults to False.

    Returns:
        None.

    Raises:
        RuntimeError: Error reading ICC profile from the input file.
    """
    output_filepath = output_dir.joinpath(input_file.name)
    # read image file
    image = Image.open(input_file)

    # check if image already has embedded icc profile
    if image.info.get("icc_profile", None):
        logger.warning("Image already has an embedded ICC profile.")
        return

    # read icc profile from file
    icc_profile, icc2rgb = read_build_icc(wsi_file, save=True)

    # get aperio metadata from svs file
    aperio_metadata = get_aperio_metadata(wsi_file)

    # update metadata with app_mag or scale_factor
    if isinstance(app_mag, (int, float)):
        # get ratio of app_mag / aperio_metadata[AppMag]
        scale_factor = app_mag / aperio_metadata["AppMag"]

    if isinstance(scale_factor, float) and scale_factor != 1.0:
        logger.debug(
            f"Updating MPP and AppMag with the provided scale factor: {scale_factor}"
        )
        aperio_metadata["MPP"] = aperio_metadata["MPP"] * scale_factor
        aperio_metadata["AppMag"] = aperio_metadata["AppMag"] / scale_factor

    # create png metadata
    png_metadata = create_png_metadata(output_filepath, data=aperio_metadata)

    # save image to disk
    if output_filepath.exists() and not force:
        overwrite_message = f"File {output_filepath.name} already exists. Overwrite?"
        response = click.confirm(overwrite_message)
        if not response:
            # save with new name "_icc" appended
            logger.info("Existing file will not be overwritten.")
            output_filepath = output_dir.joinpath(
                f"{input_file.stem}_icc{input_file.suffix}"
            )
            logger.info(f"Saving with new filename {output_filepath.name}.")

    if icc_profile is None:
        # update png_metadata with icc profile not found
        png_metadata.add_text("ICC applied", "Not found")
        image.save(output_filepath, pnginfo=png_metadata)

    if apply:
        try:
            image = apply_icc_array(
                np.array(image), icc_profile=icc_profile, icc2rgb=icc2rgb
            )

            # convert back to PIL image
            image = Image.fromarray(image)
        except Exception as e:
            logger.warning(f"Error applying icc profile: {e}")
            raise e

        # save with png metadata and ICC applied to pixel data
        png_metadata.add_text("ICC applied", "Applied")
        image.save(output_filepath, pnginfo=png_metadata)

    else:
        # save with png metadata, original pixels, and embedded icc
        png_metadata.add_text("ICC applied", "Embedded")
        image.save(
            output_filepath,
            pnginfo=png_metadata,
            icc_profile=icc_profile.tobytes(),
        )
