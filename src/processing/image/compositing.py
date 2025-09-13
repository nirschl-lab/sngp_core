#!/usr/bin/env python3
"""compositing.py in src/sngp_core/processing/image."""
import re
from typing import Optional

import cv2
import numpy as np

from argusdp.processing.image.conversions import convert_to_grayscale
from argusdp.processing.image.conversions import invert_image


cv2.setUseOptimized(True)


def blend_composite(img1_gray: np.ndarray, img2_gray: np.ndarray) -> np.ndarray:
    """Blend two grayscale images using alpha blending.

    Args:
        img1_gray: First grayscale image.
        img2_gray: Second grayscale image.

    Returns:
        Composite image.
    """
    raise NotImplementedError("blend_composite not implemented yet.")


def mask_composite(
    img1_gray: np.ndarray, img2_gray: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create a composite of two grayscale images using a masked pattern.

    Args:
        img1_gray: First grayscale image.
        img2_gray: Second grayscale image.
        mask: Mask to apply to the composite image.

    Returns:
        Composite image.
    """
    if mask is None:
        # default to alternating checkerboard pattern of size 10x10
        # TODO
        mask = np.zeros_like(img1_gray)

    raise NotImplementedError("blend_composite not implemented yet.")


def diff_composite(img1_gray: np.ndarray, img2_gray: np.ndarray) -> np.ndarray:
    """Create a composite of two grayscale images showing the difference between them.

    Args:
        img1_gray: First grayscale image.
        img2_gray: Second grayscale image.

    Returns:
        Composite image.

    Raises:
        NotImplementedError: mask_composite not implemented yet.
    """
    raise NotImplementedError("blend_composite not implemented yet.")


def montage_composite(
    img1_gray: np.ndarray, img2_gray: np.ndarray, orientation: str = "horizontal"
) -> np.ndarray:
    """Join two grayscale images horizontally or vertically to create a montage.

    Args:
        img1_gray: First grayscale image.
        img2_gray: Second grayscale image.
        orientation: Orientation of the montage. Options: "horizontal" or "vertical".

    Returns:
        Composite image.
    """
    raise NotImplementedError("montage_composite not implemented yet.")


def color_channel_composite(
    img1_gray: np.ndarray, img2_gray: np.ndarray, color: str = "magenta_green"
) -> np.ndarray:
    """Create an RGB composite of two grayscale images using the specified colors.

    This function replaces the separate magenta_green_composite, red_green_composite,
    and red_cyan_composite functions with a single, more modular approach.

    Args:
        img1_gray: First grayscale image.
        img2_gray: Second grayscale image.
        color: Color for compositing. Options: "magenta_green", "red_green", "red_cyan".

    Returns:
        Composite image as an RGB numpy array.
    """
    channel_map = {
        "magenta_green": ((1, 0, 1), (0, 1, 0)),
        "red_green": ((1, 0, 0), (0, 1, 0)),
        "red_cyan": ((1, 0, 0), (0, 1, 1)),
    }

    if color not in channel_map:
        raise ValueError(f"Unsupported compositing color: {color}")

    img1_channels = channel_map[color][0]
    img2_channels = channel_map[color][1]

    img1_rgb = np.stack(
        [
            img1_gray if channel else np.zeros_like(img1_gray)
            for channel in img1_channels
        ],
        axis=-1,
    )

    img2_rgb = np.stack(
        [
            img2_gray if channel else np.zeros_like(img2_gray)
            for channel in img2_channels
        ],
        axis=-1,
    )

    return cv2.add(img1_rgb, img2_rgb)


def create_composite(
    img1: np.ndarray,
    img2: np.ndarray,
    color: Optional[str] = "magenta_green",
    method: Optional[str] = "color_channel",
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Create a composite image from two input images using the specified method.

    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.
        color (str, optional): Output color for the composite image (default: "magenta_green").
        method (str, optional): Method for compositing images (default: "color_channel").
        mask (np.ndarray, optional): Mask to apply to the composite image for masked method.
        **kwargs: Additional keyword arguments for optional preprocessing.

        # method
        "color_channel"	Creates a composite RGB image showing A and B overlaid in
                        different color bands. Gray regions in the composite image
                        show where the two images have the same intensities.
        "blend"	Overlays A and B using alpha blending.
        "mask"	Creates an image using masked regions from A and B.
        "diff"	Creates a difference image from A and B.
        "montage"	Puts A and B next to each other in the same image.

    Returns:
        Composite image.
    """
    # Check that input images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")

    # Check that the specified method is supported
    supported_methods = {"color_channel", "blend", "mask", "diff", "montage"}
    if method not in supported_methods:
        raise ValueError(
            f"Unsupported compositing method: {method}. Expected one of {supported_methods}."
        )

    # update color string and regex replace space or dash with underscore
    color = re.sub(r"[-\s]", "_", color.lower())

    # Convert images to grayscale if they are not already
    img1_gray = convert_to_grayscale(img1)
    img2_gray = convert_to_grayscale(img2)

    # Apply optional preprocessing
    if kwargs.get("invert", False):
        img1_gray = invert_image(img1_gray)
        img2_gray = invert_image(img2_gray)

    # apply histogram equalization if specified
    if kwargs.get("hist_match", False):
        img1_gray = cv2.equalizeHist(img1_gray) if img1_gray.mean() < 225 else img1_gray
        img2_gray = cv2.equalizeHist(img2_gray) if img2_gray.mean() < 225 else img2_gray

    # Create composite image using the specified method
    if method == "color_channel":
        return color_channel_composite(img1_gray, img2_gray, color=color)
    elif method == "blend":
        return blend_composite(img1_gray, img2_gray)
    elif method == "mask":
        return mask_composite(img1_gray, img2_gray, mask=mask)
    elif method == "diff":
        return diff_composite(img1_gray, img2_gray)
    elif method == "montage":
        return montage_composite(img1_gray, img2_gray)
