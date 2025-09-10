#!/usr/bin/env python3
"""conversions.py in src/argusdp/processing/image."""

import cv2
import numpy as np


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Grayscale image.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Unsupported format.")

    if img.ndim == 3 and img.shape[2] == 3:  # Check if image is BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2 or (
        img.ndim == 3 and img.shape[2] == 1
    ):  # Image is already grayscale
        return img
    else:
        raise ValueError("Unsupported format.")


def invert_image(img: np.ndarray) -> np.ndarray:
    """Invert an image.

    Args:
        img (np.ndarray): Input image, can be binary, grayscale, or true color.

    Returns:
        np.ndarray: Complemented image, same type and size as input.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be an np.ndarray.")

    if not np.issubdtype(img.dtype, np.number) and not np.issubdtype(img.dtype, bool):
        raise TypeError("Input must be a numeric type.")

    if np.iscomplexobj(img):
        raise ValueError("Complex input not supported.")

    if isinstance(img, bool) or np.issubdtype(img.dtype, bool):
        return ~img
    elif np.issubdtype(img.dtype, np.integer):
        # For integer types, subtract from maximum value
        return np.iinfo(img.dtype).max - img
    elif np.issubdtype(img.dtype, np.floating):
        # For floating-point types, subtract from 1
        return 1.0 - img
    else:
        raise TypeError(
            f"Expected input to be of type bool, int, or float, got {img.dtype}."
        )
