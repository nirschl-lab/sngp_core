#!/usr/bin/env python3
"""__init__.py in src/argusdp/processing/image."""

import cv2
import numpy as np


cv2.setUseOptimized(True)


def to_float(img: np.ndarray) -> np.ndarray:
    """Normalize image based on input image bit depth and convert to float."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype == np.float32:
        return img
    else:
        raise TypeError("Image must be uint8, uint16, or float32")


def get_slice(offset: int, img: np.ndarray, name: str = ""):
    """Get slice for cropping."""
    # error checking
    if not isinstance(offset, (int, np.integer)):
        raise TypeError(f"Invalid offset: {offset} with type ({type(offset)})")

    # get slice_obj to remove img padding from registration warp/transform
    if offset < 0:
        # negative values have padding on the left/top, which needs to be removed
        # slice moving_img starting from abs(x offset) and ending at patch_size
        slice_obj = slice(np.abs(offset), img.shape[1])
    elif offset > 0:
        # positive values have padding on the right/bottom, which needs to be removed
        # slice moving_img starting from 0 and ending at patch_size - x offset
        slice_obj = slice(0, img.shape[1] - offset)
    elif offset == 0:
        # slice moving_patch starting from 0 and ending at patch_size
        slice_obj = slice(0, img.shape[1])
    else:
        raise ValueError(f"Invalid {name} offset: {offset}")

    return slice_obj
