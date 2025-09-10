# #!/usr/bin/env python3
# """process_sa_json.py in src/argusdp/processing/annotations."""
# from typing import Tuple
#
# import cv2
# import numpy as np
#
#
# cv2.setUseOptimized(True)
#
#
# def rle2mask(rle_str: str, output_shape: Tuple[int, int]) -> np.ndarray:
#     """Convert a run-length encoded (RLE) mask string to a binary mask.
#
#     Args:
#         rle_str: The run-length encoded mask string.
#         output_shape: The desired output shape of the binary mask as a tuple (height, width).
#
#     Returns:
#         The binary mask array with the specified output shape.
#
#     Raises:
#         TypeError: If rle_str is not a string or output_shape is not a tuple of length 2.
#     """
#     if not isinstance(rle_str, str):
#         raise TypeError("rle_str must be a string.")
#
#     if not isinstance(output_shape, tuple) or len(output_shape) != 2:
#         raise TypeError("output_shape must be a Tuple[int, int].")
#
#     # get height and width from output_shape
#     height, width = output_shape
#
#     # split string on whitespace
#     s = rle_str.split()
#     starts, lengths = (np.asarray(x, dtype=int) for x in (s[:][::2], s[1:][::2]))
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(height * width, dtype=np.uint8)
#     for lo, hi in zip(starts, ends, strict=True):
#         img[lo:hi] = 1
#     return img.reshape((height, width))
#
#
# def mask2rle(mask: np.ndarray, output_format: str = "str") -> str:
#     """Convert binary mask image to run length encoding."""
#     if mask is None:
#         return ""
#
#     # flatten mask image column-wise
#     pixels = mask.flatten(order="F")  # column major
#
#     # set first and last pixels to 0
#     # avoids issues with '1' at the start or end
#     pixels[0] = 0
#     pixels[-1] = 0
#
#     # find runs
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     runs[1::2] = runs[1::2] - runs[:-1:2]
#
#     # convert runs to string
#     return " ".join(str(x) for x in runs)
#
# def instances2mask(instances: list, height: int, width: int) -> np.ndarray:
#     """Convert instances to mask."""
#     mask = np.zeros((height, width), dtype=np.uint8)
#     for instance in instances:
#         rle_mask = rle2mask(instance["mask"], (height, width))
#         mask += rle_mask
#     return mask
