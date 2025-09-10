#!/usr/bin/env python3
"""color_deconvolution.py in src/argusdp/processing/image."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import PIL
import skimage.color
from loguru import logger
from scipy import linalg
from skimage.color import combine_stains
from skimage.color import hdx_from_rgb
from skimage.color import hed2rgb
from skimage.color import hed_from_rgb
from skimage.color import rgb2hed
from skimage.color import rgb2lab
from skimage.color import rgb_from_hdx
from skimage.color import rgb_from_hed
from skimage.color import separate_stains
from skimage.color.colorconv import _prepare_colorarray

from argusdp.fileio.text import is_empty_file
from argusdp.fileio.text.readers import yaml_loader
from argusdp.processing.image.imutils import image_show


cv2.setUseOptimized(True)

# # TODO: need to do simulated cross product to get correct stain matrix
# STAIN_MATRIX = {
#     "he1": np.array(
#         [[0.644211, 0.716556, 0.266844], [0.092789, 0.954111, 0.283111], [0, 0, 0]]
#     ),
#     "he2": np.array(
#         [
#             [0.49015734, 0.76897085, 0.41040173],
#             [0.04615336, 0.8420684, 0.5373925],
#             [0, 0, 0],
#         ]
#     ),
#     "hdx": np.array(
#         # [
#         #     [0.644211, 0.092789, 0],
#         #     [0.716556, 0.954111, 0],
#         #     [0.266844, 0.283111, 0],
#         # ]
#         [  # transposed
#             [0.644211, 0.716556, 0.266844],
#             [0.092789, 0.954111, 0.283111],
#             [0.0, 0.0, 0.0],
#         ]
#     ),
#     "hdx2": np.array(
#         # run("Colour Deconvolution2", "vectors=[User values] output=[8bit_Transmittance] simulated cross
#         # [r1]=0.2837116478549852 [g1]=0.5702843140515896 [b1]=0.7708978544645956
#         # [r2]=0.5937422310487157 [g2]=0.6209907072536136 [b2]=0.511703727340294 [r3]=0.5915635335265995 [g3]=0.59052733113409 [b3]=0.548935385072936");
#         [
#             [0.2837116478549852, 0.5937422310487157, 0.5915635335265995],
#             [0.5702843140515896, 0.6209907072536136, 0.59052733113409],
#             [0.7708978544645956, 0.511703727340294, 0.548935385072936],
#         ],
#     ),
#     "hdx3": np.array(
#         # run("Colour Deconvolution2", "vectors=[User values] output=[8bit_Transmittance] simulated cross [r1]=0.6938642393210356 [g1]=0.6545893289327942 [b1]=0.30010869337417534 [r2]=0.40766744445208086 [g2]=0.6021645502341337 [b2]=0.6864438135603184 [r3]=0.5806871426563507 [g3]=0.6054110965383234 [b3]=0.544315943677813");
#         [
#             [0.6938642393210356, 0.40766744445208086, 0.5806871426563507],
#             [0.6545893289327942, 0.6021645502341337, 0.6054110965383234],
#             [0.30010869337417534, 0.6864438135603184, 0.544315943677813],
#         ],
#     ),
# }
available_channels = ["h", "e", "d"]


def check_invertible(matrix: np.ndarray) -> bool:
    """Check if matrix is invertible."""
    return np.linalg.det(matrix) != 0


def invert_stain(stain_matrix: np.ndarray, eps=0.0001) -> np.ndarray:
    # compute 3rd color component using cross-product of first two vectors
    # Using the cross product for determining the 3rd colour can result in matrices
    # with negative coeficients (i.e. 'impossible' colours), and therefore the LUT
    # of the 3rd component determined this way will not represent a color in the image.
    stain_matrix[2, :] = np.cross(stain_matrix[0, :], stain_matrix[1, :])

    # TODO:  Check whether the matrix can be inverted (non-zero determinant).
    # If not, the program will attempt to make it invertible by adding a small value
    # (eps = 0.0001) to the rows, columns or diagonals that add to 0.
    if not check_invertible(stain_matrix):
        # replace only zero values with small value
        logger.warning("Matrix is not invertible. Attempting to make it invertible.")
        for i in range(3):
            if np.sum(stain_matrix[i, :]) == 0:
                stain_matrix[i, :] += eps
            if np.sum(stain_matrix[:, i]) == 0:
                stain_matrix[:, i] += eps

    # invert the stain matrix
    return linalg.inv(stain_matrix)


def load_stain_matrix(
    stain: str = "he1", config: Optional[str] = None, invert: bool = True
) -> np.ndarray:
    """Load color deconvolution matrix from configuration."""
    project_dir = Path(__file__).resolve().parents[4]
    config_dir = project_dir.joinpath("src", "argusdp", "conf")
    config = config or config_dir.joinpath("stain_matrix.yaml")
    if is_empty_file(config):
        logger.error(f"Config file {config} is empty.")
        raise FileNotFoundError(f"Config file {config} is empty.")

    config = yaml_loader(config)
    stain_matrix = config.get(stain, None)
    if stain_matrix is None:
        logger.error(f"Stain {stain} not found in config file {config}.")
        logger.info(f"Available stains: {config.keys()}")
        raise ValueError(f"Stain {stain} not found in config file {config}.")

    # log matrix determinant
    logger.debug(f"Matrix determinant for {stain}: {np.linalg.det(stain_matrix)}")

    # convert to array and invert the stain matrix
    stain_matrix = np.array(stain_matrix)
    if invert:
        logger.debug(f"Inverting stain matrix for {stain}.\n{stain_matrix}")
        stain_matrix = invert_stain(stain_matrix)
        logger.debug(f"Inverted stain matrix for {stain}.\n{stain_matrix}")

    return stain_matrix


def get_stain(
    img: np.ndarray,
    stain: str = "he1",
    channel: str = "h",
    normalize: bool = True,
    invert: bool = True,
    grayscale: bool = True,
) -> np.ndarray:
    """Convert histology stained image to single stain grayscale image."""

    # separate stain using appropriate color deconvolution matrix
    stain_matrix = load_stain_matrix(stain, invert=invert)

    if channel not in available_channels:
        logger.error(f"Channel {channel} not found in available channels.")
        logger.info(f"Available channels: {available_channels}")
        raise ValueError(f"Channel {channel} not found in available channels.")

    # separate stains
    img_hed = separate_stains(img, stain_matrix)
    null = np.zeros_like(img_hed[:, :, 0])

    # get stain channel
    img_stain = {}
    null_list = [null, null, null]
    for idx, key in enumerate(["h", "e", "d"]):
        temp_null_list = null_list.copy()
        temp_null_list[idx] = img_hed[:, :, idx]
        img_stain[key] = np.stack(temp_null_list, axis=-1)

    # recreate RGB
    # note: I slightly prefer "he1" for aligning two images on hematoxylin channel
    # need to compare rgb2hed(img) vs. custom stain_matrix "he1" for aligning on eosin channel
    if "he" in stain:
        img_rgb = skimage.color.hed2rgb(img_stain[channel])
    elif "hd" in stain or "custom" in stain:  # TODO: fix this
        img_rgb = skimage.color.hed2rgb(img_stain[channel])
        # img_rgb = combine_stains(img_stain[channel], load_stain_matrix(stain, invert=True))

        # stains = _prepare_colorarray(img_stain[channel], channel_axis=-1)
        #
        # # log_adjust here is used to compensate the sum within separate_stains().
        # log_adjust = -np.log(1e-3)
        # log_rgb = -(stains * log_adjust) @ stain_matrix
        # img_rgb = np.exp(log_rgb)
        # image_show(img_rgb)
    else:
        img_rgb = skimage.color.hed2rgb(img_stain[channel])

    # debug = True
    # if debug:
    #     image_show(img_rgb)

    # get LAB L channel (grayscale)
    img_l = rgb2lab(img_rgb)[:, :, 0]

    # return normalized stain L channel
    if grayscale and normalize:
        logger.debug(f"Grayscale: Normalizing {channel} channel.")
        return cv2.normalize(
            img_l,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
    elif grayscale:
        logger.debug(f"Grayscale: Not normalizing {channel} channel.")
        return img_l
    else:
        logger.debug(f"RGB: Not normalizing {channel} channel.")
        return img_rgb

    #
    # def show_rgb(img, matrix, ch="h"):
    #     img_hed = separate_stains(img, matrix)
    #     img_stain = {}
    #     null = np.zeros_like(img_hed[:, :, 0])
    #     null_list = [null, null, null]
    #     for idx, key in enumerate(["h", "e", "d"]):
    #         temp_null_list = null_list.copy()
    #         temp_null_list[idx] = img_hed[:, :, idx]
    #         img_stain[key] = np.stack(temp_null_list, axis=-1)
    #
    #     # recreate RGB
    #     img_rgb = skimage.color.hed2rgb(img_stain[ch])
    #     image_show(img_rgb)
