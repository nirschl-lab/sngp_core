#!/usr/bin/env python3
"""compute_mean.py in src/bcv/img."""
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from argusdp import LOG_DIR as log_dir
from argusdp.processing.image.imutils import check_dataset_type
from argusdp.processing.image.imutils import resize_image
from argusdp.processing.image.imutils import to_float


cv2.setUseOptimized(True)


def process_mean_image(
    img: np.ndarray,
    output_shape: Tuple[int, int, int],
    dtype: np.dtype,
    is_bgr: bool = True,
) -> np.ndarray:
    """Process the input image for mean computation to ensure common height and width.

    Args:
        img: The input image as a numpy array.
        output_shape: The desired shape of the output image.
        dtype: The data type of the output image.

    Returns:
        The processed image as a numpy array.

    Examples:
        >>> img = cv2.imread("path/to/image.png")
        >>> output_shape = (100, 100, 3)
        >>> dtype = np.float32
        >>> processed_img = process_mean_image(img, output_shape, dtype)
    """
    if not output_shape or not isinstance(output_shape, tuple):
        raise ValueError("output_shape must be a tuple")
    elif len(output_shape) not in (2, 3):
        raise ValueError("output_shape must have 2 or 3 elements")

    # expect output_shape to be (H, W, C)
    if len(output_shape) == 2:
        output_shape = (output_shape[0], output_shape[1], 3)
    elif len(output_shape) == 3 and output_shape[2] not in (1, 3):
        raise ValueError("output_shape must have 3 elements and C must be 1 or 3")

    # check channel order expect HWC not CHW
    if len(img.shape) == 3 and img.shape[2] not in (1, 3):
        logger.error(f"Expecting HWC not CHW, image shape: {img.shape}")
        raise ValueError("Expecting image dimensions arranged as HWC not CHW")

    # assume using opencv
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # convert to float
    img = to_float(img, dtype=dtype)

    # resize image to output_shape
    img, _scale_factor = resize_image(img, output_shape)

    return img


def compute_mean(  # sourcery skip
    data: Dataset,
    output_shape: Optional[Tuple[int, int]] = None,
    save_dir: Union[Path, str] = None,
    dtype: np.dtype = np.float64,
) -> tuple:
    """Compute the mean image of a dataset.

    Args:
        data: The dataset to compute the mean image for.
        output_shape: Optional. The desired shape of the mean image.
        save_dir: Optional. The directory to save the mean image.
        dtype: Optional. The data type of the mean image.

    Returns:
        The mean image as a numpy array.

    Raises:
        TypeError: If the data is not a torch.utils.data.Dataset object.

    Examples:
        >>> dataset = torchvision.datasets.ImageFolder(root="path/to/data")
        >>> compute_mean(dataset)
    """
    logger.add(
        log_dir.joinpath(f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )
    check_dataset_type(data)
    is_bgr = False

    if isinstance(data.samples[0][0], (str, Path)):
        data_format = "path"
        is_bgr = True
        original_shape = cv2.imread(str(data.samples[0][0])).shape
        output_shape = output_shape or original_shape
    elif isinstance(data.samples[0][0], np.ndarray):
        data_format = "numpy"
        original_shape = data.samples[0][0].shape
        output_shape = output_shape or original_shape
    else:
        raise TypeError("Invalid data type for image.")

    mean_im = np.zeros(output_shape, dtype=dtype)

    logger.info(f"Computing image mean for {data}")
    logger.info(f"Number of images in dataset: {len(data)}")
    logger.info(f"\tShape: {mean_im.shape} and dtype: {dtype}")
    logger.info(f"\tAll images will be resized to {output_shape}")

    # get the numbers for 0.25, 0.5, 0.75 of the dataset
    log_idx = len(data) // 4

    num_errors = 0
    for idx in tqdm(range(len(data))):
        if data_format == "path":
            img = cv2.imread(str(data.samples[idx][0]))
        else:
            img = data.samples[idx][0]

        if img is None:
            logger.warning(f"Error loading image {str(data.samples[idx][0])}.")
            num_errors += 1
            continue

        # convert to rgb, convert to float, and resize to output_shape
        img = process_mean_image(img, mean_im.shape, dtype, is_bgr=is_bgr)
        mean_im = cv2.accumulate(img, mean_im)

        # log every 25%, 50%, 75% of dataset
        if idx in (log_idx, 2 * log_idx, 3 * log_idx, len(data) - 1):
            # TODO: clean/refactor in to separate fn
            mean_image_by_channel = np.mean((mean_im / (idx + 1)), axis=(0, 1))
            logger.info(f"Processed {idx} images.")
            logger.info(f"Mean image by channel: {mean_image_by_channel}")
            temp_save_dir = Path(save_dir).joinpath(".temp")
            temp_save_dir.mkdir(parents=True, exist_ok=True)
            temp_im = mean_im.copy()
            temp_im = cv2.divide(temp_im, idx + 1 - num_errors)
            temp_im = cv2.convertScaleAbs(temp_im, alpha=255, beta=0)
            temp_im = cv2.cvtColor(temp_im, cv2.COLOR_RGB2BGR)
            temp_save_path = temp_save_dir.joinpath(f"mean_{idx}.png")
            cv2.imwrite(str(temp_save_path), temp_im)

    mean_im = cv2.divide(mean_im, idx + 1 - num_errors)

    # get channel-wise mean if mean_im is 3D
    if len(mean_im.shape) == 3:
        mean_image_by_channel = np.mean(mean_im, axis=(0, 1))
        logger.info(f"Mean image by channel: {mean_image_by_channel}")
        print(f"Mean image by channel: {tuple(mean_image_by_channel)}")

    # TODO update for arbitrary dtype and max int value
    mean_im = cv2.convertScaleAbs(mean_im, alpha=255, beta=0)

    if save_dir is not None and isinstance(save_dir, (str, Path)):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # cv2 uses bgr, so convert to bgr before saving
        mean_im = cv2.cvtColor(mean_im, cv2.COLOR_RGB2BGR)
        filename = data.name or data.__class__.__name__  # _base_folder
        save_filepath = save_dir.joinpath(f"{filename}_mean.png")
        cv2.imwrite(str(save_filepath), mean_im)  # TODO save with embedded metadata
        logger.info(f"Saved mean image to {save_filepath}")

    if mean_im.dtype != dtype:
        mean_im = mean_im.astype(dtype)

    return mean_im
