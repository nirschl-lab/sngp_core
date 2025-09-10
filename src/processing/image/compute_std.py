#!/usr/bin/env python3
"""compute_std.py in src/bcv/img."""
# noqa: C901
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
from argusdp.processing.image.imutils import load_mean_image
from argusdp.processing.image.imutils import to_float


cv2.setUseOptimized(True)


def compute_std(  # noqa: C901 # sourcery skip
    data: Dataset,
    mean_img: Union[np.ndarray, Tuple[int, int, int], str, Path],
    save_dir: Union[Path, str] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Compute std of images in data_dir and save to save_dir."""
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
        original_shape = cv2.imread(str(data.samples[0][0])).shape
        output_shape = output_shape or original_shape
    else:
        raise TypeError("Invalid data type for image.")

    mean_img = load_mean_image(mean_img, dtype)
    # preserve mean img num channels
    if output_shape == mean_img.shape:
        output_shape = mean_img.shape
    elif len(mean_img.shape) == 3 and mean_img.shape[2] == 3:
        output_shape = output_shape[:2] + (mean_img.shape[2],)
    else:
        output_shape = output_shape or mean_img.shape

    sum_sq_diff = np.zeros(output_shape, dtype=dtype)

    logger.info(f"Computing image standard deviation for {data}")
    logger.info(f"Number of images in dataset: {len(data)}")
    logger.info(f"\tShape: {original_shape} and dtype: {dtype}")
    logger.info(f"\tAll images will be resized to {output_shape}")
    logger.debug(f"\tMean image shape: {mean_img.shape} and dtype: {dtype}")
    logger.debug(f"\tSum Sq Diff shape: {sum_sq_diff.shape} and dtype: {dtype}")

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

        # TODO: refactor into new fn to convert to rgb, to_float, and resize
        if is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = to_float(img)

        if img.shape[:2] != output_shape[:2]:
            w, h = output_shape[:2]
            # use INTER_AREA for shrinking and INTER_CUBIC for enlarging
            # NOTE: assumes all images have been previously resized
            # preserving aspect ratio(e.g., if img.shape[0] < output_shape[0]
            # then img.shape[1] < output_shape[1] is also true)
            interp = (
                cv2.INTER_AREA if img.shape[0] < output_shape[0] else cv2.INTER_CUBIC
            )
            img = cv2.resize(img, (h, w), interpolation=interp)

        diff_img = np.subtract(img, mean_img)
        sq_diff_img = np.power(diff_img, 2)
        sum_sq_diff = cv2.accumulate(sq_diff_img.astype(dtype), sum_sq_diff)

        # log every 25%, 50%, 75% of dataset
        if idx in (log_idx, 2 * log_idx, 3 * log_idx, len(data) - 1):
            # TODO: clean/refactor in to separate fn
            temp_sum_sq_diff = sum_sq_diff.copy()
            temp_mean_sq_diff = temp_sum_sq_diff / (idx + 1 - num_errors)
            temp_std_im = np.sqrt(temp_mean_sq_diff)
            temp_std_image_by_channel = np.mean(temp_std_im, axis=(0, 1))
            logger.info(f"Processed {idx} images.")
            logger.info(f"Std image by channel: {temp_std_image_by_channel}")

            temp_std_im = cv2.convertScaleAbs(temp_std_im, alpha=255, beta=0)
            temp_std_im = cv2.cvtColor(temp_std_im, cv2.COLOR_RGB2BGR)

            temp_save_dir = Path(save_dir).joinpath(".temp")
            temp_save_dir.mkdir(parents=True, exist_ok=True)
            temp_save_path = temp_save_dir.joinpath(f"std_{idx}.png")
            cv2.imwrite(str(temp_save_path), temp_std_im)

    mean_sq_diff = sum_sq_diff / (idx + 1 - num_errors)
    std_im = np.sqrt(mean_sq_diff)
    std_image_by_channel = np.mean(std_im, axis=(0, 1))
    logger.info(f"Processed {idx} images.")
    logger.info(f"Std image by channel: {std_image_by_channel}")
    print(f"Std image by channel: {tuple(std_image_by_channel)}")

    std_im = cv2.convertScaleAbs(std_im, alpha=255, beta=0)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        std_im = cv2.cvtColor(std_im, cv2.COLOR_RGB2BGR)
        filename = data.name or data.__class__.__name__
        save_filepath = save_dir.joinpath(f"{filename}_std.png")
        cv2.imwrite(
            save_filepath.as_posix(), std_im
        )  # TODO save with embedded metadata
        logger.info(f"Saved std image to {save_filepath}")

    return std_im
