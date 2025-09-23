#!/usr/bin/env python3
"""imutils.py in src/sngp_core/processing/image."""

from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import PIL
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.vision import VisionDataset


cv2.setUseOptimized(True)


def get_file_size(filepath: Union[Path, str], unit: str = "bytes") -> float | None:
    """Stat to get image size on disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    size_bytes = filepath.stat().st_size
    if unit.lower() in {"b", "bytes", "byte"}:
        return size_bytes
    elif unit.lower() in {"kb", "kilobytes", "kilobyte"}:
        return size_bytes / 1024
    elif unit.lower() in {"mb", "megabytes", "megabyte"}:
        return size_bytes / 1024 / 1024
    elif unit.lower() in {"g", "gb", "gigabytes", "gigabyte"}:
        return size_bytes / 1024 / 1024 / 1024


def resize_image(
    img: np.ndarray,
    dst_size: Union[Tuple, List],
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize image."""
    # only resize if image is not already the correct size
    if img is None:
        logger.error("Image is None.")
        raise ValueError("Image is None.")

    if dst_size is None or img.shape[:2] == dst_size:
        return img

    return cv2.resize(img, dst_size, interpolation=interpolation)


def apply_clahe(
    img: np.ndarray, clip_limit: float, grid_size: int = 8, colorspace: str = "lab"
) -> np.ndarray:
    """Apply CLAHE to image."""
    clahe_ojb = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
    )

    if img.shape[2] == 1:
        return clahe_ojb.apply(img)
    elif img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe_ojb.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        raise NotImplementedError(f"Colorspace {colorspace} not implemented.")


def bbox_dist_to_border(
    bbox: Dict[str, int], image_shape: Tuple[int, int], border_size: int = 0
):
    if bbox["x1"] < border_size:
        # dist to left border
        return bbox["x1"]
    elif bbox["y1"] < border_size:
        # dist to top border
        return bbox["y1"]
    elif bbox["x2"] > image_shape[1] - border_size:
        # dist to right border
        return image_shape[1] - bbox["x2"]
    elif bbox["y2"] > image_shape[0] - border_size:
        # dist to bottom border
        return image_shape[0] - bbox["y2"]
    else:
        # bbox is within border, return min dist to any border
        return min(
            bbox["x1"],
            bbox["y1"],
            image_shape[1] - bbox["x2"],
            image_shape[0] - bbox["y2"],
        )


def center_bbox(bbox: Dict[str, int], size: int = 256):
    """Create a centered bbox of size `size` around the bbox center."""
    if not isinstance(size, int):
        raise ValueError("Size must be an integer.")
    elif not isinstance(bbox, dict):
        raise ValueError("Bbox must be a dictionary.")

    required_keys = {"x1", "y1", "x2", "y2"}
    if not required_keys.issubset(bbox.keys()):
        raise ValueError(f"Missing keys in bbox: {required_keys - bbox.keys()}")

    ctr_x = int((bbox["x1"] + bbox["x2"]) / 2)
    ctr_y = int((bbox["y1"] + bbox["y2"]) / 2)
    bbox["x1"] = ctr_x - size // 2
    bbox["x2"] = ctr_x + size // 2
    bbox["y1"] = ctr_y - size // 2
    bbox["y2"] = ctr_y + size // 2
    return bbox


def crop_to_bbox(
    image: np.ndarray,
    bbox: Dict[str, int],
    size: Optional[int] = 256,
    border_size: Optional[int] = None,
    center: Optional[bool] = False,
):
    """Crop an image to the specified bounding box."""
    border_size = border_size or size // 2
    if center:
        bbox_ctr = center_bbox(bbox, size)
        # check if coords are too close to the border
        safe_dist = bbox_dist_to_border(bbox_ctr, image.shape) > border_size
        bbox = bbox_ctr if safe_dist else bbox

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    return image[y1:y2, x1:x2]


def resize_image(image: np.ndarray, output_shape: tuple) -> Tuple[np.ndarray, float]:
    """Resize the image to the specified output size.

    Args:
        image: The input image as a NumPy array.
        output_shape: The desired output size as a tuple (width, height).

    Returns:
        tuple: A tuple containing the resized image and the adjusted microns per pixel (float).
    """
    if not isinstance(output_shape, tuple):
        raise ValueError("Output size must be a tuple.")

    # resize image and adjust microns per pixel
    if image.shape[:2] != output_shape[:2]:
        w, h = output_shape[:2]
        # use INTER_AREA for shrinking and INTER_CUBIC for enlarging
        # NOTE: assumes all images have been previously resized
        # preserving aspect ratio(e.g., if img.shape[0] < output_shape[0]
        # then img.shape[1] < output_shape[1] is also true)
        interp = cv2.INTER_AREA if image.shape[0] < output_shape[0] else cv2.INTER_CUBIC
        image = cv2.resize(image, (h, w), interpolation=interp)

    scale_factor = image.shape[0] / output_shape[0]

    return image, scale_factor


def image_show(img: np.ndarray, scale_intensity: bool = True) -> None:
    """Display an image using the default image viewer.

    Args:
        img: Input image as a NumPy array.

    Returns:
        None.

    Raises:
        ValueError: Invalid image.
    """
    if not validate_image(img):
        raise ValueError("Invalid image")

    # rescale to min-max intensity using opencv
    if scale_intensity:
        img = cv2.convertScaleAbs(img, alpha=255.0 / img.max(), beta=0)

    PIL.Image.fromarray(img).show()


def to_float(img: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Converts the image array to float type and scale pixel values to [0, 1].

    Args:
        img: The input image array.

    Returns:
        The image array converted to float type.

    Raises:
        TypeError: If the image array is not of type uint8, uint16, or float32.

    Examples:
        >>> image = np.array([[0, 255], [65535, 32768]], dtype=np.uint16)
        >>> to_float(image)
        array([[0.00000000e+00, 3.87430168e-05],
               [1.00000000e+00, 5.00000000e-01]], dtype=float32)
    """
    # validate image
    validate_image(img)

    # convert to float
    if img.dtype == np.uint8 and check_in_range(img, (0, 255)):
        return img.astype(dtype) / 255.0
    elif img.dtype == np.uint16 and check_in_range(img, (0, 65535)):
        return img.astype(dtype) / 65535.0
    elif img.dtype == np.float32 and check_in_range(img, (0, 1)):
        return img.astype(dtype)
    elif img.dtype == np.float64 and check_in_range(img, (0, 1)):
        return img.astype(dtype)
    else:
        raise TypeError(
            f"Invalid out dtype: {img.dtype} for dynamic range [{img.min()}, {img.max()}]"
        )


def validate_image(
    img: np.ndarray, range_minmax: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> bool:
    """Validate the image array and return True if valid."""
    if not is_numeric(img):
        raise TypeError("Image values must be numeric")

    if not is_array_2d(img) and not is_array_3d(img):
        raise ValueError("Image must be 2D or 3D")

    if not is_fininte(img):
        raise ValueError("Image values must be finite (no NaN or Inf)")

    if range_minmax and not check_in_range(img, range_minmax):
        raise ValueError(
            f"Image values must be in range [{range_minmax[0]}, {range_minmax[1]}]"
        )

    return True


def check_in_range(
    img: np.ndarray, range_minmax: Tuple[np.ndarray, np.ndarray]
) -> bool:
    """Validate the range of the image array."""
    return img.min() >= np.min(range_minmax) and img.max() <= np.max(range_minmax)


def is_numeric(img: np.ndarray) -> bool:
    """Check if image values are numeric."""
    return np.issubdtype(img.dtype, np.number)


def is_fininte(img: np.ndarray) -> bool:
    """Check if image values are finite."""
    return not np.isnan(img).any() and not np.isinf(img).any()


def is_array_2d(img: np.ndarray) -> bool:
    """Check if the input image array is 2D."""
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)


def is_array_3d(img: np.ndarray) -> bool:
    """Check if the input image array is 3D."""
    return len(img.shape) == 3 and img.shape[2] in (3, 4)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8."""
    # validate image
    validate_image(img, (0, 1))

    return (img * 255).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    """Convert image to uint16."""
    # validate image
    validate_image(img, (0, 1))

    return (img * 65535).astype(np.uint8)


def get_slice(offset: int, img: np.ndarray, name: str = ""):
    """Get a slice object to remove image padding based on the offset.

    Args:
        offset: The offset value to determine the padding removal.
        img: The input image.
        name: The name of the offset value (default "").

    Returns:
        The slice object to remove image padding.

    Raises:
        TypeError: If the offset is not an integer.
        ValueError: If the offset is invalid.
    """
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


def check_dataset_type(data: Dataset) -> None:
    """Check if the input data is a torch.utils.data.Dataset object.

    Args:
        data: The input data to check.

    Raises:
        TypeError: If the input data is not a torch.utils.data.Dataset object.

    Examples:
        >>> dataset = torchvision.datasets.ImageFolder(root="path/to/data")
        >>> check_dataset_type(dataset)
    """
    if not isinstance(data, (Dataset, VisionDataset, ImageFolder)):
        raise TypeError(
            f"data must be a torch.utils.data.Dataset object, got {type(data)}"
        )


def load_mean_image(
    mean_im: Union[np.ndarray, Tuple[int, int, int], str], dtype: np.dtype = np.float32
) -> np.ndarray:
    """Load mean image from file or create empty array."""
    if isinstance(mean_im, (str, Path)):
        mean_im = cv2.imread(str(mean_im))
        mean_im = cv2.cvtColor(mean_im, cv2.COLOR_BGR2RGB)
        mean_im = to_float(mean_im)
    elif isinstance(mean_im, Tuple) and len(mean_im) == 3:
        mean_im = np.zeros(mean_im, dtype=dtype)
    elif not isinstance(mean_im, np.ndarray):
        raise TypeError(
            "mean_im must be a numpy.ndarray, tuple of array size, "
            "or str with filepath to an image."
        )
    return mean_im
