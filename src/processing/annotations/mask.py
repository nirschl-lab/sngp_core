#!/usr/bin/env python3
"""mask.py in src/argusdp/processing/annotations."""
import os
import uuid
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from loguru import logger

from argusdp.processing.data_utils import timestamp
from argusdp.processing.image.detection import get_polygons
from argusdp.processing.image.detection import smooth_contours
from argusdp.processing.image.imutils import is_array_3d


cv2.setUseOptimized(True)


def rle2mask(rle_str: str, output_shape: Tuple[int, int]) -> np.ndarray:
    """Convert a run-length encoded (RLE) mask string to a binary mask.

    Args:
        rle_str: The run-length encoded mask string.
        output_shape: The desired output shape of the binary mask as a tuple (height, width).

    Returns:
        The binary mask array with the specified output shape.

    Raises:
        TypeError: If rle_str is not a string or output_shape is not a tuple of length 2.
    """
    if not isinstance(rle_str, str):
        raise TypeError("rle_str must be a string.")

    if not isinstance(output_shape, tuple) or len(output_shape) != 2:
        raise TypeError("output_shape must be a Tuple[int, int].")

    # get height and width from output_shape
    height, width = output_shape

    # split string on whitespace
    s = rle_str.split()
    starts, lengths = (np.asarray(x, dtype=int) for x in (s[:][::2], s[1:][::2]))
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends, strict=True):
        img[lo:hi] = 1
    return img.reshape((height, width))


def mask2rle(mask: np.ndarray) -> str:
    """Convert binary mask image to run length encoding."""
    if mask is None:
        raise ValueError("mask is None")

    # flatten mask image column-wise
    pixels = mask.flatten(order="F")  # column major

    # set first and last pixels to 0
    # avoids issues with '1' at the start or end
    pixels[0] = 0
    pixels[-1] = 0

    # find runs
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]

    # convert runs to string
    return " ".join(str(x) for x in runs)


def instances2mask(
    instance_list: list,
    output_shape: Tuple[int, int],
    filter_class: Optional[str] = None,
    filter_type: Optional[str] = None,
    min_area: Optional[float] = None,
) -> np.ndarray:
    """Create a labeled mask from a list of instances."""
    if not isinstance(output_shape, tuple) or len(output_shape) != 2:
        raise TypeError("output_shape must be a Tuple[int, int].")

    # allow comma-separated filter_class string
    if isinstance(filter_class, str):
        filter_class = filter_class.lower().split(",")

    if isinstance(filter_type, str):
        filter_type = filter_type.lower().split(",")

    # set dtype based on number of instances
    if len(instance_list) > 2**16:
        raise ValueError(
            f"Expected less than 2^16 instances, got {len(instance_list)}."
        )

    dtype = np.uint8 if len(instance_list) < 256 else np.uint16

    # create empty mask
    mask = np.zeros(output_shape, dtype=dtype)
    for idx, instance in enumerate(instance_list):
        if filter_type is not None and instance["type"] not in filter_type:
            continue

        if (
            filter_class is not None
            and instance["className"].lower() not in filter_class
        ):
            continue

        # get polygon points
        if instance["type"] == "polygon":  # sourcery skip
            poly_pts = np.array(instance["points"]).reshape((-1, 2))

            # filter by area
            if min_area is not None and min_area > 0:
                area = cv2.contourArea(poly_pts)
                if area < min_area:
                    continue

            cv2.fillPoly(mask, [poly_pts], idx + 1)
        else:
            raise NotImplementedError("Bbox not implemented.")

    return mask


def mask2instances(
    mask: np.ndarray,
    class_names: Union[list, str],
    creation_type: str = "Preannotation",
    locked: bool = True,
    add_centroid: bool = False,
    add_bbox: bool = False,
    approximate: bool = False,
    eps: float = 0.0001,
    # smooth: bool = False,
) -> list:
    """Convert a labeled mask to a list of instances."""
    if not mask.any() or not isinstance(mask, np.ndarray):
        return []

    if not is_array_3d(mask):
        mask = np.expand_dims(mask, axis=-1)

    if isinstance(class_names, str):
        class_names = class_names.split(",")

    # find polygons for each class
    dtype = np.uint8 if len(class_names) < 256 else np.uint16
    empty_array = np.zeros(mask.shape[:2], dtype=dtype)
    for class_id, class_name in enumerate(class_names):
        # get unique values in mask
        instance_ids = np.unique(mask[:, :, class_id]).astype(np.uint8)
        instance_ids = instance_ids[instance_ids != 0]

        # create list of instances
        instances = []
        for instance_id in instance_ids:
            temp_mask = empty_array.copy()
            temp_mask[mask[:, :, class_id] == instance_id] = 255

            # get polygons for a specific class
            if np.max(temp_mask) == 0:
                continue  # skip if temp_mask is empty

            # get polygons for a specific class
            polygon_list = get_polygons(
                temp_mask, keep_largest=False, approximate=approximate
            )
            # if smooth:
            #     polygon_list = smooth_contours(polygon_list)

            try:
                # add instances to file_json, should only be 1 polygon per instance
                for idx, poly in enumerate(polygon_list):
                    if len(poly) <= 2:
                        (
                            logger.warning(
                                f"Class:{class_id} instance:{instance_id} "
                                f"polygon:{idx} has less than 3 points! Skipping..."
                            )
                            if os.getenv("DEBUG")
                            else None
                        )
                        continue

                    created_time = timestamp()

                    # flattened contour
                    flattened_contour = poly.reshape(-1).tolist()

                    # add instance to file_json
                    instances.append(
                        {
                            "id": str(uuid.uuid4()),
                            "type": "polygon",
                            "probability": 100,
                            "points": flattened_contour,
                            "groupId": 0,
                            "pointLabels": {},
                            "locked": True,
                            "attributes": [],
                            "error": None,
                            "createdAt": created_time,
                            "updatedAt": created_time,
                            "creationType": creation_type.capitalize(),
                            "exclude": [],
                            "className": class_name,
                        }
                    )

                    # add centroid to instances
                    if add_centroid:
                        # get centroid of polygon
                        M = cv2.moments(poly)
                        # check if M["m00"] is 0
                        if M["m00"] == 0:  # avoid division by zero
                            logger.warning(f"Class {class_id} has no area!")
                        else:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            instances.append(
                                {
                                    "id": str(uuid.uuid4()),
                                    "type": "point",
                                    "probability": None,
                                    "x": cx,
                                    "y": cy,
                                    "groupId": 0,
                                    "pointLabels": {},
                                    "locked": locked,
                                    "attributes": [],
                                    "error": None,
                                    "createdAt": created_time,
                                    "updatedAt": created_time,
                                    "creationType": creation_type.capitalize(),
                                    "exclude": [],
                                    "className": class_name,
                                }
                            )

                    # add bounding box to instances
                    if add_bbox:
                        # get bounding box
                        x, y, w, h = cv2.boundingRect(poly)

                        # add bbox to file_json
                        # “points”: objects - Points of the bounding box.
                        # The list of floats is:
                        #   "x1, y1" for the left upper corner
                        #   "x2, y2" for the right lower corner
                        instances.append(
                            {
                                "id": str(uuid.uuid4()),
                                "type": "bbox",
                                "probability": None,
                                "points": {
                                    "x1": x,
                                    "y1": y,
                                    "x2": x + w,
                                    "y2": y + h,
                                },
                                "groupId": 0,
                                "pointLabels": {},
                                "locked": locked,
                                "attributes": [],
                                "error": None,
                                "createdAt": created_time,
                                "updatedAt": created_time,
                                "creationType": creation_type.capitalize(),
                                "exclude": [],
                                "className": class_name,
                            }
                        )

            except Exception as e:
                logger.error(
                    f"Error processing class {class_id} instance {instance_id}: {e}"
                )
                raise e

    return instances
