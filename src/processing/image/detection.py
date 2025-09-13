#!/usr/bin/env python3
"""detection.py in src/sngp_core/processing/image."""

from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import cv2
import diplib as dip
import numpy as np
from loguru import logger
from scipy.interpolate import splev
from scipy.interpolate import splprep


cv2.setUseOptimized(True)


def get_bounding_box(
    mask: np.ndarray, sort: bool = True, keep_largest: bool = False
) -> list[Sequence[int]]:
    """Get bounding boxes for all objects in the mask (sorted by object area)."""
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort by object area
    contours = (
        sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if sort
        else contours
    )

    # get bounding boxes
    bboxes = [cv2.boundingRect(c) for c in contours]

    if keep_largest and bboxes:
        return [tuple(max(bboxes, key=lambda x: x[2] * x[3]))]
    else:
        return bboxes


def approximate_contours(
    contours: Tuple, mask: np.ndarray, eps: float = 0.001
) -> Tuple:
    """
    Approximate each contour to reduce the number of points.
    Returns a tuple of approximated contours.
    """
    if len(contours) == 0:
        return tuple(contours)

    # Convert contours to a list to support item assignment
    contours = list(contours)

    # sort contours by area
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for idx, c in enumerate(contours):
        # compute perimeter, try first with diplib (opencv overestimates)
        try:
            temp_img = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(temp_img, [c], -1, 255, thickness=cv2.FILLED)
            msr = dip.MeasurementTool.Measure(temp_img, features=["Perimeter"])
            perim = msr[255]["Perimeter"][0]
        except KeyError:
            logger.warning(f"Error computing perimeter for contour {idx}.")
            perim = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(c, eps * perim, True)

        # element-wise comparison of two arrays c and approx
        if c.shape[0] == approx.shape[0]:
            # if the number of points is the same, then the contour was not approximated
            continue

        # replace contour with approximated polygon
        contours[idx] = approx

    # Convert back to tuple
    return tuple(contours)


def smooth_contours(contours: list) -> list:
    """Smooth contours using cv2.approxPolyDP."""
    smoothened = []
    for contour in contours:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))

    return smoothened


def get_polygons(
    mask: np.ndarray,
    approximate: bool = True,
    eps: float = 0.0001,
    sort: bool = True,
    keep_largest: bool = False,
    min_area: int = 0,
) -> List[Any]:
    """Get polygon coordinates for all objects in the mask (sorted by object area)."""
    if mask is None or not mask.any():
        return []

    if min_area > 0 and keep_largest:
        raise ValueError("Cannot use both min_area and keep_largest.")

    # find contours
    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL if keep_largest else cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return []

    # keep only the largest contour
    if keep_largest:
        contours = [max(contours, key=lambda x: cv2.contourArea(x))]

    # filter by minimum area
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        # contours = [c for c in contours if cv2.contourArea(c) < 100000]

    # approximate the contour to reduce the number of points
    if approximate:
        contours = approximate_contours(contours, mask=mask, eps=eps)

    # sort by object area
    if sort:
        return sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    else:
        return contours
