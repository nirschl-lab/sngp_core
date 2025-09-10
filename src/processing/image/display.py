#!/usr/bin/env python3
"""display.py in src/argusdp/processing/image."""
import pprint
from typing import Optional
from typing import Tuple

import click
import cv2
import numpy as np
from loguru import logger

from argusdp.conf.pydantic_validators import ClassesToIdx


DEFAULT_WINDOW_SIZE = 1200
cv2.setUseOptimized(True)


def annotate_image(
    image: np.ndarray,
    classes_to_idx: ClassesToIdx,
    color: Optional[Tuple[int, int, int]] = (0, 255, 0),
    thickness=5,
    label: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    format: Optional[str] = "RGB",  # assume RGB (PIL loader)
) -> Optional[int]:
    """Display the image with the label and mask (if provided)."""
    key = 0
    idx_to_classes = {v: k for k, v in classes_to_idx.items()}
    key_to_label = {ord(str(k)): v for k, v in idx_to_classes.items()}
    window_name = "Image to label"
    if label is not None:
        label_name = idx_to_classes[label]
        window_name += f"predicted label: {label_name} ({label})"

    # convert image to RGB if in BGR format
    if format == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # if mask is provided, draw outline on image
    if mask is not None:
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image, contours, -1, color, thickness)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE)

    logger.info("Assign the class label for the image")
    logger.info(f"\t{pprint.pformat(classes_to_idx, indent=4, sort_dicts=False)}")
    new_label = None
    while key != ord("q") or new_label is not None:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(0) & 0xFF
        # exit on 'q' key or ESC key
        if key == ord("q") or key == 27:
            logger.info(f"Exiting {window_name}...")
            cv2.destroyWindow(window_name)
            break
        elif key in key_to_label:
            label_name = key_to_label.get(key)
            if label_name is None:
                logger.warning(f"Invalid key: {key}")
                continue

            new_label = classes_to_idx[label_name]
            logger.info(f"New label: {label_name} ({new_label})")
            cv2.destroyWindow(window_name)
            break
        # create new class if key > max(ord(str(len(classes_to_idx))))
        elif key > max(key_to_label.keys()):
            # click get user input
            new_label = len(classes_to_idx) + 1
            # get label name, do not allow empty string or abort
            label_name = click.prompt(
                "Enter new label name", type=str, confirmation_prompt=True
            )
            classes_to_idx[label_name] = new_label

            # check that sorted classes_to_idx.values are monotonic
            if not np.all(np.diff(sorted(classes_to_idx.values())) > 0):
                logger.error(f"Classes_to_idx not monotonic: {classes_to_idx}")
                raise ValueError(f"Classes_to_idx not monotonic: {classes_to_idx}")

            cv2.destroyWindow(window_name)
            break
        else:
            logger.warning(f"Invalid key: {key}")

    if new_label != label:
        logger.debug(f"Updated label: {label} -> {new_label}")
        logger.debug(
            f"Updated label: {idx_to_classes[label]} -> {idx_to_classes[new_label]}"
        )
        click.confirm(
            "Keep the new label?", abort=True, default=True, show_default=True
        )

    return new_label
