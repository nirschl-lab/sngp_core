#!/usr/bin/env python3
"""utils.py in src/biovlmdata/custom_datasets."""
from pathlib import Path
from typing import Any

from loguru import logger

from src.fileio.search import find_image_json_pairs
from src.fileio.text.readers import json_loader


def create_samples_targets(
    data_root: Path,
    split: str,
    image_ext: str,
    recursive: bool,
    jsonl_filepath: Path,
    force: bool,
    target_key: str,
) -> tuple[list[Any], list[Any]]:
    json_files, image_files = find_image_json_pairs(
        data_root,
        ext=image_ext,
        recursive=recursive,
        jsonl_filepath=jsonl_filepath,
        force=force,
    )
    image_files = [data_root.joinpath(f) for f in image_files]
    json_files = [data_root.joinpath(f) for f in json_files]
    logger.info(f"Found {len(json_files)} image-json pairs for split '{split}'")
    samples = list(zip(image_files, json_files, strict=True))
    json_list = [s[1] for s in samples]
    label_key = "label" if target_key == "all" else target_key
    targets = [
        json_loader(t).get("custom_metadata", {}).get(label_key) for t in json_list
    ]

    # filter by split
    if split != "all":
        samples = [s for s in samples if Path(s[0]).parent.stem == f"{split}"]
        targets = [
            t for t, s in zip(targets, samples) if Path(s[0]).parent.stem == f"{split}"
        ]
        logger.debug(f"Filtered samples for split '{split}'")

    if len(samples) != len(targets):
        logger.error(
            f"Samples and targets length mismatch: {len(samples)} != {len(targets)}"
        )
        raise RuntimeError

    return (samples, targets)
