#!/usr/bin/env python3
"""harmonize_text.py in src/sngp_core/processing/text."""
from typing import List
from typing import Union


def validate_split(split: str) -> str:
    if not isinstance(split, str):
        raise ValueError(f"Expected split to be str, got {type(split)}")

    # harmonize split name
    if split.lower() in {"train", "training"}:
        split = "train"
    elif split.lower() in {"train_all", "train+val", "train_and_val", "trainval"}:
        split = "train+val"
    elif split.lower() in {"dev", "val", "validation"}:
        split = "validation"
    elif split.lower() in {"test", "hold-out", "held-out", "held-out_test"}:
        split = "test"
    elif split.lower() in {"predict", "inference"}:
        split = "predict"
    elif split.lower() in {"all", "all_data", "train+val+test", "trainvaltest"}:
        split = "all"
    else:
        raise ValueError(f"Split {split} not found.")

    return split


def validate_split_list(split: Union[List[str], str]) -> list:
    if isinstance(split, str):
        split = validate_split(split)
        split = [split] if split != "all" else ["train", "validation", "test"]
    elif isinstance(split, list):
        split = [validate_split(s) for s in split]
    else:
        raise ValueError(f"Expected split to be str or list, got {type(split)}")

    return split
