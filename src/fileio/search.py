#!/usr/bin/env python3
"""search.py in src/biovlmdata/fileio."""

import re
from pathlib import Path
from typing import Optional
from typing import Union

import pandas as pd
import ujson as json
from loguru import logger
from tqdm import tqdm

from src.fileio.text import is_empty_file
from src.fileio.text.readers import json_loader


def is_relative_path(path):
    """Check if a path is relative."""
    return not Path(path).is_absolute()


def find_image_json_pairs(
    input_dir: Union[Path, str],
    ext: str,
    recursive: bool,
    jsonl_filepath: Optional[Union[Path, str]] = None,
    ignore: Optional[str] = None,
    exclude_empty: bool = True,
    force: bool = False,
    random_seed: int = 8675309,
) -> tuple:
    """Find JSON and image pairs."""
    input_dir = Path(input_dir)
    if jsonl_filepath is None:
        jsonl_filepath = input_dir.joinpath("image_json_pairs.jsonl")

    ext = ext if ext.startswith(".") else f".{ext}"

    if jsonl_filepath.exists() and not force:
        # load jsonl file with image-json pairs
        df = pd.read_json(jsonl_filepath, orient="records", lines=True)

        # update path with input_dir if relative path, otherwise assume
        num_sample = min(max(100, min(int(len(df) * 0.01), 100)), len(df))
        random_subset = df["json_file"].sample(num_sample)
        if all(is_relative_path(f) for f in random_subset):
            df["json_file"] = (
                df["json_file"].apply(lambda x: input_dir.joinpath(x)).tolist()
            )
            df["image_file"] = (
                df["image_file"].apply(lambda x: input_dir.joinpath(x)).tolist()
            )

        # check subset of files exist
        random_subset = df["json_file"].sample(num_sample)
        missing_files = random_subset[~random_subset.apply(lambda x: Path(x).exists())]
        if missing_files.any():
            # find missing files
            logger.error(f"Missing files: {missing_files.to_list()}")
            logger.error(
                f"Expected json files not found in {input_dir}. "
                "Please rerun with force=True to regenerate image_json_pairs."
            )
            raise FileNotFoundError("Expected json files not found")

        # get json and image files
        json_files = df["json_file"].tolist()
        image_files = df["image_file"].tolist()
    else:
        # set ignore flags (defaults to any hidden files, autosave or backup files
        emacs_autosave = r"~#|#.*|#$"  # emacs autosave files
        # hidden files, autosave, backup files
        hidden_autosave_backup = r"^\..*|\~$|\.swp$"
        if ignore:
            ignore_pattern = f"{hidden_autosave_backup}|{emacs_autosave}|{ignore}"
        else:
            ignore_pattern = f"{hidden_autosave_backup}|{emacs_autosave}"

        # find json files
        json_files = sorted(
            list(input_dir.rglob("*.json"))
            if recursive
            else list(input_dir.glob("*.json"))
        )
        # ignore json file with same name as parent directory
        json_files = [elem for elem in json_files if elem.stem != elem.parent.name]

        # remove ignored files
        json_files = [
            elem for elem in json_files if not re.match(ignore_pattern, elem.name)
        ]

        # remove files that are empty (stat size == 0)
        if exclude_empty:
            json_files_clean = [elem for elem in json_files if not is_empty_file(elem)]
            if len(json_files) != len(json_files_clean):
                missing_files = set(json_files) - set(json_files_clean)
                logger.warning(
                    f"Removed {len(missing_files)} empty json files: {missing_files}"
                )
                raise FileNotFoundError(f"Empty json files found: {missing_files}")

        # error checks
        if not json_files:
            logger.error(f"No json files found in {input_dir}")
            return [], []
            # raise FileNotFoundError(f"No json files found in {input_dir}")

        # find image files assuming image is json_file.with_suffix(ext)
        image_files = [
            json_file.parent.joinpath(json_file.with_suffix(ext))
            for json_file in json_files
        ]
        # error checks
        for elem in image_files:
            if not elem.exists():
                logger.error(f"Expected image-json pairs. Image not found: {elem}")
                raise FileNotFoundError(f"Image not found: {elem}")

        if len(json_files) != len(image_files):
            logger.error(
                f"Number of json files {len(json_files)} "
                f"!= number of image files {len(image_files)}"
            )
            raise ValueError("Number of json files != number of image files")

        # read json files to get split and label
        split = []
        label = []
        for json_file in tqdm(json_files):
            # with open(json_file) as f:
            #     data = json.load(f)
            data = json_loader(json_file)
            # # verify SADict
            # data = SADict(**data)

            split.append(data["custom_metadata"].get("split"))
            label.append(data["custom_metadata"].get("label"))

        # save jsonl of all image-json pairs
        df = pd.DataFrame(
            {
                "json_file": [f.relative_to(input_dir).as_posix() for f in json_files],
                "image_file": [
                    f.relative_to(input_dir).as_posix() for f in image_files
                ],
                "split": split,
                "label": label,
            }
        )
        # if any split or label is None/nan then log error
        if df["split"].isnull().any():
            logger.error(f"Missing split in {input_dir}")
        if df["label"].isnull().any():
            logger.error(f"Missing label in {input_dir}")

        # shuffle
        df = df.sample(frac=1, random_state=random_seed)

        # save to jsonl
        json_str = df.to_json(orient="records", lines=True)
        with jsonl_filepath.open("w") as f:
            f.write(json_str)

        # save split
        for split in ["train", "validation", "test"]:
            jsonl_split_filepath = jsonl_filepath.with_name(f"{split}.jsonl")
            df_split = df[df["split"] == split]
            json_str_split = df_split.to_json(orient="records", lines=True)
            with jsonl_split_filepath.open("w") as f:
                f.write(json_str_split)

        # subsample image_json_df to get either 100 per class for classes with more than 100
        # and get all samples for classes with less than 100
        # get only test samples for test split
        for num_sample, split in zip(
            [700, 100, 200], ["train", "validation", "test"], strict=True
        ):
            df_test = df[df["split"] == split]
            df_group = df_test.groupby("label")
            df_group = df_group.apply(
                lambda x: pd.DataFrame(x).sample(
                    n=min(len(x), num_sample), random_state=random_seed
                )
            )
            # summarize counts per class
            df_subset = df_group.reset_index(drop=True)
            df_group_counts = df_subset.groupby("label").size()
            logger.info(f"Counts per class: {df_group_counts}")

            # create json str with subset
            json_str_subset_filepath = jsonl_filepath.with_name(
                f"{split}_{num_sample}.jsonl"
            )
            df_subset = df_subset.reset_index(drop=True)
            df_subset = df_subset.sample(frac=1, random_state=random_seed)
            json_str_subset = df_subset.to_json(orient="records", lines=True)
            with json_str_subset_filepath.open("w") as f:
                f.write(json_str_subset)

        # ##
        # # groupby split and save each split to train, validation, test.json
        # df_group = df.groupby("split")
        # for split, df_split in df_group:
        #     jsonl_split_filepath = jsonl_filepath.with_name(f"{split}.jsonl")
        #     json_str_split = df_split.to_json(orient="records", lines=True)
        #     with jsonl_split_filepath.open("w") as f:
        #         f.write(json_str_split)

    return json_files, image_files

    #     # error checks
    #     if not json_files:
    #         logger.error(f"No json files found in {input_dir}")
    #         return [], []
    #         # raise FileNotFoundError(f"No json files found in {input_dir}")
    #
    #     # find image files assuming image is json_file.with_suffix(ext)
    #     image_files = [
    #         json_file.parent.joinpath(json_file.with_suffix(ext))
    #         for json_file in json_files
    #     ]
    #     # error checks
    #     for elem in image_files:
    #         if not elem.exists():
    #             logger.error(f"Expected image-json pairs. Image not found: {elem}")
    #             raise FileNotFoundError(f"Image not found: {elem}")
    #
    #     if len(json_files) != len(image_files):
    #         logger.error(
    #             f"Number of json files {len(json_files)} "
    #             f"!= number of image files {len(image_files)}"
    #         )
    #         raise ValueError("Number of json files != number of image files")
    #
    #     # save jsonl of all image-json pairs
    #     image_json_df = pd.DataFrame(
    #         {
    #             "json_file": [f.relative_to(input_dir).as_posix() for f in json_files],
    #             "image_file": [
    #                 f.relative_to(input_dir).as_posix() for f in image_files
    #             ],
    #         }
    #     )
    #     json_str = image_json_df.to_json(orient="records", lines=True)
    #     with jsonl_filepath.open("w") as f:
    #         f.write(json_str)
    #
    # return json_files, image_files
