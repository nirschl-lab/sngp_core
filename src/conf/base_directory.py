#!/usr/bin/env python3
"""base_directory.py in src/biovlmdata/conf."""

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import DirectoryPath
from pydantic import Field
from pydantic import FilePath
from pydantic import field_validator

from argusdp.conf.pydantic_validators import _check_input
from argusdp.fileio.text.readers import get_metadata_file


def _get_first_file(input_dir: Path):
    # get first non-directory and non-hidden file
    dir_contents = input_dir.iterdir()
    for f in dir_contents:
        if f.is_file() and f.suffix in {".json", ".png"}:
            return f


def _check_paired_files(first_file: Path):
    paired_ext = ".json" if first_file.suffix == ".png" else ".png"
    paired_file = first_file.with_suffix(paired_ext)

    if not paired_file.exists():
        error_msg = f"Expected paired file {paired_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def _check_subfolders(root: Path):
    subfolders = ["train", "validation", "test"]
    for split in subfolders:
        if not root.joinpath(split).is_dir():
            logger.error(f"Subfolder {split} not found in {root}.")
            raise ValueError(f"{root} does not contain subfolder {split}.")

        # check at least one image:json pair in each subfolder
        split_dir = root.joinpath(split)
        first_file = _get_first_file(split_dir)
        if first_file is None:
            logger.error(f"No files found in {split_dir}.")
            raise ValueError(f"{root.joinpath(split)} does not contain any files.")

        # if json file, check for paired png, else check for paired json
        _check_paired_files(first_file)


def _check_metadata_files(root: Path):
    # at least one of yaml or json metadata files should be present
    metadata_file = get_metadata_file(root, metadata_format=".yaml")

    if metadata_file is None:
        logger.error(f"Expected metadata file not found in {root}.")
        raise FileNotFoundError(f"Expected metadata file not found in {root}.")

    # check for feather file with same name as parent
    feather_file = root.joinpath(f"{root.stem}.feather")
    if not feather_file.exists():
        logger.warning(f"Expected feather file not found in {root}.")


def _check_files(
    root: Path, ext: str = ".png", recursive: bool = False, random_seed: int = 8675309
):
    # jsonl files recommended but can be created later
    jsonl_files = [
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "image_json_pairs.jsonl",
    ]
    for f in jsonl_files:
        if not root.joinpath(f).exists():
            logger.warning(f"Expected file {f} not found in {root}.")


class BioVLMDataDirectory(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_encoders={Path: str},
    )

    root: DirectoryPath = Field(
        description="BioVLM root directory containing subfolders train, validation, and test with image-json pairs.",
        validation_alias=AliasChoices("root", "input_dir"),
        serialization_alias="root",
    )
    name: str = Field(default=None, description="Name of the dataset.")
    metadata_file: FilePath = Field(
        default=None,
        description="Metadata yaml/json file with same name as parent directory.",
        validation_alias=AliasChoices("metadata_file", "dataset_json"),
        serialization_alias="metadata_file",
    )
    jsonl_files: dict = Field(
        default=None,
        description="Dictionary of jsonl files for train, validation, test, and subsets.",
        validation_alias=AliasChoices("jsonl_files", "jsonl_files_dict"),
        serialization_alias="jsonl_files",
    )
    # # dataset name, no reserved characters or words, should be filepath safe
    # dataset_name: str = Field(
    #     description="Name of the dataset.",
    #     min_length=3,
    # )

    @field_validator("root")
    def _check_root(cls, value):
        v = _check_input(value)
        if not v.is_dir() or not v.exists():
            raise ValueError(f"{v} is not a valid directory.")

        # check for subfolders train, validation, and test
        _check_subfolders(v)

        # check for jsonl files
        _check_files(v)

        return v

    def model_post_init(self, __context: Any) -> None:
        self.name = self.root.stem
        self.metadata_file = get_metadata_file(self.root, metadata_format=".yaml")
        self.jsonl_files = {
            s: self.root.joinpath(f"{s}.jsonl") for s in ["train", "validation", "test"]
        } | {
            f"{s}_subset": self.root.joinpath(f"{s}_subset.jsonl")
            for s in ["train", "validation", "test"]
        }

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return (
            f"{self.__class__.__name__}(\n"
            f"\troot={self.root}\n"
            f")"  # noqa: B907
        )

    def __str__(self) -> str:
        """An easy-to-read string representation of the class instance."""
        base_str = (
            f"Dataset {self.name.replace('_',' ').title()}\n"
            f"\tRoot:\t\t{Path(self.root)}\n"
        )

        additional_str = ""
        for key in ["metadata_file"]:
            safe_key = key.lower().replace(" ", "_")
            if not getattr(self, safe_key):
                continue
            else:
                value = getattr(self, safe_key)

            key = key.replace("_", " ").capitalize()
            sep = "\t" if len(key) < 8 else ""
            additional_str += f"\t{key}:{sep}{value}\n"

        return base_str + additional_str


# # check
# from pathlib import Path
#
#
# data_root = Path("/media/jjn/jjn_raid/data/bcv/data/processed/acevedo_et_al_2020")
# data_root = BioVLMDataDirectory(root=data_root)
# print(data_root)
