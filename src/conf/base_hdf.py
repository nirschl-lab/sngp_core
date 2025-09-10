#!/usr/bin/env python3
"""base_hdf.py in src/argusdp/conf."""
import ast
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from loguru import logger
from numpy._typing import NDArray
from pydantic import AliasChoices
from pydantic import AliasGenerator
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Extra
from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import field_validator

from src.conf.compiled_regex import RE_LIST
from src.conf.pydantic_validators import ClassesToIdx
from src.conf.pydantic_validators import Domain
from src.conf.pydantic_validators import ImageMeanStd
from src.conf.pydantic_validators import ImageShape
from src.conf.pydantic_validators import License
from src.conf.pydantic_validators import ListBool
from src.conf.pydantic_validators import ListStr
from src.conf.pydantic_validators import Modality
from src.conf.pydantic_validators import Split
from src.conf.pydantic_validators import _check_input


class HDFMetadata(BaseModel):
    """Metadata for the HDF file."""

    model_config = ConfigDict(
        extra=Extra.ignore,
        str_strip_whitespace=True,
        json_encoders={Path: str},
    )

    # Metadata fields
    bibtex: Optional[Union[List[str], str]] = Field(
        default=None, description="BibTeX citation for the dataset."
    )
    classes_to_idx: ClassesToIdx = Field(
        description="Mapping of class names to integer indices."
    )
    dataset_name: str = Field(
        description="Name of the dataset.",
        min_length=4,
    )
    description: Optional[str] = Field(
        default=None, description="Description of the dataset."
    )
    domain: ListStr = Field(
        description="Domain (field of study) of the dataset.",
        validation_alias=AliasChoices("domain", "field_of_study"),
        serialization_alias="domain",
    )
    image_mean_std: Optional[ImageMeanStd] = Field(
        default=None, description="Mean and standard deviation of the dataset images."
    )
    image_size: Optional[ImageShape] = Field(
        default=None, description="Shape of the images in the dataset."
    )
    institution: Optional[ListStr] = Field(
        default=None,
        description="Institution or source of the dataset.",
        validation_alias=AliasChoices("institution", "source"),
        serialization_alias="institution",
    )
    keywords: Optional[ListStr] = Field(
        default=None, description="Keywords for the dataset."
    )
    last_updated: str
    license: Union[List[License], License] = Field(
        description="License for the dataset."
    )
    microns_per_pixel: Optional[Union[List[PositiveFloat], PositiveFloat, str]] = Field(
        default=None,
        description="Microns per pixel.",
        validation_alias=AliasChoices("microns_per_pixel", "micron_per_pixel", "mpp"),
        serialization_alias="microns_per_pixel",
        coerce_str_to_numbers=True,
    )
    modality: ListStr = Field(
        description="Microscopy modality of the dataset.",
        validation_alias=AliasChoices("modality", "microscopy_type"),
        serialization_alias="modality",
    )
    model: str
    num_samples: Optional[PositiveInt] = None
    num_classes: PositiveInt = Field(
        default=None, description="Number of classes in the dataset."
    )
    split: Split
    stain: Optional[ListStr] = Field(
        default=None, description="Stain used in the dataset."
    )
    subdomain: Optional[ListStr] = Field(
        default=None,
        description="Subdomain (subfield of study) of the dataset.",
        validation_alias=AliasChoices("subdomain", "sub_domain"),
        serialization_alias="subdomain",
    )
    submodality: Optional[ListStr] = Field(
        default=None,
        description="Submodality of the dataset.",
        validation_alias=AliasChoices("submodality", "sub_modality"),
        serialization_alias="submodality",
    )
    random_seed: int
    version: Optional[str] = Field(
        default="0.0.1",
        description="Version of the dataset.",
        pattern=r"^\d+\.\d+\.\d+$",
    )
    version_comment: Optional[str] = Field(
        default=None, description="Comment on the dataset version."
    )

    @field_validator("microns_per_pixel")
    @classmethod
    def validate_microns_per_pixel(
        cls, v
    ) -> Optional[Union[List[PositiveFloat], PositiveFloat]]:
        v = _check_input(v)
        if v is None or v == "" or not v:
            return None
        elif isinstance(v, Union[float, int]):
            return [v]
        elif isinstance(v, str) and RE_LIST.match(v):
            # if elem is a string representation of a list, convert to list
            v = ast.literal_eval(v)
        elif isinstance(v, str):
            v = [v]
        elif isinstance(v, list):
            pass
        else:
            raise ValueError(
                f"Invalid input for microns_per_pixel: {v} with type ({type(v)})"
            )

        return v


class HDFSplit(BaseModel):
    """Split for the HDF file."""

    model_config = ConfigDict(
        # extra=Extra.ignore,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
        alias_generator=AliasGenerator(
            serialization_alias=lambda field_name: field_name.replace(
                "active_learning", "al"
            )
        ),
    )

    features: NDArray[np.float32]
    labels: NDArray[np.int64]
    image_ids: ListStr
    instance_ids: ListStr
    image_files: ListStr
    al_selected: NDArray[np.bool_]
    al_scores: NDArray[np.float32]
    al_labels: NDArray[np.int32]


class BioVLMHDF(BaseModel):
    """BioVLM HDF file contents."""

    metadata: HDFMetadata
    train: Optional[HDFSplit] = None
    validation: Optional[HDFSplit] = None
    test: Optional[HDFSplit] = None

    def model_post_init(self, __context: Any) -> None:
        """Post-init hook for the model."""
        if self.train is None and self.validation is None and self.test is None:
            raise ValueError("At least one split must be provided.")

        if self.metadata.num_samples is None:
            num_samples = sum(
                len(split.features)
                for split in [self.train, self.validation, self.test]
                if split is not None
            )
            self.metadata.num_samples = num_samples
