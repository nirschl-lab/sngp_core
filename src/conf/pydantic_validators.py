#!/usr/bin/env python3
"""pydantic_validators.py in src/argusdp/conf."""
import ast
import re
import uuid
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import UUID4
from pydantic import AliasChoices
from pydantic import AliasGenerator
from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import EmailStr
from pydantic import Extra
from pydantic import Field
from pydantic import FilePath
from pydantic import NaiveDatetime
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import alias_generators
from pydantic import field_validator
from pydantic import validators
from pydantic.functional_validators import AfterValidator
from pydantic.networks import AnyUrl
from pydantic.networks import EmailStr
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import Annotated

from src.conf import DOMAINS as valid_domains
from src.conf import MODALITIES as valid_modalities
from src.conf.compiled_regex import RE_CC0
from src.conf.compiled_regex import RE_CCBY
from src.conf.compiled_regex import RE_IMAGE_MEAN_STD
from src.conf.compiled_regex import RE_IMAGE_SHAPE
from src.conf.compiled_regex import RE_LIST
from src.conf.compiled_regex import RE_RESERVED_CHARS
from src.conf.compiled_regex import RE_UUID
from src.processing.text.conversions import str2num
from src.processing.text.harmonize_text import validate_split
from src.processing.text.harmonize_text import validate_split_list


# from argusdp.processing.data_utils import _is_valid_timestamp

modalities = list(valid_modalities.keys())
submodalities = [elem for sublist in valid_modalities.values() for elem in sublist]
all_modalities = modalities + submodalities

domains = list(valid_domains.keys())
subdomains = [elem for sublist in valid_domains.values() for elem in sublist]
all_domains = valid_domains


# Use Annotated to bind validation to a type rather than model or field
# helper function
def _check_input(v: Any, return_value: Optional[Any] = None) -> Union[str, None]:
    null_str = {"null", "none", "na", "nan", "n/a"}
    try:
        if v is None or not v:
            return return_value
        elif isinstance(v, (list, tuple, set)):
            return return_value if pd.isnull(v).any() else v
        elif isinstance(v, str) and v.lower() in null_str:
            return return_value
        elif isinstance(v, float) and np.isnan(v):
            return return_value
    except Exception as e:
        logger.error(f"Error checking input {v}:\n{e}")
        raise ValueError(f"Error checking input {v}") from e

    return v


# Custom Annotated types
def _check_age(v: Any) -> NonNegativeFloat:
    v = _check_input(v)
    if v is None:
        return np.nan

    if isinstance(v, str):
        v = str2num(v)

    if not isinstance(v, (int, float)):
        raise ValueError("Age must be a number.")
    elif v < 0:
        raise ValueError("Age must be a positive number.")
    elif v > 150:
        raise ValueError("Age must be less than 150.")

    return float(v)


def _validate_classes_to_idx(v: Any) -> Dict[str, int]:
    if v is None or _check_input(v) is None:
        raise ValueError("classes_to_idx is required.")

    # if dict stored as string, try to convert to dict
    if isinstance(v, str):
        v = _check_input(v)
        try:
            v = eval(v)
        except Exception as e:
            raise ValueError(f"Error evaluating classes_to_idx: {e}") from e

    if not isinstance(v, dict):
        raise ValueError("classes_to_idx must be a dictionary.")

    # coerce str to num
    for key, val in v.items():
        if isinstance(val, str):
            val = val.replace(",", "") if "," in val else val
            v[key] = str2num(val)

    # values must be non-negative integers
    if not all(isinstance(val, int) and val >= 0 for val in v.values()):
        raise ValueError("Values in classes_to_idx must be non-negative integers.")

    # keys must be str, otherwise convert to str
    if not all(isinstance(key, str) for key in v.keys()):
        v = {str(key): val for key, val in v.items()}

    # values must be sequential monotonic increasing when sorted
    if not np.all(np.diff(sorted(v.values())) == 1):
        raise ValueError("Values in classes_to_idx must be sequential integers.")

    return v


def _check_for_reserved_chars(v: str) -> str:
    # check for reserved characters that are not allowed in filenames
    v = _check_input(v)
    if RE_RESERVED_CHARS.search(v):
        raise ValueError(f"Reserved characters are not allowed: {v}")
    return v


def _validate_licence(v: str) -> str:
    v = _check_input(v)
    if v is None:
        raise ValueError("License is required.")

    if isinstance(v, str) and RE_LIST.match(v):
        # if elem is a string representation of a list, convert to list
        v = ast.literal_eval(v)
    elif isinstance(v, str):
        # replace spaces with hyphens for CC licenses
        v = v.strip().replace(" ", "-") if "CC" in v else v.strip()
        v = [v]
    elif not isinstance(v, str):
        raise ValueError("License must be a string")
    elif isinstance(v, list):
        v = [elem.strip() for elem in v]

    # unpack list of lists
    v = [elem for sublist in v for elem in sublist] if isinstance(v[0], list) else v

    alt_licenses = [
        "CC0-1.0",
        "Public Domain",
        "Proprietary",
        "Other",
        "Unknown",
        "Non-commercial",
    ]
    alt_licenses = [elem.lower() for elem in alt_licenses]
    output_list = []
    if isinstance(v, list) and isinstance(v[0], list):
        v = [elem for sublist in v for elem in sublist]

    for elem in v:
        # regex for CC license
        if RE_CCBY.match(elem) or RE_CC0.match(elem):
            output_list.append(elem)
            continue
        elif elem.lower() in alt_licenses:
            output_list.append(elem)
        else:
            logger.error(f"Invalid license: {elem}")
            logger.error(f"Expected one of CC-BY or {alt_licenses}")
            # raise ValueError(f"Expected: CC-BY {alt_licenses} Actual: {elem}")

    return v


def _validate_ontology_id(v: Any) -> list:
    # check for reserved characters that are not allowed in filenames
    if isinstance(v, str):
        v = _check_input(v)
        v = v.split(",") if "," in v else [v]
    elif isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return []
        else:
            v = [str(v)]
    elif isinstance(v, list):
        # remove "nan" elem from list
        v = [_check_input(elem) for elem in v]
        v = [elem for elem in v if elem is not None]

    # hack
    if v is None or v == "" or not v:
        return []

    # replace ":" with "_"
    v = [elem.replace(":", "_") for elem in v]

    # process list
    output_list = []
    for elem in v:
        if isinstance(elem, str) and ":" in v and "_" not in v:
            v = v.replace(":", "_")
        output_list.append(elem)

    return output_list


def _validate_md5(v: Any) -> Union[str, None]:
    v = _check_input(v)
    if v is None:
        return None

    # must be a 32 character hex string
    if not re.match(r"^[a-f0-9]{32}$", v):
        raise ValueError("Invalid image_md5 format")
    elif v == "d41d8cd98f00b204e9800998ecf8427e":
        logger.error(f"Empty image_md5 hash: {v}")
        v = None

    return v


def _validate_image_shape(v: Any) -> List[PositiveInt]:
    v = _check_input(v)
    if v is None:
        return None

    # check if it is a valid list formatted as str (e.g., '[250, 250, 3]')
    if isinstance(v, str) and RE_IMAGE_SHAPE.match(v):
        v = eval(v)
        if len(v) == 3:
            v = v[:2] if v[2] == 1 else v
    elif not isinstance(v, list):
        raise ValueError("Image shape must be a list")

    if len(v) not in [2, 3]:
        raise ValueError("Image shape must be a list of length 2 or 3")
    elif len(v) == 3 and v[2] != 3:
        raise ValueError("Image shape must be a list of length 2 or 3 with 3 channels")

    try:
        v = [int(elem) for elem in v]
    except Exception as e:
        raise ValueError("Error converting image size to integers.") from e

    if not all(isinstance(elem, int) for elem in v):
        raise ValueError("Image size must be a list of two positive integers.")

    if any(elem <= 0 for elem in v):
        raise ValueError("Image size must be a list of two positive integers.")

    return v


def _validate_image_mean_std(v: Any) -> List[PositiveInt]:
    v = _check_input(v)
    if v is None:
        return None

    # check if it is a valid list formatted as str (e.g., '[250, 250, 3]')

    if isinstance(v, str) and RE_IMAGE_MEAN_STD.match(v):
        v = eval(v)
    elif not isinstance(v, list):
        raise ValueError("Image shape must be a list")

    if len(v) not in [2, 3]:
        raise ValueError("Image shape must be a list of length 2 or 3")
    elif len(v) == 3 and v[2] != 3:
        raise ValueError("Image shape must be a list of length 2 or 3 with 3 channels")

    try:
        v[0] = [float(elem) for elem in v[0]]  # mean
        v[1] = [float(elem) for elem in v[1]]  # std
    except Exception as e:
        raise ValueError("Error converting image mean std to float.") from e

    if any(elem <= 0 for elem in v[0]):
        raise ValueError("Image mean must be a list of three positive integers.")

    if any(elem <= 0 for elem in v[1]):
        raise ValueError("Image std must be a list of three positive integers.")

    return v


def _validate_bool(v: Any) -> List[bool]:
    v = _check_input(v)
    if v is None or v == "":
        return None
    elif not isinstance(v, (bool, list)):
        raise ValueError("Expected bool or list of bool.")

    return v if isinstance(v, list) else [v]


def _validate_listint(v: Any) -> List[int]:
    v = _check_input(v)
    if v is None or v == "":
        return None
    elif not isinstance(v, (int, list)):
        raise ValueError("Expected int or list of int.")

    if isinstance(v, list):
        return v

    return [int(v)]


def _validate_liststr(v: Any, info: ValidationInfo) -> List[str]:
    v = _check_input(v)
    if v is None or v == "":
        return []
    elif not isinstance(v, (str, list)):
        raise ValueError("Expected string or list of strings.")

    if isinstance(v, list):
        v = [str(elem) for elem in v if elem is not None]
    elif isinstance(v, str):
        v = v.split(",")

    # remove na, nan, None, empty strings
    v = [elem for elem in v if elem not in ["na", "nan", "None", ",", ""]]

    return v


def _validate_split(v: Union[str, List[str]]) -> Union[str, List[str]]:
    v = _check_input(v)
    if v is None or v == "":
        raise ValueError("Split is required.")

    if not isinstance(v, (str, list)):
        raise ValueError("Expected string or list of strings.")

    return validate_split(v) if isinstance(v, str) else validate_split_list(v)


def _check_for_elem_in_dict(v: List[str], ref_dict: Dict[str, Any]) -> str:
    if not isinstance(v, list):
        raise ValueError("Expected a dictionary.")

    for elem in v:
        if elem not in ref_dict:
            raise ValueError(f"Invalid domain: {elem} not found in {ref_dict.keys()}")

        # check values of all_domains to see if subdomain was given
        # if match found in subdomain, return subdomain
        for key, val in ref_dict.items():
            if elem in val:
                logger.info(
                    f"Input `{elem}` is a subset of `{key}`. Returning `{key}`."
                )
                return key

    return v


def _validate_domain(v: Union[str, list]) -> str:
    v = _check_input(v)
    if v is None or v == "" or not v:
        raise ValueError("Domain is required.")

    if isinstance(v, str):
        v = [v]
    elif not isinstance(v, list):
        raise ValueError("Expected string or list of strings.")

    return _check_for_elem_in_dict(v, valid_domains)


def _validate_modality(v: Any) -> str:
    v = _check_input(v)
    if v is None or v == "":
        raise ValueError("Modality is required.")

    if v not in all_modalities:
        raise ValueError(f"Invalid modality: {v}")

    # check values of all_modalities to see if submodality was given
    # if match found in submodality, return submodality
    for key, val in valid_modalities.items():
        if v in val:
            logger.info(f"Input `{v}` is a submodality of `{key}`. Returning `{key}`.")
            return key

    return v


def _validate_split(v: str) -> str:
    v = _check_input(v)
    if v is None or v == "":
        raise ValueError("Split is required.")

    if not isinstance(v, str):
        raise ValueError("Split must be a string.")

    valid_splits = {"train", "validation", "test", "held-out"}
    if v.lower() not in valid_splits:
        raise ValueError(f"Invalid split: {v}. Must be one of {valid_splits}.")

    return v.lower()


def _validate_pmid(v):
    v = _check_input(v)
    if v is None or v == "":
        return None

    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        else:
            v = str(int(v))
    elif isinstance(v, int):
        v = str(v)

    if not isinstance(v, str):
        raise ValueError(f"Expected PMID to be type str or int. Got {type(v)}")

    return v


# Annotated types
ListInt = Annotated[Union[List[int], int], AfterValidator(_validate_listint)]
ListStr = Annotated[Union[List[str], str], AfterValidator(_validate_liststr)]
ListURL = Annotated[
    Union[List[AnyUrl], AnyUrl, str],
    # AfterValidator(str),
    AfterValidator(_validate_liststr),
]

# Custom Annotated types
Age = Annotated[Any, AfterValidator(_check_age)]
ClassesToIdx = Annotated[Any, AfterValidator(_validate_classes_to_idx)]
Domain = Annotated[Union[ListStr, str], AfterValidator(_validate_domain)]
ImageShape = Annotated[
    Union[List[PositiveInt], str], AfterValidator(_validate_image_shape)
]
ImageMeanStd = Annotated[Any, AfterValidator(_validate_image_mean_std)]
# LastUpdated = Annotated[Union[AwareDatetime, NaiveDatetime], AfterValidator(_is_valid_timestamp)]
ListBool = Annotated[Union[List[bool], bool], AfterValidator(_validate_liststr)]
ListStr = Annotated[Union[List[str], str], AfterValidator(_validate_liststr)]
License = Annotated[str, AfterValidator(_validate_licence)]
ListLicense = Annotated[
    Union[List[License], License], AfterValidator(_validate_liststr)
]
MD5 = Annotated[str, AfterValidator(_validate_md5)]
Modality = Annotated[str, AfterValidator(_validate_modality)]
OntologyID = Annotated[Any, AfterValidator(_validate_ontology_id)]
PMID = Annotated[Union[str, ListStr, int, ListInt], AfterValidator(_validate_pmid)]
Split = Annotated[Union[str, ListStr], AfterValidator(_validate_split)]
ValidURL = Annotated[
    Union[AnyUrl, str],
    AfterValidator(lambda x: AnyUrl(x) if x else None),
    AfterValidator(lambda x: str(x) if x else None),
]


# Custom pydantic models
# SAMetadata
class LastAction(BaseModel):
    email: EmailStr
    timestamp: int  # assuming timestamp is in milliseconds since epoch


# SAComment
class UserDetail(BaseModel):
    email: EmailStr
    role: str


class Correspondence(BaseModel):
    text: str
    email: EmailStr


# Custom Metadata
class BaseQuestion(BaseModel):
    # strip whitespace from strings
    model_config = ConfigDict(str_strip_whitespace=True)

    id: Union[UUID4, str] = Field(default_factory=uuid.uuid4, alias="question_id")
    name: Optional[str] = Field(
        validation_alias=AliasChoices("name", "question_name"),
        serialization_alias="name",
    )
    question: str
    options: List[str]
    answer: str
    answer_idx: NonNegativeInt  # do not allow negative index
    tags: Optional[List[str]] = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if v is None or v == "":
            return str(uuid.uuid4())
        elif not isinstance(v, (str, uuid.UUID)):
            logger.error(f"Expected str or UUID4, got {type(v)}")
            raise ValueError("Image ID must be a string or UUID4")

        if not RE_UUID.match(v):
            raise ValueError("Invalid UUID4 format")
        # convert to str
        return str(v)


class DomainQuestion(BaseModel):
    domain: BaseQuestion
    subdomain: BaseQuestion


class ModalityQuestion(BaseModel):
    modality: BaseQuestion
    submodality: BaseQuestion


class BioVLMQuestion(BaseModel):
    classification: Optional[Dict[str, BaseQuestion]] = None
    microscopy_domain: Optional[Domain] = Field(
        default=None,
        validation_alias=AliasChoices("domain", "microscopy_domain"),
        serialization_alias="microscopy_domain",
    )
    microscopy_modality: Optional[Modality] = Field(
        default=None,
        validation_alias=AliasChoices("modality", "microscopy_modality"),
        serialization_alias="microscopy_modality",
    )


# SAInstance
class Attribute(BaseModel):
    name: str
    groupName: str


class BboxPoints(BaseModel):
    # positive float allow zero
    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    x2: PositiveFloat
    y2: PositiveFloat

    @field_validator("x2", "y2")
    @classmethod
    def validate_x2y2(cls, v, info: ValidationInfo):
        # x2 > x1 and y2 > y1
        if info.field_name == "x2" and v <= info.data.get("x1"):
            raise ValueError("x2 must be greater than x1")
        if info.field_name == "y2" and v <= info.data.get("y1"):
            raise ValueError("y2 must be greater than y1")

        return v
