#!/usr/bin/env python3
"""sa_json_dataclass.py in src/custom_datasets."""
import re
import uuid
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import yaml
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
from pydantic import field_validator
from pydantic.functional_validators import AfterValidator
from pydantic.networks import AnyUrl
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import Annotated

# import tiktoken
from src import DATA_ROOT
from src import MODULE_ROOT
from src.conf.compiled_regex import RE_UUID
from src.conf.pydantic_validators import MD5
from src.conf.pydantic_validators import PMID
from src.conf.pydantic_validators import Age
from src.conf.pydantic_validators import Attribute
from src.conf.pydantic_validators import BaseQuestion
from src.conf.pydantic_validators import BboxPoints
from src.conf.pydantic_validators import BioVLMQuestion
from src.conf.pydantic_validators import ClassesToIdx
from src.conf.pydantic_validators import Correspondence
from src.conf.pydantic_validators import Domain
from src.conf.pydantic_validators import LastAction
from src.conf.pydantic_validators import License
from src.conf.pydantic_validators import ListStr
from src.conf.pydantic_validators import Modality
from src.conf.pydantic_validators import OntologyID
from src.conf.pydantic_validators import Split
from src.conf.pydantic_validators import UserDetail
from src.conf.pydantic_validators import _check_input
from src.fileio.text import RE_FILENAME
from src.fileio.text import VALID_EXTENSIONS
from src.fileio.text import is_valid_filename
from src.processing.data_utils import _is_valid_timestamp
from src.processing.data_utils import compute_checksum
from src.processing.data_utils import timestamp
from src.processing.data_utils import unpack_list
from src.processing.text.conversions import str2num


# from sngp_core.conf import DOMAINS as valid_domains
# from sngp_core.conf import MODALITIES as valid_modalities
# from sngp_core.conf import STAINS as valid_stains
# from sngp_core.conf import TASKS as valid_tasks
def compute_tokens(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    raise NotImplementedError
    # encoding = tiktoken.encoding_for_model(model_name)
    # return len(encoding.encode(string))


def _get_instance_tasks(instances: List[dict], supported_tasks: ListStr = None) -> list:
    """Get supported tasks based on instance types."""
    if not instances:
        return supported_tasks or []

    instances = [dict(inst) for inst in instances]
    supported_tasks = supported_tasks or []
    polygon_inst = any(inst.get("type") == "polygon" for inst in instances)
    bbox_inst = any(inst.get("type") == "bbox" for inst in instances)
    centroid_inst = any(inst.get("type") == "point" for inst in instances)
    if polygon_inst:
        supported_tasks += ["semantic", "instance"]

    if bbox_inst:
        supported_tasks += ["bounding_box"]

    if centroid_inst:
        supported_tasks += ["centroid"]

    return supported_tasks


def update_tasks(custom_metadata: dict, instances: Optional[List[dict]] = None) -> list:
    # concatenate into list, then return comma-separated string
    supported_tasks = custom_metadata.get("supported_tasks") or []

    if "caption" in custom_metadata:
        supported_tasks += ["captioning"]

    if instances is not None:
        instances = instances
        supported_tasks = _get_instance_tasks(
            instances, supported_tasks=supported_tasks
        )

    # update classification tasks
    if "label" in custom_metadata:
        supported_tasks += ["multi_class"]

    # remove empty str
    supported_tasks = [task.strip() for task in supported_tasks if task.strip() != ""]
    # set to remove duplicates
    return list(set(supported_tasks))


def update_tags(custom_metadata: dict, tags: list) -> list:
    """Update tags based on custom_metadata."""
    # combine all
    essential_keys = [
        "stain",
        "institution",
        "domain",
        "subdomain",
        "modality",
        "submodality",
        "snomedct_id",
    ]
    # ontology keys for any key with suffix "*_id" and not ["image_id", "patient_id"]

    ontology_ids = [
        key
        for key in custom_metadata.keys()
        if key.endswith("_id") and key not in ["image_id", "patient_id"]
    ]
    # add ontology name keys and their corresponding "*_name" keys (if set)
    ontology_names = [
        key
        for key in custom_metadata.keys()
        if key.endswith("_name")
        and key not in ["dataset_name", "patient_name", "label_name"]
    ]

    # get essential keys if the value is set
    output_tags = [custom_metadata.get(key, "") for key in essential_keys]
    # add ontology ids
    for key in ontology_ids:
        temp = custom_metadata.get(key, "")
        if isinstance(temp, list):
            output_tags += unpack_list(temp)
        else:
            output_tags.append(temp)

    output_tags += [unpack_list(custom_metadata.get(key, "")) for key in ontology_ids]
    # add ontology names
    output_tags += [custom_metadata.get(key, "") for key in ontology_names]
    # add tags from input arg tags
    output_tags += tags if tags else []

    output_tags = unpack_list(output_tags)

    # remove empty str, strip trailing whitespace
    return [elem.strip() for elem in output_tags if elem != ""]


# Metadata
class SAMetadata(BaseModel):
    """Validate the SuperAnnotate json metadata field."""

    height: PositiveInt
    width: PositiveInt
    name: Union[str, Path]  # FilePath
    lastAction: Optional[LastAction] = None
    projectId: Optional[int] = None
    isPredicted: Optional[bool] = False
    status: Optional[str] = "NotStarted"
    pinned: Optional[bool] = False
    annotatorEmail: Optional[EmailStr] = None
    qaEmail: Optional[EmailStr] = None
    format: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        # ensure format is the same as Path(name).suffix
        if Path(self.name).suffix.lower() != self.format.lower():
            logger.warning(
                f"Overwriting format to match filename extension {Path(self.name).suffix}"
            )
            self.format = Path(self.name).suffix.lower()

        # add createdAt and updatedAt if
        if self.createdAt is None:
            self.createdAt = timestamp()
        if self.updatedAt is None:
            self.createdAt = timestamp()

    @field_validator("name", "format")
    @classmethod
    def validate_format(cls, v, info: ValidationInfo) -> str:
        if info.field_name == "name" and not is_valid_filename(v):
            raise ValueError(f"Invalid filename: {v}")
        elif info.field_name == "format" and v is None or v == "":
            filename = info.data.get("name")
            return Path(filename).suffix.lower()
        elif info.field_name == "format":
            # ensure format is a valid extension
            v = f".{v.lower().strip().replace('.', '')}"
            if v not in VALID_EXTENSIONS:
                raise ValueError(
                    f"Invalid format: {v}. Valid extensions are {VALID_EXTENSIONS}"
                )

        return v

    @field_validator("createdAt", "updatedAt")
    @classmethod
    def validate_timestamp(cls, v):
        if v is None:
            return timestamp()  # use current timestamp
        elif not isinstance(v, str):
            raise ValueError("Timestamp must be a string")

        if not _is_valid_timestamp(v):
            raise ValueError(f"Invalid timestamp format {v}")

        return v

    class ConfigDict:
        str_strip_whitespace = (
            True  # Strips whitespace from strings, useful for filenames and emails
        )


# Comment
class SAComment(BaseModel):
    correspondence: List[Correspondence]
    x: PositiveFloat
    y: PositiveFloat
    resolved: bool
    createdAt: Union[NaiveDatetime, AwareDatetime, str]
    createdBy: UserDetail
    creationType: str = Field(
        default="Preannotation", description="Manual or Preannotation"
    )
    updatedAt: Union[NaiveDatetime, AwareDatetime, str]
    updatedBy: UserDetail

    @field_validator("creationType")
    @classmethod
    def validate_creation_type(cls, v):
        if v not in ["Manual", "Preannotation"]:
            raise ValueError("CreationType must be 'Manual' or 'Preannotation'")
        return v

    @field_validator("createdAt", "updatedAt")
    @classmethod
    def validate_timestamp(cls, v):
        if v is None:
            return v
        elif not isinstance(v, str):
            raise ValueError("Timestamp must be a string")

        if not _is_valid_timestamp(v):
            raise ValueError(f"Invalid timestamp format {v}")

    @field_validator("createdBy", "updatedBy")
    @classmethod
    def validate_user_detail(cls, v, info: ValidationInfo):
        # If the objects createdBy and updatedBy exist, then email and role are mandatory.
        if v is None:
            return v

        if not info.data.get("email") or not info.data.get("role"):
            raise ValueError("Email and role are mandatory for createdBy and updatedBy")


PREFIX_MAP = {
    "allenbrain_id": "ALLENBRAIN",
    # "antibody_id",
    "bgee_id": "BGEE",
    "biogrid_id": "BIOGRID",
    "bto_id": "BTO",
    "cellontology_id": "CL",
    "cellosaurus_id": "CVCL",
    "chebi_id": "CHEBI",
    "cmpo_id": "CMPO",
    "ctd_id": "CTD",
    "cvdo_id": "CVDO",
    "diseaseontology_id": "DO",
    "drugbank_id": "DB",
    "efo_id": "EFO",
    "ensemblgene_id": "ENSG",
    "ensemblprotein_id": "ENSP",
    "entrezgene_id": "EG",
    "fma_id": "FMA",
    "go_id": "GO",
    "hpo_id": "HP",
    "icdo_id": "ICDO",
    "icd11_id": "ICD11",
    "icd11_uri": "ICD11URI",
    "icd10_id": "ICD10",
    "icd9_id": "ICD9",
    "icdo_id": "ICDO",
    "kegg_id": "KEGG",
    "loinc_id": "LOINC",
    "medgen_id": "MEDGEN",
    "mesh_id": "MESH",
    "mondo_id": "MONDO",
    "nan": "nan",
    "ncbitaxon_id": "NCBITaxon",
    "ncit_id": "NCIT",
    "nextprot_id": "NX",
    "orphanet_id": "ORPHANET",
    "pato_id": "PATO",
    "reactome_id": "REACTOME",
    "rrid_id": "RRID",
    "snomedct_id": "SCTID",  # renamed from SNOMEDCT
    "uberon_id": "UBERON",
    "umlscui_id": "UMLSCUI",
    "uniprot_id": "UP",
}


def update_prefix(prefix, v, field_name):
    # check for lowercase prefix and update if necessary
    if v in {"na", "nan", "None", "none", "N/A"}:
        return None

    try:
        code = v.split("_")[1]
    except IndexError:
        # logger.warning(f"Incorrect format for {field_name}: {v}")
        v = f"{prefix}_{v}"

    if v.startswith(prefix.lower()):
        v = f"{prefix}_{code}"
    return v


class BioVLMCaption(BaseModel):
    id: Union[UUID4, str] = Field(default_factory=uuid.uuid4, alias="caption_id")
    name: str = None
    caption: str = Field(description="Caption for the image", max_length=500)
    tokens: Optional[NonNegativeInt] = Field(
        default=None,
        validation_alias=AliasChoices("tokens", "num_tokens"),
        serialization_alias="tokens",
        description="GPT 3.5 turbo tokens (enumerated by tiktoken)",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.tokens is None:
            self.tokens = compute_tokens(self.caption, model_name="gpt-3.5-turbo")

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


class SACustomMetadata(BaseModel):
    model_config = ConfigDict(
        extra=Extra.ignore,
        str_strip_whitespace=True,
        alias_generator=AliasGenerator(
            serialization_alias=lambda field_name: field_name.replace("sub_", "sub")
        ),
    )

    # Union[str, float, NonNegativeInt, NonNegativeFloat]
    age: Optional[Age] = Field(
        default=None,
        validation_alias=AliasChoices("age", "patient_age"),
        serialization_alias="age",
    )
    area_fraction: Optional[NonNegativeFloat] = Field(
        default=None,
        validation_alias=AliasChoices("area_fraction", "area"),
        serialization_alias="area_fraction",
    )
    allenbrain_id: Optional[OntologyID] = Field(
        default=None,
        validation_alias=AliasChoices("allenbrain_id", "allen_brain_id"),
        serialization_alias="allenbrain_id",
    )
    allenbrain_acronym: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("allenbrain_acronym", "allenbrain_id_acronym"),
        serialization_alias="allenbrain_acronym",
    )
    allenbrain_name: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("allenbrain_name", "allenbrain_id_name"),
        serialization_alias="allenbrain_name",
    )
    antibody_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("antibody_id", "antibody_rrid"),
        serialization_alias="antibody_id",
    )
    antibody_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("antibody_name", "antibody"),
        serialization_alias="antibody_name",
    )
    bto_id: Optional[OntologyID] = None
    bto_name: Optional[ListStr] = None
    biogrid_id: Optional[ListStr] = None
    biogrid_name: Optional[ListStr] = None
    bgee_id: Optional[ListStr] = None
    bgee_name: Optional[ListStr] = None
    behavior_code: Optional[str] = None
    caption: Optional[List[Union[dict, BioVLMCaption]]] = None
    cellontology_id: Optional[ListStr] = None
    cellontology_name: Optional[ListStr] = None
    cellosaurus_id: Optional[OntologyID] = None
    cellosaurus_name: Optional[ListStr] = None
    classes_to_idx: ClassesToIdx = None
    cmpo_id: Optional[OntologyID] = None
    cmpo_name: Optional[ListStr] = None
    cvdo_id: Optional[Union[str, float, List[str], List[Any]]] = Field(
        default=None,
    )
    cvdo_name: Optional[Union[str, float, List[str], List[Any]]] = Field(
        default=None,
    )
    comment: Optional[str] = None
    ctd_id: Optional[ListStr] = None  # comparative toxicogenomics
    ctd_name: Optional[ListStr] = None
    disease: Optional[str] = None
    diseaseontology_id: Optional[ListStr] = None
    diseaseontology_name: Optional[ListStr] = None
    dataset_name: str = Field(
        description="Name of the dataset.",
        min_length=4,
    )
    dataset_slug: Optional[str] = Field(
        default=None,
        description="Slug for the dataset.",
        min_length=4,
        pattern=r"^[a-z0-9_]+$",
    )
    dataset_parent: Optional[str] = Field(
        default=None,
        description="Parent dataset, if derived from another dataset.",
        min_length=4,
    )
    domain: Domain
    drugbank_id: Optional[ListStr] = None
    drugbank_name: Optional[ListStr] = None
    efo_id: Optional[ListStr] = None
    efo_name: Optional[ListStr] = None
    ensemblgene_id: Optional[ListStr] = None
    ensemblgene_name: Optional[ListStr] = None
    ensemblprotein_id: Optional[ListStr] = None
    ensemblprotein_name: Optional[ListStr] = None
    entrezgene_id: Optional[ListStr] = None
    entrezgene_name: Optional[ListStr] = None
    ethnicity: Optional[str] = None
    filename: Optional[str] = Field(  # TODO: check match with SAMetadata.name
        deafult=None,
        validation_alias=AliasChoices("filename", "image_name", "name"),
        serialization_alias="filename",
        description="Filename of the image, which should match SAMetadata.name.",
    )
    file_size: Optional[NonNegativeInt] = None
    fma_id: Optional[ListStr] = None
    fma_name: Optional[ListStr] = None
    foreground_mask: Optional[str] = Field(  # TODO add RLE string validation
        default=None,
        validation_alias=AliasChoices("foreground_mask", "rle_mask"),
        serialization_alias="foreground_mask",
    )
    gene: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("gene", "gene_name"),
        serialization_alias="gene",
    )
    go_id: Optional[ListStr] = None
    go_name: Optional[ListStr] = None
    hpo_id: Optional[ListStr] = None
    hpo_name: Optional[ListStr] = None
    icdo_id: Optional[ListStr] = None
    icdo_name: Optional[ListStr] = None
    icd9_id: Optional[ListStr] = None
    icd9_name: Optional[ListStr] = None
    icd10_id: Optional[ListStr] = None
    icd10_name: Optional[ListStr] = None
    icd11_id: Optional[ListStr] = None
    icd11_name: Optional[ListStr] = None
    icd11_uri: Optional[ListStr] = None
    instance_count: Optional[Union[NonNegativeInt, str]] = None
    institution: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("institution", "institute"),
        serialization_alias="institution",
    )
    image_id: Union[UUID4, str] = Field(
        validation_alias=AliasChoices("image_id", "image_uuid"),
        serialization_alias="image_id",
    )
    image_md5: Optional[MD5] = Field(
        default=None,
        validation_alias=AliasChoices("image_md5", "image_checksum"),
        serialization_alias="image_md5",
    )
    keywords: Optional[ListStr] = None
    kegg_id: Optional[ListStr] = None
    kegg_name: Optional[ListStr] = None
    label: int = Field(
        gt=-1,
        validation_alias=AliasChoices("label", "label_int"),
        serialization_alias="label",
    )
    label_name: str = Field(
        validation_alias=AliasChoices("label_name", "class_name"),
        serialization_alias="label_name",
    )
    label_subname: Optional[Union[str, float]] = Field(
        default=None,
        validation_alias=AliasChoices("label_subname", "sublabel"),
        serialization_alias="label_subname",
        coerce_numbers_to_str=True,
    )
    label_task: Optional[str] = None
    label_additional_info: Optional[str] = None
    label_description: Optional[str] = None
    label_synonyms: Optional[ListStr] = None
    last_updated: Optional[Union[NaiveDatetime, AwareDatetime, str]] = None
    loinc_id: Optional[ListStr] = None
    loinc_name: Optional[ListStr] = None
    license: License
    medgen_id: Optional[ListStr] = None
    medgen_name: Optional[ListStr] = None
    mesh_id: Optional[ListStr] = None
    mesh_name: Optional[ListStr] = None
    mondo_id: Optional[ListStr] = None  # mondo disease ontology
    mondo_name: Optional[ListStr] = None
    microns_per_pixel: Optional[PositiveFloat] = Field(
        validation_alias=AliasChoices("microns_per_pixel", "mpp"),
        serialization_alias="microns_per_pixel",
    )
    modality: Modality
    # TODO check that len(multilabel) == len(multilabel_names)
    multilabel: Optional[List[NonNegativeFloat]] = None
    multilabel_names: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "multilabel_names", "multilabel_name", "multilabel_labels"
        ),
    )
    ncbitaxon_id: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("ncbitaxon_id", "ncbi_taxon_id"),
        serialization_alias="ncbitaxon_id",
    )
    ncbitaxon_name: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("ncbitaxon_name", "organism"),
        serialization_alias="ncbitaxon_name",
    )
    ncit_id: Optional[ListStr] = None
    ncit_name: Optional[ListStr] = None
    nextprot_id: Optional[ListStr] = None
    nextprot_name: Optional[ListStr] = None
    normal_or_abnormal: Optional[str] = None
    original_filename: Optional[str] = None
    original_labels: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("original_labels", "orig_labels"),
        serialization_alias="original_labels",
    )
    orphanet_id: Optional[ListStr] = None
    orphanet_name: Optional[ListStr] = None
    parent_image_id: Optional[Union[UUID4, str]] = None
    patient_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("patient_id", "participant_id"),
        serialization_alias="patient_id",
        coerce_numbers_to_str=True,
    )
    # phenotype and trait ontology (PATO)
    pato_id: Optional[Union[str, float, List[str]]] = None
    pato_name: Optional[Union[str, float, List[str]]] = None
    pmi: Optional[PositiveFloat] = Field(
        default=None,
        validation_alias=AliasChoices("pmi", "postmortem_interval"),
        serialization_alias="pmi",
    )
    pmid: Optional[PMID] = Field(default=None)
    related_genes: Optional[ListStr] = None
    reactome_id: Optional[ListStr] = None
    reactome_name: Optional[ListStr] = None
    rrid_id: Optional[ListStr] = None
    rrid_name: Optional[ListStr] = None
    sex: Optional[Union[str, float]] = Field(
        default=None,
        validation_alias=AliasChoices("sex", "patient_sex"),
        serialization_alias="sex",
    )
    snomedct_id: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("snomedct_id", "snomed_ct_id", "scit_id"),
        serialization_alias="snomedct_id",
    )
    snomedct_name: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("snomedct_name", "snomed_ct_name", "scit_name"),
        serialization_alias="snomedct_name",
    )
    split: Split = Field(
        validation_alias=AliasChoices("split", "fold"),
        serialization_alias="split",
    )
    specimen_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("specimen_id", "specimen_accession"),
        serialization_alias="specimen_id",
    )
    stain: str = Field(
        validation_alias=AliasChoices("stain", "staining"),
        serialization_alias="stain",
        min_length=2,
    )
    subdomain: str = Field(
        validation_alias=AliasChoices("subdomain", "sub_domain"),
        serialization_alias="subdomain",
    )
    submodality: str = Field(
        validation_alias=AliasChoices("submodality", "sub_modality"),
        serialization_alias="submodality",
    )
    synthetic: bool = False
    tissue: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("tissue", "tissue_organ"),
        serialization_alias="tissue",
    )
    supported_tasks: ListStr = Field(
        validation_alias=AliasChoices("supported_tasks", "tasks", "task"),
        serialization_alias="supported_tasks",
    )
    uberon_id: Optional[ListStr] = Field(default=None)
    uberon_name: Optional[ListStr] = None
    umlscui_id: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("umlscui_id", "snomedct_concept_id"),
        serialization_alias="umlscui_id",
    )
    umlscui_name: Optional[ListStr] = None
    uniprot_id: Optional[ListStr] = None
    uniprot_name: Optional[ListStr] = None
    url: Optional[AnyUrl] = Field(
        default=None, validation_alias=AliasChoices("url", "image_url")
    )
    url_md5: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("url_md5", "url_hash")
    )
    questions: Optional[BioVLMQuestion] = None

    # quantity_location: Optional[str] = None
    # staining_location: Optional[str] = None
    # staining_intensity: Optional[str] = None
    # staining_type: Optional[str] = None
    # @field_validator("age")
    # @classmethod
    # def validate_age(cls, v):
    #     # must be positive number < 125
    #     if isinstance(v, str):
    #         v = str2num(v)
    #
    #     if v is not None and (v < 0 or v > 125):
    #         raise ValueError("Age must be a positive number less than 125.")
    #     return v

    #
    @field_validator("instance_count")
    @classmethod
    def validate_instance_count(cls, v) -> Optional[NonNegativeInt]:
        if v is None or v == "" or not v:
            return None
        elif isinstance(v, str):
            v = str2num(v)

        if not isinstance(v, (int, float)):
            raise ValueError("Instance count must be a number")
        elif v < 0:
            raise ValueError("Instance count must be a positive number")

        return int(v)

    @field_validator("cvdo_name", "cvdo_id")
    @classmethod
    def validate_cvdo(cls, v):
        # return None if empty, "", or None
        if v is None or v == "" or not v:
            return None

        if isinstance(v, str):
            return [v]
        elif isinstance(v, list):
            return v

        # return none if inf/nan
        if np.isinf(v) or np.isnan(v):
            return None

    @field_validator("split")
    @classmethod
    def validate_split(cls, v):
        # must be lowercase and in train, test, or validation
        v = v.lower()
        if v in {"holdout", "hold-out", "hold_out"}:
            # rename holdout to test
            v = "test"

        if v not in ["train", "test", "validation"]:
            raise ValueError("Split must be one of ['train', 'test', 'validation']")

        return v

    @field_validator("label_synonyms", "keywords")
    @classmethod
    def keep_unique(cls, v: ListStr, info: ValidationInfo):
        v = _check_input(v)
        if v is None or v == "" or not v:
            return None

        # remove empty str, nan, None
        v = [elem.strip() for elem in v if elem.strip() != ""]
        v = [elem for elem in v if elem not in {"na", "nan", "None", "none", "N/A"}]

        # expecting list of str, convert to set to remove duplicates
        return list(set(v))

    @field_validator(
        "domain",
        "license",
        "modality",
        "stain",
        "subdomain",
        "submodality",
        "microns_per_pixel",
    )
    @classmethod
    def convert_list_to_elem(cls, v):
        v = _check_input(v)
        if v is None or v == "" or not v:
            return None

        if isinstance(v, list):
            # won't work for list of str or list of list
            return v[0] if len(v) == 1 else ",".join(v)

        return v

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, v):
        # test if valid uuid4 format
        if not isinstance(v, str):
            raise ValueError("Image ID must be a string")

        if not RE_UUID.match(v):
            raise ValueError("Invalid UUID4 format")
        # convert to str
        return str(v)

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        # must be any item in the dict
        mapping_dict = {"m": "male", "f": "female", "u": "unknown", "o": "other"}
        # hack
        if v is None or v == "" or not v:
            return None

        # hack
        if isinstance(v, str):
            if v in {"na", "nan", "None", "none", "N/A"}:
                return None
        elif isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return None
        elif isinstance(v, list):
            # remove "nan" elem from list
            v = [elem for elem in v if elem not in {"na", "nan", "None", "none", "N/A"}]

        # hack
        if not v:
            return None

        # get map to full name
        v = mapping_dict.get(v.lower(), v) if len(v) == 1 else v.lower()

        if v in mapping_dict.values():
            return v
        else:
            raise ValueError(
                f"Invalid sex value: {v}. Must be one of {mapping_dict.keys()}"
            )

    # @field_validator("domain")  # "subdomain"
    # @classmethod
    # def validate_domain(cls, v):
    #     if v not in list(valid_domains.keys()):
    #         raise ValueError(
    #             f"Invalid domain: {v}. Must be one of {list(valid_domains.keys())}."
    #         )
    #     return v

    # @field_validator("modality")  # "submodality"
    # @classmethod
    # def validate_modality(cls, v):
    #     valid_modalities = list(valid_modalities.keys())
    #     valid_modalities += ["other", "unknown", "mixed(light,fluorescence)", "mixed"]
    #     if v not in valid_modalities:
    #         raise ValueError(
    #             f"Invalid modality: {v}. Must be one of {valid_modalities}."
    #         )
    #     return v

    @field_validator("last_updated")  # "createdAt"
    @classmethod
    def validate_timestamp(cls, v):
        if v is None:
            return v
        elif not isinstance(v, str):
            raise ValueError("Timestamp must be a string")

        if not _is_valid_timestamp(v):
            raise ValueError(f"Invalid timestamp format {v}")

        return v

    @field_validator("image_md5")
    @classmethod
    def validate_image_md5(cls, v, info: ValidationInfo):
        if v is None or v == "":
            # will attempt to compute image_md5 in SADict post-init
            return None

        # must be a 32 character hex string
        if not re.match(r"^[a-f0-9]{32}$", v):
            raise ValueError("Invalid image_md5 format")
        elif v == "d41d8cd98f00b204e9800998ecf8427e":
            logger.error(f"Empty image_md5 for {info.data.get('image_id')}")

        return v

    # TODO: is this still needed?
    @field_validator("supported_tasks", "ncit_id", check_fields=False)
    @classmethod
    def validate_list_str(cls, v, info: ValidationInfo):
        if v is None or v == "" or not v:
            return None

        if isinstance(v, str):
            v = v.split(",")
        elif not isinstance(v, list):
            raise ValueError("Tasks must be a list of strings")

        if info.field_name == "supported_tasks":
            # remove "classification" if "multi_class" is present
            if "multi_class" in v:
                v = [elem for elem in v if elem != "classification"]

            # remove empty str
            v = [elem.strip() for elem in v if elem.strip() != ""]
            # sort
            v = sorted(list(set(v)))

        return v

    @field_validator(
        "allenbrain_id",
        "antibody_id",
        "bgee_id",
        "biogrid_id",
        "bto_id",
        "cellontology_id",
        "cellosaurus_id",
        "cmpo_id",
        "ctd_id",
        "cvdo_id",
        "diseaseontology_id",
        "drugbank_id",
        "efo_id",
        "ensemblgene_id",
        "ensemblprotein_id",
        "entrezgene_id",
        "fma_id",
        "go_id",
        "hpo_id",
        "icd10_id",
        "icd9_id",
        "icdo_id",
        "kegg_id",
        "loinc_id",
        "medgen_id",
        "mesh_id",
        "mondo_id",
        "ncbitaxon_id",
        "ncit_id",
        "nextprot_id",
        "orphanet_id",
        "pato_id",
        "reactome_id",
        "rrid_id",
        "snomedct_id",
        "uberon_id",
        "umlscui_id",
        "uniprot_id",
        check_fields=False,
    )
    @classmethod
    def validate_ontology_id(cls, v, info: ValidationInfo) -> Union[list, None]:
        if isinstance(v, str):
            if v in {"na", "nan", "None", "none", "N/A"}:
                return None
            elif "," in v:
                v = v.split(",")
            else:
                v = [v]
        elif isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return None
            else:
                v = [str(v)]
        elif isinstance(v, list):
            # remove "nan" elem from list
            v = [elem for elem in v if elem not in {"na", "nan", "None", "none", "N/A"}]

        # hack
        if v is None or v == "" or not v:
            return None

        prefix = PREFIX_MAP.get(info.field_name)
        if prefix is None or prefix == "nan":
            logger.warning(f"Error getting prefix for {info.field_name}")
            return None

        # process list
        output_list = []
        for elem in v:
            if ":" in elem and "_" not in elem:
                elem = elem.replace(":", "_")

            if info.field_name in {"allenbrain_id"} and "[" in elem:
                # if str contains list brackets "[" only keep text within brackets
                # "ALLENBRAIN_['ALLENBRAIN_12139']" -> "ALLENBRAIN_12139"
                elem = re.match(r".*['(](.*)[')]", elem)
                if elem is None:
                    logger.warning(f"Error processing {info.field_name} with {v}")
                    continue

                elem = elem.group(1)

            if isinstance(elem, str) and elem.count("_") > 1:
                # only have one _ in the string
                raise ValueError(
                    f"Only one underscore allowed in {info.field_name}: {v}"
                )

            if info.field_name in {"snomedct_id"}:
                elem = elem.replace("SNOMEDCT", prefix)

            elem = update_prefix(prefix, elem, info.field_name)
            if elem is None:
                logger.warning(f"Error getting prefix for {info.field_name} with {v}")
                continue

            if not elem.startswith(prefix):
                logger.error(f"{info.field_name} invalid format: {elem}")
                logger.error(f"Expected format for {v}: {prefix}_{{id}}")
                raise ValueError(f"{info.field_name} must start with {prefix}_")

            output_list.append(elem)

        return output_list

    @field_validator(
        "allenbrain_name",
        "bto_name",
        "cellontology_name",
        "cellosaurus_name",
        "cmpo_name",
        "drubgbank_name",
        "efo_name",
        "fma_name",
        "go_name",
        "hpo_name",
        "loinc_name",
        "medgen_name",
        "mesh_name",
        "ncbitaxon_name",
        "ncit_name",
        "nextprot_name",
        "orphanet_name",
        "pato_name",
        "reactome_name",
        "rrid_name",
        "snomedct_name",
        "uberon_name",
        "umls_name",
        "umlscui_name",
        check_fields=False,
    )
    @classmethod
    def validate_ontology_name(cls, v, info: ValidationInfo) -> list:
        v = _check_input(v)
        if v is None or v == "" or not v:
            return None

        if v == ["None"] or v == ["nan"]:
            return None

        # convert to list
        if isinstance(v, str):
            return [v] if v not in {"na", "nan", "None", "none", "N/A"} else None
        elif isinstance(v, list):
            v = [elem for elem in v if elem not in {"na", "nan", "None", "none", "N/A"}]
            return v if v else None
        else:
            raise ValueError(f"{info.field_name} must be a string or list of strings")

    # @field_validator("license")
    # @classmethod
    # def validate_license(cls, v):
    #     # must be str
    #     if not isinstance(v, str):
    #         raise ValueError("License must be a string")
    #     elif "CC" in v:
    #         v = v.strip().replace(" ", "-")
    #     else:
    #         v = v.strip()
    #
    #     # regex for CC-BY license
    #     if RE_CCBY.match(v):
    #         return v
    #
    #     valid_licenses = [
    #         "CC0-1.0",
    #         "Public Domain",
    #         "Proprietary",
    #         "Other",
    #         "Unknown",
    #         "Non-commercial",
    #     ]
    #     valid_licenses = [elem.lower() for elem in valid_licenses]
    #     if v.lower() not in valid_licenses:
    #         logger.error(f"Invalid license: {v}")
    #         logger.error(f"Expected one of CC-BY or {valid_licenses}")
    #         raise ValueError(f"Expected: CC-BY {valid_licenses} Actual: {v}")
    #
    #     return v


class SAInstance(BaseModel):
    id: Union[UUID4, str]
    type: str = Field(
        default="polygon",
        description="Must be one of ['point', 'bbox', 'polygon', 'polyline', 'ellipse']",
    )
    classId: Optional[int] = Field(default=-1)
    className: str = Field(default="")
    probability: Union[NonNegativeFloat, NonNegativeInt] = Field(
        default=100, ge=0, le=100
    )
    points: Optional[Union[List[NonNegativeFloat], dict]] = None
    x: Optional[NonNegativeFloat] = None
    y: Optional[NonNegativeFloat] = None
    groupId: Optional[int] = Field(default=0)
    pointLabels: Dict[str, str] = Field(default_factory=dict)
    attributes: List[Attribute] = Field(default_factory=list)
    error: Optional[bool] = None
    locked: bool = Field(default=False)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if v is None or v == "":
            return str(uuid.uuid4())
        elif not isinstance(v, (str, uuid.UUID)):
            logger.error(f"Expected str or UUID4, got {type(v)}")
            raise ValueError("Image ID must be a string or UUID4")

        return str(v)

    @field_validator("points")
    @classmethod
    def validate_points(cls, v, info: ValidationInfo):
        # preprocess points, if necessary
        if info.data.get("type") == "point":
            # points is not used for point type
            return None
        elif info.data.get("type") == "bbox":
            # bbox should be dict with keys x1, y1, x2, y2
            if not isinstance(v, dict):
                raise ValueError("Bbox points must be a dictionary")

            if any(k not in v for k in ["x1", "y1", "x2", "y2"]):
                raise ValueError("Bbox points must have keys x1, y1, x2, y2")

            # pass through BboxPoints for validation then return it
            return dict(BboxPoints(**v))
        elif info.data.get("type") == "polygon":
            # check list
            if not isinstance(v, list):
                raise ValueError("Polygon points must be a list of floats")

            # check if points is a list of floats
            if not all(isinstance(p, (int, float)) for p in v):
                raise ValueError("Polygon points must be a list of floats")

            # check even number of points
            if len(v) % 2 != 0:
                raise ValueError("Polygon points must have an even number of elements")

            return v
        elif info.data.get("type") in ["polyline", "ellipse"]:
            raise NotImplementedError(
                f"Type {info.data.get('type')} is not yet supported"
            )
        else:
            logger.error(f"Info: {info.data}")
            logger.error(f"Value: {v}")
            raise ValueError("Invalid type")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v not in ["point", "bbox", "polygon"]:
            raise ValueError("type must be 'point', 'bbox', or 'polygon'")
        return v

    @field_validator("probability", "classId", check_fields=False)
    @classmethod
    def check_numeric_values(cls, v, info: ValidationInfo):
        if info.field_name == "probability" and not (0 <= v <= 100):
            raise ValueError("Probability must be between 0 and 100")
        if info.field_name == "classId" and (v is not None and v < -1):
            raise ValueError("classId must be -1 or a positive integer")
        return int(v) if v is not None else v

    @field_validator("className", "pointLabels")
    @classmethod
    def check_empty_strings(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("String fields should not be empty or just whitespace")
        return v

    @field_validator("attributes")
    @classmethod
    def validate_attributes(cls, v, info: ValidationInfo):
        for attr in v:
            if not attr.name or not attr.groupName:
                raise ValueError("Attribute name and groupName must not be empty")

        return v


class SADict(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_encoders={Path: str},
    )

    metadata: SAMetadata
    comments: Optional[List[SAComment]] = []
    custom_metadata: SACustomMetadata
    instances: Optional[List[Union[SAInstance, dict]]] = []
    tags: Optional[ListStr] = []

    def model_post_init(self, __context: Any) -> None:
        # setup
        file_updated = False
        orig_custom_metadata = self.custom_metadata.model_dump(
            exclude_none=True, exclude_unset=True
        ).copy()
        split = self.custom_metadata.split
        if self.custom_metadata.dataset_slug is None:
            self.custom_metadata.dataset_slug = (
                self.custom_metadata.dataset_name.lower().replace(" ", "_")
            )

        dataset_slug = self.custom_metadata.dataset_slug
        # assume filepath DATA_ROOT/{dataset_name}/{split}/{file.png|json}
        image_filepath = Path(DATA_ROOT).joinpath(
            dataset_slug, split, self.metadata.name
        )
        json_filepath = image_filepath.with_suffix(".json")
        debug = self.custom_metadata.dataset_name.lower() in {"pytest", "debug"}

        # check if image file exists
        if not debug and not image_filepath.is_file():
            logger.warning(f"Image file not found: {image_filepath}")
            # raise FileNotFoundError(f"Image file not found: {image_filepath}")

        # check "split-{}" in filename matches custom_metadata.split
        filename_split = RE_FILENAME.match(self.metadata.name).group("split")
        if filename_split != split:
            # rename image filename to match custom_metadata.split
            new_filename = self.metadata.name.replace(filename_split, split)
            new_filepath = image_filepath.with_name(new_filename)
            try:
                if not debug and image_filepath.is_file():
                    logger.info(f"Renaming {image_filepath} to {new_filepath}")
                    # image_filepath.rename(new_filepath)
                if not debug and json_filepath.is_file():
                    logger.info(
                        f"Renaming {json_filepath} to {new_filepath.with_suffix('.json')}"
                    )
                    # json_filepath.rename(new_filepath.with_suffix(".json"))

                # check image_filepath and json_filepath name equal
                if image_filepath.name != new_filepath.name:
                    error_msg = f"Image and JSON filepaths do not match: {image_filepath} != {new_filepath}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # update metadata name if successful
                self.metadata.name = Path(new_filename)
                file_updated = True
            except Exception as e:
                error_msg = f"Error renaming {image_filepath} to {new_filepath}:\n\t{e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        # check file_size
        if not debug and not self.custom_metadata.file_size:
            # update file_size
            self.custom_metadata.file_size = image_filepath.stat().st_size

        # compute image checksum if not provided
        if not debug and not self.custom_metadata.image_md5:
            # logger.warning(f"No checksum provided for {image_filepath}. Computing...")
            image_md5 = compute_checksum(image_filepath, method="md5")
            self.custom_metadata.image_md5 = image_md5
            self.custom_metadata.file_size = image_filepath.stat().st_size

        if not debug and not self.custom_metadata.file_size:
            # update file_size
            self.custom_metadata.file_size = image_filepath.stat().st_size

        # update supported_tasks based on custom_metadata keys
        custom_metadata = self.custom_metadata.model_dump(
            exclude_unset=True, exclude_none=True
        )
        supported_tasks = update_tasks(custom_metadata, self.instances)
        self.custom_metadata.supported_tasks = supported_tasks

        # update tags based on custom_metadata keys
        tags = update_tags(custom_metadata, tags=self.tags)
        self.tags = tags

        # convert instances to list of dicts
        if self.instances:
            self.instances = [dict(inst) for inst in self.instances]

        # update last_updated timestamp if any changes were made
        file_updated = file_updated or orig_custom_metadata != custom_metadata
        if file_updated or not self.custom_metadata.last_updated:
            self.custom_metadata.last_updated = timestamp()

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: ListStr):
        v = _check_input(v)
        if v is None or v == "" or not v:
            return None

        # expecting list of str, convert to set to remove duplicates
        return list(set(v))

        # if isinstance(v, str):
        #     return v.split(",")
        # elif isinstance(v, list):
        #     return v
        # else:
        #     raise ValueError("Tags must be a list of strings")

    @field_validator("metadata")
    @classmethod
    def validate_split(cls, v, info: ValidationInfo):
        # ensure metadata.name 'split-{}' matches custom_metadata.split
        if not info.data.get("custom_metadata"):
            return v

        # metadata.name and custom_metadata.split already
        # validated in SAMetadata and SACustomMetadata
        filename = info.data.get("metadata").get("name")
        filename_split = RE_FILENAME.match(filename).group("split")
        metadata_split = info.data.get("custom_metadata").get("split")

        if filename_split != metadata_split:
            raise ValueError(
                f"Filename split {filename_split} does not match metadata split {metadata_split}"
            )

    # TODO: validate instances
    # bbox: x2y2 < metadata.width, metadata.height
    # polygon: even number of points, x, y < metadata.width, metadata.height
    # point: x, y < metadata.width, metadata.height
