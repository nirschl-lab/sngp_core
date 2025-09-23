#!/usr/bin/env python3
"""base_metadata.py in src/sngp_core/conf.

Pydantic class for minimum metadata requirements for a dataset yaml/json.
"""
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Union

from loguru import logger
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Extra
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import alias_generators
from pydantic import field_validator
from pydantic.networks import AnyUrl
from pydantic_core.core_schema import ValidationInfo

from src.conf.pydantic_validators import PMID
from src.conf.pydantic_validators import ClassesToIdx
from src.conf.pydantic_validators import ImageMeanStd
from src.conf.pydantic_validators import ImageShape
from src.conf.pydantic_validators import License
from src.conf.pydantic_validators import ListInt
from src.conf.pydantic_validators import ListLicense
from src.conf.pydantic_validators import ListStr
from src.conf.pydantic_validators import ListURL
from src.conf.pydantic_validators import ValidURL
from src.conf.pydantic_validators import _check_input


class BioVLMLabelMetadata(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_encoders={Path: str},
    )

    # Metadata fields
    # TODO
    #   class_1:
    #     age: null
    #     antibody_id: null
    #     antibody_name: null
    #     bto_id: null # recommended
    #     bto_name: null
    #     cellontology_id: null
    #     cellontology_name: null
    #     cellosaurus_id: null # recommended
    #     cellosaurus_name: null
    #     cmpo_id: null
    #     cmpo_name: null
    #     disease: null
    #     efo_id: null # recommended
    #     efo_name: null
    #     ensembl_id: null
    #     ensembl_name: null
    #     ethnicity: null
    #     gene: null
    #     go_id: null # recommended
    #     go_name: null
    #     hpo_id: null # recommended
    #     hpo_name: null
    #     icdo_id: null
    #     icdo_name: null
    #     icd10_id: null
    #     icd10_name: null
    #     fma_id: null
    #     fma_name: null
    #     label_additional_info: null
    #     label_description: null
    #     label_name: null # required
    #     label_subname: null
    #     label_synonyms: null
    #     loinc_id: null
    #     loinc_name: null
    #     mesh_id: null
    #     mesh_name: null
    #     medgen_id: null
    #     medgen_name: null
    #     multilabel: null
    #     ncbitaxon_id: null
    #     ncbitaxon_name: null
    #     normal_or_abnormal: null
    #     ncit_id: null
    #     pato_id: null
    #     pato_name: null
    #     related_genes: null
    #     snomedct_id: null
    #     snomedct_name: null
    #     specimen_id: null
    #     synthetic: false
    #     tissue: null
    #     uberon_id: null
    #     uberon_name: null
    #     umlscui_id: null
    #     umlscui_name: null
    #     uniprot_id: null
    #     uniprot_name: null


class BioVLMDatasetMetadata(BaseModel):
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
    copyright_year: Optional[ListInt] = Field(
        default=None, description="Year of copyright."
    )
    # dataset name, no reserved characters or words, should be filepath safe
    dataset_name: str = Field(
        description="Name of the dataset.",
        min_length=4,
    )
    # must be lowercase and have underscores replacing spaces
    dataset_slug: Optional[str] = Field(
        default=None,
        description="Snake case version of the dataset name.",
        min_length=4,
    )
    # dataset name, no reserved characters or words, should be filepath safe
    dataset_url: Optional[ValidURL] = Field(
        default=None, description="URL to the dataset."
    )
    description: Optional[str] = Field(
        default=None, description="Description of the dataset."
    )
    doi: Optional[ListStr] = Field(default=None, description="DOI for the dataset(s).")
    domain: ListStr = Field(
        description="Domain (field of study) of the dataset.",
        validation_alias=AliasChoices("domain", "field_of_study"),
        serialization_alias="domain",
    )
    extension: str = Field(default="json")
    homepage: Optional[Union[ListURL, Any]] = Field(
        default=None, description="URL to the dataset homepage(s)."
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
    label_task: Optional[str] = Field(
        default=None, description="Description of the original task of the dataset."
    )
    label_metadata: Optional[dict] = None
    language: Optional[Union[str, ListStr]] = Field(
        default="en",
        description="Language of the dataset: one of 'en', 'es', 'fr', 'de'.",
    )
    license: Union[ListLicense, License] = Field(description="License for the dataset.")
    microns_per_pixel: Optional[Union[List[PositiveFloat], PositiveFloat]] = Field(
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
    ncbitaxon_id: Optional[ListStr] = Field(
        default=None,
        validation_alias=AliasChoices("ncbitaxon_id", "ncbi_taxon_id"),
        serialization_alias="ncbitaxon_id",
        # pattern=r"^NCBITaxon_\d+$",
    )
    ncbitaxon_name: Optional[ListStr] = Field(
        default=None,
        description="NCBI Taxonomy name.",
        validation_alias=AliasChoices("ncbitaxon_name", "ncbi_taxon_name"),
        serialization_alias="ncbitaxon_name",
    )
    ncit_id: Optional[ListStr] = Field(
        default=None,
        description="NCIT ID.",
        # , pattern=r"^NCIT_C\d+$"
    )
    ncit_name: Optional[ListStr] = Field(default=None, description="NCIT name.")
    num_classes: PositiveInt = Field(
        default=None, description="Number of classes in the dataset."
    )
    num_total: Optional[PositiveInt] = Field(
        default=None, description="Number of total samples."
    )
    num_train: Optional[PositiveInt] = Field(
        default=None, description="Number of training samples."
    )
    num_validation: Optional[PositiveInt] = Field(
        default=None, description="Number of validation samples."
    )
    num_test: Optional[PositiveInt] = Field(
        default=None, description="Number of test samples."
    )
    pmid: Optional[Union[List[PMID], PMID]] = Field(
        # 1-8 digit number with no leading zeros
        default=None,
        description="PubMed ID.",
        # pattern=r"^\d{1,8}$",
    )
    pretty_classes_to_idx: Optional[ClassesToIdx] = Field(
        default=None,
        description="Mapping of class names to integer indices with pretty class names.",
    )
    pretty_dataset_name: Optional[str] = Field(
        default=None,
        description="Pretty name of the dataset.",
    )
    snomedct_id: Optional[ListStr] = Field(
        default=None,
        description="SNOMED CT ID.",
        # , pattern=r"^SNOMEDCT_\d+$"
    )
    snomedct_name: Optional[ListStr] = Field(
        default=None, description="SNOMED CT name."
    )
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
    supported_tasks: Union[ListStr] = Field(description="Supported tasks.")
    synthetic: Optional[bool] = Field(
        default=None, description="Whether the dataset is synthetic."
    )
    uberon_id: Optional[ListStr] = Field(
        default=None,
        description="Uberon ID.",
        # , pattern=r"^UBERON_\d+$"
    )
    uberon_name: Optional[ListStr] = Field(default=None, description="Uberon name.")
    version: Optional[str] = Field(
        default="0.0.1",
        description="Version of the dataset.",
        pattern=r"^\d+\.\d+\.\d+$",
    )
    version_comment: Optional[str] = Field(
        default=None, description="Comment on the dataset version."
    )

    def model_post_init(self, __context: Any) -> None:
        # setup pretty_dataset_name, pretty_classes_to_idx
        if self.pretty_dataset_name is None:
            self.pretty_dataset_name = self.dataset_name.replace("_", " ").title()

        if self.pretty_classes_to_idx is None:
            self.pretty_classes_to_idx = {
                k.replace("_", " ").title(): v for k, v in self.classes_to_idx.items()
            }

        # # for any attr with suffix "_name", remove elements from list if None, "na", "None", "none", "N/A"
        # for key, value in self.__dict__.items():
        #     # if key == "ncit_name":
        #     #     key
        #
        #     if key.endswith("_name") and isinstance(value, list):
        #         self.__dict__[key] = [
        #             elem
        #             for elem in value
        #             if elem not in {"na", "nan", "None", "none", "N/A"}
        #         ]
        #         # if empty list, set to None
        #         if not self.__dict__[key]:
        #             self.__dict__[key] = None

    @field_validator("bibtex")
    @classmethod
    def validate_bibtex(cls, v):
        v = _check_input(v)
        if v is None or v == "" or not v:
            return None

        if isinstance(v, str):
            v = [v]

        if not isinstance(v, list):
            raise ValueError("BibTeX must be a list of strings.")

        return v

    @field_validator("dataset_slug")
    @classmethod
    def validate_dataset_slug(cls, v, info: ValidationInfo):
        # generate slug from dataset name if not provided
        v = _check_input(v)
        if v is None and info.data.get("dataset_name") is not None:
            v = info.data.get("dataset_name").lower().replace(" ", "_")

        # check that the slug matches the dataset name
        expected_slug = info.data.get("dataset_name").lower().replace(" ", "_")
        if v.lower() != expected_slug:
            err_msg = f"Dataset slug {v} does not match dataset name {info.data['dataset_name']}."
            logger.warning(err_msg)
            raise ValueError(err_msg)

        return v

    # @field_validator("pretty_classes_to_idx")
    # @classmethod
    # def validate_pretty_classes_to_idx(cls, v, info: ValidationInfo):
    #     # generate pretty_classes_to_idx from classes_to_idx if not provided
    #     v = _check_input(v)
    #     if v is None and info.data.get("classes_to_idx") is not None:
    #         v = info.data.get("classes_to_idx")
    #         v = {k.replace("_", " ").title(): v for k, v in v.items()}
    #     else:
    #         raise ValueError("classes_to_idx is required for pretty_classes_to_idx.")
    #
    #     return v

    @field_validator("license")
    @classmethod
    def license_to_list(cls, v):
        v = _check_input(v)
        if isinstance(v, str):
            v = [v]
        return v

    @field_validator("microns_per_pixel")
    @classmethod
    def microns_per_pixel_to_list(cls, v):
        if isinstance(v, float):
            v = [v]

        return v

    @field_validator("image_size")
    @classmethod
    def image_size_to_tuple(cls, v) -> ImageShape:
        # should be List[List[PositiveInt]]

        return v

    @field_validator("keywords")
    @classmethod
    def keep_unique(cls, v: ListStr):
        v = _check_input(v)
        if v is None or v == "" or not v:
            return None

        # remove empty str, nan, None
        v = [elem.strip() for elem in v if elem.strip() != ""]
        v = [elem for elem in v if elem not in {"na", "nan", "None", "none", "N/A"}]

        # expecting list of str, convert to set to remove duplicates
        return list(set(v))
