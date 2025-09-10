#!/usr/bin/env python3
"""dataset_registry.py in src/argusdp/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/registry.py
"""

from enum import Enum

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

# import all dataset wrappers
from src.custom_datasets.dataset_wrappers import *
import pdb

class DatasetType(Enum):
    """Enum of supported Datasets."""

    acevedo_et_al_2020 = Acevedo_et_al_2020Wrapper()
    # pdb.set_trace()
    # # bravura = BRAVURAWrapper()
    # burgess_et_al_2024_contour = Burgess_et_al_2024_ContourWrapper()
    # burgess_et_al_2024_eccentricity = Burgess_et_al_2024_EccentricityWrapper()
    # burgess_et_al_2024_texture = Burgess_et_al_2024_TextureWrapper()
    # # cifar10 = CIFAR10
    # # cifar100 = CIFAR100
    # colocalization_benchmark = Colocalization_BenchmarkWrapper()
    # # dataframe = BioVLMDataFrameWrapper()
    # # dtd = DTDWrapper()
    # empiar_sbfsem = EMPIAR_SBFSEMWrapper()
    # eulenberg_et_al_2017_brightfield = Eulenberg_et_al_2017_BrightfieldWrapper()
    # eulenberg_et_al_2017_darkfield = Eulenberg_et_al_2017_DarkfieldWrapper()
    # eulenberg_et_al_2017_epifluorescence = Eulenberg_et_al_2017_EpifluorescenceWrapper()
    # # fgvcaircraft = FGVCAircraftWrapper()
    # # flowers102 = Flowers102Wrapper()
    # # food101 = Food101Wrapper()
    # held_et_al_2010_galt = Held_et_al_2010_GalTWrapper()
    # held_et_al_2010_h2b = Held_et_al_2010_H2BWrapper()
    # held_et_al_2010_mt = Held_et_al_2010_MTWrapper()
    # hussain_et_al_2019 = Hussain_et_al_2019Wrapper()
    # icpr2020_pollen = ICPR2020_PollenWrapper()
    # image_folder = BioVLMImageFolderWrapper()
    # # imagenet100 = ImageNet100Wrapper()
    # jung_et_al_2022 = Jung_et_al_2022Wrapper()
    # kather_et_al_2016 = Kather_et_al_2016Wrapper()
    # kather_et_al_2018 = Kather_et_al_2018Wrapper()
    # kather_et_al_2018_val7k = Kather_et_al_2018_Val7KWrapper()
    # nirschl_et_al_2018 = Nirschl_et_al_2018Wrapper()
    # nirschl_unpub_fluorescence = Nirschl_Unpub_FluorescenceWrapper()
    # # oxford_iiit_pet = OxfordIIITPetWrapper()
    # # places365 = Places365Wrapper()
    # sa_dataset = SADatasetWrapper()
    # sainstance_dataset = SAInstanceDatasetWrapper()
    # # stanford_cars = StanfordCarsWrapper()
    # # svhn = SVHNWrapper()
    tang_et_al_2019 = Tang_et_al_2019Wrapper()
    # wong_et_al_2022 = Wong_et_al_2022Wrapper()
    # wu_et_al_2023 = Wu_et_al_2023Wrapper()
