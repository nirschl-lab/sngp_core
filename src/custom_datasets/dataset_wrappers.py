#!/usr/bin/env python3
"""dataset_wrappers.py in src/argusdp/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/dataset_wrappers.py
"""
# sourcery skip: upper-camel-case-classes, no-loop-in-tests, no-conditionals-in-tests
__all__ = [
    "SADatasetWrapper",
    "SAInstanceDatasetWrapper",
    "Acevedo_et_al_2020Wrapper",
    "BRAVURAWrapper",
    "BioVLMImageFolderWrapper",
    "Burgess_et_al_2024_ContourWrapper",
    "Burgess_et_al_2024_EccentricityWrapper",
    "Burgess_et_al_2024_TextureWrapper",
    "Colocalization_BenchmarkWrapper",
    "DTDWrapper",
    "EMPIAR_SBFSEMWrapper",
    # "Eulenberg_et_al_2017AllWrapper",
    "Eulenberg_et_al_2017_BrightfieldWrapper",
    "Eulenberg_et_al_2017_DarkfieldWrapper",
    "Eulenberg_et_al_2017_EpifluorescenceWrapper",
    "FGVCAircraftWrapper",
    "Flowers102Wrapper",
    "Food101Wrapper",
    "Held_et_al_2010_GalTWrapper",
    "Held_et_al_2010_H2BWrapper",
    "Held_et_al_2010_MTWrapper",
    "Hussain_et_al_2019Wrapper",
    "ICPR2020_PollenWrapper",
    "ImageNet100Wrapper",
    "Jung_et_al_2022Wrapper",
    "Kather_et_al_2016Wrapper",
    "Kather_et_al_2018Wrapper",
    "Kather_et_al_2018_Val7KWrapper",
    "Nirschl_et_al_2018Wrapper",
    "Nirschl_Unpub_FluorescenceWrapper",
    "OxfordIIITPetWrapper",
    "Places365Wrapper",
    "SVHNWrapper",
    "StanfordCarsWrapper",
    "Tang_et_al_2019Wrapper",
    "Wong_et_al_2022Wrapper",
    "Wu_et_al_2023Wrapper",
    # "BioVLMDataFrameWrapper",
]

from pathlib import Path
from typing import Callable
from typing import Optional
import pdb

from dotenv import find_dotenv
from dotenv import load_dotenv
from torchvision.datasets import DTD
from torchvision.datasets import SVHN
from torchvision.datasets import FGVCAircraft
from torchvision.datasets import Flowers102
from torchvision.datasets import Food101
from torchvision.datasets import ImageFolder
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets import Places365
from torchvision.datasets import StanfordCars
from torchvision.datasets import VisionDataset

from src import DATA_ROOT
from src.custom_datasets.biovlm_image_folder import BioVLMImageFolder
from src.custom_datasets.sa_dataset import SADataset
from src.custom_datasets.sa_instance_dataset import SAInstanceDataset
import pdb

# from argusdp.custom_datasets.biovlm_dataframe import BioVLMDataframe

RANDOM_SEED = 8675309

load_dotenv(find_dotenv())
load_dotenv(override=True)

# pdb.set_trace()

class BaseDataWrapper:

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        self.name: str = self.__class__.__name__.replace("Wrapper", "").lower()
        self.root: Path = self._check_root(root)
        self.kwargs: dict = kwargs

    def _check_root(self, root: str) -> Path:
        # pdb.set_trace()
        if not root or root is None:
            root = Path(DATA_ROOT).joinpath(self.name)

        root = Path(root)
        if Path(root).is_absolute() and Path(root).stem != self.name:
            root = root.joinpath(self.name)
        elif not root.is_absolute():
            root = DATA_ROOT.joinpath(root).joinpath(self.name)

        if not root.exists():
            raise FileNotFoundError(f"Directory {root} not found.")

        return root


class SADatasetWrapper:
    @staticmethod
    def __call__(
        dataset_name: str,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:
        if not root:
            root = Path(DATA_ROOT).joinpath(dataset_name)
        elif Path(root).is_absolute():
            root = Path(root).joinpath(dataset_name)
        else:
            root = DATA_ROOT.joinpath(root).joinpath(dataset_name)

        return SADataset(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
            **kwargs,
        )


class SAInstanceDatasetWrapper:
    @staticmethod
    def __call__(
        dataset_name: str,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SAInstanceDataset:
        if not root:
            root = Path(DATA_ROOT).joinpath(dataset_name)
        elif Path(root).is_absolute():
            root = Path(root).joinpath(dataset_name)
        else:
            root = DATA_ROOT.joinpath(root).joinpath(dataset_name)

        return SAInstanceDataset(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
            **kwargs,
        )


class Acevedo_et_al_2020Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
            **kwargs,
        )


# class BioVLMDataFrameWrapper:
#     @staticmethod
#     def __call__(
#         root: str = None,
#         split: str = "train",
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         random_seed: Optional[int] = RANDOM_SEED,
#         **kwargs: Optional[dict],
#     ) -> VisionDataset:
#         if not root:
#             root = DATA_ROOT
#         elif not Path(root).is_absolute():
#             root = DATA_ROOT.joinpath(root)
#
#         return BioVLMDataFrame(
#             root,
#             split=split,
#             transform=transform,
#             target_transform=target_transform,
#             random_seed=random_seed,
#         )


class BioVLMImageFolderWrapper:
    @staticmethod
    def __call__(
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> BioVLMImageFolder:
        if not root:
            root = DATA_ROOT
        elif not Path(root).is_absolute():
            root = DATA_ROOT.joinpath(root)

        return BioVLMImageFolder(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            random_seed=random_seed,
        )


class BRAVURAWrapper:
    @staticmethod
    def __call__(
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        raise NotImplementedError("BRAVURA dataset is not implemented yet.")


class Burgess_et_al_2024_ContourWrapper(BaseDataWrapper):
    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:
        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Burgess_et_al_2024_EccentricityWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Burgess_et_al_2024_TextureWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Colocalization_BenchmarkWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class EMPIAR_SBFSEMWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Eulenberg_et_al_2017_BrightfieldWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Eulenberg_et_al_2017_DarkfieldWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Eulenberg_et_al_2017_EpifluorescenceWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Jung_et_al_2022Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Held_et_al_2010_GalTWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Held_et_al_2010_H2BWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Held_et_al_2010_MTWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Hussain_et_al_2019Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class ICPR2020_PollenWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Kather_et_al_2016Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Kather_et_al_2018Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Kather_et_al_2018_Val7KWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Nirschl_et_al_2018Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:
        # if not root:
        #     root = DATA_ROOT.joinpath("nirschl_et_al_2018")
        # elif not Path(root).is_absolute():
        #     root = DATA_ROOT.joinpath(root)

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Nirschl_Unpub_FluorescenceWrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:
        # if not root:
        #     root = DATA_ROOT.joinpath("nirschl_et_al_2018")
        # elif not Path(root).is_absolute():
        #     root = DATA_ROOT.joinpath(root)

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Tang_et_al_2019Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:
        # if not root:
        #     root = DATA_ROOT.joinpath("tang_et_al_2019")
        # elif not Path(root).is_absolute():
        #     root = DATA_ROOT.joinpath(root)

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Wong_et_al_2022Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:
        # if not root:
        #     root = DATA_ROOT.joinpath("wong_et_al_2022")
        # elif not Path(root).is_absolute():
        #     root = DATA_ROOT.joinpath(root)

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


class Wu_et_al_2023Wrapper(BaseDataWrapper):

    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_ext: str = "png",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> SADataset:
        # if not root:
        #     root = DATA_ROOT.joinpath("wong_et_al_2022")
        # elif not Path(root).is_absolute():
        #     root = DATA_ROOT.joinpath(root)

        return SADataset(
            self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            image_ext=image_ext,
            random_seed=random_seed,
        )


##### General computer vision datasets ######
# structure copied from:
# https://github.com/sanketx/AL-foundation-models
class Food101Wrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        return Food101(root, split, transform, download=download)


class StanfordCarsWrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        return StanfordCars(root, split, transform, download=download)


class FGVCAircraftWrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        return FGVCAircraft(root, split, transform=transform, download=download)


class DTDWrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        return DTD(root, split, partition=1, transform=transform, download=download)


class OxfordIIITPetWrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        return OxfordIIITPet(
            root, split, target_types="category", transform=transform, download=download
        )


class Flowers102Wrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        return Flowers102(root, split, transform, download=download)


class SVHNWrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        return SVHN(root, split, transform, download=download)


# class DomainNetRealWrapper:
#     @staticmethod
#     def __call__(
#         root: str,
#         train: bool,
#         transform: Optional[transforms.Compose] = None,
#         download: bool = False,
#     ) -> VisionDataset:
#         root = os.path.join(root, "domainnet_real")
#         file = "real_train.txt" if train else "real_test.txt"
#         file = os.path.join(root, file)
#         return CustomImageFolder(root, file, transform=transform)


class ImageNet100Wrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        root = Path(root).joinpath("imagenet100", split)
        return ImageFolder(root, transform=transform)


class Places365Wrapper:
    @staticmethod
    def __call__(
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> VisionDataset:
        if download:  # Check if image archive already extracted
            try:
                Places365(
                    root, split, small=True, transform=transform, download=download
                )
            except RuntimeError:
                download = False

        return Places365(
            root, split, small=True, transform=transform, download=download
        )
