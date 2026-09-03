from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from loguru import logger

from src.data.base_image_datamodule import BaseImageDataModule
from src.data.components.hf_dataset import HFDataset, _apply_transform

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A policy shipped inside histo-artifact-sim, resolved by name. `load_config` treats the
#: value as a filesystem path when one exists and as a builtin name otherwise, so a
#: generated policy file still works here.
DEFAULT_ARTIFACT_CONFIG = "balanced"

#: Acevedo's native patch edge -- the simulator runs on the raw image, before the
#: datamodule's resize/crop. Only consulted when coverage is derived from the bank
#: (`coverage="assets"`); under the default `fixed` scaling the policy's quartiles are
#: taken literally at whatever size the patch happens to be.
DEFAULT_REFERENCE_PATCH_PX = 360

#: Assets are 40x throughout, but only the alpha/ filenames say so; the paired/ ones carry
#: tile coordinates instead. Without this the policy calibration sees mixed magnifications
#: and warns that the derived coverage mixes scales.
ASSET_MAGNIFICATION = "40x"


def build_artifact_pipeline(
    assets_root: str,
    config_spec: str = DEFAULT_ARTIFACT_CONFIG,
    seed: int = 42,
    taxonomy_csv: Optional[str] = None,
    patch_magnification: float = 40.0,
    reference_patch_px: int = DEFAULT_REFERENCE_PATCH_PX,
    coverage: str = "config",
    weights: str = "config",
    coverage_scaling: Optional[str] = "fixed",
    on_missing_category: str = "error",
):
    """Build the artifact simulator from an asset bank.

    `from_assets` indexes the bank and caches the measurements outside the project tree
    (under `$HISTO_ARTIFACTS_CACHE`), so there is no manifest CSV to keep in sync -- adding
    or removing an asset invalidates the cache on its own.

    `coverage` and `weights` choose whether artifact *size* and *frequency* come from the
    bank or from the named policy. Both default to the policy here, which keeps the
    shipped `balanced` quartiles rather than re-deriving them.

    `coverage_scaling="fixed"` is the reason this does not just call `from_assets` and
    stop. Coverage is normally a fraction of a *field of view*: the assets are 40x cutouts
    measured against a 1024px patch, so on Acevedo's 360px patch the same fragment covers
    8.09x as much, which puts `tissue_component` at 433% and `bone_calcification` at 375%
    -- total occlusion for several of eleven equally-weighted categories, and no longer a
    robustness test of anything. `fixed` takes the quartiles literally at whatever size
    the patch is, keeping artifacts visible without swallowing the cell. Deriving coverage
    from the bank instead (`coverage="assets"`) does not avoid this: it folds the same
    factor into the quartiles themselves. Pass `coverage_scaling=None` to keep whatever
    the policy declares.
    """
    try:
        from histo_artifacts import ArtifactPipeline
    except ImportError as exc:
        raise ImportError(
            "histo_artifacts is required for artifact simulation. Install it before using this module."
        ) from exc

    pipeline = ArtifactPipeline.from_assets(
        assets_root,
        config=config_spec,
        seed=seed,
        patch_magnification=patch_magnification,
        coverage=coverage,
        weights=weights,
        reference_patch_px=reference_patch_px,
        taxonomy_csv=taxonomy_csv,
        default_magnification=ASSET_MAGNIFICATION,
        # Loud by default: a configured category with no assets loses its weight and
        # silently changes the realized distribution.
        on_missing_category=on_missing_category,
    )

    if coverage_scaling is not None and pipeline.config.coverage_scaling != coverage_scaling:
        # Rebuilt rather than mutated: SimulationConfig is frozen, and the catalog is
        # reused so this costs nothing -- the indexing already happened above.
        pipeline = ArtifactPipeline(
            pipeline.catalog,
            replace(pipeline.config, coverage_scaling=coverage_scaling),
            seed=seed,
            patch_magnification=patch_magnification,
            on_missing_category=on_missing_category,
        )

    provenance = pipeline.provenance
    logger.info(
        "Artifact simulator ready: config={} config_id={} manifest_id={} "
        "assets={} coverage_scaling={} patch_magnification={}x reference_patch_px={}",
        config_spec,
        provenance["config_id"],
        provenance["manifest_id"],
        provenance["catalog_size"],
        provenance["coverage_scaling"],
        provenance["patch_magnification"],
        provenance["reference_patch_px"],
    )
    logger.debug("Artifact category counts: {}", provenance["category_counts"])
    return pipeline


def _transform_with_mask(image, mask, transform):
    """Transform a simulated image and its artifact mask, keeping the two aligned.

    Albumentations co-transforms the pair, so spatial ops apply to both. A torchvision
    `Compose` cannot -- but the ones used here (and `BaseImageDataModule`'s own default)
    are photometric, leaving the geometry alone, so the untouched mask still lines up.
    That is checked rather than assumed: a torchvision transform that does resize or crop
    produces a shape mismatch and an error, never a silently misaligned mask.
    """
    if transform is None or isinstance(transform, A.Compose):
        image_out, mask_out = _apply_transform(image, transform, mask=mask, return_mask=True)
    else:
        image_out, mask_out = _apply_transform(image, transform), mask

    mask_out = torch.as_tensor(np.asarray(mask_out))
    image_shape = tuple(image_out.shape[-2:])
    if tuple(mask_out.shape[-2:]) != image_shape:
        raise ValueError(
            f"Artifact mask {tuple(mask_out.shape)} does not line up with the transformed "
            f"image {tuple(image_out.shape)}. {type(transform).__name__} changes the image "
            f"geometry but cannot co-transform a mask -- use an albumentations Compose for "
            f"the test augmentations."
        )
    return image_out, mask_out


class ArtifactHFDataset(Dataset):
    """Pairs each real image with an artifact-simulated version from `simulator`."""

    def __init__(self, hf_dataset, simulator, transform=None, fold=None, seed: int = 42):
        self.dataset = hf_dataset
        self.simulator = simulator
        self.transform = transform
        self.fold = fold
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        image_id = item.get("image_id", idx)

        # Seed per sample, derived from content rather than position. Without this every
        # forked DataLoader worker replays the simulator's single RNG stream, so images at
        # the same within-worker offset get identical artifacts. Keying on image_id also
        # keeps a sample's artifacts stable regardless of worker count or shard order.
        from histo_artifacts.seeding import derive_seed

        simulated = self.simulator(image, seed=derive_seed(self.seed, str(image_id)))

        real_image = _apply_transform(image, self.transform)
        artifact_simulated_image, artifact_mask = _transform_with_mask(
            simulated["image"], simulated["artifact_mask"], self.transform
        )

        return {
            "image_id": image_id,
            "real_image": real_image,
            "artifact_simulated_image": artifact_simulated_image,
            # Localized artifacts only. Whole-frame effects (illumination, pixelation,
            # brightness/contrast) modify the image but are reported in artifact_labels,
            # never merged into the mask.
            "artifact_mask": artifact_mask,
            "artifact_labels": torch.from_numpy(simulated["artifact_labels"]),
            "target": label,
            "fold": self.fold,
        }


class ArtifactImageDataModule(BaseImageDataModule):
    """`LightningDataModule` that pairs real images with artifact-simulated versions
    for the test/predict stage. Train/val always use plain (unsimulated) images.
    """

    def __init__(
        self,
        dataset_name: str,
        artifact_bank_dir: str,
        num_classes: int = 8,
        class_to_idx: Optional[Dict[str, int]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_augmentations: Optional[transforms.Compose | A.Compose] = None,
        val_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_augmentations: Optional[transforms.Compose | A.Compose] = None,
        test_all_folds: Optional[bool] = False,
        sample_rate: int = 0,
        institution: Optional[str] = None,
        artifact_config_path: str = DEFAULT_ARTIFACT_CONFIG,
        artifact_taxonomy_csv: Optional[str] = None,
        artifact_seed: int = 42,
        artifact_patch_magnification: float = 40.0,
        artifact_reference_patch_px: int = DEFAULT_REFERENCE_PATCH_PX,
        artifact_coverage: str = "config",
        artifact_weights: str = "config",
        artifact_coverage_scaling: Optional[str] = "fixed",
        artifact_on_missing_category: str = "error",
        simulate_artifacts_for_test: bool = True,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            num_classes=num_classes,
            class_to_idx=class_to_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            test_all_folds=test_all_folds,
            sample_rate=sample_rate,
            institution=institution,
        )
        self.artifact_bank_dir = artifact_bank_dir
        self.artifact_config_path = artifact_config_path
        self.artifact_taxonomy_csv = artifact_taxonomy_csv
        self.artifact_seed = artifact_seed
        self.artifact_patch_magnification = artifact_patch_magnification
        self.artifact_reference_patch_px = artifact_reference_patch_px
        self.artifact_coverage = artifact_coverage
        self.artifact_weights = artifact_weights
        self.artifact_coverage_scaling = artifact_coverage_scaling
        self.artifact_on_missing_category = artifact_on_missing_category
        self.simulate_artifacts_for_test = simulate_artifacts_for_test
        self.artifact_pipeline = None
        #: Label space of `artifact_labels`: policy categories, then `global:*` effects.
        self.artifact_label_names: Optional[tuple[str, ...]] = None

    def _make_dataset(self, hf_split, fold: str, transform) -> HFDataset:
        # Train/val never see artifact simulation, regardless of simulate_artifacts_for_test.
        return HFDataset(hf_split, transform=transform, fold=fold)

    def _make_test_dataset(self, hf_split, fold: str, transform) -> Dataset:
        if not self.simulate_artifacts_for_test:
            return HFDataset(hf_split, transform=transform, fold=fold)

        # Built once, in the parent process, so forked workers inherit a ready pipeline
        # rather than each indexing the bank.
        if self.artifact_pipeline is None:
            self.artifact_pipeline = build_artifact_pipeline(
                assets_root=self.artifact_bank_dir,
                config_spec=self.artifact_config_path,
                seed=self.artifact_seed,
                taxonomy_csv=self.artifact_taxonomy_csv,
                patch_magnification=self.artifact_patch_magnification,
                reference_patch_px=self.artifact_reference_patch_px,
                coverage=self.artifact_coverage,
                weights=self.artifact_weights,
                coverage_scaling=self.artifact_coverage_scaling,
                on_missing_category=self.artifact_on_missing_category,
            )
            self.artifact_label_names = self.artifact_pipeline.label_names

        return ArtifactHFDataset(
            hf_split,
            simulator=self.artifact_pipeline,
            transform=transform,
            fold=fold,
            seed=self.artifact_seed,
        )
