"""CLI to dump paired real/artifact-simulated image samples + metadata to disk.

    uv run scripts/artifact/save_simulated_artifacts.py --num-images 20 --output-path <dir>
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import rootutils
import torch
from loguru import logger
from PIL import Image

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.artifact_image_datamodule import (
    DEFAULT_ARTIFACT_CONFIG,
    DEFAULT_REFERENCE_PATCH_PX,
    REPO_ROOT,
    build_artifact_pipeline,
)

DEFAULT_OUTPUT_PATH = "/data1/maheswararao/artifact_sim/dataloader"

#: Matches configs/data/artifact_image_classifier.yaml, so what this dumps is what the
#: artifact dataloader will feed a model.
DEFAULT_DATASET_NAME = "nirschl-lab/acevedo_et_al_2020"
DEFAULT_TAXONOMY_CSV = REPO_ROOT / "data" / "artifact" / "artifact_taxonomy_by_image.csv"


def _to_pil_image(image):
    if isinstance(image, Image.Image):
        return image

    if torch.is_tensor(image):
        array = image.detach().cpu()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = array.permute(1, 2, 0)
        array = array.numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array.squeeze(-1)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)

    if array.dtype != np.uint8:
        if array.size and array.min() >= 0 and array.max() <= 1:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    return Image.fromarray(array)


def save_simulated_artifacts(
    num_images: int,
    output_path: str = DEFAULT_OUTPUT_PATH,
    dataset_name: str = DEFAULT_DATASET_NAME,
    assets_root: Optional[str] = None,
    config_path: str = DEFAULT_ARTIFACT_CONFIG,
    taxonomy_csv: Optional[str] = None,
    seed: int = 42,
    patch_magnification: float = 40.0,
    reference_patch_px: int = DEFAULT_REFERENCE_PATCH_PX,
) -> None:
    assets_root = assets_root or os.environ.get("HISTO_ARTIFACTS_BANK")
    if not assets_root:
        raise SystemExit(
            "No asset bank given. Pass --assets-root, or set HISTO_ARTIFACTS_BANK in the "
            "environment or in this project's .env (see env_example)."
        )

    taxonomy = taxonomy_csv or (str(DEFAULT_TAXONOMY_CSV) if DEFAULT_TAXONOMY_CSV.is_file() else None)
    if taxonomy is None:
        logger.warning(
            "No taxonomy CSV at {}; categories will come from the bank's folder layout, which "
            "leaves 17% of assets unreachable. Generate one with "
            "scripts/artifact/build_taxonomy.py.",
            DEFAULT_TAXONOMY_CSV,
        )

    simulator = build_artifact_pipeline(
        assets_root=assets_root,
        config_spec=config_path,
        seed=seed,
        taxonomy_csv=taxonomy,
        patch_magnification=patch_magnification,
        reference_patch_px=reference_patch_px,
    )
    dataset = datasets.load_dataset(dataset_name)["test"]

    output_dir = Path(output_path)
    real_dir = output_dir / "real"
    artifact_dir = output_dir / "artifact"
    mask_dir = output_dir / "mask"
    metadata_dir = output_dir / "metadata"
    for directory in (real_dir, artifact_dir, mask_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    from histo_artifacts.seeding import derive_seed

    saved_images = 0
    for index in range(len(dataset)):
        if saved_images >= num_images:
            break

        item = dataset[index]
        real_image = item["image"]
        image_id = str(item.get("image_id", index)).replace("/", "_")

        # Same derivation the dataloader uses, so a given image_id gets the same
        # artifacts here as it will during inference.
        simulated = simulator(real_image, seed=derive_seed(seed, image_id))

        _to_pil_image(real_image).save(real_dir / f"{image_id}.png")
        _to_pil_image(simulated["image"]).save(artifact_dir / f"{image_id}.png")
        # Scaled to 0/255 so the mask is visible in an image viewer.
        mask = (np.asarray(simulated["artifact_mask"]) > 0).astype(np.uint8) * 255
        Image.fromarray(mask).save(mask_dir / f"{image_id}.png")

        metadata = dict(simulated.get("metadata", {}))
        metadata["image_id"] = image_id
        metadata["target"] = item.get("label")
        metadata["artifact_labels"] = {
            name: float(value)
            for name, value in zip(simulator.label_names, simulated["artifact_labels"])
        }
        metadata["mask_coverage_percent"] = float(mask.mean() / 255.0 * 100.0)
        with open(metadata_dir / f"{image_id}.json", "w") as file:
            json.dump(metadata, file, indent=2, default=str)

        saved_images += 1

    logger.success("Saved {} simulated samples to {}", saved_images, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save simulated artifact samples to disk.")
    parser.add_argument("--num-images", "-n", type=int, help="Number of simulated samples to save.")
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Directory where the simulated outputs will be written. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"HuggingFace dataset to draw test images from. Default: {DEFAULT_DATASET_NAME}",
    )
    parser.add_argument(
        "--assets-root",
        default=None,
        help="Artifact asset bank (default: $HISTO_ARTIFACTS_BANK)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_ARTIFACT_CONFIG,
        help=f"Sampling policy: a builtin name or a path. Default: {DEFAULT_ARTIFACT_CONFIG}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-magnification", type=float, default=40.0)
    parser.add_argument("--reference-patch-px", type=int, default=DEFAULT_REFERENCE_PATCH_PX)
    args = parser.parse_args()

    save_simulated_artifacts(
        num_images=args.num_images,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        assets_root=args.assets_root,
        config_path=args.config,
        seed=args.seed,
        patch_magnification=args.patch_magnification,
        reference_patch_px=args.reference_patch_px,
    )
