"""CLI to dump paired real/artifact-simulated image samples + metadata to disk.

    uv run scripts/artifact/save_simulated_artifacts.py --num-images 20 --output-path <dir>
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

from src.data.artifact_image_datamodule import (
    DEFAULT_ARTIFACT_CONFIG,
    DEFAULT_ARTIFACT_CSV,
    REPO_ROOT,
    build_artifact_pipeline,
)

DEFAULT_OUTPUT_PATH = "/data1/maheswararao/artifact_sim/dataloader"


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
    dataset_name: Optional[str] = None,
    csv_path: str = DEFAULT_ARTIFACT_CSV,
    config_path: str = DEFAULT_ARTIFACT_CONFIG,
    seed: int = 42,
) -> None:
    if dataset_name is None:
        yaml_file_path = REPO_ROOT / "configs" / "paths" / "default.yaml"
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        dataset_name = data["data_cache_dir"]

    simulator = build_artifact_pipeline(csv_path=csv_path, config_path=config_path, seed=seed)
    dataset = datasets.load_dataset(dataset_name)["test"]

    output_dir = Path(output_path)
    real_dir = output_dir / "real"
    artifact_dir = output_dir / "artifact"
    metadata_dir = output_dir / "metadata"
    for directory in (real_dir, artifact_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    saved_images = 0
    for index in range(len(dataset)):
        if saved_images >= num_images:
            break

        item = dataset[index]
        real_image = item["image"]
        artifact_applied = True
        try:
            simulated = simulator(real_image)
        except ValueError as exc:
            if "Artifact mask is empty" in str(exc):
                logger.warning(
                    "Artifact mask missing/empty for image {}. Saving original image as artifact fallback.",
                    index,
                )
                simulated = {"image": real_image, "metadata": {}}
                artifact_applied = False
            else:
                raise
        image_id = str(item.get("image_id", index)).replace("/", "_")

        _to_pil_image(real_image).save(real_dir / f"{image_id}.png")
        _to_pil_image(simulated["image"]).save(artifact_dir / f"{image_id}.png")

        metadata = dict(simulated.get("metadata", {}))
        metadata["image_id"] = image_id
        metadata["target"] = item.get("label")
        metadata["artifact_applied"] = artifact_applied
        with open(metadata_dir / f"{image_id}.json", "w") as file:
            json.dump(metadata, file, indent=2, default=str)

        saved_images += 1

    print(f"Saved {saved_images} simulated samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save simulated artifact samples to disk.")
    parser.add_argument("--num-images", "-n", type=int, help="Number of simulated samples to save.")
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Directory where the simulated outputs will be written. Default: {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()

    save_simulated_artifacts(num_images=args.num_images, output_path=args.output_path)
