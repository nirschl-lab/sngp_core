"""CLI to export per-institution, per-class sample images plus both artifact axes.

Picks `--per-class` images for every (institution, class) cell of one split, then for
each one saves the real image and the two artifact axes that
scripts/inference/run_artifact_axes.sh evaluates:

- config axis:     artifact_balanced, procedural off, exact overlay count = level
- procedural axis: artifacts off, procedural_ood, severity = level (histo_c ladder)

Each image is seeded with `derive_seed(seed, image_id)`, the key ArtifactImageDataModule
uses, so the saved images are exactly what a model sees during artifact inference. One
long-format metadata.csv (one row per PNG) maps every file back to its source image.

    uv run scripts/artifact/export_presentation_samples.py --output-path <dir>
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import datasets
import numpy as np
import pandas as pd
import rootutils
from loguru import logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.artifact.save_simulated_artifacts import DEFAULT_TAXONOMY_CSV, _to_pil_image
from src.data.artifact_image_datamodule import build_artifact_pipeline

DEFAULT_DATASET_NAME = "nirschl-lab/wong_et_al_2022"
DEFAULT_OUTPUT_PATH = "/data1/maheswararao/experiments/uncertainty-aware-ml/publishing_data/adrc"
CONFIG_AXIS_POLICY = "artifact_balanced"
PROCEDURAL_AXIS_POLICY = "procedural_ood"


def _select_samples(split_ds: datasets.Dataset, per_class: int, seed: int) -> pd.DataFrame:
    """Seeded random `per_class` rows per (institution, class), read without decoding images."""
    columns = split_ds.select_columns(["institution", "label"]).to_pandas()
    columns["dataset_index"] = np.arange(len(columns))
    rng = np.random.default_rng(seed)
    picks = []
    for (institution, label), cell in sorted(columns.groupby(["institution", "label"])):
        if len(cell) < per_class:
            logger.warning("{} / label {} has only {} images", institution, label, len(cell))
        chosen = rng.choice(cell["dataset_index"].to_numpy(), size=min(per_class, len(cell)), replace=False)
        for number, index in enumerate(sorted(chosen), start=1):
            picks.append({"institution": institution, "label": int(label), "number": number, "dataset_index": int(index)})
    return pd.DataFrame(picks)


def export_presentation_samples(
    output_path: str = DEFAULT_OUTPUT_PATH,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "test",
    per_class: int = 5,
    levels: Optional[List[int]] = None,
    assets_root: Optional[str] = None,
    seed: int = 42,
    patch_magnification: float = 40.0,
) -> None:
    levels = levels or [1, 2, 3, 4, 5]
    assets_root = assets_root or os.environ.get("HISTO_ARTIFACTS_BANK")
    if not assets_root:
        raise SystemExit("No asset bank given. Pass --assets-root or set HISTO_ARTIFACTS_BANK.")
    taxonomy = str(DEFAULT_TAXONOMY_CSV) if DEFAULT_TAXONOMY_CSV.is_file() else None

    from histo_artifacts.seeding import derive_seed

    common = dict(assets_root=assets_root, seed=seed, taxonomy_csv=taxonomy, patch_magnification=patch_magnification)
    # (axis, level, folder, pipeline) -- one pipeline per arm, built once.
    arms = [
        ("config", level, Path("config_axis") / f"count_{level}",
         build_artifact_pipeline(config_spec=CONFIG_AXIS_POLICY, procedural="none", count=level, **common))
        for level in levels
    ] + [
        ("procedural", level, Path("procedural_axis") / f"severity_{level}",
         build_artifact_pipeline(config_spec="none", procedural=PROCEDURAL_AXIS_POLICY, severity=level, **common))
        for level in levels
    ]

    split_ds = datasets.load_dataset(dataset_name)[split]
    class_names = split_ds.features["label"].names
    samples = _select_samples(split_ds, per_class, seed)
    logger.info("Selected {} images from {} [{}]", len(samples), dataset_name, split)

    output_dir = Path(output_path)
    rows = []
    for sample in samples.itertuples(index=False):
        item = split_ds[sample.dataset_index]
        class_name = class_names[sample.label]
        sample_id = f"{sample.institution}_{class_name}_{sample.number:02d}"
        image_id = str(item.get("image_id", sample.dataset_index))
        image_seed = derive_seed(seed, image_id)
        relative = Path(sample.institution) / class_name / f"{sample_id}.png"
        base = {
            "sample_id": sample_id,
            "institution": sample.institution,
            "class_name": class_name,
            "label": sample.label,
            "image_id": image_id,
            "original_filename": item.get("original_filename"),
            "split": split,
            "dataset_index": sample.dataset_index,
        }

        real = _to_pil_image(item["image"]).convert("RGB")
        real_array = np.asarray(real)
        (output_dir / "real" / relative).parent.mkdir(parents=True, exist_ok=True)
        real.save(output_dir / "real" / relative)
        rows.append({**base, "axis": "real", "level": 0, "file_path": str(Path("real") / relative), "seed": None})

        for axis, level, folder, pipeline in arms:
            simulated = pipeline(real, seed=image_seed)
            operations = simulated["metadata"]["operations"]
            path = output_dir / folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _to_pil_image(simulated["image"]).save(path)
            rows.append({
                **base,
                "axis": axis,
                "level": level,
                "file_path": str(folder / relative),
                "seed": image_seed,
                "overlay_categories": ";".join(op["parent_category"] for op in operations if op["kind"] == "overlay"),
                "procedural_effects": ";".join(op["name"] for op in operations if op["kind"] == "procedural"),
                "artifact_coverage_pct": float((np.asarray(simulated["artifact_mask"]) > 0).mean() * 100.0),
                "no_effect": bool(np.array_equal(simulated["image"], real_array)),
            })
        logger.info("Saved {} ({})", sample_id, image_id)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    logger.success("Saved {} images and metadata.csv to {}", len(metadata), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export per-institution/class samples with both artifact axes.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-class", type=int, default=5, help="Images per (institution, class) cell.")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Counts / severities.")
    parser.add_argument("--assets-root", default=None, help="Artifact asset bank (default: $HISTO_ARTIFACTS_BANK)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-magnification", type=float, default=40.0)
    args = parser.parse_args()

    export_presentation_samples(
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        split=args.split,
        per_class=args.per_class,
        levels=args.levels,
        assets_root=args.assets_root,
        seed=args.seed,
        patch_magnification=args.patch_magnification,
    )
