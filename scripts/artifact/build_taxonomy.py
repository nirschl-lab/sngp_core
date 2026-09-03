"""Generate the `image_name`-keyed taxonomy that categorises the artifact asset bank.

    uv run scripts/artifact/build_taxonomy.py

The bank at $HISTO_ARTIFACTS_BANK is laid out flat: `alpha/<parent_category>/*.png` with
no sub-category folders, plus `paired/marker/` and `paired/hair/`. Two things go wrong if
the folder layout is taken at face value, which is what `histo-artifacts` does without a
taxonomy:

- `marker` and `hair` are not categories any shipped policy mentions, so 107 assets (17%
  of the bank) can never be sampled. They are ink and fibre respectively.
- `stain_artifact` and `processing_material` are weighted by every policy but have no
  folder of their own; their assets sit inside other folders and are only distinguishable
  by filename.

`img_names_categorized.csv` already resolves both, per asset, for the whole `alpha/` tree.
This script joins it with folder-derived rows for the two `paired/` categories and writes
the result where `ArtifactPipeline.from_assets(taxonomy_csv=...)` can consume it.

The output is a derived local artifact -- `data/artifact/` is gitignored, since it
describes a bank that lives outside the repo. Regenerate it rather than copying it.
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = REPO_ROOT / "data" / "artifact" / "img_names_categorized.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "artifact" / "artifact_taxonomy_by_image.csv"

OUTPUT_COLUMNS = ["image_name", "parent_category", "sub_category"]

#: `paired/<folder>` -> the policy category its assets actually belong to.
PAIRED_CATEGORY_MAP = {
    "marker": "pigment_ink",
    "hair": "fiber_hair",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

#: The distribution this reconstruction must reproduce. These are the counts of the
#: 611-row taxonomy the histo-artifact-sim repo carried at rev bc9a5a08, before the file
#: was removed from it. Asserting them is what makes a silently wrong join impossible.
EXPECTED_COUNTS = {
    "debris": 329,
    "pigment_ink": 123,
    "fiber_hair": 45,
    "stain_artifact": 30,
    "bone_calcification": 20,
    "tissue_component": 17,
    "other_artifact": 17,
    "processing_material": 13,
    "bubble": 11,
    "focus": 3,
    "scratch_glass": 3,
}


def _bank_root(explicit: Optional[str] = None) -> Path:
    """Resolve the asset bank, preferring an explicit path over $HISTO_ARTIFACTS_BANK."""
    raw = explicit or os.environ.get("HISTO_ARTIFACTS_BANK")
    if not raw:
        raise SystemExit(
            "No asset bank given. Pass --assets-root, or set HISTO_ARTIFACTS_BANK in the "
            "environment or in this project's .env (see env_example)."
        )
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Asset bank is not a directory: {root}")
    return root.resolve()


def _read_alpha_rows(source_csv: Path) -> list[dict[str, str]]:
    """Rows for the `alpha/` tree, already categorised per asset."""
    if not source_csv.is_file():
        raise SystemExit(
            f"{source_csv} is missing. It is the only source of the stain_artifact and "
            f"processing_material labels and of every sub_category, and it cannot be "
            f"regenerated from this repo or from the asset bank -- copy it from a machine "
            f"that has it."
        )

    with source_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = {"image_name", "parent_category", "sub_category"} - set(columns)
        if missing:
            raise SystemExit(
                f"{source_csv} is missing column(s) {sorted(missing)}. Found: "
                f"{', '.join(columns) or '(no header)'}."
            )
        return [
            {
                "image_name": row["image_name"].strip(),
                "parent_category": row["parent_category"].strip(),
                "sub_category": row["sub_category"].strip(),
            }
            for row in reader
            if row.get("image_name", "").strip()
        ]


def _read_paired_rows(bank_root: Path) -> list[dict[str, str]]:
    """Rows for `paired/<folder>/images/`, remapped onto the policy's categories.

    The folder name becomes the sub_category, so the stratified sampler still tells these
    apart from the alpha assets now sharing their parent category.
    """
    rows: list[dict[str, str]] = []
    for folder, parent_category in PAIRED_CATEGORY_MAP.items():
        image_dir = bank_root / "paired" / folder / "images"
        if not image_dir.is_dir():
            raise SystemExit(f"Expected paired image directory is missing: {image_dir}")
        names = sorted(
            path.name for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not names:
            raise SystemExit(f"No images found in {image_dir}")
        logger.info("paired/{}: {} assets -> {}", folder, len(names), parent_category)
        rows.extend(
            {"image_name": name, "parent_category": parent_category, "sub_category": folder}
            for name in names
        )
    return rows


def _verify(rows: list[dict[str, str]], *, strict: bool) -> None:
    """Fail on a duplicate key or a distribution that does not match the known bank."""
    duplicates = [name for name, count in Counter(r["image_name"] for r in rows).items() if count > 1]
    if duplicates:
        raise SystemExit(
            f"{len(duplicates)} duplicate image_name value(s), which would make the taxonomy "
            f"ambiguous: {', '.join(sorted(duplicates)[:5])}"
        )

    counts = dict(Counter(row["parent_category"] for row in rows))
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        logger.info("  {:<20} {}", name, count)

    if counts != EXPECTED_COUNTS:
        message = (
            f"Category distribution does not match the expected bank composition.\n"
            f"  expected: {EXPECTED_COUNTS}\n"
            f"  got:      {counts}\n"
            f"The bank has changed. Re-derive the expectation deliberately rather than "
            f"loosening it, then update EXPECTED_COUNTS."
        )
        if strict:
            raise SystemExit(message)
        logger.warning(message)


def build_taxonomy(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    source_csv: Path = DEFAULT_SOURCE_CSV,
    assets_root: Optional[str] = None,
    strict: bool = True,
) -> int:
    """Write the taxonomy CSV and return its row count."""
    bank_root = _bank_root(assets_root)
    logger.info("Asset bank: {}", bank_root)

    alpha_rows = _read_alpha_rows(source_csv)
    logger.info("alpha/: {} assets from {}", len(alpha_rows), source_csv.name)
    rows = alpha_rows + _read_paired_rows(bank_root)

    logger.info("Total: {} assets", len(rows))
    _verify(rows, strict=strict)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.success("Wrote {} rows to {}", len(rows), output_csv)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--assets-root",
        default=None,
        help="Artifact asset bank (default: $HISTO_ARTIFACTS_BANK)",
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Warn instead of failing when the category distribution has changed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_taxonomy(
        output_csv=args.output,
        source_csv=args.source_csv,
        assets_root=args.assets_root,
        strict=not args.allow_drift,
    )
