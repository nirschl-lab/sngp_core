from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# Explicit mapping keeps the labels reproducible and easy to review.
# Keys are normalized to lowercase with underscores converted to hyphens.
LABEL_MAP: dict[str, tuple[str, str, str, bool]] = {
    "acellular-fibrin": ("tissue_component", "acellular_fibrin", "", False),
    "afb-debris": ("debris", "afb_debris", "", False),
    "air-bubble": ("bubble", "air_bubble", "", False),
    "artifact": ("other_artifact", "unspecified_artifact", "", True),
    "artifact-debris": ("debris", "generic_debris", "", False),
    "artifact-fiber": ("fiber_hair", "fiber", "", False),
    "artifact-foam": ("processing_material", "foam", "", False),
    "artifact-hair": ("fiber_hair", "hair", "", False),
    "artifact-ihc-debris": ("debris", "ihc_debris", "", False),
    "artifact-ink": ("pigment_ink", "ink", "", False),
    "artifact-sutures": ("processing_material", "suture", "", False),
    "artifact-retic": ("stain_artifact", "reticulin_artifact", "", False),
    "biel-debris": ("debris", "bielschowsky_debris", "", False),
    "bielschovsky-debris-acellular": ("debris", "acellular_bielschowsky_debris", "", False),
    "blue-ink": ("pigment_ink", "blue_ink", "", False),
    "bone-debris": ("debris", "bone_debris", "bone_calcification", False),
    "bone-dust": ("debris", "bone_dust", "bone_calcification", False),
    "bone-fragment": ("bone_calcification", "bone_fragment", "", False),
    "bone-fragments": ("bone_calcification", "bone_fragments", "", False),
    "bone-fragments-fibers": ("bone_calcification", "bone_fragments_with_fibers", "fiber_hair", False),
    "bubble": ("bubble", "bubble", "", False),
    "calc": ("bone_calcification", "calcification", "", False),
    "calc-tiss-debris": ("debris", "calcified_tissue_debris", "bone_calcification", False),
    "calcium-debris": ("debris", "calcium_debris", "bone_calcification", False),
    "cereb-dentate-negative-igm": ("stain_artifact", "negative_igm_background", "", True),
    "congo-red-debris": ("debris", "congo_red_debris", "", False),
    "dab-artifact": ("pigment_ink", "dab_artifact", "stain_artifact", False),
    "dab-debris": ("debris", "dab_debris", "pigment_ink", False),
    "debis": ("debris", "generic_debris", "", False),
    "debris": ("debris", "generic_debris", "", False),
    "debris-cells": ("debris", "cellular_debris", "tissue_component", False),
    "debris-gram": ("debris", "gram_stain_debris", "stain_artifact", False),
    "debris-gunk": ("debris", "gunk_debris", "", False),
    "debris-ihc": ("debris", "ihc_debris", "", False),
    "debris-pigment": ("debris", "pigmented_debris", "pigment_ink", False),
    "debris-rbc": ("debris", "rbc_debris", "tissue_component", False),
    "dust": ("debris", "dust", "", False),
    "fiber": ("fiber_hair", "fiber", "", False),
    "fiber-hair": ("fiber_hair", "fiber_with_hair", "", False),
    "fiber-gms": ("fiber_hair", "gms_fiber", "stain_artifact", False),
    "fibers": ("fiber_hair", "fiber", "", False),
    "fine-debris": ("debris", "fine_debris", "", False),
    "foreign-material": ("processing_material", "foreign_material", "", False),
    "gel-foam": ("processing_material", "gel_foam", "", False),
    "glass-scratch": ("scratch_glass", "glass_scratch", "", False),
    "gms-background": ("stain_artifact", "gms_background", "", False),
    "green-ink": ("pigment_ink", "green_ink", "", False),
    "hair": ("fiber_hair", "hair", "", False),
    "ink": ("pigment_ink", "ink", "", False),
    "ink-black": ("pigment_ink", "black_ink", "", False),
    "ink-green": ("pigment_ink", "green_ink", "", False),
    "ink-orange": ("pigment_ink", "orange_ink", "", False),
    "keratin-debris": ("debris", "keratin_debris", "tissue_component", False),
    "meningioma-infarct-dystrophic-calc": (
        "bone_calcification",
        "dystrophic_calcification",
        "tissue_component",
        True,
    ),
    "out-of-focus": ("focus", "out_of_focus", "", False),
    "out-of-focus-debris": ("focus", "out_of_focus_with_debris", "debris", False),
    "pigment": ("pigment_ink", "pigment", "", False),
    "red-blood-cells-ihc": ("tissue_component", "red_blood_cells_ihc", "", False),
    "red-ink": ("pigment_ink", "red_ink", "", False),
    "scratch-glue": ("scratch_glass", "scratch_with_glue", "processing_material", False),
    "squamous-debris": ("debris", "squamous_debris", "tissue_component", False),
    "surgical-sponge": ("processing_material", "surgical_sponge", "fiber_hair", False),
    "tiss-debris-nuclei": ("debris", "tissue_nuclear_debris", "tissue_component", False),
    "yellow-ink": ("pigment_ink", "yellow_ink", "", False),
}


def normalize_label(label: str) -> str:
    label = label.strip().lower().replace("_", "-")
    label = re.sub(r"\s+", "-", label)
    return re.sub(r"-+", "-", label)


def extract_artifact_label(image_name: str) -> str:
    """Extract the text between `_part-..._` and the magnification token."""
    match = re.search(
        r"_\s*part-[^_]+_(.+?)_(?:\d+(?:\.\d+)?x)(?:_|\.)",
        image_name,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def fallback_classification(label_key: str) -> tuple[str, str, str, bool]:
    """Conservative keyword fallback for future labels not in LABEL_MAP."""
    subtype = label_key.replace("-", "_") or "unparsed_label"
    if "out-of-focus" in label_key or "blur" in label_key:
        return "focus", subtype, "", True
    if "bubble" in label_key:
        return "bubble", subtype, "", True
    if "scratch" in label_key or "glass" in label_key:
        return "scratch_glass", subtype, "", True
    if any(token in label_key for token in ("ink", "pigment", "dab")):
        return "pigment_ink", subtype, "", True
    if any(token in label_key for token in ("fiber", "hair")):
        return "fiber_hair", subtype, "", True
    if any(token in label_key for token in ("bone", "calc")):
        return "bone_calcification", subtype, "", True
    if any(token in label_key for token in ("foam", "sponge", "suture", "foreign", "glue")):
        return "processing_material", subtype, "", True
    if any(token in label_key for token in ("debris", "debis", "dust", "gunk")):
        return "debris", subtype, "", True
    return "other_artifact", subtype, "", True


def categorize(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    # Accept the user's current column name and a few common alternatives.
    candidates = ["img_name", "image_name", "filename", "file_name"]
    image_col = next((c for c in candidates if c in df.columns), None)
    if image_col is None:
        raise ValueError(f"No image-name column found. Expected one of: {candidates}")

    out = pd.DataFrame()
    out["image_name"] = df[image_col].astype(str).str.strip()
    out["artifact_label_raw"] = out["image_name"].map(extract_artifact_label)
    out["artifact_label_key"] = out["artifact_label_raw"].map(normalize_label)

    classifications = []
    for key in out["artifact_label_key"]:
        classifications.append(LABEL_MAP.get(key, fallback_classification(key)))

    out[["parent_category", "sub_category", "secondary_category", "needs_review"]] = pd.DataFrame(
        classifications, index=out.index
    )
    out["classification_source"] = out["artifact_label_key"].map(
        lambda key: "explicit_filename_mapping" if key in LABEL_MAP else "keyword_fallback"
    )

    # Keep rows with genuinely missing image names out of the final data.
    out = out[out["image_name"].ne("") & out["image_name"].ne("nan")].reset_index(drop=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize histopathology artifact image filenames.")
    parser.add_argument("input_csv", type=Path, nargs="?", help="CSV containing an img_name/image_name column",
                        default='data/artifact/img_names.csv')
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/artifact/img_names_categorized.csv"),
        help="Output CSV path"
    )
    args = parser.parse_args()

    result = categorize(args.input_csv, args.output)
    print(f"Wrote {len(result)} rows to {args.output}")
    print("\nParent-category counts:")
    print(result["parent_category"].value_counts().to_string())
    print(f"\nRows marked for review: {int(result['needs_review'].sum())}")


if __name__ == "__main__":
    main()
