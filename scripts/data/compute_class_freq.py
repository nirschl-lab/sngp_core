"""Compute per-class training-split sample counts for a dataset's `class_freq`.

Reuses the counting logic in `src/visualization/datasets_class_distribution.py`
rather than reimplementing HF-dataset loading. Output is ordered by the
dataset's `class_to_idx` so it can be pasted directly into
`configs/data/<dataset>.yaml` as a top-level `class_freq:` key (a sibling of
`datamodule:`, NOT nested inside it -- everything under `datamodule:` is passed
as a `ClassificationImageDataModule` constructor kwarg via Hydra `_target_`
instantiation, which doesn't accept `class_freq`) and referenced from
experiment configs as `${data.class_freq}` -- the single source of truth every
model family's `model.class_freq` resolves to.

Usage:
    uv run scripts/data/compute_class_freq.py --dataset tang
"""
import argparse

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.datasets_class_distribution import (  # noqa: E402
    available_datasets,
    load_split_frame,
    resolve_dataset_config,
)


def compute_class_freq(dataset: str) -> list[int]:
    """Train-split sample count per class, ordered by `class_to_idx` index.

    Counts by the dataset's integer `label` column, matched directly to
    `class_to_idx`'s index values -- not by joining on class name against
    `label_name`. At least one dataset's `label_name` spelling doesn't match its
    own `classes_to_idx`/`class_to_idx` naming (acevedo's "immature granulocyte"
    vs "immature_granulocyte"), while `label` reliably matches `classes_to_idx`.
    """
    cfg = resolve_dataset_config(dataset)
    num_classes = len(cfg.class_to_idx)

    df = load_split_frame(cfg.dataset_name)
    train_labels = df.loc[df["split"] == "train", "label"]
    if train_labels.empty:
        raise ValueError(f"Dataset {dataset!r} has no 'train' split rows.")

    counts = train_labels.value_counts()
    missing = [idx for idx in range(num_classes) if idx not in counts.index]
    if missing:
        raise ValueError(
            f"No train-split samples found for label index/indices {missing} in dataset "
            f"{dataset!r}; configs/data/{dataset}.yaml's class_to_idx (expected indices "
            f"0..{num_classes - 1}) doesn't match the dataset's label range."
        )

    return [int(counts[idx]) for idx in range(num_classes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help=f"One of: {', '.join(available_datasets())}")
    args = parser.parse_args()

    class_freq = compute_class_freq(args.dataset)
    print(f"\nDataset: {args.dataset}")
    print(f"class_freq (ordered by class_to_idx): {class_freq}")
    print(f"\nAdd to configs/data/{args.dataset}.yaml as a top-level key (sibling of `datamodule:`):")
    print(f"class_freq: {class_freq}")


if __name__ == "__main__":
    main()
