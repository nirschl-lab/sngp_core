"""Class-distribution statistics for a project dataset: by split and by institution.

Usage:
    uv run src/visualization/datasets_class_distribution.py --dataset tang
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import rootutils
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

DATA_CONFIG_DIR = ROOT / "configs" / "data"
ROW_COLUMNS = ["label", "label_name", "institution"]


def available_datasets() -> list[str]:
    """Short dataset names with a configs/data/<name>.yaml entry."""
    return sorted(p.stem for p in DATA_CONFIG_DIR.glob("*.yaml") if p.stem != "artifact_image_classifier")


def resolve_dataset_config(name: str) -> DictConfig:
    """Load configs/data/<name>.yaml and return its `datamodule` sub-config."""
    config_path = DATA_CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {', '.join(available_datasets())}")
    return OmegaConf.load(config_path).datamodule


def load_split_frame(hf_dataset_name: str) -> pd.DataFrame:
    """Load a raw HF dataset and return one row per sample: split/label/label_name/institution."""
    raw = datasets.load_dataset(hf_dataset_name)
    frames = []
    for split_name, split_ds in raw.items():
        cols = [c for c in ROW_COLUMNS if c in split_ds.column_names]
        sub = split_ds.select_columns(cols).to_pandas()
        sub["split"] = split_name
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def class_distribution_by_split(df: pd.DataFrame) -> pd.DataFrame:
    """Sample count per (split, class) -- rows=split, columns=class name."""
    return df.groupby(["split", "label_name"]).size().unstack("label_name", fill_value=0)


def institution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Sample count per institution, sorted descending."""
    return (
        df.groupby("institution")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def class_by_institution(df: pd.DataFrame) -> pd.DataFrame:
    """Sample count per (institution, class) -- rows=institution, columns=class name."""
    return df.groupby(["institution", "label_name"]).size().unstack("label_name", fill_value=0)


def institution_by_split(df: pd.DataFrame) -> pd.DataFrame:
    """Sample count per (split, institution) -- shows whether institutions are shared across splits."""
    return df.groupby(["split", "institution"]).size().unstack("institution", fill_value=0)


def plot_class_distribution_by_split(counts: pd.DataFrame, dataset_label: str) -> plt.Figure:
    """Grouped bar chart of class counts, one group of bars per split."""
    set_default_style()
    ax = counts.T.plot(kind="bar", figsize=(max(8.0, 1.2 * counts.shape[1]), 6.0))
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample count")
    ax.set_title(f"{dataset_label}: class distribution by split")
    ax.legend(title="Split", frameon=False)
    ax.tick_params(axis="x", rotation=30)
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def plot_class_by_institution(counts: pd.DataFrame, dataset_label: str) -> plt.Figure:
    """Heatmap of class counts per institution."""
    set_default_style()
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * counts.shape[1]), max(4.0, 0.6 * counts.shape[0])))
    sns.heatmap(counts, annot=True, fmt="d", cmap="viridis", ax=ax, cbar_kws={"label": "Sample count"})
    ax.set_xlabel("Class")
    ax.set_ylabel("Institution")
    ax.set_title(f"{dataset_label}: class distribution by institution")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-distribution statistics for a project dataset.")
    parser.add_argument("--dataset", required=True, help=f"One of: {', '.join(available_datasets())}")
    parser.add_argument("--csv-dir", default=str(ROOT / "csv" / "dataset_stats"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures" / "dataset_stats"))
    args = parser.parse_args()

    cfg = resolve_dataset_config(args.dataset)
    df = load_split_frame(cfg.dataset_name)

    by_split = class_distribution_by_split(df)
    inst_counts = institution_summary(df)
    by_institution = class_by_institution(df)
    by_split_institution = institution_by_split(df)

    csv_dir = Path(args.csv_dir) / args.dataset
    csv_dir.mkdir(parents=True, exist_ok=True)
    by_split.to_csv(csv_dir / "class_by_split.csv")
    inst_counts.to_csv(csv_dir / "institution_counts.csv", index=False)
    by_institution.to_csv(csv_dir / "class_by_institution.csv")
    by_split_institution.to_csv(csv_dir / "institution_by_split.csv")

    figures_dir = Path(args.figures_dir) / args.dataset
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_class_distribution_by_split(by_split, args.dataset)
    fig.savefig(figures_dir / "class_by_split.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    if len(inst_counts) > 1:
        fig = plot_class_by_institution(by_institution, args.dataset)
        fig.savefig(figures_dir / "class_by_institution.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"Only one institution present ({inst_counts.loc[0, 'institution']!r}) -- skipping institution plot.")

    print(f"\nDataset: {args.dataset} ({cfg.dataset_name})")
    print(f"Institutions: {len(inst_counts)}")
    print(inst_counts.to_string(index=False))
    print("\nClass distribution by split:")
    print(by_split)
    print(f"\nSaved CSVs to {csv_dir}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
