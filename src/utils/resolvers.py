"""Custom OmegaConf resolvers, registered as an import-time side effect.

Imported from `src/utils/__init__.py` so registration happens before any Hydra
config is composed/resolved in `src/train.py`/`src/eval.py`.
"""

from omegaconf import OmegaConf


def _tags_with(base, *extra):
    """Append any given non-null/non-empty extra tags onto `base`, deduplicated.

    Lets a W&B tags list be composed dynamically (e.g. append an institution id or
    "hpo") without every config needing to fully re-declare the base tags list --
    OmegaConf interpolations can't concatenate lists directly (`${tags} + [x]` is not
    valid syntax).
    """
    tags = list(base) if base else []
    for item in extra:
        if item is None or item == "":
            continue
        if item not in tags:
            tags.append(item)
    return tags


if not OmegaConf.has_resolver("tags_with"):
    OmegaConf.register_new_resolver("tags_with", _tags_with)


def _dataset_label(name, institution=None):
    """Institution-scoped dataset identifier for task_name/output-directory naming:
    `<name>_<institution>` when `data.datamodule.institution` is set, else plain
    `<name>` -- so a per-institution train/eval/infer run lands in its own directory
    instead of collapsing into the parent dataset's shared folder (previously required
    manually overriding `data.name=<name>_<institution>` per run to avoid that
    collision; see docs/OUTPUT_LAYOUT.md).
    """
    return f"{name}_{institution}" if institution else name


if not OmegaConf.has_resolver("dataset_label"):
    OmegaConf.register_new_resolver("dataset_label", _dataset_label)
