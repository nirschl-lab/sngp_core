"""Plain-data checkpoint metadata contract.

Every checkpoint this project writes carries a `checkpoint["sngp_core"]` block
alongside Lightning's own `hyper_parameters`/`state_dict` — pure primitives only, so
`read_meta()` (in `io.py`) never needs to unpickle a live `net` object or import path
to know what architecture a checkpoint holds.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

FORMAT_VERSION = 2


@dataclass
class CheckpointMeta:
    format_version: int
    lit_module: str  # fully-qualified "module.ClassName" of the LightningModule
    net_spec: Dict[str, Any]
    num_classes: int
    idx_to_class: Optional[Dict[int, str]] = None
    dataset_name: Optional[str] = None


def build_meta(
    lit_module: Any,
    net_spec: Dict[str, Any],
    num_classes: int,
    idx_to_class: Optional[Dict[int, str]] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    cls = type(lit_module)
    return {
        "format_version": FORMAT_VERSION,
        "lit_module": f"{cls.__module__}.{cls.__qualname__}",
        "net_spec": net_spec,
        "num_classes": num_classes,
        "idx_to_class": idx_to_class,
        "dataset_name": dataset_name,
    }
