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
    # Provenance of a post-hoc calibration written by
    # `scripts/checkpoints/calibrate_checkpoint.py` (via `io.write_checkpoint_with_net_spec`):
    # which knob (`temperature` / `mean_field_factor`) was fit, on which split of which
    # dataset, from which source checkpoint, with metrics before/after. The fitted value
    # itself lives in `net_spec` -- this block only records where it came from. `None`
    # for a checkpoint straight out of training. Additive, so pre-existing checkpoints
    # (no key) still read.
    calibration: Optional[Dict[str, Any]] = None


def build_meta(
    lit_module: Any,
    net_spec: Dict[str, Any],
    num_classes: int,
    idx_to_class: Optional[Dict[int, str]] = None,
    dataset_name: Optional[str] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cls = type(lit_module)
    return {
        "format_version": FORMAT_VERSION,
        "lit_module": f"{cls.__module__}.{cls.__qualname__}",
        "net_spec": net_spec,
        "num_classes": num_classes,
        "idx_to_class": idx_to_class,
        "dataset_name": dataset_name,
        "calibration": calibration,
    }
