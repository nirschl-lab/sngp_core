"""The single canonical checkpoint-loading API.

Every consumer that needs to read a trained checkpoint -- `src/inference/infer.py`,
`scripts/hf/export_to_hub.py`, ad hoc analysis scripts -- goes through this module.
Never raw `torch.load` + manual state-dict key surgery, never a bare
`LightningModule.load_from_checkpoint()` scattered across call sites: those were the
3-4 divergent, differently-fragile checkpoint-loading recipes this module replaces.

The one sanctioned *writer* of a modified checkpoint is `write_checkpoint_with_net_spec`
(used by `scripts/checkpoints/calibrate_checkpoint.py` to bake a post-hoc-fitted knob
into `net_spec`); `scripts/checkpoints/migrate_checkpoints.py` is the other, one-off,
writer.
"""
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.checkpointing.spec import FORMAT_VERSION, CheckpointMeta
from src.models.registry import NET_REGISTRY, build_net

_MIGRATE_HINT = (
    "Run `uv run scripts/checkpoints/migrate_checkpoints.py --in <path> --out <dir>` to migrate it."
)


def _load_raw(ckpt_path: Union[str, Path], map_location: str) -> Dict[str, Any]:
    return torch.load(str(ckpt_path), map_location=map_location, weights_only=False)


def _import_class(dotted_path: str):
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _resolve_device(device: Optional[str]) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def read_meta(ckpt_path: Union[str, Path]) -> CheckpointMeta:
    """Read a checkpoint's architecture/format metadata without unpickling a net object."""
    checkpoint = _load_raw(ckpt_path, map_location="cpu")
    sngp_core = checkpoint.get("sngp_core")
    if not isinstance(sngp_core, dict):
        raise ValueError(f"{ckpt_path} has no 'sngp_core' metadata block (pre-v{FORMAT_VERSION} checkpoint). {_MIGRATE_HINT}")
    if sngp_core.get("format_version", 0) < FORMAT_VERSION:
        raise ValueError(
            f"{ckpt_path} is format_version {sngp_core.get('format_version')}, "
            f"this project requires >= {FORMAT_VERSION}. {_MIGRATE_HINT}"
        )
    # Known fields only: a checkpoint written by a newer minor revision may carry an
    # additive metadata key this code does not know yet, and that must not make the
    # whole checkpoint unreadable.
    known = {key: value for key, value in sngp_core.items() if key in CheckpointMeta.__dataclass_fields__}
    return CheckpointMeta(**known)


def _validate_net_spec_updates(net_spec: Dict[str, Any], updates: Dict[str, Any]) -> None:
    if "name" in updates:
        raise ValueError("`name` selects the net class and cannot be rewritten; build a new checkpoint instead")
    if not updates:
        raise ValueError("no net_spec updates given")
    name = net_spec.get("name")
    if name not in NET_REGISTRY:
        raise KeyError(f"Unknown net registry key {name!r} in the checkpoint's net_spec")
    cls = NET_REGISTRY[name]
    # `inspect.signature` rather than `build_net(spec)`: cheap, and it cannot trigger a
    # torchvision weight download the way constructing a `pretrained=True` spec could.
    params = inspect.signature(cls.__init__).parameters
    for key, value in updates.items():
        if key not in params:
            raise ValueError(f"{key!r} is not a constructor argument of {cls.__name__} (net_spec keys: {sorted(params)})")
        try:
            json.dumps(value)
        except TypeError as exc:
            raise ValueError(f"net_spec value for {key!r} must be JSON-serializable, got {type(value).__name__}") from exc


def write_checkpoint_with_net_spec(
    src_ckpt: Union[str, Path],
    dst_ckpt: Union[str, Path],
    net_spec_updates: Dict[str, Any],
    *,
    calibration: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Copy a checkpoint with some `net_spec` keys rewritten -- the sanctioned way to bake a
    post-hoc-fitted, inference-only knob (`temperature`, `mean_field_factor`,
    `ridge_penalty`) into a checkpoint without retraining or a config edit.

    Both copies of the spec are updated -- `checkpoint["sngp_core"]["net_spec"]`, which
    `load_net` rebuilds from, and `checkpoint["hyper_parameters"]["net_spec"]`, which
    `load_lit_module` rebuilds from -- so every loader sees the new value. `state_dict`,
    optimizer/loop state and callbacks are copied verbatim (weights are bit-identical).
    `calibration` (plain JSON data, see `CheckpointMeta.calibration`) records where the
    value came from.

    Refuses to write over the source path (pass a sibling such as `best.calibrated.ckpt`)
    and over an existing destination unless `overwrite=True`.
    """
    src = Path(src_ckpt)
    dst = Path(dst_ckpt)
    if dst.resolve() == src.resolve():
        raise ValueError(f"refusing to rewrite {src} in place; write a sibling (e.g. best.calibrated.ckpt) instead")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists; pass overwrite=True to replace it")

    meta = read_meta(src)  # also enforces FORMAT_VERSION
    _validate_net_spec_updates(meta.net_spec, net_spec_updates)
    if calibration is not None:
        try:
            json.dumps(calibration)
        except TypeError as exc:
            raise ValueError("calibration provenance must be JSON-serializable") from exc

    checkpoint = _load_raw(src, map_location="cpu")
    new_spec = {**checkpoint["sngp_core"]["net_spec"], **net_spec_updates}
    checkpoint["sngp_core"]["net_spec"] = new_spec
    hparams = checkpoint.get("hyper_parameters")
    if isinstance(hparams, dict) and "net_spec" in hparams:
        hparams["net_spec"] = dict(new_spec)
    if calibration is not None:
        checkpoint["sngp_core"]["calibration"] = dict(calibration)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(dst))
    return dst


def load_net(ckpt_path: Union[str, Path], *, device: Optional[str] = None, strict: bool = True) -> nn.Module:
    """Load just the underlying net (no Lightning/optimizer machinery) for inference."""
    meta = read_meta(ckpt_path)
    load_spec = dict(meta.net_spec)
    if "pretrained" in load_spec:
        load_spec["pretrained"] = False  # weights are overwritten by state_dict below
    net = build_net(load_spec)

    checkpoint = _load_raw(ckpt_path, map_location="cpu")
    prefix = "net."
    net_state = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    net.load_state_dict(net_state, strict=strict)

    device = _resolve_device(device)
    net.to(device)
    net.eval()
    return net


def load_lit_module(
    ckpt_path: Union[str, Path],
    *,
    device: Optional[str] = None,
    strict: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
):
    """Load a full LightningModule (net + criterion + logged hparams).

    The LightningModule class is resolved from the checkpoint's own metadata, so
    callers never need to know in advance which model family a checkpoint holds.
    """
    meta = read_meta(ckpt_path)
    cls = _import_class(meta.lit_module)

    checkpoint = _load_raw(ckpt_path, map_location="cpu")
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    if overrides:
        hparams.update(overrides)

    lit_model = cls(**hparams)
    lit_model.load_state_dict(checkpoint["state_dict"], strict=strict)

    device = _resolve_device(device)
    lit_model.to(device)
    lit_model.eval()
    return lit_model
