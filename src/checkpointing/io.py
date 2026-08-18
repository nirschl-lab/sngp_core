"""The single canonical checkpoint-loading API.

Every consumer that needs to read a trained checkpoint -- `src/inference/infer.py`,
`scripts/hf/export_to_hub.py`, ad hoc analysis scripts -- goes through this module.
Never raw `torch.load` + manual state-dict key surgery, never a bare
`LightningModule.load_from_checkpoint()` scattered across call sites: those were the
3-4 divergent, differently-fragile checkpoint-loading recipes this module replaces.
"""
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.checkpointing.spec import FORMAT_VERSION, CheckpointMeta
from src.models.registry import build_net

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
    return CheckpointMeta(**sngp_core)


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
