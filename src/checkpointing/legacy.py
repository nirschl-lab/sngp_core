"""Legacy (pre-v2) checkpoint support.

Imported only by `scripts/checkpoints/migrate_checkpoints.py`. Production code never
reads a v1 checkpoint directly -- see `src/checkpointing/io.py`, which refuses anything
below `FORMAT_VERSION`. This module is allowed to be uglier than the rest of
`src/checkpointing/` since it exists to be run once per old checkpoint, not maintained
as a permanent code path.
"""
import importlib
import sys
from typing import Any, Dict

import torch

# Old pickled `net`/`hyper_parameters` objects can reference import paths that no
# longer exist post-refactor (e.g. the pre-Stage-1 `src.models.torch_vision_base`
# compat shim). Alias them so torch.load can still unpickle the object long enough to
# read its plain-data attributes.
_MODULE_ALIASES = {
    "src.models.torch_vision_base": "src.models.baseline.baseline_models",
}


def _install_legacy_aliases() -> None:
    for old_name, new_name in _MODULE_ALIASES.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = importlib.import_module(new_name)


def load_legacy_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    _install_legacy_aliases()
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


def derive_net_spec(net: Any, *, arch_hint: str = None) -> Dict[str, Any]:
    """Best-effort: recover a plain-data `net_spec` from a legacy pickled net object."""
    if hasattr(net, "spec"):
        return net.spec

    cls_name = type(net).__name__
    if cls_name == "BaselineClassifier":
        return {
            "name": "baseline_classifier",
            "arch": getattr(net, "backbone_name", arch_hint),
            "num_classes": net.num_classes,
            "dropout_p": getattr(net, "dropout_p", 0.5),
            "pretrained": getattr(net, "pretrained", False),
        }
    if cls_name == "SNGPClassifier":
        if arch_hint is None:
            raise ValueError(
                "Legacy SNGPClassifier checkpoints did not record their backbone arch. "
                "Pass --arch explicitly for this checkpoint."
            )
        gp = net.gp_head
        return {
            "name": "sngp_classifier",
            "num_classes": net.num_classes,
            "arch": arch_hint,
            "pretrained": False,
            "rff_dim": gp.rff_dim,
            "length_scale": gp.length_scale,
            "ridge_penalty": gp.ridge,
            "cov_momentum": gp.cov_momentum,
            "mean_field": gp.mean_field,
        }
    raise TypeError(f"Don't know how to derive a net_spec for legacy net type {cls_name!r}")


# torchvision ResNet's `children()` order (fixed across resnet18/34/50): everything up
# to but excluding `fc`. Pre-Stage-1 SNGPClassifier wrapped exactly this slice in a
# plain nn.Sequential, so its state_dict keys are positional ("backbone.4.0.conv1...")
# instead of named ("backbone.layer1.0.conv1..."). Stage 1 unified SNGP onto the same
# named-backbone module baseline already used, so a legacy SNGP checkpoint's backbone
# weights need this prefix remap or they silently fail to load (missing/unexpected
# keys) even after the net_spec itself is correctly recovered above.
_LEGACY_SNGP_RESNET_BACKBONE_INDEX_TO_NAME = {
    "0": "conv1",
    "1": "bn1",
    "2": "relu",
    "3": "maxpool",
    "4": "layer1",
    "5": "layer2",
    "6": "layer3",
    "7": "layer4",
    "8": "avgpool",
}


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor], net_spec: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Rewrite a legacy SNGP checkpoint's `net.backbone.<idx>....` keys to the
    named-module keys the post-Stage-1 SNGPClassifier expects. No-op for every other
    net family (their backbone module structure did not change)."""
    if net_spec.get("name") != "sngp_classifier":
        return state_dict

    prefix = "net.backbone."
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            index, _, tail = rest.partition(".")
            new_name = _LEGACY_SNGP_RESNET_BACKBONE_INDEX_TO_NAME.get(index)
            if new_name is None:
                # Already-named key (e.g. re-running migration on a partially-migrated
                # checkpoint) -- leave as-is rather than guessing.
                remapped[key] = value
                continue
            remapped[f"{prefix}{new_name}.{tail}" if tail else f"{prefix}{new_name}"] = value
        else:
            remapped[key] = value
    return remapped
