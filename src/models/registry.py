"""Central net registry: maps a plain string key to a net class.

This is what lets checkpoints store architecture identity as data (a `net_spec` dict with
a `name` key) instead of pickling a live `nn.Module` object or relying on an import path
that breaks the moment a class moves or is renamed.
"""
from typing import Callable, Dict, Type

import torch.nn as nn

NET_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_net(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Class decorator: registers `cls` under `name` and stamps `cls.registry_name = name`."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in NET_REGISTRY and NET_REGISTRY[name] is not cls:
            raise ValueError(f"Net name '{name}' is already registered to {NET_REGISTRY[name]!r}")
        cls.registry_name = name
        NET_REGISTRY[name] = cls
        return cls

    return decorator


def build_net(spec: dict) -> nn.Module:
    """Instantiate a net from its plain-data `spec` (as produced by `net.spec`)."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in NET_REGISTRY:
        raise KeyError(f"Unknown net registry key '{name}'. Registered: {sorted(NET_REGISTRY)}")
    return NET_REGISTRY[name](**spec)
