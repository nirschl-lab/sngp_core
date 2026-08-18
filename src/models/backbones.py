"""Shared backbone construction for every net family (baseline, SNGP, HF export).

Deliberately dependency-light: only `torch`/`torchvision`, no `hydra`/`lightning`/other
`src.*` imports. This lets `scripts/hf/export_to_hub.py` concatenate this file's source
verbatim into a self-contained HF `trust_remote_code` bundle.
"""
from typing import NamedTuple, Tuple

import torch.nn as nn
from torchvision import models


class BackboneEntry(NamedTuple):
    ctor: object
    weights_enum_name: str
    default_weight_attr: str
    kind: str  # "resnet" | "vit"


BACKBONES = {
    "resnet18": BackboneEntry(models.resnet18, "ResNet18_Weights", "IMAGENET1K_V1", "resnet"),
    "resnet34": BackboneEntry(models.resnet34, "ResNet34_Weights", "IMAGENET1K_V1", "resnet"),
    "resnet50": BackboneEntry(models.resnet50, "ResNet50_Weights", "IMAGENET1K_V2", "resnet"),
    "vit_b_16": BackboneEntry(models.vit_b_16, "ViT_B_16_Weights", "IMAGENET1K_V1", "vit"),
    "vit_b_32": BackboneEntry(models.vit_b_32, "ViT_B_32_Weights", "IMAGENET1K_V1", "vit"),
    "vit_l_16": BackboneEntry(models.vit_l_16, "ViT_L_16_Weights", "IMAGENET1K_V1", "vit"),
    "vit_l_32": BackboneEntry(models.vit_l_32, "ViT_L_32_Weights", "IMAGENET1K_V1", "vit"),
    "vit_h_14": BackboneEntry(models.vit_h_14, "ViT_H_14_Weights", "IMAGENET1K_V1", "vit"),
}

# SNGP applies recursive spectral-norm wrapping to every Conv2d/Linear in the backbone.
# ViT internals (LayerNorm/attention) are not validated/supported for this — resnet only.
SPECTRAL_NORM_COMPATIBLE = frozenset(name for name, e in BACKBONES.items() if e.kind == "resnet")


def build_backbone(arch: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Build a torchvision backbone with its classification head stripped.

    Returns `(module, feat_dim)` where `module(x)` yields already-flattened
    `[B, feat_dim]` features for both resnet (avgpool+flatten baked into forward) and
    ViT (class-token representation) archs — callers never need to pool/flatten themselves.
    """
    if arch not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {arch}. Supported: {sorted(BACKBONES)}")
    entry = BACKBONES[arch]

    weights = None
    if pretrained:
        enum = getattr(models, entry.weights_enum_name, None)
        if enum is not None:
            weights = getattr(enum, entry.default_weight_attr, None)
    try:
        model = entry.ctor(weights=weights if pretrained else None)
    except TypeError:
        model = entry.ctor(pretrained=pretrained)

    if entry.kind == "resnet":
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    # ViT: torchvision's `heads` is a Sequential ending in a Linear; read its in_features,
    # then strip it so forward returns the raw class-token representation [B, feat_dim].
    feat_dim = None
    if hasattr(model, "heads") and hasattr(model.heads, "head") and hasattr(model.heads.head, "in_features"):
        feat_dim = model.heads.head.in_features
    else:
        for m in model.heads.modules():
            if isinstance(m, nn.Linear):
                feat_dim = m.in_features
                break
    if feat_dim is None:
        feat_dim = getattr(model, "hidden_dim", None)
    if feat_dim is None:
        raise RuntimeError(f"Could not infer feature dimension for {arch}")

    model.heads = nn.Identity()
    return model, feat_dim
