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


# ---------------------------------------------------------------------------
# WideResNet-28-10 (CIFAR)
# ---------------------------------------------------------------------------
# Vendored rather than imported. `timm` has no CIFAR WideResNet (only the ImageNet
# bottleneck `wide_resnet50_2`/`wide_resnet101_2`), and neither does torchvision;
# `torch_uncertainty.models...wideresnet.wideresnet28x10` exists but is post-activation
# and, more importantly, this module's source is inlined verbatim into the HF
# `trust_remote_code` bundle by scripts/hf/export_to_hub.py -- a third-party import here
# would follow it into every exported model.
#
# Transcribed from the SNGP reference,
# google/uncertainty-baselines `uncertainty_baselines/models/wide_resnet_sngp.py`:
# pre-activation blocks, 3x3 stride-1 stem with no maxpool, filter-wise dropout,
# bias-free convs, he-normal init, three groups of 4 blocks at 16k/32k/64k filters.
# BatchNorm there is explicitly configured to PyTorch's own defaults
# (`epsilon=1e-5, momentum=0.9` in Keras convention), so `nn.BatchNorm2d()` matches.


class _WideBasicBlock(nn.Module):
    """Pre-activation residual block: (BN-ReLU-dropout-conv) x 2 plus a shortcut.

    `nn.Dropout2d` zeroes whole channels, which is what the reference's
    `Dropout(noise_shape=[B, 1, 1, C])` ("filter-wise dropout") does.
    """

    def __init__(self, in_planes: int, planes: int, stride: int, dropout_rate: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        y = self.dropout(nn.functional.relu(self.bn1(x)))
        y = self.conv1(y)
        y = self.dropout(nn.functional.relu(self.bn2(y)))
        y = self.conv2(y)
        if self.shortcut is not None:
            x = self.shortcut(x)
            # Reference quirk, kept deliberately: when (and only when) the shortcut is a
            # projection, a third dropout is applied -- to the residual branch `y`, not to
            # the projected `x`. See `basic_block` in wide_resnet_sngp.py.
            y = self.dropout(y)
        return x + y


class WideResNet(nn.Module):
    """Zagoruyko & Komodakis WideResNet in the SNGP reference's configuration.

    The classification head is named `fc` so that `build_backbone`'s `kind == "resnet"`
    branch strips it the same way it strips a torchvision ResNet's.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.1,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth must be 6n+4 (16, 22, 28, 40, ...), got {depth}")
        num_blocks = (depth - 4) // 6
        widths = [16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        in_planes = 16
        groups = []
        for width, stride in zip(widths, [1, 2, 2]):
            blocks = []
            for block_stride in [stride] + [1] * (num_blocks - 1):
                blocks.append(_WideBasicBlock(in_planes, width, block_stride, dropout_rate))
                in_planes = width
            groups.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3 = groups

        self.bn = nn.BatchNorm2d(widths[-1])
        # Adaptive rather than the reference's fixed `AveragePooling2D(pool_size=8)`, so a
        # forward pass at a resolution other than 32px still yields flat [B, feat_dim].
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(widths[-1], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.dropout(self.conv1(x))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = nn.functional.relu(self.bn(x))
        return self.fc(self.flatten(self.avgpool(x)))


def wide_resnet28_10(
    weights=None, num_classes: int = 1000, dropout_rate: float = 0.1
) -> WideResNet:
    """WRN-28-10 with the SNGP CIFAR reference's filter-wise dropout rate (0.1).

    `weights` exists only so this matches the torchvision ctor signature `build_backbone`
    calls; there are no pretrained weights for a CIFAR WideResNet.
    """
    if weights is not None:
        raise ValueError(
            "wide_resnet28_10 has no pretrained weights; build it with pretrained=False"
        )
    return WideResNet(
        depth=28, widen_factor=10, dropout_rate=dropout_rate, num_classes=num_classes
    )


BACKBONES = {
    "resnet18": BackboneEntry(models.resnet18, "ResNet18_Weights", "IMAGENET1K_V1", "resnet"),
    "resnet34": BackboneEntry(models.resnet34, "ResNet34_Weights", "IMAGENET1K_V1", "resnet"),
    "resnet50": BackboneEntry(models.resnet50, "ResNet50_Weights", "IMAGENET1K_V2", "resnet"),
    "vit_b_16": BackboneEntry(models.vit_b_16, "ViT_B_16_Weights", "IMAGENET1K_V1", "vit"),
    "vit_b_32": BackboneEntry(models.vit_b_32, "ViT_B_32_Weights", "IMAGENET1K_V1", "vit"),
    "vit_l_16": BackboneEntry(models.vit_l_16, "ViT_L_16_Weights", "IMAGENET1K_V1", "vit"),
    "vit_l_32": BackboneEntry(models.vit_l_32, "ViT_L_32_Weights", "IMAGENET1K_V1", "vit"),
    "vit_h_14": BackboneEntry(models.vit_h_14, "ViT_H_14_Weights", "IMAGENET1K_V1", "vit"),
    # `kind="resnet"`: same head layout (`.fc`) and the same all-conv body, so both the
    # head-stripping branch below and spectral-norm wrapping apply unchanged. The empty
    # weights-enum name makes the `pretrained=True` lookup fall through to `weights=None`.
    "wide_resnet28_10": BackboneEntry(wide_resnet28_10, "", "", "resnet"),
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
