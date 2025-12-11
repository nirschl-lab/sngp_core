# pip install torch torchvision
from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

Backbone = Literal[
    "resnet18", "resnet34", "resnet50",
    "vit_b_16", "vit_b_32",
    "vit_l_16", "vit_l_32",
    "vit_h_14",
    "swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b",
]

class BaselineClassifier(nn.Module):
    """
    Classification model with selectable ResNet/ViT backbone.
    Grabs penultimate features, then applies Dropout + Linear.
    Includes helpers for Monte Carlo Dropout inference.
    """

    def __init__(
        self,
        arch: Backbone = "resnet50",
        num_classes: int = 2,
        dropout_p: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = arch
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.pretrained = pretrained

        if arch.startswith("resnet"):
            self.feature_extractor, feat_dim = self._build_resnet(arch, pretrained)
        elif arch.startswith("vit_"):
            self.feature_extractor, feat_dim = self._build_vit(arch, pretrained)
        elif arch.startswith("swin"):
            self.feature_extractor, feat_dim = self._build_swin(arch, pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {arch}")

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(feat_dim, num_classes),
        )

    # -------------------------
    # Backbones
    # -------------------------
    def _build_resnet(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        ctor_map = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }
        weights_enums = {
            "resnet18": getattr(models, "ResNet18_Weights", None),
            "resnet34": getattr(models, "ResNet34_Weights", None),
            "resnet50": getattr(models, "ResNet50_Weights", None),
        }
        default_weights_attr = {
            "resnet18": "IMAGENET1K_V1",
            "resnet34": "IMAGENET1K_V1",
            "resnet50": "IMAGENET1K_V2",
        }

        ctor = ctor_map[name]
        weights = None
        if pretrained:
            enum = weights_enums[name]
            if enum is not None:
                # Try torchvision>=0.13-style enums
                try:
                    weights = getattr(enum, default_weights_attr[name])
                except Exception:
                    weights = None  # fallback to None on older APIs
        # Older torchvision expects weights=None or pretrained=True.
        try:
            model = ctor(weights=weights if pretrained else None)
        except TypeError:
            model = ctor(pretrained=pretrained)

        feat_dim = model.fc.in_features
        model.fc = nn.Identity()  # expose pooled features
        return model, feat_dim

    def _build_vit(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        ctor_map = {
            "vit_b_16": models.vit_b_16,
            "vit_b_32": models.vit_b_32,
            "vit_l_16": models.vit_l_16,
            "vit_l_32": models.vit_l_32,
            "vit_h_14": models.vit_h_14,
        }
        weights_enums = {
            "vit_b_16": getattr(models, "ViT_B_16_Weights", None),
            "vit_b_32": getattr(models, "ViT_B_32_Weights", None),
            "vit_l_16": getattr(models, "ViT_L_16_Weights", None),
            "vit_l_32": getattr(models, "ViT_L_32_Weights", None),
            "vit_h_14": getattr(models, "ViT_H_14_Weights", None),
        }
        # Most ViT enums use IMAGENET1K_V1 as the default classification pretrain
        default_attr = "IMAGENET1K_V1"

        ctor = ctor_map[name]
        weights = None
        if pretrained:
            enum = weights_enums[name]
            if enum is not None:
                try:
                    weights = getattr(enum, default_attr)
                except Exception:
                    weights = None

        try:
            vit = ctor(weights=weights if pretrained else None)
        except TypeError:
            vit = ctor(pretrained=pretrained)

        # vit.heads is a Sequential; grab in_features of final linear
        feat_dim: Optional[int] = None
        if hasattr(vit, "heads") and hasattr(vit.heads, "head") and hasattr(vit.heads.head, "in_features"):
            feat_dim = vit.heads.head.in_features
        else:
            last_linear = None
            for m in vit.heads.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                feat_dim = last_linear.in_features
        if feat_dim is None:
            raise RuntimeError(f"Could not infer feature dimension for {name}")

        vit.heads = nn.Identity()
        return vit, feat_dim

    def _build_swin(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        ctor_map = {
            "swin_t": models.swin_t,
            "swin_s": models.swin_s,
            "swin_b": models.swin_b,
            "swin_v2_t": models.swin_v2_t,
            "swin_v2_s": models.swin_v2_s,
            "swin_v2_b": models.swin_v2_b,
        }
        weights_enums = {
            "swin_t": getattr(models, "Swin_T_Weights", None),
            "swin_s": getattr(models, "Swin_S_Weights", None),
            "swin_b": getattr(models, "Swin_B_Weights", None),
            "swin_v2_t": getattr(models, "Swin_V2_T_Weights", None),
            "swin_v2_s": getattr(models, "Swin_V2_S_Weights", None),
            "swin_v2_b": getattr(models, "Swin_V2_B_Weights", None),
        }
        default_attr = "IMAGENET1K_V1"

        ctor = ctor_map[name]
        weights = None
        if pretrained:
            enum = weights_enums[name]
            if enum is not None:
                try:
                    weights = getattr(enum, default_attr)
                except Exception:
                    weights = None

        try:
            swin = ctor(weights=weights if pretrained else None)
        except TypeError:
            swin = ctor(pretrained=pretrained)

        # Swin Transformer head is usually a Linear layer; get its in_features
        feat_dim: Optional[int] = None
        if hasattr(swin, "head") and hasattr(swin.head, "in_features"):
            feat_dim = swin.head.in_features
        else:
            # Fallback: try to find any Linear layer
            for m in swin.modules():
                if isinstance(m, nn.Linear):
                    feat_dim = m.in_features
                    break
        
        if feat_dim is None:
            raise RuntimeError(f"Could not infer feature dimension for {name}")

        swin.head = nn.Identity()
        return swin, feat_dim

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.feature_extractor(x)
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            feats = feats.flatten(1)  # safety for rare shapes
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits

    # -------------------------
    # MC Dropout utilities
    # -------------------------
    @staticmethod
    def _set_batchnorm_eval(module: nn.Module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()

    @staticmethod
    def _set_dropout_train(module: nn.Module):
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    def enable_mc_dropout(self):
        """Activate dropout layers while leaving other layers as-is."""
        self.apply(self._set_dropout_train)

    @torch.no_grad()
    def mc_predict(
        self,
        x: torch.Tensor,
        T: int = 20,
        return_std: bool = True,
        apply_softmax: bool = True,
    ):
        """
        Perform T stochastic passes with dropout active and BN frozen.
        Returns mean (and optional std) of probs (or logits if apply_softmax=False).
        """
        was_training = self.training
        try:
            self.train(True)
            self.apply(self._set_batchnorm_eval)
            self.apply(self._set_dropout_train)

            all_logits = []
            all_probs = []
            for _ in range(T):
                logits = self.forward(x)
                all_logits.append(logits)
                all_probs.append(F.softmax(logits, dim=-1) if apply_softmax else logits)

            logits_stack = torch.stack(all_logits, 0)  # (T, B, C)
            probs_stack = torch.stack(all_probs, 0)    # (T, B, C)
            mean_logits = logits_stack.mean(0)
            mean_probs = probs_stack.mean(0)
            if return_std:
                std = logits_stack.std(0, unbiased=False)
                return mean_logits, mean_probs, std
            return mean_logits, mean_probs
        finally:
            self.train(was_training)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    for arch in ["swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b", "resnet18", "resnet34", "resnet50", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]:
        print(f"Testing BaselineClassifier with arch={arch}")
        model = BaselineClassifier(
            arch=arch,
            num_classes=10,
            dropout_p=0.3,
            pretrained=False,
        )
        x = torch.randn(4, 3, 224, 224)

        logits = model(x)  # standard forward
        assert logits.shape == (4, 10)
        # mean_logits, mean_probs, std = model.mc_predict(x, T=5, return_std=True, apply_softmax=True)
        # assert mean_logits.shape == (4, 10)
        # assert mean_probs.shape == (4, 10)
        # assert std.shape == (4, 10)
