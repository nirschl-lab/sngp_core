from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone
from src.models.outputs import ModelOutput
from src.models.registry import register_net

Backbone = Literal[
    "resnet18", "resnet34", "resnet50",
    "vit_b_16", "vit_b_32",
    "vit_l_16", "vit_l_32",
    "vit_h_14",
]


@register_net("baseline_classifier")
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

        self.feature_extractor, feat_dim = build_backbone(arch, pretrained)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(feat_dim, num_classes),
        )

    @property
    def spec(self) -> dict:
        """Plain-data description of this net, sufficient to rebuild it via `build_net`."""
        return {
            "name": self.registry_name,
            "arch": self.backbone_name,
            "num_classes": self.num_classes,
            "dropout_p": self.dropout_p,
            "pretrained": self.pretrained,
        }

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor, return_features: bool = False) -> ModelOutput:
        feats = self.feature_extractor(x)
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            feats = feats.flatten(1)  # safety for rare shapes
        logits = self.classifier(feats)
        return ModelOutput(logits=logits, features=feats if return_features else None)

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
                logits = self.forward(x).logits
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
