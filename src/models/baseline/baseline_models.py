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

    `temperature` is this family's single post-hoc calibration knob (Guo et al. 2017):
    `logits / T`. It is never trained or swept -- training always runs at the config
    default `1.0` -- and is fit afterwards on validation NLL by
    `scripts/checkpoints/calibrate_checkpoint.py`, which writes the fitted value into the
    checkpoint's `net_spec` so inference picks it up without a config edit. It is the
    counterpart of SNGP's `mean_field_factor` (the reference's collapsed lambda * sigma^2), so the
    two families are compared with the same one-scalar post-hoc freedom. Applied
    unconditionally (train, eval and MC-Dropout passes alike): `/ 1.0` is bit-exact, and a
    single code path means `mc_forward_samples` inherits the fitted temperature too.
    """

    def __init__(
        self,
        arch: Backbone = "resnet50",
        num_classes: int = 2,
        dropout_p: float = 0.5,
        pretrained: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.backbone_name = arch
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.pretrained = pretrained
        self.temperature = float(temperature)

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
            "temperature": self.temperature,
        }

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor, return_features: bool = False) -> ModelOutput:
        feats = self.feature_extractor(x)
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            feats = feats.flatten(1)  # safety for rare shapes
        # Post-hoc temperature; a plain attribute (not a buffer) so the state dict of
        # every existing checkpoint still loads with strict=True.
        logits = self.classifier(feats) / self.temperature
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
    def mc_forward_samples(self, x: torch.Tensor, T: int = 20) -> torch.Tensor:
        """Perform T stochastic passes with dropout active and BN frozen.

        Returns the per-pass logits, shape `(T, B, C)` -- the raw material both
        `mc_predict` (mean/std reduction) and inference's optional per-pass
        persistence (`infer.save.save_member_logits`) build on, so each pass is
        computed exactly once regardless of which of those a caller wants.
        """
        was_training = self.training
        try:
            self.train(True)
            self.apply(self._set_batchnorm_eval)
            self.apply(self._set_dropout_train)

            all_logits = [self.forward(x).logits for _ in range(T)]
            return torch.stack(all_logits, 0)  # (T, B, C)
        finally:
            self.train(was_training)

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
        logits_stack = self.mc_forward_samples(x, T=T)  # (T, B, C)
        probs_stack = F.softmax(logits_stack, dim=-1) if apply_softmax else logits_stack

        mean_logits = logits_stack.mean(0)
        mean_probs = probs_stack.mean(0)
        if return_std:
            std = logits_stack.std(0, unbiased=False)
            return mean_logits, mean_probs, std
        return mean_logits, mean_probs
