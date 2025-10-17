#!/usr/bin/env python3
"""TIMM classifier models with Spectral-normalized Gaussian Process head."""
import os
from typing import Iterable, Optional, Tuple

import timm
import torch
import torch.nn as nn

from src.models.sngp.sngp_classification_layer import SNGP
from torch.nn.utils import spectral_norm
import torch.nn.utils.parametrize as parametrize
from loguru import logger


class ScaledWeightParam(torch.nn.Module):
    """Applies a scalar multiplier to a layer's weight for spectral normalization bound."""
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, w):
        return w * self.scale


def apply_spectral_norm_to_model(
    model: nn.Module,
    spec_norm_bound: float = 1.0,
    spec_norm_iteration: int = 1,
    verbose: bool = False,
) -> nn.Module:
    """
    Recursively apply spectral normalization to Conv2d and Linear layers.

    Args:
        model: Model or submodule to modify.
        spec_norm_bound: Maximum spectral norm (like TF norm_multiplier).
        spec_norm_iteration: Power iteration count (like TF iteration).
        verbose: Print wrapped layers if True.

    Returns:
        Model with spectral normalization applied in-place.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if not hasattr(module, "weight_u"):
                try:
                    # apply spectral normalization
                    spectral_norm(module, n_power_iterations=spec_norm_iteration)
                    if spec_norm_bound != 1.0 and isinstance(module, nn.Linear) and hasattr(module, "weight_u") :
                        # apply scaling parametrization (currently only works for nn.Linear)
                        parametrize.register_parametrization(
                            module, "weight", ScaledWeightParam(spec_norm_bound)
                        )

                    # ensure no NaNs in weights after init
                    if torch.isnan(module.weight).any():
                        raise ValueError(f"NaN detected in weights of layer {name} after spectral norm application")

                    if verbose:
                        print(f"Applied SN to {name}: iter={spec_norm_iteration}, bound={spec_norm_bound}")
                except Exception as e:
                    error_msg = f"Failed to wrap {name}: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

        else:
            apply_spectral_norm_to_model(module, spec_norm_bound, spec_norm_iteration, verbose)
    return model


class TimmBasicClassifier(nn.Module):
    """Standard TIMM model with classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        in_chans: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TimmDropOutBasicClassifier(nn.Module):
    """TIMM model with dropout-based classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        in_chans: int = 3,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )
        in_features = (
            self.model.get_classifier().in_features
            if hasattr(self.model.get_classifier(), "in_features")
            else self.model.fc.in_features
        )
        self.model.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TimmSNGPClassifier(nn.Module):
    """TIMM model with all conv and linear layers spectral-normalized, as well as a spectral normalized Gaussian Process head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        in_chans: int = 3,
        reduction_dim: int = 512,
        use_spec_norm: bool = True,
        spec_norm_bound: float =0.95,
        spec_norm_iteration: int =1,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.error_flag = False

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
            in_chans=in_chans,
        )

        # apply spectral normalization recursively
        if use_spec_norm:
            logger.debug("Applying spectral normalization to backbone")
            self.backbone = apply_spectral_norm_to_model(
                self.backbone,
                spec_norm_bound=spec_norm_bound,
                spec_norm_iteration=spec_norm_iteration,
            )

        # build proj + SNGP head
        in_feats = getattr(self.backbone, "num_features", None)
        if in_feats is None:
            raise ValueError(f"Backbone {model_name} has no num_features attribute")

        # TODO check TF impl to see if necessary
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce_dim = (
            nn.utils.spectral_norm(nn.Linear(in_feats, reduction_dim))
            if use_spec_norm
            else nn.Linear(in_feats, reduction_dim)
        )

        self.sngp_classifier = SNGP(
            in_features=reduction_dim,
            num_classes=num_classes,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, return_covariance: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x = self.backbone.forward_features(x)
        # Handle all-NaN input by replacing with small stddev random normal values
        if torch.isnan(x).all():
            if not self.error_flag:
                logger.warning("Forward features from TimmSNGPClassifier are all NaN; replacing with random normal tensor (std=1e-2)")
                x = torch.randn_like(x) * 1e-2
                self.error_flag = True
            else:
                # raise error if happens again
                logger.error("Forward features from TimmSNGPClassifier are all NaN; previously replaced once, raising error now")
                raise ValueError("Forward features from TimmSNGPClassifier are all NaN")

        x = self.pool(x).flatten(1)
        x = self.reduce_dim(x)
        outputs =  self.sngp_classifier(x) # returns dict with logits, logits_raw, cov, features
        if not self.training and return_covariance:
            # always return both logits and covariance at eval/test time for mean-field scaling
            return outputs.get("logits"), outputs.get("covariance")
        else:
            # only return logits during training
            return outputs.get("logits")

            


if __name__ == "__main__":
    # Smoke test
    x = torch.randn(2, 3, 224, 224)
    model = TimmSNGPClassifier(
        "resnet50", num_classes=8, pretrained=True, reduction_dim=512
    )
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
