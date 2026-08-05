"""
HuggingFace-compatible model wrapper for BaselineClassifier.
This allows model loading without dependency on the source code.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BaselineClassifierConfig(PretrainedConfig):
    """Configuration class for BaselineClassifier."""
    model_type = "baseline_classifier"
    
    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 2,
        dropout_p: float = 0.5,
        pretrained: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.arch = arch
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.pretrained = pretrained


class BaselineClassifier(nn.Module):
    """
    Classification model with selectable ResNet/ViT backbone.
    Grabs penultimate features, then applies Dropout + Linear.
    Includes helpers for Monte Carlo Dropout inference.
    """

    def __init__(
        self,
        arch: str = "resnet50",
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
        else:
            raise ValueError(f"Unsupported backbone: {arch}")

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(feat_dim, num_classes),
        )

    def _build_resnet(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        from torchvision import models
        
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
                try:
                    weights = getattr(enum, default_weights_attr[name])
                except Exception:
                    weights = None
        
        try:
            model = ctor(weights=weights if pretrained else None)
        except TypeError:
            model = ctor(pretrained=pretrained)

        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    def _build_vit(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        from torchvision import models
        
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

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.feature_extractor(x)
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            feats = feats.flatten(1)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits

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

            logits_stack = torch.stack(all_logits, 0)
            probs_stack = torch.stack(all_probs, 0)
            mean_logits = logits_stack.mean(0)
            mean_probs = probs_stack.mean(0)
            if return_std:
                std = logits_stack.std(0, unbiased=False)
                return mean_logits, mean_probs, std
            return mean_logits, mean_probs
        finally:
            self.train(was_training)


class BaselineClassifierForImageClassification(PreTrainedModel):
    """
    HuggingFace-compatible wrapper for BaselineClassifier.
    
    This allows the model to be loaded with:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("org/my-model", trust_remote_code=True)
    """
    config_class = BaselineClassifierConfig
    base_model_prefix = "model"

    def __init__(self, config: BaselineClassifierConfig):
        super().__init__(config)
        self.model = BaselineClassifier(
            arch=config.arch,
            num_classes=config.num_classes,
            dropout_p=config.dropout_p,
            pretrained=config.pretrained,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
        return_features: bool = False,
    ):
        """
        Args:
            pixel_values: Input tensor of shape (batch_size, 3, 224, 224)
            return_dict: Whether to return dict or tuple
            return_features: Whether to return intermediate features
        """
        if return_features:
            logits, features = self.model(pixel_values, return_features=True)
            if return_dict:
                return {"logits": logits, "features": features}
            return logits, features
        
        logits = self.model(pixel_values)
        if return_dict:
            return {"logits": logits}
        return logits
