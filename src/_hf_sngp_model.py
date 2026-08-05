"""
HuggingFace-compatible SNGP (Spectral-normalized Neural Gaussian Process) model wrapper.
Enables model loading without dependency on source code.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import logging
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights,
    ViT_L_32_Weights, ViT_H_14_Weights,
)

logger = logging.get_logger(__name__)


class SNGPConfig(PretrainedConfig):
    """Configuration class for SNGP model."""
    model_type = "sngp_classifier"

    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 2,
        rff_dim: int = 1024,
        length_scale: float = 1.0,
        ridge_penalty: float = 1e-3,
        cov_momentum: float = 0.999,
        mean_field: bool = True,
        n_power_iterations_sn: int = 1,
        pretrained: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.arch = arch
        self.num_classes = num_classes
        self.rff_dim = rff_dim
        self.length_scale = length_scale
        self.ridge_penalty = ridge_penalty
        self.cov_momentum = cov_momentum
        self.mean_field = mean_field
        self.n_power_iterations_sn = n_power_iterations_sn
        self.pretrained = pretrained


def apply_spectral_norm_to_convs(
    module: nn.Module,
    n_power_iterations: int = 1
) -> None:
    """Recursively apply spectral normalization to Conv/Linear layers."""
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            if not hasattr(child, 'weight_u'):
                sn = spectral_norm(child, n_power_iterations=n_power_iterations)
                setattr(module, name, sn)
        else:
            apply_spectral_norm_to_convs(child, n_power_iterations=n_power_iterations)


class RandomFeatureGaussianProcess(nn.Module):
    """
    RFF-GP output layer for uncertainty quantification.
    
    Uses Random Fourier Features with Gaussian Process posterior to provide:
    - Mean-field logits (calibrated predictions)
    - Raw logits (unscaled)
    - Predictive variance (uncertainty estimates)
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        rff_dim: int = 1024,
        length_scale: float = 1.0,
        ridge_penalty: float = 1e-3,
        cov_momentum: float = 0.999,
        mean_field: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.rff_dim = rff_dim
        self.length_scale = length_scale
        self.ridge = ridge_penalty
        self.cov_momentum = cov_momentum
        self.mean_field = mean_field

        # Random Fourier feature parameters (fixed)
        W = torch.randn(in_dim, rff_dim, dtype=dtype) / length_scale
        b = 2 * math.pi * torch.rand(rff_dim, dtype=dtype)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        # Linear classifier over RFFs (learned)
        self.classifier = nn.Linear(rff_dim, num_classes, bias=True)

        # EMA covariance of features
        C = torch.zeros(rff_dim, rff_dim, dtype=dtype)
        self.register_buffer("cov_ema", C)
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        # Identity for covariance computation
        eye = torch.eye(rff_dim, dtype=dtype)
        self.register_buffer("I", eye)

        self.rff_scale = math.sqrt(2.0 / rff_dim)

    @torch.no_grad()
    def _update_cov(self, phi: torch.Tensor) -> None:
        """Update exponential moving average of feature covariance."""
        B = phi.shape[0]
        batch_cov = (phi.T @ phi) / max(1, B)
        if self.num_updates == 0:
            self.cov_ema.copy_(batch_cov)
        else:
            self.cov_ema.mul_(self.cov_momentum).add_(
                (1.0 - self.cov_momentum) * batch_cov
            )
        self.num_updates += 1

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Random Fourier Features: phi(x) = sqrt(2/m) * cos(x W + b)."""
        proj = x @ self.W + self.b
        phi = torch.cos(proj) * self.rff_scale
        return phi

    def forward(
        self,
        x: torch.Tensor,
        update_cov: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [B, in_dim]
            update_cov: Update covariance during training

        Returns:
            mean_field_logits: Calibrated logits
            raw_logits: Uncalibrated logits
            pred_var: Predictive variance [B, 1]
        """
        phi = self._features(x)

        # Update covariance during training
        if self.training and update_cov:
            with torch.no_grad():
                self._update_cov(phi)

        raw_logits = self.classifier(phi)

        # Compute predictive variance via GP approximation
        with torch.no_grad():
            A = (self.cov_ema + self.ridge * self.I).to(phi.dtype).to(phi.device)
            L = torch.linalg.cholesky(A)
            phi_T = phi.T
            y = torch.linalg.solve_triangular(L, phi_T, upper=False)
            z = torch.linalg.solve_triangular(L.T, y, upper=True)
            solved = z.T
            pred_var = (phi * solved).sum(dim=1, keepdim=True)
            pred_var = torch.clamp(pred_var, min=0.0)

        # Apply mean-field logit correction
        if self.mean_field:
            denom = torch.sqrt(1.0 + pred_var)
            mean_field_logits = raw_logits / denom
        else:
            mean_field_logits = raw_logits

        return mean_field_logits, raw_logits, pred_var


class SNGPClassifier(nn.Module):
    """SNGP classifier with spectral normalization and RFF-GP head."""

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        pretrained: bool = False,
        rff_dim: int = 1024,
        length_scale: float = 1.0,
        ridge_penalty: float = 1e-3,
        cov_momentum: float = 0.999,
        mean_field: bool = True,
        n_power_iterations_sn: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.arch = arch

        # Build backbone
        self.backbone, feat_dim = self._build_backbone(
            arch, pretrained, n_power_iterations_sn
        )

        # RFF-GP head
        self.gp_head = RandomFeatureGaussianProcess(
            in_dim=feat_dim,
            num_classes=num_classes,
            rff_dim=rff_dim,
            length_scale=length_scale,
            ridge_penalty=ridge_penalty,
            cov_momentum=cov_momentum,
            mean_field=mean_field,
        )

    def _build_backbone(
        self,
        arch: str,
        pretrained: bool,
        n_power_iterations_sn: int
    ) -> Tuple[nn.Module, int]:
        """Build backbone and return (model, feature_dim)."""

        if arch in {"resnet18", "resnet34", "resnet50"}:
            weights_map = {
                "resnet18": (resnet18, ResNet18_Weights),
                "resnet34": (resnet34, ResNet34_Weights),
                "resnet50": (resnet50, ResNet50_Weights),
            }
            ctor, weights_enum = weights_map[arch]
            weights = weights_enum.IMAGENET1K_V1 if pretrained else None

            base = ctor(weights=weights)
            feat_dim = base.fc.in_features

            # Remove classifier
            modules = list(base.children())[:-1]
            backbone = nn.Sequential(*modules)

        elif arch in {"vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"}:
            weights_map = {
                "vit_b_16": (vit_b_16, ViT_B_16_Weights),
                "vit_b_32": (vit_b_32, ViT_B_32_Weights),
                "vit_l_16": (vit_l_16, ViT_L_16_Weights),
                "vit_l_32": (vit_l_32, ViT_L_32_Weights),
                "vit_h_14": (vit_h_14, ViT_H_14_Weights),
            }
            ctor, weights_enum = weights_map[arch]
            weights = weights_enum.IMAGENET1K_V1 if pretrained else None

            base = ctor(weights=weights)

            # Extract feature dimension
            feat_dim = None
            for m in base.heads.modules():
                if isinstance(m, nn.Linear):
                    feat_dim = m.in_features
                    break
            if feat_dim is None:
                feat_dim = getattr(base, "hidden_dim", 768)

            base.heads = nn.Identity()
            backbone = base

        else:
            raise ValueError(f"Unsupported arch: {arch}")

        # Apply spectral normalization
        apply_spectral_norm_to_convs(backbone, n_power_iterations=n_power_iterations_sn)

        return backbone, feat_dim

    def forward(self, x: torch.Tensor, update_cov: bool = True):
        """
        Forward pass.

        Returns:
            mean_field_logits, raw_logits, pred_var
        """
        feats = self.backbone(x)

        # Handle tuple returns
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        # ResNet: [B, C, 1, 1] -> [B, C]
        if feats.dim() == 4:
            feats = feats.flatten(1)

        # ViT sequence: [B, N, D] -> [B, D] (class token)
        elif feats.dim() == 3:
            feats = feats[:, 0]

        return self.gp_head(feats, update_cov=update_cov)


class SNGPForImageClassification(PreTrainedModel):
    """
    HuggingFace-compatible wrapper for SNGP.

    Load with:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("org/my-model", trust_remote_code=True)
    """
    config_class = SNGPConfig
    base_model_prefix = "model"

    def __init__(self, config: SNGPConfig):
        super().__init__(config)
        self.model = SNGPClassifier(
            num_classes=config.num_classes,
            arch=config.arch,
            pretrained=config.pretrained,
            rff_dim=config.rff_dim,
            length_scale=config.length_scale,
            ridge_penalty=config.ridge_penalty,
            cov_momentum=config.cov_momentum,
            mean_field=config.mean_field,
            n_power_iterations_sn=config.n_power_iterations_sn,
        )

    def _load_state_dict_into_model(self, model, state_dict, model_name_or_path, *args, **kwargs):
        """
        Override to filter out spectral norm parametrization keys that may not exist
        in freshly created models.
        """
        # Filter out parametrization keys (from spectral norm)
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if "parametrizations" not in k
        }

        # Load the filtered state dict
        return super()._load_state_dict_into_model(
            model, filtered_state_dict, model_name_or_path, *args, **kwargs
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
        update_cov: bool = True,
    ):
        """
        Args:
            pixel_values: Input tensor [B, 3, 224, 224]
            return_dict: Whether to return dict
            update_cov: Update GP covariance during training

        Returns:
            Dict with mean_field_logits, raw_logits, pred_var
        """
        mean_field_logits, raw_logits, pred_var = self.model(
            pixel_values,
            update_cov=update_cov
        )

        if return_dict:
            return {
                "mean_field_logits": mean_field_logits,
                "logits": mean_field_logits,  # For compatibility
                "raw_logits": raw_logits,
                "pred_var": pred_var,
            }

        return mean_field_logits, raw_logits, pred_var
