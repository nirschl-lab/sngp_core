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
    n_power_iterations: int = 1,
    skip_if_has_weight_orig: bool = False
) -> None:
    """
    Recursively apply spectral normalization to Conv/Linear layers.
    
    Args:
        module: Module to apply spectral norm to
        n_power_iterations: Power iterations for spectral norm
        skip_if_has_weight_orig: If True, skip applying SN to layers that already have weight_orig
                                  (useful when loading checkpoints with pre-existing SN)
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            # Skip if already has spectral norm applied
            if hasattr(child, 'weight_u'):
                continue
            # Skip if weight_orig exists (pre-existing spectral norm from checkpoint)
            if skip_if_has_weight_orig and hasattr(child, 'weight_orig'):
                continue
            
            sn = spectral_norm(child, n_power_iterations=n_power_iterations)
            setattr(module, name, sn)
        else:
            apply_spectral_norm_to_convs(child, n_power_iterations=n_power_iterations, skip_if_has_weight_orig=skip_if_has_weight_orig)


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
    """
    ResNet backbone (torchvision) with spectral normalization + RFF-GP head.
    """

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
        apply_spectral_norm: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # --- Backbone ---
        if arch in {"resnet18", "resnet34", "resnet50"}:
            if arch == "resnet18":
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                base = resnet18(weights=weights)
                feat_dim = base.fc.in_features
            elif arch == "resnet34":
                weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                base = resnet34(weights=weights)
                feat_dim = base.fc.in_features
            elif arch == "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                base = resnet50(weights=weights)
                feat_dim = base.fc.in_features

            # Remove original classifier
            modules = list(base.children())[:-1]  # keep up to global avgpool
            self.backbone = nn.Sequential(*modules)  # outputs [B, feat_dim, 1, 1]

            # Pool + flatten
            self.pool = nn.Identity()  # resnet already has avgpool at [-2]
            self.flatten = nn.Flatten()

        elif arch in {"vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"}:
            # Construct ViT with optional ImageNet weights
            if arch == "vit_b_16":
                weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
                base = vit_b_16(weights=weights)
            elif arch == "vit_b_32":
                weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
                base = vit_b_32(weights=weights)
            elif arch == "vit_l_16":
                weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
                base = vit_l_16(weights=weights)
            elif arch == "vit_l_32":
                weights = ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None
                base = vit_l_32(weights=weights)
            elif arch == "vit_h_14":
                weights = ViT_H_14_Weights.IMAGENET1K_V1 if pretrained else None
                base = vit_h_14(weights=weights)

            # Grab the incoming feature size from the existing head, then strip it
            # torchvision ViT uses a Heads block -> final Linear; we read its in_features
            feat_dim = None
            for m in base.heads.modules():
                if isinstance(m, nn.Linear):
                    feat_dim = m.in_features
                    break
            if feat_dim is None:
                # Fallback to hidden_dim if present
                feat_dim = getattr(base, "hidden_dim", 768)

            base.heads = nn.Identity()  # expose class-token representation [B, feat_dim]

            self.backbone = base        # forward now returns [B, feat_dim]
            self.pool = nn.Identity()   # no pooling for ViT
            self.flatten = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        # Apply spectral norm to all convs/linears in the backbone
        # Skip this if loading from checkpoint where weights are already in spectral norm form
        if apply_spectral_norm:
            apply_spectral_norm_to_convs(self.backbone, n_power_iterations=n_power_iterations_sn)


        # --- RFF-GP head ---
        self.gp_head = RandomFeatureGaussianProcess(
            in_dim=feat_dim,
            num_classes=num_classes,
            rff_dim=rff_dim,
            length_scale=length_scale,
            ridge_penalty=ridge_penalty,
            cov_momentum=cov_momentum,
            mean_field=mean_field,
        )

    def forward(self, x: torch.Tensor, update_cov: bool = True):
        """
        Returns:
        mean_field_logits, raw_logits, pred_var
        """
        feats = self.backbone(x)

        # Some backbones may return tuples (e.g., aux outputs). Keep the main tensor.
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        # ResNet: [B, C, 1, 1]  -> flatten to [B, C]
        if feats.dim() == 4:
            feats = self.pool(feats)    # no-op for your ResNet setup, keeps [B, C, 1, 1]
            feats = self.flatten(feats) # -> [B, C]

        # Transformer variants that might return sequences: [B, N, D]
        elif feats.dim() == 3:
            # Prefer class token if present; otherwise fallback to mean-pool the sequence
            feats = feats[:, 0] if getattr(self, "use_cls_token", True) else feats.mean(dim=1)

        # ViT (torchvision with heads=Identity) already returns [B, D]; nothing to do for dim()==2

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
            apply_spectral_norm=False,  # No spectral norm - weights are pre-normalized
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model from pretrained checkpoint."""
        # Use the standard HuggingFace loading mechanism
        # The checkpoint has weight_orig, weight_u, weight_v which match the spectral_norm structure
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Standard state dict loading - checkpoint has spectral norm components."""
        # Call parent implementation
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
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
