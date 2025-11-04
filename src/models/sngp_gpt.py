from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple
import timm
import math

def apply_spectral_norm_to_convs(module: nn.Module, n_power_iterations: int = 1) -> None:
    """
    Recursively wrap Conv/Linear layers with spectral normalization.
    Bias is left untouched. BatchNorm layers are skipped.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            # Avoid wrapping twice
            if not hasattr(child, 'weight_u'):
                sn = spectral_norm(child, n_power_iterations=n_power_iterations)
                setattr(module, name, sn)
        else:
            apply_spectral_norm_to_convs(child, n_power_iterations=n_power_iterations)


# ---------------------------
# Random Fourier Feature GP head
# ---------------------------

class RandomFeatureGaussianProcess(nn.Module):
    """
    RFF-GP output layer as used in SNGP:
      - Fixed random Fourier features phi(x) ~ sqrt(2/m)*cos(Wx + b)
      - Linear classifier over phi(x) trained via standard CE
      - Maintain EMA covariance of phi(x) to compute predictive variance
      - Mean-field logit correction: logits / sqrt(1 + var), improving calibration

    This module returns:
      - mean_field_logits: calibrated logits for CE / evaluation
      - raw_logits: unscaled logits (useful for analysis)
      - pred_var: predictive variance per example (shape: [B, 1])
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

        # Random Fourier feature parameters (fixed, not learned)
        # W ~ N(0, 1/length_scale^2), b ~ Uniform(0, 2pi)
        W = torch.randn(in_dim, rff_dim, dtype=dtype) / length_scale
        b = 2 * math.pi * torch.rand(rff_dim, dtype=dtype)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        # Linear classifier over RFFs (learned)
        self.classifier = nn.Linear(rff_dim, num_classes, bias=True)

        # EMA covariance of phi(x): C ≈ E[phi^T phi] / B
        # We maintain the second-moment matrix S = E[phi^T phi] (size rff_dim x rff_dim)
        C = torch.zeros(rff_dim, rff_dim, dtype=dtype)
        self.register_buffer("cov_ema", C)
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        # Add a small jitter to stabilize initial inverses
        eye = torch.eye(rff_dim, dtype=dtype)
        self.register_buffer("I", eye)

        # Pre-scaling constant for RFFs
        self.rff_scale = math.sqrt(2.0 / rff_dim)

    @torch.no_grad()
    def _update_cov(self, phi: torch.Tensor) -> None:
        """
        Update the EMA of feature covariance using the current batch of features.
        phi: [B, rff_dim]
        """
        B = phi.shape[0]
        batch_cov = (phi.T @ phi) / max(1, B)  # [rff_dim, rff_dim]
        if self.num_updates == 0:
            self.cov_ema.copy_(batch_cov)
        else:
            self.cov_ema.mul_(self.cov_momentum).add_((1.0 - self.cov_momentum) * batch_cov)
        self.num_updates += 1

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RFFs: phi(x) = sqrt(2/m) * cos(x W + b)
        x: [B, in_dim]
        returns: [B, rff_dim]
        """
        proj = x @ self.W  # [B, rff_dim]
        proj = proj + self.b  # broadcast
        phi = torch.cos(proj) * self.rff_scale
        return phi

    def forward(self, x: torch.Tensor, update_cov: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, in_dim] pooled features
        update_cov: set False during pure eval if you want a frozen covariance
        """
        phi = self._features(x)  # [B, rff_dim]

        # Update running covariance estimate
        if self.training and update_cov:
            with torch.no_grad():
                self._update_cov(phi)

        # Classifier logits
        raw_logits = self.classifier(phi)  # [B, num_classes]

        # Predictive variance via GP posterior variance approximation:
        # var(x) ≈ phi(x) @ ( (cov + ridge*I)^(-1) ) @ phi(x)^T
        # Compute A = cov + ridge*I with EMA covariance
        with torch.no_grad():
            A = (self.cov_ema + self.ridge * self.I).to(phi.dtype).to(phi.device)
            L = torch.linalg.cholesky(A)                        # [rff_dim, rff_dim]

            # Solve A^{-1} @ phi_T via two triangular solves:
            # A = L L^T ⇒ A^{-1} @ B = L^{-T} (L^{-1} B)
            phi_T = phi.T                                       # [rff_dim, B]
            y = torch.linalg.solve_triangular(L,     phi_T, upper=False)  # [rff_dim, B]
            z = torch.linalg.solve_triangular(L.T,   y,     upper=True)   # [rff_dim, B]

            solved = z.T                                        # [B, rff_dim]
            pred_var = (phi * solved).sum(dim=1, keepdim=True)  # [B, 1]
            pred_var = torch.clamp(pred_var, min=0.0)

        if self.mean_field:
            # Mean-field logit correction
            denom = torch.sqrt(1.0 + pred_var)  # [B, 1]
            mean_field_logits = raw_logits / denom
        else:
            mean_field_logits = raw_logits

        return mean_field_logits, raw_logits, pred_var


# ---------------------------
# SNGP-ResNet wrapper
# ---------------------------

class SNGPResNet(nn.Module):
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
    ):
        super().__init__()
        self.num_classes = num_classes

        # --- Backbone ---
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
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        # Remove original classifier
        modules = list(base.children())[:-1]  # keep up to global avgpool
        self.backbone = nn.Sequential(*modules)  # outputs [B, feat_dim, 1, 1]

        # Apply spectral norm to all convs/linears in the backbone
        apply_spectral_norm_to_convs(self.backbone, n_power_iterations=n_power_iterations_sn)

        # Pool + flatten
        self.pool = nn.Identity()  # resnet already has avgpool at [-2]
        self.flatten = nn.Flatten()

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
        feats = self.backbone(x)            # [B, C, 1, 1]
        feats = self.flatten(feats)         # [B, C]
        return self.gp_head(feats, update_cov=update_cov)

class TimmSNGPClassifier(nn.Module):
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
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model(arch,
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
            in_chans=3,
        )

        
        in_feats = getattr(self.backbone, "num_features", None)
        if in_feats is None:
            raise ValueError(f"Backbone {arch} has no num_features attribute")


        apply_spectral_norm_to_convs(self.backbone, n_power_iterations=n_power_iterations_sn)

        # Pool + flatten
        self.pool = nn.Identity()  # resnet already has avgpool at [-2]
        self.flatten = nn.Flatten()

        # --- RFF-GP head ---
        self.gp_head = RandomFeatureGaussianProcess(
            in_dim=in_feats,
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
        feats = self.backbone(x)            # [B, C, 1, 1]
        feats = self.flatten(feats)         # [B, C]
        return self.gp_head(feats, update_cov=update_cov)

if __name__ == "__main__":
    # simple test
    model = TimmSNGPClassifier(
        num_classes=10,
        arch="resnet18",
        pretrained=False,
        rff_dim=512,
        length_scale=1.0,
        ridge_penalty=1e-3,
        cov_momentum=0.999,
        mean_field=True,
        n_power_iterations_sn=1,
    )
    x = torch.randn(4, 3, 224, 224)
    mean_field_logits, raw_logits, pred_var = model(x)
    print("mean_field_logits:", mean_field_logits.shape)
    print("raw_logits:", raw_logits.shape)
    print("pred_var:", pred_var.shape)
