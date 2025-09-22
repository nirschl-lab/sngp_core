import timm
import torch
import torch.nn as nn
from typing import Optional, Iterable
import pdb
# OPTIONAL: control where timm caches model weights
# timm exposes this via timm.models.hub.set_model_cache_dir
def _maybe_set_timm_cache_dir(cache_dir: Optional[str]):
    if cache_dir:
        try:
            from timm.models import hub as timm_hub
            timm_hub.set_model_cache_dir(cache_dir)
        except Exception as e:
            # Fall back silently if the API surface changes; user can still set TORCH_HOME
            print(f"[timm-cache] Could not set cache dir to '{cache_dir}': {e}")

class TimmBasicClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, in_chans=3, cache_dir='timm_cache_dir'):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            cache_dir=cache_dir
        )
    
    def forward(self, x):
        return self.backbone(x)


class TimmBackboneWithProbe(nn.Module):
    """
    Build a classifier-free timm backbone, optionally freeze it, normalize
    features to a fixed dimension, and attach a custom probe.

    Key options:
      - cache_dir: where timm stores weights (e.g., '/data/timm_cache')
      - freeze_backbone: set requires_grad=False for all backbone params
      - use_model_head_pool: try model.forward_head(..., pre_logits=True)
      - custom_pool: 'avg' | 'max' | 'gem' if not using model head pooling
    """
    def __init__(
        self,
        model_name: str,
        proj_dim: int = 512,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        use_model_head_pool: bool = False,
        custom_pool: str = "avg",
        in_chans: int = 3,
        cache_dir: Optional[str] = None,   # NEW: timm cache path
        freeze_backbone: bool = True,     # NEW: freeze switch
    ):
        super().__init__()

        # (1) Set timm weight cache directory (optional)
        # _maybe_set_timm_cache_dir(cache_dir)

        # (2) Build classifier-free backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # removes classifier head
            in_chans=in_chans,
            global_pool='',   # we control pooling
            cache_dir=cache_dir 
        )
        self.use_model_head_pool = use_model_head_pool and hasattr(self.backbone, "forward_head")

        # (3) Optional custom pooling if not using model head
        if not self.use_model_head_pool:
            if custom_pool == "avg":
                self.pool = nn.AdaptiveAvgPool2d(1)
            elif custom_pool == "max":
                self.pool = nn.AdaptiveMaxPool2d(1)
            elif custom_pool == "gem":
                class GeM(nn.Module):
                    def __init__(self, p=3.0, eps=1e-6):
                        super().__init__()
                        self.p = nn.Parameter(torch.ones(1) * p)
                        self.eps = eps
                    def forward(self, x):
                        x = x.clamp(min=self.eps).pow(self.p)
                        return x.mean(dim=(-1, -2)).pow(1.0 / self.p)  # [B, C]
                self.pool = GeM()
            else:
                raise ValueError(f"Unknown custom_pool: {custom_pool}")

        # (4) Projection to fixed dimension
        in_feats = getattr(self.backbone, "num_features", None)
        print('in_feats -> ', in_feats)
        self.proj = nn.Linear(in_feats, proj_dim) if (in_feats is not None and in_feats > 0) else nn.LazyLinear(proj_dim)

        # (5) Classification probe (optional)
        self.probe = nn.Linear(proj_dim, num_classes) if num_classes is not None else None

        # (6) Freeze policy
        if freeze_backbone:
            self.freeze_backbone()

    # ---------- Freezing Utilities ----------

    def freeze_backbone(self, except_modules: Optional[Iterable[str]] = None, verbose: bool = False):
        """
        Freeze all backbone params except those whose module name contains any token in `except_modules`.
        Useful for leaving norms or a last stage trainable.
        """
        except_modules = set(except_modules or [])
        for name, p in self.backbone.named_parameters():
            keep = any(tok in name for tok in except_modules) if except_modules else False
            p.requires_grad = keep is True
            if verbose and (keep or not p.requires_grad):
                print(f"[freeze] {name}: requires_grad={p.requires_grad}")
        return self

    def unfreeze_backbone(self, only_modules: Optional[Iterable[str]] = None, verbose: bool = False):
        """
        Unfreeze all backbone params, or only those whose module name contains any token in `only_modules`.
        """
        only_modules = set(only_modules or [])
        for name, p in self.backbone.named_parameters():
            if only_modules:
                if any(tok in name for tok in only_modules):
                    p.requires_grad = True
                    if verbose:
                        print(f"[unfreeze] {name}: requires_grad=True")
            else:
                p.requires_grad = True
        return self

    # ---------- Forward Path ----------

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a fixed-dim feature vector [B, proj_dim].
        """
        f = self.backbone.forward_features(x)
        # print('backbone features shape', f.shape)
        # pdb.set_trace()

        if self.use_model_head_pool:
            # Many timm models support forward_head(..., pre_logits=True) -> [B, C]
            f = self.backbone.forward_head(f, pre_logits=True) ##### This is not working for all models 
        else:
            # Fallback pooling depending on shape
            if f.ndim == 4:            # [B, C, H, W]
                if hasattr(self, "pool"):
                    if isinstance(self.pool, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                        f = self.pool(f).flatten(1)
                    else:  # GeM returns [B, C]
                        f = self.pool(f)
                else:
                    f = f.mean(dim=(-1, -2))  # safe default
            elif f.ndim == 3:          # [B, N, C] (token models)
                f = f.mean(dim=1)      # mean-pool tokens (you can swap for CLS)
            elif f.ndim == 2:          # [B, C]
                pass
            else:
                raise RuntimeError(f"Unexpected feature shape: {f.shape}")

        f = self.proj(f)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.extract_features(x)        # [B, proj_dim]
        return self.probe(f) if self.probe is not None else f

if __name__ == '__main__':
    cache_dir = 'timm_cache_dir'
    model = TimmBackboneWithProbe(
        "resnet50",
        proj_dim=512,
        num_classes=10,
        pretrained=True,
        cache_dir=cache_dir,   # NEW
        freeze_backbone=True            # NEW
    )
    # Test the model with random input
    batch_size = 2
    in_chans = 3
    height = width = 224
    x = torch.randn(batch_size, in_chans, height, width)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")



