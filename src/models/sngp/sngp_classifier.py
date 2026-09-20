import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from src.models.backbones import BACKBONES, build_backbone
from src.models.components.spectral_norm import apply_spectral_norm, assert_spectral_norm_compatible
from src.models.outputs import ModelOutput
from src.models.registry import register_net

# Mean-field multiplicative factor. This is the paper's "kernel amplitude" sigma (Liu et
# al. 2022, Table 10): the reference implementation collapses it into the single tunable
# `gp_mean_field_factor` -- 1.0 in the ImageNet SNGP baseline, 20.0 in the CIFAR one --
# so this default matches ImageNet (the closer setting to 224px resnets) and is meant to
# be fit post-hoc on validation NLL, exactly as the paper recommends for sigma. pi/8 is
# the textbook probit constant, but at realistic dataset sizes it makes the correction
# nearly inert (~2% logit shrink); see scripts/checkpoints/calibrate_checkpoint.py.
DEFAULT_MEAN_FIELD_FACTOR = 1.0
PROBIT_MEAN_FIELD_FACTOR = math.pi / 8

_LEGACY_BUFFERS = ("cov_ema", "num_updates")

_LEGACY_CKPT_MSG = (
    "This checkpoint was trained with the pre-correction SNGP head (buffers {found}). "
    "That head applied the mean-field correction inside the training loss and used an "
    "EMA of the mean second moment instead of the paper's precision matrix, so its "
    "weights are not compatible with the corrected head and cannot be meaningfully "
    "migrated. Check out the `sngp-pre-correction` tag (or the `isbi2026` branch) to "
    "use it, or retrain on this branch."
)

# ---------------------------
# Random Fourier Feature GP head
# ---------------------------

class RandomFeatureGaussianProcess(nn.Module):
    """RFF-GP output layer, following Liu et al. 2020 ("Simple and Principled
    Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness")
    and the `edward2` reference implementation.

      - Fixed random Fourier features phi(x) = sqrt(2/m) * cos(Wx + b)
      - Linear classifier over phi(x), trained with standard CE on the *raw* logits
      - Laplace precision matrix accumulated over one epoch of training
      - Mean-field logit correction applied **at inference only**

    Precision lifecycle
    -------------------
    The paper writes the posterior precision as a single sum over the whole training
    set at converged weights::

        P = ridge * I + sum_i w_i * phi_i phi_i^T

    The reference implementation does not make a separate pass for this: it resets the
    accumulator at the start of every epoch and accumulates during ordinary training
    forwards, so at any epoch boundary the accumulator holds exactly one full pass over
    the training data. The last epoch's is the one that counts -- the per-epoch reset is
    what makes it final-epoch-only. `SNGPLitModule.on_train_epoch_start` drives the
    reset; see that module for why per-epoch reset (rather than last-epoch-only) is also
    what makes `save_top_k=1` checkpointing self-consistent here.

    One approximation is inherited from the reference and kept deliberately: the
    accumulation happens while weights are still moving within the epoch and with the
    backbone in `train()` mode, so BatchNorm uses batch statistics and augmentation is
    on -- meaning `phi` differs slightly from the inference-time features the covariance
    is later applied to. This is not a bug; every published SNGP result carries it.

    The inverse is computed **lazily and cached** (`_cov_stale`), not per batch. That
    keeps validation, test, predict and post-checkpoint-load correct through one code
    path, and it matters because Lightning runs the validation loop *before*
    `on_train_epoch_end` -- finalizing in that hook would leave validation a full epoch
    behind.

    Returns `(logits, raw_logits, variance)`, where during training `logits is
    raw_logits` and `variance` is `None` (no variance is computed at all).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        rff_dim: int = 1024,
        length_scale: float = 1.0,
        ridge_penalty: float = 1e-3,
        cov_momentum: float = -1.0,
        mean_field: bool = True,
        mean_field_factor: float = DEFAULT_MEAN_FIELD_FACTOR,
        normalize_input: bool = True,
        likelihood: str = "gaussian",
        output_bias: bool = False,
        random_feature_type: str = "orf",
        scale_random_features: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if likelihood not in ("gaussian", "binary_logistic"):
            raise ValueError(f"Unsupported likelihood: {likelihood!r}. Use 'gaussian' or 'binary_logistic'.")
        if random_feature_type not in ("rff", "orf"):
            raise ValueError(f"Unsupported random_feature_type: {random_feature_type!r}. Use 'rff' or 'orf'.")
        if ridge_penalty <= 0:
            raise ValueError(f"ridge_penalty must be > 0 (it seeds the precision matrix), got {ridge_penalty}")

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.rff_dim = rff_dim
        self.length_scale = length_scale
        self.ridge = ridge_penalty
        self.cov_momentum = cov_momentum
        self.mean_field = mean_field
        self.mean_field_factor = mean_field_factor
        self.normalize_input = normalize_input
        self.likelihood = likelihood
        self.output_bias = output_bias
        self.random_feature_type = random_feature_type
        self.scale_random_features = scale_random_features

        # LayerNorm on the GP input ("similar to applying ARD", per the reference's own
        # description) -- also keeps the RFF kernel well-scaled as backbone feature
        # magnitudes drift during training.
        self.input_norm = nn.LayerNorm(in_dim) if normalize_input else None

        # Random Fourier feature parameters (fixed, not learned)
        # b ~ Uniform(0, 2pi); W columns are random directions with chi-distributed
        # norms => RBF kernel with length-scale l.
        W = self._sample_projection(in_dim, rff_dim, random_feature_type, dtype) / length_scale
        b = 2 * math.pi * torch.rand(rff_dim, dtype=dtype)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        # Linear classifier over RFFs (learned). The reference uses no trainable output
        # bias (a fixed zero `gp_output_bias`); `output_bias=True` restores one.
        self.classifier = nn.Linear(rff_dim, num_classes, bias=output_bias)

        # Precision accumulator: holds ONLY `sum_i w_i phi_i phi_i^T`. The ridge term is
        # added on demand, so the DDP all-reduce below cannot multiply it by world_size.
        self.register_buffer("precision_accum", torch.zeros(rff_dim, rff_dim, dtype=dtype))
        # Cached inverse of `ridge * I + precision_accum`.
        self.register_buffer("covariance", torch.eye(rff_dim, dtype=dtype) / ridge_penalty)
        # Non-persistent: a freshly loaded checkpoint recomputes once from
        # `precision_accum` and self-heals, so the cache can never go stale across a load.
        self.register_buffer("_cov_stale", torch.tensor(True), persistent=False)

        # Pre-scaling constant for RFFs. `scale_random_features=False` drops it (1.0),
        # which is what the reference CIFAR baseline does and what its own comment
        # recommends "when using GP layer as the output layer of a neural network ... to
        # prevent it from changing the learning rate to the hidden layers": the factor
        # sqrt(2/m) is ~0.044 at m=1024, and it multiplies the gradient flowing back into
        # the backbone. Adam rescales that away per-parameter, so it is invisible under
        # the project's AdamW protocol; plain SGD at a fixed lr just trains the backbone
        # ~22x too slowly. Default stays True so existing checkpoints are untouched.
        self.rff_scale = math.sqrt(2.0 / rff_dim) if scale_random_features else 1.0

    # -- random feature map --------------------------------------------------

    @staticmethod
    def _sample_projection(in_dim: int, rff_dim: int, kind: str, dtype: torch.dtype) -> torch.Tensor:
        """Draw the fixed `[in_dim, rff_dim]` projection whose columns are the random
        directions of the RBF feature map.

        `"rff"` draws them i.i.d. Gaussian. `"orf"` (orthogonal random features, Yu et
        al. 2016, and the default in both reference SNGP baselines) draws orthonormal
        blocks and rescales each column by an independent chi(in_dim) norm. A standard
        Gaussian vector decomposes into a uniform direction times a chi norm, so the
        per-column marginal is identical to `"rff"` -- what changes is that directions
        within a block no longer partially duplicate each other, which lowers the
        variance of the kernel approximation at a given `rff_dim`. Cost is paid once,
        at construction; the forward pass is unchanged.
        """
        if kind == "rff":
            return torch.randn(in_dim, rff_dim, dtype=dtype)

        blocks = []
        remaining = rff_dim
        while remaining > 0:
            # `torch.linalg.qr` of a square Gaussian gives Q with orthonormal columns,
            # Haar-distributed over the orthogonal group.
            q, _ = torch.linalg.qr(torch.randn(in_dim, in_dim, dtype=dtype))
            blocks.append(q[:, :min(in_dim, remaining)])
            remaining -= in_dim
        directions = torch.cat(blocks, dim=1)

        # chi(in_dim) samples: the norm of an in_dim-dimensional standard Gaussian.
        norms = torch.randn(in_dim, rff_dim, dtype=dtype).norm(dim=0)
        return directions * norms.unsqueeze(0)

    # -- precision lifecycle -------------------------------------------------

    def reset_precision(self) -> None:
        """Zero the precision accumulator. Called at the start of every training epoch."""
        self.precision_accum.zero_()
        self._cov_stale.fill_(True)

    @torch.no_grad()
    def update_precision(self, phi: torch.Tensor, logits: torch.Tensor) -> None:
        """Accumulate this batch's contribution to the Laplace precision matrix.

        phi: [B, rff_dim], logits: [B, num_classes]
        """
        if self.likelihood == "gaussian":
            # Reference default for image classification: unit weight.
            weighted = phi
        else:
            # Multinomial-logistic Laplace weight p(1-p), reduced over classes via the
            # max-probability convention.
            prob = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
            weighted = phi * torch.sqrt(prob * (1.0 - prob)).unsqueeze(-1).to(phi.dtype)

        batch_precision = weighted.T @ phi
        if self.cov_momentum < 0:
            # Exact sum over the epoch -- the paper's scheme, and the reference default.
            self.precision_accum.add_(batch_precision)
        else:
            batch_precision = batch_precision / max(1, phi.shape[0])
            self.precision_accum.mul_(self.cov_momentum).add_((1.0 - self.cov_momentum) * batch_precision)
        self._cov_stale.fill_(True)

    @torch.no_grad()
    def _ensure_covariance(self) -> None:
        """Recompute and cache the posterior covariance if the accumulator has moved."""
        if not bool(self._cov_stale):
            return

        accum = self.precision_accum
        if dist.is_available() and dist.is_initialized():
            # Each rank accumulates only its own shard, and DDP's default
            # `broadcast_buffers=True` would otherwise leave every rank with rank 0's
            # copy. Summing the ridge-free accumulator keeps the ridge un-scaled.
            accum = accum.clone()
            dist.all_reduce(accum, op=dist.ReduceOp.SUM)
            self.precision_accum.copy_(accum)

        eye = torch.eye(self.rff_dim, dtype=accum.dtype, device=accum.device)
        precision = self.ridge * eye + accum
        try:
            self.covariance.copy_(torch.cholesky_inverse(torch.linalg.cholesky(precision)))
        except RuntimeError:
            # Rare, but a near-singular precision must not take down a training run.
            self.covariance.copy_(torch.linalg.pinv(precision.double()).to(accum.dtype))
        self._cov_stale.fill_(False)

    # -- forward -------------------------------------------------------------

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RFFs: phi(x) = sqrt(2/m) * cos(x W + b), or cos(x W + b) when
        `scale_random_features` is off.

        x: [B, in_dim] -> [B, rff_dim]
        """
        if self.input_norm is not None:
            x = self.input_norm(x)
        proj = x @ self.W  # [B, rff_dim]
        proj = proj + self.b  # broadcast
        return torch.cos(proj) * self.rff_scale

    def predictive_variance(self, phi: torch.Tensor) -> torch.Tensor:
        """var(x) = phi(x)^T (ridge*I + sum_i w_i phi_i phi_i^T)^-1 phi(x). [B, 1]"""
        self._ensure_covariance()
        cov = self.covariance.to(dtype=phi.dtype, device=phi.device)
        return (phi @ cov * phi).sum(dim=1, keepdim=True).clamp(min=0.0)

    def forward(
        self, x: torch.Tensor, update_precision: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """x: [B, in_dim] pooled features.

        `update_precision` only has effect in train mode; eval never mutates state.
        """
        phi = self._features(x)
        raw_logits = self.classifier(phi)

        if self.training:
            if update_precision:
                self.update_precision(phi, raw_logits)
            # Mean-field is an inference-time correction. Training sees raw logits, and
            # no variance is computed at all -- which is also what removes the former
            # per-batch `rff_dim x rff_dim` Cholesky from the training loop.
            return raw_logits, raw_logits, None

        with torch.no_grad():
            pred_var = self.predictive_variance(phi)

        if self.mean_field:
            mean_field_logits = raw_logits / torch.sqrt(1.0 + self.mean_field_factor * pred_var)
        else:
            mean_field_logits = raw_logits

        return mean_field_logits, raw_logits, pred_var

    # -- checkpoint guard ----------------------------------------------------

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        found = [name for name in _LEGACY_BUFFERS if f"{prefix}{name}" in state_dict]
        if found:
            raise RuntimeError(_LEGACY_CKPT_MSG.format(found=", ".join(found)))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


# ---------------------------
# SNGP-ResNet wrapper
# ---------------------------

@register_net("sngp_classifier")
class SNGPClassifier(nn.Module):
    """
    ResNet backbone (torchvision) with spectral normalization + RFF-GP head.

    Constructor defaults are the Liu et al. 2022 (Table 9) / reference-implementation
    constants, so a bare `SNGPClassifier(num_classes=...)` is the paper's model:
    `rff_dim=1024`, `length_scale=1.4142` (sqrt(2): edward2's `gp_kernel_scale=2.0` scales
    the GP input by 1/sqrt(2); here `W` is divided by `length_scale`, so sqrt(2) is the
    same kernel width), exact per-epoch precision (`cov_momentum=-1`), orthogonal random
    features, one power iteration. `configs/model/sngp_classifier.yaml` restates them and
    tests/test_configs.py keeps the two in sync.

    The one deliberate exception is `spectral_norm_bound`: the paper's `c` (eq. 15), the
    only SNGP-specific knob that is *swept* (it changes the trained function, unlike
    `mean_field_factor`/`ridge_penalty`, which are inference-only). Its ctor default is
    `None` -- stock hard normalization, sigma == 1 -- so that checkpoints written before
    the bound existed (whose `net_spec` lacks the key) rebuild bit-identically. The model
    config sets the paper/reference default of 6.0 explicitly.

    `use_spectral_norm=False` builds the same backbone + GP head with **no** spectral
    normalization at all (plain `weight` parameters, no hooks). It exists for the
    spectral-*regularization* variant (`SNGPSpectralRegLitModule`), which bounds the
    backbone's singular values through a loss term instead of weight rescaling -- the two
    mechanisms must never be combined. `spectral_norm_bound` is ignored when it is off.
    Default `True`, so every spec written before the key existed rebuilds unchanged.
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        pretrained: bool = False,
        rff_dim: int = 1024,
        length_scale: float = 1.4142,
        ridge_penalty: float = 1e-3,
        cov_momentum: float = -1.0,
        mean_field: bool = True,
        n_power_iterations_sn: int = 1,
        mean_field_factor: float = DEFAULT_MEAN_FIELD_FACTOR,
        normalize_input: bool = True,
        likelihood: str = "gaussian",
        output_bias: bool = False,
        random_feature_type: str = "orf",
        scale_random_features: bool = True,
        spectral_norm_bound: Optional[float] = None,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.arch = arch
        self.pretrained = pretrained
        self.rff_dim = rff_dim
        self.length_scale = length_scale
        self.ridge_penalty = ridge_penalty
        self.cov_momentum = cov_momentum
        self.mean_field = mean_field
        self.n_power_iterations_sn = n_power_iterations_sn
        self.mean_field_factor = mean_field_factor
        self.normalize_input = normalize_input
        self.likelihood = likelihood
        self.output_bias = output_bias
        self.random_feature_type = random_feature_type
        self.scale_random_features = bool(scale_random_features)
        self.spectral_norm_bound = None if spectral_norm_bound is None else float(spectral_norm_bound)
        self.use_spectral_norm = bool(use_spectral_norm)

        if arch not in BACKBONES:
            raise ValueError(f"Unsupported backbone: {arch}. Supported: {sorted(BACKBONES)}")
        if self.use_spectral_norm:
            assert_spectral_norm_compatible(arch)

        # `build_backbone` returns a module whose forward already yields flat [B, feat_dim]
        # features for both resnet and ViT archs.
        self.backbone, feat_dim = build_backbone(arch, pretrained)

        # Apply spectral norm to all convs/linears in the backbone. `bound=None` is the
        # stock sigma == 1 normalization; a float is the paper's eq. 15 upper bound c.
        # Skipped entirely for the spectral-regularization variant (see class docstring).
        if self.use_spectral_norm:
            apply_spectral_norm(
                self.backbone, n_power_iterations=n_power_iterations_sn, bound=self.spectral_norm_bound
            )

        # --- RFF-GP head ---
        self.gp_head = RandomFeatureGaussianProcess(
            in_dim=feat_dim,
            num_classes=num_classes,
            rff_dim=rff_dim,
            length_scale=length_scale,
            ridge_penalty=ridge_penalty,
            cov_momentum=cov_momentum,
            mean_field=mean_field,
            mean_field_factor=mean_field_factor,
            normalize_input=normalize_input,
            likelihood=likelihood,
            output_bias=output_bias,
            random_feature_type=random_feature_type,
            scale_random_features=scale_random_features,
        )

    def reset_precision(self) -> None:
        """Delegate to the GP head; called from `SNGPLitModule.on_train_epoch_start`."""
        self.gp_head.reset_precision()

    @property
    def spec(self) -> dict:
        """Plain-data description of this net, sufficient to rebuild it via `build_net`."""
        return {
            "name": self.registry_name,
            "num_classes": self.num_classes,
            "arch": self.arch,
            "pretrained": self.pretrained,
            "rff_dim": self.rff_dim,
            "length_scale": self.length_scale,
            "ridge_penalty": self.ridge_penalty,
            "cov_momentum": self.cov_momentum,
            "mean_field": self.mean_field,
            "n_power_iterations_sn": self.n_power_iterations_sn,
            "mean_field_factor": self.mean_field_factor,
            "normalize_input": self.normalize_input,
            "likelihood": self.likelihood,
            "output_bias": self.output_bias,
            "random_feature_type": self.random_feature_type,
            "scale_random_features": self.scale_random_features,
            "spectral_norm_bound": self.spectral_norm_bound,
            "use_spectral_norm": self.use_spectral_norm,
        }

    def forward(self, x: torch.Tensor, update_precision: bool = True) -> ModelOutput:
        feats = self.backbone(x)

        # Some backbones may return tuples (e.g., aux outputs). Keep the main tensor.
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        # Safety net for any backbone that still returns a spatial [B, C, H, W] map.
        if feats.dim() == 4:
            feats = feats.flatten(1)
        elif feats.dim() == 3:
            feats = feats[:, 0]

        mean_field_logits, raw_logits, pred_var = self.gp_head(feats, update_precision=update_precision)
        return ModelOutput(logits=mean_field_logits, raw_logits=raw_logits, variance=pred_var)
