import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from src.models.backbones import BACKBONES, build_backbone
from src.models.components.spectral_norm import apply_spectral_norm, assert_spectral_norm_compatible
from src.models.outputs import ModelOutput
from src.models.registry import register_net

# Mean-field multiplicative factor: the lambda in `logits / sqrt(1 + lambda * var)` (Liu
# et al. 2022, eq. 19), which the paper sets to pi/8. The reference implementation does
# NOT expose the paper's kernel amplitude sigma^2 (eq. 8/10) at all; it collapses lambda
# and sigma^2 into the single post-hoc `gp_mean_field_factor` -- 1.0 in the ImageNet SNGP
# baseline, 20.0 in the CIFAR one -- and Table 10's "kernel amplitude" column reports that
# collapsed value. The collapse is exact only where the ridge prior dominates the
# precision (far from the data); in general sigma^2 in the feature map is equivalent to
# ridge -> ridge / sigma^2 and also changes training. `kernel_amplitude` (below) is the
# paper's sigma^2 itself. This default matches ImageNet (the closer setting to 224px
# resnets) and is meant to be fit post-hoc on validation NLL. pi/8 is the paper's
# constant, but with sigma^2 = 1 it makes the correction nearly inert at realistic
# dataset sizes (~2% logit shrink); see scripts/checkpoints/calibrate_checkpoint.py.
DEFAULT_MEAN_FIELD_FACTOR = 1.0
PROBIT_MEAN_FIELD_FACTOR = math.pi / 8

# Laplace weight `w_i` in `P = ridge*I + sum_i w_i phi_i phi_i^T`. The multinomial-logistic
# Hessian is `diag(p) - p p^T` per example -- a [K, K] matrix -- while the precision
# accumulator this project keeps is a single [rff_dim, rff_dim], so the Hessian has to be
# reduced to one scalar per example. These are the reductions:
#   gaussian         -- unit weight; the reference default for image classification.
#   binary_logistic  -- p_max (1 - p_max), the edward2 reference's own reduction: read the
#                       top class as a one-vs-rest Bernoulli and ignore the rest.
#   trace_logistic   -- tr(diag(p) - p p^T) = sum_k p_k(1 - p_k) = 1 - ||p||^2. Uses every
#                       class rather than only the argmax, and is the tightest scalar
#                       summary of the full Hessian that costs nothing extra.
# `per_class_logistic` (the unreduced [B, K] diagonal) is implemented in `_laplace_weights`
# but refused at construction -- see the guard in `RandomFeatureGaussianProcess.__init__`.
SUPPORTED_LIKELIHOODS = ("gaussian", "binary_logistic", "trace_logistic")
PER_CLASS_LIKELIHOOD = "per_class_logistic"

# Random-feature maps for the RBF kernel; see `RandomFeatureGaussianProcess._features`.
FEATURE_MAPS = ("cos", "positive", "hyperbolic")

_PER_CLASS_MSG = (
    "likelihood='per_class_logistic' is not supported yet. Its [B, K] weight needs a "
    "[num_classes, rff_dim, rff_dim] precision accumulator and yields a [B, K] predictive "
    "variance, which neither the [B, 1] `ModelOutput.variance` contract nor the "
    "one-scalar-per-row `predictions.csv` schema carries today. The weight itself is "
    f"implemented in `_laplace_weights`. Use one of {SUPPORTED_LIKELIHOODS}."
)

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

      - Fixed random Fourier features phi(x) = sqrt(2/m) * cos(Wx + b), or positive /
        hyperbolic random features (`feature_map`, see `_features`)
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
        feature_map: str = "cos",
        kernel_amplitude: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if likelihood == PER_CLASS_LIKELIHOOD:
            raise NotImplementedError(_PER_CLASS_MSG)
        if likelihood not in SUPPORTED_LIKELIHOODS:
            raise ValueError(f"Unsupported likelihood: {likelihood!r}. Use one of {SUPPORTED_LIKELIHOODS}.")
        if random_feature_type not in ("rff", "orf", "simrf"):
            raise ValueError(
                f"Unsupported random_feature_type: {random_feature_type!r}. Use 'rff', 'orf' or 'simrf'."
            )
        if random_feature_type == "simrf" and in_dim < 2:
            raise ValueError(f"random_feature_type='simrf' needs in_dim >= 2 (a simplex needs 2+ vertices), got {in_dim}")
        if feature_map not in FEATURE_MAPS:
            raise ValueError(f"Unsupported feature_map: {feature_map!r}. Use one of {FEATURE_MAPS}.")
        if feature_map == "hyperbolic" and rff_dim % 2:
            raise ValueError(f"feature_map='hyperbolic' needs an even rff_dim (two features per direction), got {rff_dim}")
        if ridge_penalty <= 0:
            raise ValueError(f"ridge_penalty must be > 0 (it seeds the precision matrix), got {ridge_penalty}")
        if kernel_amplitude <= 0:
            raise ValueError(f"kernel_amplitude must be > 0 (it is the GP prior variance sigma^2), got {kernel_amplitude}")

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
        self.feature_map = feature_map
        self.kernel_amplitude = kernel_amplitude

        # LayerNorm on the GP input ("similar to applying ARD", per the reference's own
        # description) -- also keeps the RFF kernel well-scaled as backbone feature
        # magnitudes drift during training.
        self.input_norm = nn.LayerNorm(in_dim) if normalize_input else None

        # Random Fourier feature parameters (fixed, not learned)
        # b ~ Uniform(0, 2pi); W columns are random directions with chi-distributed
        # norms => RBF kernel with length-scale l.
        # `hyperbolic` emits two features per direction, so it draws rff_dim/2 directions
        # and the feature count -- hence the classifier and precision -- stays rff_dim.
        # `b` is only used by `cos`, but is always drawn and stored so state_dict keys and
        # the RNG stream stay the same across feature maps.
        n_directions = rff_dim // 2 if feature_map == "hyperbolic" else rff_dim
        W = self._sample_projection(in_dim, n_directions, random_feature_type, dtype) / length_scale
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
        #
        # Scaled, every feature map satisfies phi(x).phi(y) ~= k(x, y): sqrt(2/m) for cos,
        # sqrt(1/m) for positive / hyperbolic (m = rff_dim features in all three). Unscaled
        # multiplies each by sqrt(m/2), so phi.phi ~= (m/2) k for every map -- 1.0 for cos,
        # as before -- which keeps the ridge prior and the mean-field factor on the same
        # footing whichever map is chosen.
        if scale_random_features:
            self.rff_scale = math.sqrt((2.0 if feature_map == "cos" else 1.0) / rff_dim)
        else:  # written out, not derived, so an unscaled cos head stays exactly 1.0
            self.rff_scale = 1.0 if feature_map == "cos" else math.sqrt(0.5)
        # Kernel amplitude sigma^2 (paper eq. 8/10: phi = sqrt(2 sigma^2 / m) cos(Wx + b)), so
        # phi.phi ~= sigma^2 k scaled and (m/2) sigma^2 k unscaled -- the unscaled base already
        # carries an implicit m/2. It multiplies the features themselves, so it acts on the
        # training logits and the backbone gradient as well as on the variance; it is not a
        # post-hoc knob like `mean_field_factor`. Skipped at 1.0 so existing heads stay exact.
        if kernel_amplitude != 1.0:
            self.rff_scale *= math.sqrt(kernel_amplitude)

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

        `"simrf"` (simplex random features, Reid et al. 2023, arXiv:2301.13856) goes one
        step further: each block's directions are the vertices of a regular simplex under
        a Haar rotation, so every pair has dot product -1/(in_dim-1) instead of 0. Norms
        and per-column marginals are the same as for `"orf"`, so the estimate stays
        unbiased. The paper proves SimRF has the minimum MSE for *positive* random
        features; for the cos features used here that guarantee does not carry over, and
        any gain over `"orf"` has to be measured.
        """
        if kind == "rff":
            return torch.randn(in_dim, rff_dim, dtype=dtype)

        if kind == "simrf":
            # Centred standard basis, normalised: in_dim unit vectors summing to zero,
            # pairwise dot -1/(in_dim-1). Equals the paper's explicit simplex matrix up to
            # a fixed rotation, which the Haar Q below absorbs.
            simplex = torch.eye(in_dim, dtype=dtype) - 1.0 / in_dim
            simplex = simplex / simplex.norm(dim=0, keepdim=True)

        blocks = []
        remaining = rff_dim
        while remaining > 0:
            # QR of a square Gaussian gives Q with orthonormal columns. It is Haar-distributed
            # over the orthogonal group only once each column is multiplied by sign(R_kk)
            # (Mezzadri 2007): LAPACK's sign convention otherwise skews column k's k-th
            # coordinate negative. The cos features are even in w, so that skew never biased
            # them; positive random features are not, so it biased ORF there.
            q, r = torch.linalg.qr(torch.randn(in_dim, in_dim, dtype=dtype))
            signs = torch.sign(torch.diagonal(r))
            q = q * torch.where(signs == 0, torch.ones_like(signs), signs)
            block = q @ simplex if kind == "simrf" else q
            blocks.append(block[:, :min(in_dim, remaining)])
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

    @staticmethod
    @torch.no_grad()
    def _laplace_weights(logits: torch.Tensor, likelihood: str) -> Optional[torch.Tensor]:
        """The Laplace weight `w` for one batch of logits, or `None` for unit weights.

        Returns `[B]` for the scalar reductions and `[B, K]` for `per_class_logistic`;
        see the `SUPPORTED_LIKELIHOODS` comment at the top of this module for what each
        reduction is. Static (rather than reading `self.likelihood`) so the per-class
        branch stays reachable from tests while the constructor refuses that mode.

        logits: [B, num_classes]
        """
        if likelihood == "gaussian":
            # Reference default for image classification: unit weight.
            return None

        prob = torch.softmax(logits.float(), dim=-1)
        if likelihood == "binary_logistic":
            # Top class read as a one-vs-rest Bernoulli -- the edward2 convention.
            p = prob.max(dim=-1).values
            return p * (1.0 - p)
        if likelihood == "trace_logistic":
            # tr(diag(p) - p p^T) = sum_k p_k(1 - p_k) = 1 - ||p||^2.
            return 1.0 - (prob * prob).sum(dim=-1)
        if likelihood == PER_CLASS_LIKELIHOOD:
            # The unreduced Hessian diagonal, [B, K].
            return prob * (1.0 - prob)
        raise ValueError(f"Unsupported likelihood: {likelihood!r}. Use one of {SUPPORTED_LIKELIHOODS}.")

    @torch.no_grad()
    def update_precision(self, phi: torch.Tensor, logits: torch.Tensor) -> None:
        """Accumulate this batch's contribution to the Laplace precision matrix.

        phi: [B, rff_dim], logits: [B, num_classes]
        """
        weights = self._laplace_weights(logits, self.likelihood)
        if weights is None:
            batch_precision = phi.T @ phi
        elif weights.dim() == 1:
            # `sum_i w_i phi_i phi_i^T` -- the weight is applied ONCE, to one side of the
            # outer product, matching `P = ridge*I + sum_i w_i phi_i phi_i^T` in the class
            # docstring and edward2's `sqrt(w) * phi` squared against itself.
            batch_precision = (phi * weights.unsqueeze(-1).to(phi.dtype)).T @ phi
        else:
            # Unreachable: the constructor refuses the only mode that returns [B, K].
            raise NotImplementedError(_PER_CLASS_MSG)

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
        """Random features for the RBF kernel k(x, y) = exp(-||x - y||^2 / 2 l^2), times
        `rff_scale`. W is already divided by l, so ||x / l||^2 = ||x||^2 / l^2.

          cos         cos(x W + b)                                       (Rahimi & Recht)
          positive    exp(x W - ||x/l||^2)                               (FAVOR+)
          hyperbolic  [exp(x W - ||x/l||^2), exp(-x W - ||x/l||^2)]      (FAVOR+ "++")

        All three are unbiased. The positive maps pair with `simrf`, whose MSE optimality
        (Reid et al. 2023) is proven for positive features, and in the kernel study
        (`scripts/metrics/random_feature_kernel_mse.py`) SimRF cut positive-feature MSE to
        ~0.1x ORF at rho = ||x||/l = 0.5 while tying ORF for cos. Positive features need
        rho <~ 1: their variance grows like exp(||x + y||^2 / l^2), and past rho ~ 2 the
        estimate collapses to ~0 (and float32 underflows at rho ~ 16, the LayerNorm +
        l = sqrt(2) protocol). The CIFAR-100 recipe (no input norm, l = 20) sits at rho ~ 0.57.

        x: [B, in_dim] -> [B, rff_dim]
        """
        if self.input_norm is not None:
            x = self.input_norm(x)
        proj = x @ self.W  # [B, rff_dim], or [B, rff_dim/2] for hyperbolic
        if self.feature_map == "cos":
            return torch.cos(proj + self.b) * self.rff_scale
        sq = x.pow(2).sum(dim=-1, keepdim=True) / self.length_scale**2
        if self.feature_map == "positive":
            return torch.exp(proj - sq) * self.rff_scale
        return torch.cat([torch.exp(proj - sq), torch.exp(-proj - sq)], dim=-1) * self.rff_scale

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
        feature_map: str = "cos",
        kernel_amplitude: float = 1.0,
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
        self.feature_map = feature_map
        self.kernel_amplitude = float(kernel_amplitude)
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
            feature_map=feature_map,
            kernel_amplitude=self.kernel_amplitude,
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
            "feature_map": self.feature_map,
            "kernel_amplitude": self.kernel_amplitude,
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
