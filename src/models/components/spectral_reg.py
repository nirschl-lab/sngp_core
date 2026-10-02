"""Spectral-norm *regularization* (rep-spectral) -- the loss-term alternative to spectral
normalization.

Yang, Zavatone-Veth & Pehlevan, "Spectral regularization for adversarially-robust
representation learning" (arXiv 2405.17181), eq. (5):

    L = CE + gamma/2 * sum_{l < L} sigma_max^2(W_l)

penalizes the squared top singular value of every *representation* layer and deliberately
leaves the readout out. Where `spectral_norm.py` rescales weights in a forward pre-hook so
that sigma <= c holds by construction, this module only *measures* sigma and hands back a
differentiable penalty for the training loss; weights are never touched. Training-only
code -- nothing here is needed to rebuild or run a trained net.

Two estimators for a conv layer, selected by `conv_mode`:

  * `"operator"` (default) -- the spectral norm of the *linearized convolution* K~ (the
    paper's Appendix B.1 object), estimated by power iteration with `conv2d` /
    `conv_transpose2d` on the layer's actual input shape, as edward2's
    `SpectralNormalizationConv2D` does. Exact for zero padding and any stride; needs the
    input spatial size, which a forward pre-hook records on the first forward pass.
  * `"reshape"` -- the Miyato et al. estimate on the `[out, in*k*k]` kernel matrix, the
    quantity `torch.nn.utils.spectral_norm` (and hence `spectral_norm_bound`) works with.
    A lower bound on the operator norm (by up to a factor of k); kept for ablation.

The reference implementation computes K~'s top eigenvalue exactly (FFT blocks +
`eigvalsh`, under circular padding) and amortizes that cost every N updates; power
iteration with `n_power_iterations=1` is the paper's own recommendation ("N=1 is enough,
since the parameter moves slowly").

Gradient: sigma = <u, K~ v> with u, v the (detached) top singular pair, so autograd yields
d sigma^2 / dW = 2 sigma u v^T -- the analytic gradient the paper differentiates. Biases,
BatchNorm and every other layer type are ignored: they do not enter the Jacobian bound the
regularizer is motivated by (BN's affine scale is uncontrolled by *both* methods).
`BatchNormSpectralRegularizer` below is the separate, opt-in term for BN's gain.

Folded BN->conv (`input_bn`): a conv fed by a BatchNorm through ReLU/dropout -- every
conv of a pre-activation block -- can be penalized as the *composite* instead. With
G = diag(|gamma| / sqrt(running_var + eps)) and D the 0/1 ReLU mask, the block's Jacobian
is W D G = (W G) D (diagonals commute), so ||W D G|| <= ||W G||: the norm of the conv
applied to a per-channel-rescaled input bounds the BN-ReLU-conv map exactly. Penalizing
W G rather than W and G separately matters because BN gain alone is not scale-invariant
(its running variance tracks the norm of whatever precedes it), and capping only W leaves
BN free to undo the cap. Gradient reaches both `conv.weight` and `bn.weight`; the running
variance is detached, as in `BatchNormSpectralRegularizer`. `find_bn_conv_pairs` builds
the map for this project's WideResNet.

`u`/`v` are non-persistent buffers: they are re-estimated when a run resumes
(`warmup_iterations` power iterations on first use) and never enter a checkpoint's
`state_dict`, so a regularized net loads with `strict=True` like any other.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import _WideBasicBlock

CONV_MODES = ("operator", "reshape")

# Power iterations run the first time the penalty is computed, so the very first
# regularized step already sees a converged sigma estimate (same rationale as
# `spectral_norm.DEFAULT_SN_WARMUP_ITERATIONS`: random u/v badly under-shoot sigma).
DEFAULT_SPEC_REG_WARMUP_ITERATIONS = 20


@dataclass(frozen=True)
class SpectralRegOutput:
    """`penalty` is the differentiable scalar `sum_l sigma_l^2`; `sigmas` the detached
    per-layer estimates (ordered like `SpectralRegularizer.layer_names`) for logging."""

    penalty: torch.Tensor
    sigmas: torch.Tensor


class SpectralRegularizer(nn.Module):
    """`sum_l sigma_max^2(W_l)` over the `Conv2d`/`Linear` leaves of `module`.

    Pass the *representation* part of a net (e.g. `SNGPClassifier.backbone`), not the
    whole net: the readout is excluded by construction, as in the paper. The layers are
    held in a plain list -- never registered as submodules, which would duplicate their
    parameters in `state_dict()` -- so `module` keeps sole ownership of its weights.

    Call `forward()` (no arguments) after at least one forward pass through `module` in
    operator mode, so the per-conv input shapes are known; it advances every layer's
    power iteration by `n_power_iterations` and returns the penalty.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        n_power_iterations: int = 1,
        conv_mode: str = "operator",
        warmup_iterations: int = DEFAULT_SPEC_REG_WARMUP_ITERATIONS,
        eps: float = 1e-12,
        input_bn: Optional[Dict[str, nn.BatchNorm2d]] = None,
    ) -> None:
        """`input_bn` maps a conv's name (relative to `module`) to the BatchNorm that feeds
        it through a ReLU/dropout; those convs are penalized folded with that BN's gain
        (see the module docstring). Convs not in the map keep their plain sigma."""
        super().__init__()
        if conv_mode not in CONV_MODES:
            raise ValueError(f"conv_mode must be one of {CONV_MODES}, got {conv_mode!r}")
        if n_power_iterations < 1:
            raise ValueError(f"n_power_iterations must be >= 1, got {n_power_iterations}")
        if warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be >= 0, got {warmup_iterations}")
        self.n_power_iterations = int(n_power_iterations)
        self.conv_mode = conv_mode
        self.warmup_iterations = int(warmup_iterations)
        self.eps = float(eps)

        self._layers: List[Tuple[str, nn.Module]] = [
            (name, m)
            for name, m in module.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ]
        if not self._layers:
            raise ValueError(
                "SpectralRegularizer: `module` has no Conv2d/Linear layers to regularize"
            )
        for name, layer in self._layers:
            if hasattr(layer, "weight_u"):
                raise ValueError(
                    f"SpectralRegularizer: layer {name!r} carries a spectral_norm hook. Spectral "
                    "normalization and spectral regularization must not be combined -- build the "
                    "net without spectral norm (SNGPClassifier(use_spectral_norm=False))."
                )
            if isinstance(layer, nn.Conv2d) and conv_mode == "operator":
                if isinstance(layer.padding, str):
                    raise NotImplementedError(
                        f"SpectralRegularizer: conv {name!r} uses padding={layer.padding!r}; "
                        "operator mode needs integer padding (use conv_mode='reshape' otherwise)."
                    )
                if layer.padding_mode != "zeros":
                    raise NotImplementedError(
                        f"SpectralRegularizer: conv {name!r} uses "
                        f"padding_mode={layer.padding_mode!r}; operator mode is exact for zero "
                        "padding only."
                    )

        # Folded input gains, keyed by layer index. BN modules are held in a plain dict for
        # the same reason as `_layers`: `module` keeps sole ownership of their parameters.
        self._input_bn: Dict[int, nn.BatchNorm2d] = {}
        index = {name: idx for idx, (name, _) in enumerate(self._layers)}
        for name, bn in (input_bn or {}).items():
            if name not in index:
                raise ValueError(f"SpectralRegularizer: input_bn names unknown layer {name!r}")
            layer = self._layers[index[name]][1]
            if not isinstance(layer, nn.Conv2d) or layer.groups != 1:
                raise ValueError(
                    f"SpectralRegularizer: input_bn layer {name!r} must be an ungrouped Conv2d"
                )
            if not (isinstance(bn, nn.BatchNorm2d) and bn.affine and bn.track_running_stats):
                raise ValueError(
                    f"SpectralRegularizer: input_bn[{name!r}] must be an affine BatchNorm2d "
                    "with running stats"
                )
            if bn.num_features != layer.in_channels:
                raise ValueError(
                    f"SpectralRegularizer: input_bn[{name!r}] has {bn.num_features} channels, "
                    f"conv expects {layer.in_channels}"
                )
            self._input_bn[index[name]] = bn

        # Per-conv input (h, w), recorded by pre-hooks; only needed in operator mode.
        self._input_shapes: Dict[int, Tuple[int, int]] = {}
        # Layers whose u/v must be (re)initialised: everything at construction, plus any
        # conv whose input shape changed since its vectors were drawn.
        self._stale: Set[int] = set(range(len(self._layers)))
        self._hooks = []
        if conv_mode == "operator":
            for idx, (_, layer) in enumerate(self._layers):
                if isinstance(layer, nn.Conv2d):
                    self._hooks.append(layer.register_forward_pre_hook(self._make_shape_hook(idx)))

    # ------------------------------------------------------------------ introspection
    @property
    def layer_names(self) -> List[str]:
        """Names (relative to the regularized module) of the penalized layers, in order."""
        return [name for name, _ in self._layers]

    @property
    def folded_layer_names(self) -> List[str]:
        """Names of the layers penalized folded with their input BatchNorm's gain."""
        return [self._layers[idx][0] for idx in sorted(self._input_bn)]

    def __len__(self) -> int:
        return len(self._layers)

    def extra_repr(self) -> str:
        return (
            f"num_layers={len(self._layers)}, num_folded={len(self._input_bn)}, "
            f"conv_mode={self.conv_mode!r}, "
            f"n_power_iterations={self.n_power_iterations}, "
            f"warmup_iterations={self.warmup_iterations}"
        )

    # ------------------------------------------------------------------ shape capture
    def _make_shape_hook(self, idx: int):
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            shape = (int(inputs[0].shape[-2]), int(inputs[0].shape[-1]))
            if self._input_shapes.get(idx) != shape:
                self._input_shapes[idx] = shape
                self._stale.add(idx)

        return hook

    # ------------------------------------------------------------------ vector storage
    def _vectors(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f"u_{idx}"), getattr(self, f"v_{idx}")

    def _set_vectors(self, idx: int, u: torch.Tensor, v: torch.Tensor) -> None:
        # `register_buffer` accepts re-registration of an existing buffer name; the
        # buffers stay non-persistent so nothing reaches the checkpoint state_dict.
        self.register_buffer(f"u_{idx}", u, persistent=False)
        self.register_buffer(f"v_{idx}", v, persistent=False)

    def _unit(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.reshape(-1), dim=0, eps=self.eps).reshape(x.shape)

    # ------------------------------------------------------------------ linear maps
    def _input_gain(self, idx: int) -> Optional[torch.Tensor]:
        """Per-input-channel BN gain `|gamma| / sqrt(running_var + eps)` as [1, C, 1, 1],
        or None for an unfolded layer. Differentiable in gamma only."""
        bn = self._input_bn.get(idx)
        if bn is None:
            return None
        gain = bn.weight.abs() * (bn.running_var.detach() + bn.eps).rsqrt()
        return gain.reshape(1, -1, 1, 1)

    def _matrix(self, idx: int) -> torch.Tensor:
        """Weight as a 2-D matrix: Linear as is, conv reshaped to [out, in*k*k], with a
        folded layer's input columns scaled by its BN gain."""
        weight = self._layers[idx][1].weight
        gain = self._input_gain(idx)
        if gain is not None:
            weight = weight * gain
        return weight.reshape(weight.shape[0], -1)

    def _folded_forward(self, idx: int, v: torch.Tensor) -> torch.Tensor:
        """(K~ G) v for a conv in operator mode; G = I for an unfolded layer."""
        gain = self._input_gain(idx)
        return self._conv_forward(v if gain is None else v * gain, self._layers[idx][1])

    def _folded_adjoint(self, idx: int, u: torch.Tensor) -> torch.Tensor:
        """(K~ G)^T u = G K~^T u, back to the conv's input shape."""
        out = self._conv_transpose(u, self._layers[idx][1], self._input_shapes[idx])
        gain = self._input_gain(idx)
        return out if gain is None else out * gain

    def _uses_operator(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Conv2d) and self.conv_mode == "operator"

    @staticmethod
    def _conv_forward(v: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        """K~ v: apply the conv (no bias) to a [1, c_in, h, w] input."""
        return F.conv2d(
            v, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups
        )

    @staticmethod
    def _conv_transpose(u: torch.Tensor, conv: nn.Conv2d, in_hw: Tuple[int, int]) -> torch.Tensor:
        """K~^T u: the adjoint, back to the [1, c_in, h, w] input shape.

        `output_padding` recovers the exact input size that a strided conv floors away;
        the formula inverts torch's Conv2d output-size rule.
        """
        out_hw = u.shape[-2:]
        output_padding = tuple(
            in_hw[i]
            - (
                (out_hw[i] - 1) * conv.stride[i]
                - 2 * conv.padding[i]
                + conv.dilation[i] * (conv.kernel_size[i] - 1)
                + 1
            )
            for i in range(2)
        )
        return F.conv_transpose2d(
            u,
            conv.weight,
            None,
            conv.stride,
            conv.padding,
            output_padding,
            conv.groups,
            conv.dilation,
        )

    # ------------------------------------------------------------------ power iteration
    @torch.no_grad()
    def _init_vectors(self, idx: int) -> None:
        _, layer = self._layers[idx]
        weight = layer.weight
        if self._uses_operator(layer):
            if idx not in self._input_shapes:
                raise RuntimeError(
                    f"SpectralRegularizer: input shape of conv {self._layers[idx][0]!r} is "
                    "unknown. Run one forward pass through the regularized module *after* "
                    "constructing the regularizer (its shape-recording hooks must be attached "
                    "first) and before computing the penalty."
                )
            h, w = self._input_shapes[idx]
            c_in = weight.shape[1] * layer.groups
            v = torch.randn(1, c_in, h, w, device=weight.device, dtype=weight.dtype)
            u = self._folded_forward(idx, v)
        else:
            mat = self._matrix(idx)
            u = torch.randn(mat.shape[0], device=weight.device, dtype=weight.dtype)
            v = torch.randn(mat.shape[1], device=weight.device, dtype=weight.dtype)
        self._set_vectors(idx, self._unit(u), self._unit(v))
        if self.warmup_iterations:
            self._power_iterate(idx, self.warmup_iterations)
        self._stale.discard(idx)

    @torch.no_grad()
    def _power_iterate(self, idx: int, n_iterations: int) -> None:
        """Advance layer `idx`'s top singular pair by `n_iterations` rounds, in place.

        Same update order as `torch.nn.utils.spectral_norm`: v <- unit(W^T u), then
        u <- unit(W v), so `sigma = <u, W v>` below uses a `u` computed from the current `v`.
        """
        _, layer = self._layers[idx]
        u, v = self._vectors(idx)
        if self._uses_operator(layer):
            for _ in range(n_iterations):
                v = self._unit(self._folded_adjoint(idx, u))
                u = self._unit(self._folded_forward(idx, v))
        else:
            mat = self._matrix(idx)
            for _ in range(n_iterations):
                v = self._unit(torch.mv(mat.t(), u))
                u = self._unit(torch.mv(mat, v))
        self._set_vectors(idx, u, v)

    def _sigma(self, idx: int) -> torch.Tensor:
        """Differentiable sigma_max estimate `<u, W v>` (`<u, W G v>` when folded) with u, v
        held fixed."""
        _, layer = self._layers[idx]
        u, v = self._vectors(idx)
        if self._uses_operator(layer):
            return (u * self._folded_forward(idx, v)).sum()
        return torch.dot(u, torch.mv(self._matrix(idx), v))

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def warm_up(self, n_iterations: Optional[int] = None) -> None:
        """(Re)converge every layer's power iteration -- e.g. after resuming a run."""
        n = self.warmup_iterations if n_iterations is None else n_iterations
        for idx in range(len(self._layers)):
            if idx in self._stale:
                self._init_vectors(idx)
            else:
                self._power_iterate(idx, n)

    def forward(self) -> SpectralRegOutput:
        sigmas = []
        for idx in range(len(self._layers)):
            if idx in self._stale:
                self._init_vectors(idx)
            self._power_iterate(idx, self.n_power_iterations)
            sigmas.append(self._sigma(idx))
        sigma = torch.stack(sigmas)
        return SpectralRegOutput(penalty=(sigma**2).sum(), sigmas=sigma.detach())


class BatchNormSpectralRegularizer(nn.Module):
    """`sum_l L_l^2` over the `BatchNorm2d` layers of `module`, where

        L_l = max_i |gamma_i| / sqrt(running_var_i + eps)

    is the layer's Lipschitz constant (Gouk et al. 2021, Machine Learning 110:393, eq. 28) --
    the soft, loss-term counterpart of `spectral_norm.SpectralBatchNorm2d`'s hard cap, in the
    same `sum (top singular value)^2` form as `SpectralRegularizer` (a BN layer is diagonal,
    so its top singular value is its largest per-channel gain). As in Gouk et al. and DUE the
    *running* variance is used, so the gradient reaches gamma only, and only its arg-max
    channel per layer.

    Layers are held in a plain list (never registered as submodules), so the regularized
    module keeps sole ownership of its parameters and nothing enters `state_dict()`.
    `forward()` takes no arguments and returns a `SpectralRegOutput` whose `sigmas` are the
    per-layer L, so it is a drop-in for `SpectralRegularizer` in `SNGPSpectralRegLitModule`.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._layers: List[Tuple[str, nn.Module]] = [
            (name, m) for name, m in module.named_modules() if isinstance(m, nn.BatchNorm2d)
        ]
        if not self._layers:
            raise ValueError("BatchNormSpectralRegularizer: `module` has no BatchNorm2d layers to regularize")
        for name, layer in self._layers:
            if not layer.affine or not layer.track_running_stats:
                raise ValueError(
                    f"BatchNormSpectralRegularizer: BN {name!r} needs affine=True and "
                    "track_running_stats=True (the penalty is on gamma / running std)"
                )

    @property
    def layer_names(self) -> List[str]:
        """Names (relative to the regularized module) of the penalized layers, in order."""
        return [name for name, _ in self._layers]

    def __len__(self) -> int:
        return len(self._layers)

    def extra_repr(self) -> str:
        return f"num_layers={len(self._layers)}"

    def forward(self) -> SpectralRegOutput:
        gains = torch.stack([
            (bn.weight.abs() * (bn.running_var.detach() + bn.eps).rsqrt()).max() for _, bn in self._layers
        ])
        return SpectralRegOutput(penalty=(gains**2).sum(), sigmas=gains.detach())


def find_bn_conv_pairs(module: nn.Module) -> Dict[str, nn.BatchNorm2d]:
    """The `input_bn` map for `SpectralRegularizer`: every BN -> ReLU -> dropout -> conv
    pair of `module`, keyed by the conv's name relative to `module`.

    Only this project's pre-activation WideResNet block is recognized: `bn1 -> conv1` and
    `bn2 -> conv2` in each `_WideBasicBlock` (`src/models/backbones.py`). Deliberately not
    a generic name match -- the folded bound needs nothing but a 1-Lipschitz diagonal map
    between the BN and the conv, which a post-activation ResNet (conv -> BN) does not
    have. The stem conv, the projection shortcuts (they read the block input, before
    `bn1`) and the final BN (it feeds the GP head, not a conv) are left out.
    """
    pairs: Dict[str, nn.BatchNorm2d] = {}
    for name, block in module.named_modules():
        if isinstance(block, _WideBasicBlock):
            prefix = f"{name}." if name else ""
            pairs[f"{prefix}conv1"] = block.bn1
            pairs[f"{prefix}conv2"] = block.bn2
    if not pairs:
        raise ValueError(
            f"find_bn_conv_pairs: no pre-activation WideResNet blocks in {type(module).__name__}; "
            "folded BN->conv regularization is only defined for wide_resnet28_10 backbones"
        )
    return pairs
