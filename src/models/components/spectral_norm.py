"""Spectral-norm wrapping used by SNGP nets.

Dependency-light (only `torch`, plus `src.models.backbones` for the compatibility set) so
this file can be vendored into HF export bundles alongside `backbones.py`.

Two normalization regimes, selected by `apply_spectral_norm(..., bound=...)`:

  * `bound=None` -- stock `torch.nn.utils.spectral_norm`: every wrapped weight is divided
    by its estimated spectral norm, so sigma(W_eff) == 1 always. This was the only
    behaviour before the bound existed and is kept byte-identical for reproducibility of
    checkpoints whose `net_spec` predates `spectral_norm_bound`.
  * `bound=c` -- Liu et al. 2022 eq. 15 (`BoundedSpectralNorm`): `W <- c * W / sigma_hat`
    only when `sigma_hat > c`, otherwise `W` is left untouched, so sigma(W_eff) <= c. `c`
    is the paper's tunable "spectral norm bound": it trades the expressiveness of the
    residual blocks against their distance preservation, and the paper recommends a grid
    search for the smallest `c` that retains accuracy (c = 6 for a WideResNet). This is
    exactly the reference implementation's `SpectralNormalization(norm_multiplier=c)`.

Known approximation, inherited from `torch.nn.utils.spectral_norm`: conv kernels are
reshaped to `[out, in * k * k]` and sigma_hat is the spectral norm of *that matrix*, not the
operator norm of the convolution (which edward2's `SpectralNormalizationConv2D` estimates
with a conv/conv-transpose power iteration). The paper's Appendix A.2 already notes that
spectral normalization "does not have precise control of the true spectral norm of the
convolutional kernel", which is why `c` is treated as a hyperparameter to sweep rather than
a constant to derive -- but it does mean `c` values are not numerically transferable
between the two estimators.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)

from src.models.backbones import SPECTRAL_NORM_COMPATIBLE

# Power iterations run at construction to seed u/v. `spectral_norm`'s hook only iterates
# in train mode, so without this a freshly built net normalizes by a spectral-norm
# estimate taken from *random* u/v -- see `warm_up_spectral_norm`.
DEFAULT_SN_WARMUP_ITERATIONS = 20


class BoundedSpectralNorm(SpectralNorm):
    """`torch.nn.utils.spectral_norm.SpectralNorm` with the paper's upper bound `c`.

    Identical to the stock hook -- same power iteration, same in-place `u`/`v` update
    semantics, same `weight_orig`/`weight_u`/`weight_v` state-dict layout, same
    state-dict hooks -- except that the returned weight is `W / max(1, sigma_hat / c)`:
    scaled down to spectral norm `c` when it exceeds it, otherwise returned as is.
    Division by exactly `1.0` leaves `W` bit-identical.
    """

    def __init__(
        self,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        bound: float = 1.0,
    ) -> None:
        super().__init__(name, n_power_iterations, dim, eps)
        if bound <= 0:
            raise ValueError(f"spectral-norm bound must be > 0, got {bound}")
        self.bound = float(bound)

    def compute_weight(self, module: nn.Module, do_power_iteration: bool) -> torch.Tensor:
        # Power-iteration block reproduced from torch's SpectralNorm.compute_weight: the
        # in-place `out=` update of u/v (DataParallel replicas share storage) followed by
        # a clone (so two forward passes can be back-propagated through) must be kept
        # exactly as upstream has it.
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # Liu et al. 2022 eq. 15 as a single divide: c*W/sigma if sigma > c else W.
        return weight / torch.clamp(sigma / self.bound, min=1.0)

    @staticmethod
    def apply(
        module: nn.Module, name: str, n_power_iterations: int, dim: int, eps: float, bound: float
    ) -> "BoundedSpectralNorm":
        # Reproduces torch's SpectralNorm.apply, which hard-codes `SpectralNorm(...)` for
        # the hook object; everything else (parameter rename, u/v buffers, the three hook
        # registrations) is the stock recipe so that state dicts stay interchangeable.
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(f"Cannot register two spectral_norm hooks on the same parameter {name}")

        fn = BoundedSpectralNorm(name, n_power_iterations, dim, eps, bound)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f"`BoundedSpectralNorm` cannot be applied as parameter `{name}` is None")
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `BoundedSpectralNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying spectral normalization"
            )

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def bounded_spectral_norm(
    module: nn.Module,
    bound: float,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> nn.Module:
    """Drop-in analogue of `torch.nn.utils.spectral_norm` with an upper bound `c` (eq. 15)."""
    if dim is None:
        dim = 1 if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)) else 0
    BoundedSpectralNorm.apply(module, name, n_power_iterations, dim, eps, bound)
    return module


def warm_up_spectral_norm(module: nn.Module, n_iterations: int = DEFAULT_SN_WARMUP_ITERATIONS) -> None:
    """Converge every spectral-norm layer's power iteration in place.

    `torch.nn.utils.spectral_norm` only advances its power iteration during *train-mode*
    forward passes, so a freshly constructed net divides each weight by `u^T W v` for
    random unit `u`, `v` -- an estimate that badly under-shoots the true largest singular
    value. The error compounds multiplicatively across layers: an untrained
    spectral-normed resnet18 emits activations on the order of 1e30 in eval mode, versus
    ~1 once the iteration has converged.

    That is harmless during training (the first train-mode forwards fix it), but it
    corrupts anything that evaluates a net before training it -- Lightning's
    `num_sanity_val_steps` pass, and any construct-then-`eval()` call.

    Covers both the stock hook and `BoundedSpectralNorm` (a subclass).
    """
    for submodule in module.modules():
        for hook in submodule._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm):
                with torch.no_grad():
                    for _ in range(n_iterations):
                        weight = hook.compute_weight(submodule, do_power_iteration=True)
                    # `compute_weight` advances u/v in place but only *returns* the
                    # renormalized weight; without this the cached `.weight` attribute
                    # stays at its random-u/v value until the next forward pre-hook runs.
                    setattr(submodule, hook.name, weight)


def apply_spectral_norm(
    module: nn.Module,
    n_power_iterations: int = 1,
    warmup_iterations: int = DEFAULT_SN_WARMUP_ITERATIONS,
    bound: Optional[float] = None,
) -> None:
    """Recursively wrap Conv2d/Linear layers with spectral normalization. Bias and
    BatchNorm layers are left untouched.

    `bound=None` is stock hard normalization (sigma == 1); a float is the paper's upper
    bound `c` (sigma <= c) -- see the module docstring.

    `warmup_iterations` seeds the power iteration so the net is usable before its first
    train-mode forward; pass 0 to skip.
    """
    _wrap_spectral_norm(module, n_power_iterations=n_power_iterations, bound=bound)
    if warmup_iterations:
        warm_up_spectral_norm(module, n_iterations=warmup_iterations)


def _wrap_spectral_norm(module: nn.Module, n_power_iterations: int = 1, bound: Optional[float] = None) -> None:
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            if not hasattr(child, "weight_u"):
                if bound is None:
                    wrapped = spectral_norm(child, n_power_iterations=n_power_iterations)
                else:
                    wrapped = bounded_spectral_norm(child, bound=bound, n_power_iterations=n_power_iterations)
                setattr(module, name, wrapped)
        else:
            _wrap_spectral_norm(child, n_power_iterations=n_power_iterations, bound=bound)


def assert_spectral_norm_compatible(arch: str) -> None:
    """Raise if `arch` is known to be incompatible with recursive spectral-norm wrapping.

    ViT backbones are excluded: SNGP's spectral normalization has not been validated
    against ViT internals (LayerNorm/attention), unlike resnet's Conv2d/Linear layers.
    """
    if arch not in SPECTRAL_NORM_COMPATIBLE:
        raise ValueError(
            f"Spectral normalization (SNGP) is not supported for backbone '{arch}'. "
            f"Supported backbones: {sorted(SPECTRAL_NORM_COMPATIBLE)}"
        )
