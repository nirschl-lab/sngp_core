"""Spectral-norm wrapping used by SNGP nets.

Dependency-light (only `torch`, plus `src.models.backbones` for the compatibility set) so
this file can be vendored into HF export bundles alongside `backbones.py`.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn.utils.spectral_norm import SpectralNorm

from src.models.backbones import SPECTRAL_NORM_COMPATIBLE

# Power iterations run at construction to seed u/v. `spectral_norm`'s hook only iterates
# in train mode, so without this a freshly built net normalizes by a spectral-norm
# estimate taken from *random* u/v -- see `warm_up_spectral_norm`.
DEFAULT_SN_WARMUP_ITERATIONS = 20


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
) -> None:
    """Recursively wrap Conv2d/Linear layers with spectral normalization. Bias and
    BatchNorm layers are left untouched.

    `warmup_iterations` seeds the power iteration so the net is usable before its first
    train-mode forward; pass 0 to skip.
    """
    _wrap_spectral_norm(module, n_power_iterations=n_power_iterations)
    if warmup_iterations:
        warm_up_spectral_norm(module, n_iterations=warmup_iterations)


def _wrap_spectral_norm(module: nn.Module, n_power_iterations: int = 1) -> None:
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            if not hasattr(child, "weight_u"):
                setattr(module, name, spectral_norm(child, n_power_iterations=n_power_iterations))
        else:
            _wrap_spectral_norm(child, n_power_iterations=n_power_iterations)


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
