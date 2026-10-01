"""Optimizers not shipped by the pinned torch.

`MuonWithAuxAdamW` / `MuonWithAuxSGD` are a single-device Muon (Jordan et al. 2024,
https://kellerjordan.github.io/posts/muon/) with an AdamW or SGD fallback for every
parameter Muon should not see, in one `torch.optim.Optimizer` so the project's
single-optimizer `configure_optimizers` and its per-epoch LR schedulers work unchanged.
torch 2.8 (the pinned version) has no `torch.optim.Muon`, and the 2.9 one accepts 2D
parameters only, while a ResNet's matrices are 4D conv kernels.

The aux optimizer matters more than its <0.1% share of the parameters suggests: it owns
every BatchNorm affine and the GP output layer, which set the feature scale the RBF kernel
sees and the logit scale. On CIFAR-100 (250 epochs to 100% train accuracy, no early
stopping) the weakly decayed AdamW group let the final BN gamma grow 3.5x and the GP output
layer 4x against the SGD arms' L2 6e-4, narrowing the kernel and inflating the logits.
`MuonWithAuxSGD` gives that group the SGD arms' exact update rule (the airbench Muon recipe),
so Muon-vs-SGD differs only on the hidden convs.

Muon replaces each hidden weight's momentum update with its nearest semi-orthogonal
matrix (Newton-Schulz), so every step has spectral norm ~lr regardless of the gradient's
magnitude. It bounds the *update*, not the weight: only decoupled weight decay caps the
weight's spectral norm, softly, at roughly `1 / weight_decay` in steady state.
"""
from typing import Iterable, Tuple

import torch
from loguru import logger
from torch.optim import Optimizer

# Quintic Newton-Schulz coefficients of the reference implementation. They trade exact
# convergence for speed: singular values land in roughly [0.7, 1.2], not exactly 1.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def is_muon_param(p: torch.Tensor) -> bool:
    """A hidden conv kernel: 4D and not the RGB stem (`in_channels == 3`).

    `LitModuleBase.configure_optimizers` passes a flat `parameters()` iterator, so the
    split is by shape, not name. For the torchvision ResNets / WRN this sends every conv
    but the stem to Muon; the stem, BatchNorm/LayerNorm affine params, biases and the 2D
    GP output layer (the reference keeps embeddings and heads on Adam) go to the aux
    optimizer (AdamW or SGD). The
    random-feature projection and precision matrix are buffers and never reach here.
    """
    return p.ndim == 4 and p.shape[1] > 3


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor U V^T of a 2D matrix G = U S V^T, in bfloat16."""
    a, b, c = _NS_COEFFS
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class _MuonHybrid(Optimizer):
    """Muon on hidden conv kernels (group 0), an aux optimizer on everything else (group 1).

    Each group has its own `lr` / `weight_decay`, so an LR scheduler scales both from their
    own initial LR, and `param_groups[0]['lr']` (what `LitModuleBase` logs as `lr`) is the
    Muon LR. Subclasses build the aux group and implement `_aux_step`.
    """

    _aux_name = "aux"

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        aux_group: dict,
        lr: float,
        momentum: float,
        weight_decay: float,
        nesterov: bool,
        ns_steps: int,
    ) -> None:
        params = [p for p in params if p.requires_grad]
        muon_params = [p for p in params if is_muon_param(p)]
        aux_params = [p for p in params if not is_muon_param(p)]
        if not muon_params:
            raise ValueError(
                f"{type(self).__name__} found no hidden conv kernels (4D, in_channels > 3) -- "
                "this optimizer targets conv backbones; use a plain optimizer for this model."
            )
        groups = [
            dict(params=muon_params, use_muon=True, lr=lr, momentum=momentum,
                 weight_decay=weight_decay, nesterov=nesterov, ns_steps=ns_steps),
            dict(params=aux_params, use_muon=False, **aux_group),
        ]
        super().__init__(groups, defaults={})
        logger.info(
            f"{type(self).__name__}: Muon on {len(muon_params)} tensors "
            f"({sum(p.numel() for p in muon_params):,} params), {self._aux_name} on "
            f"{len(aux_params)} tensors ({sum(p.numel() for p in aux_params):,} params)"
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._aux_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        beta = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - beta)
            update = p.grad.lerp(buf, beta) if group["nesterov"] else buf
            update = update.reshape(len(update), -1)  # conv (out, in, kh, kw) -> (out, in*kh*kw)
            update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
            update = update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update.reshape(p.shape).to(p.dtype), alpha=-group["lr"])

    def _aux_step(self, group: dict) -> None:
        raise NotImplementedError


class MuonWithAuxAdamW(_MuonHybrid):
    """Muon on hidden conv kernels, AdamW on everything else.

    :param params: The model's parameters (split by `is_muon_param`).
    :param lr: Muon learning rate (reference default 0.02).
    :param momentum: Muon momentum.
    :param weight_decay: Muon decoupled weight decay.
    :param nesterov: Nesterov-style momentum for Muon.
    :param ns_steps: Newton-Schulz iterations.
    :param adamw_lr: AdamW learning rate.
    :param adamw_betas: AdamW betas.
    :param adamw_eps: AdamW epsilon.
    :param adamw_weight_decay: AdamW decoupled weight decay.
    """

    _aux_name = "AdamW"

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 1e-3,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
    ) -> None:
        aux_group = dict(lr=adamw_lr, betas=tuple(adamw_betas), eps=adamw_eps,
                         weight_decay=adamw_weight_decay)
        super().__init__(params, aux_group, lr, momentum, weight_decay, nesterov, ns_steps)

    def _aux_step(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step = state["step"]
            state["exp_avg"].lerp_(p.grad, 1 - beta1)
            state["exp_avg_sq"].lerp_(p.grad.square(), 1 - beta2)
            m_hat = state["exp_avg"] / (1 - beta1**step)
            v_hat = state["exp_avg_sq"] / (1 - beta2**step)
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(m_hat / (v_hat.sqrt() + group["eps"]), alpha=-group["lr"])


class MuonWithAuxSGD(_MuonHybrid):
    """Muon on hidden conv kernels, SGD (momentum, optional Nesterov) on everything else.

    The aux step is `torch.optim.SGD` with dampening 0, including its *coupled* L2
    (`grad + weight_decay * p`) -- the same rule, and so the same meaning of
    `sgd_weight_decay`, as the SGD arms' `weight_decay`.

    :param params: The model's parameters (split by `is_muon_param`).
    :param lr: Muon learning rate (reference default 0.02).
    :param momentum: Muon momentum.
    :param weight_decay: Muon decoupled weight decay.
    :param nesterov: Nesterov-style momentum for Muon.
    :param ns_steps: Newton-Schulz iterations.
    :param sgd_lr: SGD learning rate.
    :param sgd_momentum: SGD momentum.
    :param sgd_nesterov: Nesterov momentum for SGD.
    :param sgd_weight_decay: SGD L2 penalty, added to the gradient.
    """

    _aux_name = "SGD"

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        sgd_lr: float = 0.04,
        sgd_momentum: float = 0.9,
        sgd_nesterov: bool = True,
        sgd_weight_decay: float = 0.0,
    ) -> None:
        aux_group = dict(lr=sgd_lr, momentum=sgd_momentum, nesterov=sgd_nesterov,
                         weight_decay=sgd_weight_decay)
        super().__init__(params, aux_group, lr, momentum, weight_decay, nesterov, ns_steps)

    def _aux_step(self, group: dict) -> None:
        mu = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            d = p.grad.add(p, alpha=group["weight_decay"]) if group["weight_decay"] else p.grad
            if mu:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = d.clone()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).add_(d)
                d = d.add(buf, alpha=mu) if group["nesterov"] else buf
            p.add_(d, alpha=-group["lr"])
