#!/usr/bin/env python3
"""calibration_losses.py in src/metrics."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-5


# ---------- utilities ----------

def _softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)

def _entropy_from_probs(p: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # p: [N, C]
    p_safe = (1.0 - eps) * p + eps / p.size(-1)
    return -(p_safe * torch.log(p_safe)).sum(-1)  # [N]

def _accuracy_indicator(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # returns float tensor of shape [N] with 1. for correct, 0. for incorrect
    pred = logits.argmax(dim=-1)
    return (pred == y).float()


# ---------- 1) SB-ECElb (label-binned, L2) ----------
def soft_binned_ece_label_squared(
    logits: torch.Tensor,
    y: torch.Tensor,
    m: int,
    temperature: float,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Squared soft-binned ECElb (Eq. 12 in Soft Calibration Objectives).
    Vectorized PyTorch refactor of `compute_squared_error_label_binning_tensorflow`.

    Args:
        logits: [N, C]
        y:      [N] int64
        m: number of bins
        temperature: soft bin temperature T (>0)
    Returns:
        scalar tensor
    """
    device = logits.device
    N, C = logits.shape
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)            # [N]
    acc = (pred == y).float()                  # [N]

    # bin centers b in (1/C .. 1) equally spaced like TF impl
    # TF anchors: midpoints of linspace(1/C, 1, m+1)
    edges = torch.linspace(1.0 / C, 1.0, steps=m + 1, device=device)
    b = 0.5 * (edges[1:] + edges[:-1])        # [m]

    # soft memberships u* (N,m) via softmax of -(c - b)^2 / T
    # broadcasting: (N,1) - (1,m) -> (N,m)
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    logits_membership = -((conf.unsqueeze(1) - b.unsqueeze(0)) ** 2) / temperature
    c = _softmax_stable(logits_membership, dim=1)          # [N,m]
    c = (1 - eps) * c + eps * (1.0 / m)                    # smoothing like TF

    # a_bar per bin: weighted mean of accuracy by memberships
    num = (c * acc.unsqueeze(1)).sum(dim=0)                # [m]
    den = c.sum(dim=0).clamp_min(eps)                      # [m]
    a_bar = num / den                                      # [m]

    # squared error across examples and bins (Eq. 12 inner term)
    se = (c * (a_bar.unsqueeze(0) - conf.unsqueeze(1))**2).sum()  # scalar
    return se / float(N)


# ---------- 2) SB-ECEbin (confidence-binned, L2 root) ----------
def soft_binned_ece_confidence(
    confidences: torch.Tensor,  # [N] scalar confidence per example
    accuracies: torch.Tensor,   # [N] 0/1 correctness per example
    m: int,
    use_decay: bool,
    decay_factor: float,
    temperature: float,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Soft-binned ECE (Eq. 11). Refactor of TF `get_soft_binning_ece_tensor`.

    Returns:
        scalar tensor: sqrt( sum_j w_j * (conf_bin_j - acc_bin_j)^2 )
    """
    device = confidences.device
    N = confidences.numel()

    # anchors at midpoints: (1/(2m), 3/(2m), ..., (2m-1)/(2m))
    anchors = torch.arange(1, 2*m, 2, device=device, dtype=confidences.dtype) / (2.0 * m)  # [m]

    if use_decay:
        # paper’s/UB’s heuristic: T from desired geometric decay between successive bins
        # T = 1 / (log(decay_factor) * m^2)
        temperature = 1.0 / (math.log(decay_factor) * m * m)

    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    # membership logits L_{i,j} = - (c_i - ξ_j)^2 / T
    L = -((confidences.unsqueeze(1) - anchors.unsqueeze(0)) ** 2) / temperature  # [N,m]
    U = _softmax_stable(L, dim=1)  # memberships u*_{i,j}  [N,m]

    sum_coeffs_for_bin = U.sum(dim=0).clamp_min(eps)  # [m]

    # per-bin average confidence and accuracy (weighted by U)
    conf_bin = (U * confidences.unsqueeze(1)).sum(dim=0) / sum_coeffs_for_bin  # [m]
    acc_bin  = (U * accuracies.unsqueeze(1)).sum(dim=0)  / sum_coeffs_for_bin  # [m]

    # bin weights normalized to 1 (L1) like TF
    bin_weights = (sum_coeffs_for_bin / sum_coeffs_for_bin.sum()).clamp_min(eps)

    # L2 distance aggregated with weights; UB returns sqrt of weighted L2
    ece = torch.sqrt(((conf_bin - acc_bin) ** 2 * bin_weights).sum())
    return ece


# ---------- 3) AvUC ----------
def avuc_loss(
    probabilities: torch.Tensor,  # [N,C] probs
    labels: torch.Tensor,         # [N]
    stop_prob_gradients: bool = False,
    entropy_threshold: float = 0.5,   # u_th; tune per dataset
    eps: float = EPS,
) -> torch.Tensor:
    """
    Refactor of TF `get_avuc_loss` (AvUC with optional prob stop-grad).
    Uses same tanh(entropy) shaping and log(1 + (AU + IC)/(AC + IU)) form.
    """
    N, C = probabilities.shape
    if stop_prob_gradients:
        probs = probabilities.detach()
    else:
        probs = probabilities

    conf, pred = probs.max(dim=1)                 # [N]
    acc = (pred == labels).float()                # [N]
    ent = _entropy_from_probs(probs, eps=eps)     # [N]

    # four soft buckets (mirrors TF implementation)
    # AC: accurate & entropy < th
    nac = (conf * (1 - torch.tanh(ent))) * ((ent < entropy_threshold) & (acc > 0.5)).float()
    # AU: accurate & entropy >= th
    nau = (conf * torch.tanh(ent)) * ((ent >= entropy_threshold) & (acc > 0.5)).float()
    # IC: inaccurate & entropy < th
    nic = ((1 - conf) * (1 - torch.tanh(ent))) * ((ent < entropy_threshold) & (acc < 0.5)).float()
    # IU: inaccurate & entropy >= th
    niu = ((1 - conf) * torch.tanh(ent)) * ((ent >= entropy_threshold) & (acc < 0.5)).float()

    nac_diff = nac.sum()
    nau_diff = nau.sum()
    nic_diff = nic.sum()
    niu_diff = niu.sum()

    loss = torch.log1p((nau_diff + nic_diff) / (nac_diff + niu_diff).clamp_min(eps))
    return loss


# ---------- 4) Soft-AvUC ----------
def soft_avuc_loss(
    probabilities: torch.Tensor,  # [N,C]
    labels: torch.Tensor,         # [N]
    use_deprecated_v0: bool = False,
    temp: float = 1.0,            # T in Eq. 15 (soft AvUC)
    theta: float = 0.5,           # kappa in Eq. 15, (0,1)
    eps: float = EPS,
) -> torch.Tensor:
    """
    Refactor of TF `get_soft_avuc_loss`. Uses normalized-entropy based
    soft gating between certain/uncertain.
    """
    N, C = probabilities.shape
    probs = probabilities
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()                # [N]
    ent = _entropy_from_probs(probs, eps=eps)     # [N]
    entmax = math.log(C)

    if use_deprecated_v0:
        # older 2-way softmax on [- (H - Hmax)^2, -H^2]
        xus = -((ent - entmax) ** 2)
        xcs = -(ent ** 2)
        qucs = _softmax_stable(torch.stack([xus, xcs], dim=1), dim=1)   # [N,2]
        qus = qucs[:, 0]
        qcs = qucs[:, 1]
    else:
        # soft uncertainty from normalized entropy \tilde{e} = H/ln(C)
        ebar = (ent / entmax).clamp(0.0, 1.0)
        # sigmoid( (1/T) * log( e*(1-θ) / ((1-e)*θ) ) )
        logit = torch.log((ebar * (1 - theta)).clamp_min(eps)) - torch.log(((1 - ebar) * theta).clamp_min(eps))
        qus = torch.sigmoid(logit / max(temp, eps))  # uncertainty weight
        qcs = 1.0 - qus

    # Use the same tanh(entropy) shaping as TF for A/C split
    tanh_ent = torch.tanh(ent)

    # AC/AU/IC/IU soft masses using qcs/qus instead of hard thresholding
    nac = (qcs * (1 - tanh_ent)) * (acc > 0.5)
    nau = (qus * tanh_ent)       * (acc > 0.5)
    nic = (qcs * (1 - tanh_ent)) * (acc < 0.5)
    niu = (qus * tanh_ent)       * (acc < 0.5)

    nac_diff = nac.sum()
    nau_diff = nau.sum()
    nic_diff = nic.sum()
    niu_diff = niu.sum()

    loss = torch.log1p((nau_diff + nic_diff) / (nac_diff + niu_diff).clamp_min(eps))
    return loss


# ---------- 5) A small wrapper to combine with CE ----------

@dataclass
class CalibrationLossConfig:
    sb_ece_label_weight: float = 0.0
    sb_ece_label_bins: int = 15
    sb_ece_label_temp: float = 1e-2

    sb_ece_conf_weight: float = 0.0
    sb_ece_conf_bins: int = 15
    sb_ece_conf_temp: float = 1e-2
    sb_ece_conf_use_decay: bool = False
    sb_ece_conf_decay_factor: float = 0.9

    avuc_weight: float = 0.0
    avuc_stop_prob_grad: bool = False
    avuc_entropy_threshold: float = 0.5

    soft_avuc_weight: float = 0.0
    soft_avuc_temp: float = 1.0
    soft_avuc_theta: float = 0.5
    soft_avuc_deprecated_v0: bool = False


def calibration_losses(
    logits: torch.Tensor,             # [N,C]
    labels: torch.Tensor,             # [N]
    cfg: CalibrationLossConfig,
) -> Tuple[torch.Tensor, dict]:
    """
    Returns total calibration penalty and a dict of individual terms.
    """
    terms = {}
    probs = F.softmax(logits, dim=-1)
    conf = probs.max(dim=-1).values
    acc = _accuracy_indicator(logits, labels)

    total = logits.new_tensor(0.0)

    if cfg.sb_ece_label_weight > 0:
        sb_lb = soft_binned_ece_label_squared(logits, labels, cfg.sb_ece_label_bins, cfg.sb_ece_label_temp)
        terms["sb_ece_label_sq"] = sb_lb
        total = total + cfg.sb_ece_label_weight * sb_lb

    if cfg.sb_ece_conf_weight > 0:
        sb_bin = soft_binned_ece_confidence(
            conf, acc, cfg.sb_ece_conf_bins, cfg.sb_ece_conf_use_decay,
            cfg.sb_ece_conf_decay_factor, cfg.sb_ece_conf_temp
        )
        terms["sb_ece_conf"] = sb_bin
        total = total + cfg.sb_ece_conf_weight * sb_bin

    if cfg.avuc_weight > 0:
        avuc = avuc_loss(probs, labels, cfg.avuc_stop_prob_grad, cfg.avuc_entropy_threshold)
        terms["avuc"] = avuc
        total = total + cfg.avuc_weight * avuc

    if cfg.soft_avuc_weight > 0:
        savuc = soft_avuc_loss(
            probs, labels, cfg.soft_avuc_deprecated_v0, cfg.soft_avuc_temp, cfg.soft_avuc_theta
        )
        terms["soft_avuc"] = savuc
        total = total + cfg.soft_avuc_weight * savuc

    return total, terms
