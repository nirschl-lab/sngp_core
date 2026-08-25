"""Class-balanced focal loss for imbalanced multiclass classification.

Combines effective-number-of-samples reweighting (Cui et al. 2019,
"Class-Balanced Loss Based on Effective Number of Samples") with focal loss
(Lin et al. 2017, "Focal Loss for Dense Object Detection"), so imbalance
correction is one mechanism with two tunable hyperparameters (`beta`, `gamma`),
built identically for every model family via `LitModuleBase._init_criterion`.
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """Focal loss with per-class weights derived from the effective number of samples.

    `beta` controls how aggressively rare classes are upweighted (`beta=0`
    degenerates to uniform weights; values close to 1, e.g. 0.999, approach
    inverse-frequency weighting). `gamma` controls the focal down-weighting of
    already-easy examples (`gamma=0` degenerates to plain class-balanced CE).
    """

    def __init__(
        self,
        class_freq: Sequence[float],
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1); got {beta}")

        counts = torch.tensor(list(class_freq), dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, counts)
        weight = (1.0 - beta) / effective_num
        weight = weight * (weight.numel() / weight.sum())
        # Registered as `weight` (not `class_weights`) to match
        # `nn.CrossEntropyLoss`'s own state-dict key, so
        # `LitModuleBase.load_state_dict`'s `criterion.weight` handling keeps
        # working unchanged.
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Focal modulation must be derived from the *unweighted* per-sample CE --
        # folding `weight=` into this call would scale `pt` itself and corrupt
        # the (1 - pt) ** gamma term.
        ce = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.weight[targets]
        return (alpha_t * (1.0 - pt) ** self.gamma * ce).mean()
