"""Lightning module for training Deep Ensembles.

Training-time strategy: cycle through ensemble members epoch-by-epoch. Everything
else -- forward, loss, test-time uncertainty (`ModelOutput.variance`/`.member_logits`,
populated by `DeepEnsemble.forward()` in eval mode) -- is handled generically by
`LitModuleBase`, so this class only needs the member-cycling logic below.
"""
from typing import List, Optional

import torch
from loguru import logger

from src.models.lit_module_base import LitModuleBase


class DeepEnsembleLitModule(LitModuleBase):
    """Lightning Module for Deep Ensemble training.

    During training, cycles through ensemble members epoch-by-epoch. During
    evaluation/testing, `DeepEnsemble.forward()` (net-level, keyed off `self.training`)
    already runs the full ensemble and reports `ModelOutput.variance`/`.member_logits`
    -- no lit-module-level override needed for that.
    """

    def __init__(
        self,
        net: Optional[torch.nn.Module] = None,  # Should be a DeepEnsemble instance
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        num_estimators: int = 5,
        num_classes: int = 8,
        class_freq: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        cb_beta: float = 0.999,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        train_strategy: str = "sequential",
        **kwargs,
    ) -> None:
        """
        Args:
            net: DeepEnsemble model instance
            optimizer: Optimizer (will be applied per ensemble member)
            scheduler: Learning rate scheduler
            num_estimators: Number of ensemble members
            train_strategy: How to train the ensemble. Only "sequential" (train one
                member at a time, cycling through epochs) is implemented today.
        """
        if train_strategy != "sequential":
            raise NotImplementedError(
                f"train_strategy={train_strategy!r} is not implemented; only 'sequential' "
                "is supported today. Simultaneous multi-member training ('all') is a future TODO."
            )

        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            num_classes=num_classes,
            class_freq=class_freq,
            class_weights=class_weights,
            cb_beta=cb_beta,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            **kwargs,
        )

        self.num_estimators = num_estimators
        self.train_strategy = train_strategy

        # Track which ensemble member is being trained
        self.current_member_idx = 0

    def on_train_epoch_start(self) -> None:
        """Set active ensemble member at the start of each training epoch."""
        total_epochs = self.trainer.max_epochs
        epochs_per_member = total_epochs // self.num_estimators

        if epochs_per_member > 0:
            self.current_member_idx = self.current_epoch // epochs_per_member
            # Cap at last member if we exceed
            self.current_member_idx = min(self.current_member_idx, self.num_estimators - 1)
        else:
            # If not enough epochs, just use modulo
            self.current_member_idx = self.current_epoch % self.num_estimators

        self.net.set_active_member(self.current_member_idx)
        logger.info(
            f"Epoch {self.current_epoch}: Training ensemble member "
            f"{self.current_member_idx + 1}/{self.num_estimators}"
        )

        # Log which member is being trained
        self.log(
            "train/current_member",
            float(self.current_member_idx),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_end(self) -> None:
        """Log ensemble-level validation progress on top of the shared val metrics."""
        super().on_validation_epoch_end()
        progress = (self.current_member_idx + 1) / self.num_estimators * 100
        self.log("ensemble/training_progress", progress, prog_bar=False)
