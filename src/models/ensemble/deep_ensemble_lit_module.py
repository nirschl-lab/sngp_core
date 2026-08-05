"""
Lightning module for training and evaluating Deep Ensembles.

This module handles:
- Sequential training of ensemble members
- Inference with ensemble averaging
- Uncertainty quantification
- Logging of individual member and ensemble metrics
"""
import time
from typing import List, Optional, Tuple

import torch
from loguru import logger

from src.models.lit_module_base import LitModuleBase


class DeepEnsembleLitModule(LitModuleBase):
    """
    Lightning Module for Deep Ensemble training and evaluation.
    
    During training, cycles through ensemble members epoch-by-epoch.
    During evaluation/testing, uses full ensemble for predictions.
    """
    
    def __init__(
        self,
        net: torch.nn.Module,  # Should be a DeepEnsemble instance
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_estimators: int = 5,
        num_classes: int = 8,
        hist_bins: int = 10,
        calibration_curve_bins: int = 10,
        test_name: str = "test_predictions",
        log_csv: bool = False,
        csv_save_path: str = "csv/",
        log_metrics_per_class: bool = False,
        log_test_metrics: bool = True,
        class_freq: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        uncertainty_type: str = "variance",  # or "entropy", "mutual_info"
        train_strategy: str = "sequential",  # or "all" (experimental)
        **kwargs
    ) -> None:
        """
        Args:
            net: DeepEnsemble model instance
            optimizer: Optimizer (will be applied per ensemble member)
            scheduler: Learning rate scheduler
            num_estimators: Number of ensemble members
            train_strategy: How to train ensemble
                - "sequential": Train one member at a time (recommended)
                - "all": Train all members simultaneously (requires more memory)
            uncertainty_type: Type of uncertainty to compute
                - "variance": Variance across ensemble predictions
                - "entropy": Entropy of mean prediction
                - "mutual_info": Mutual information (epistemic uncertainty)
        """
        
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            num_classes=num_classes,
            hist_bins=hist_bins,
            calibration_curve_bins=calibration_curve_bins,
            test_name=test_name,
            log_csv=log_csv,
            csv_save_path=csv_save_path,
            log_metrics_per_class=log_metrics_per_class,
            log_test_metrics=log_test_metrics,
            class_freq=class_freq,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            **kwargs
        )
        
        self.num_estimators = num_estimators
        self.uncertainty_type = uncertainty_type
        self.train_strategy = train_strategy
        
        # Track which ensemble member is being trained
        self.current_member_idx = 0
        self.inference_times = []
        
        # For tracking individual member performance
        self.member_train_losses = [[] for _ in range(num_estimators)]
        self.member_val_losses = [[] for _ in range(num_estimators)]
        
    def on_train_epoch_start(self) -> None:
        """Set active ensemble member at the start of each training epoch."""
        if self.train_strategy == "sequential":
            # Cycle through ensemble members
            # Each member gets trained for full epochs
            # You can customize this logic based on your needs
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
                prog_bar=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ensemble."""
        return self.net(x)
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data.
        
        During training: Uses only the active ensemble member
        During eval: Uses full ensemble
        """
        img_ids, x, targets, fold = batch
        batch_size = x.size(0)
        
        # ---- Forward Pass with Optional Timing ----
        start_time = None
        if self.log_test_metrics and not self.training:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
        
        # Forward pass
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        
        if self.log_test_metrics and not self.training:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
        
        # Compute predictions
        preds = torch.argmax(probs, dim=1)
        
        # ---- Loss Computation ----
        loss = self.criterion(logits, targets)
        
        return img_ids, loss, logits, probs, preds, targets, fold
    
    def on_validation_epoch_end(self) -> None:
        """Log ensemble-level validation metrics."""
        super().on_validation_epoch_end()
        
        # Log current training progress
        if self.train_strategy == "sequential":
            progress = (self.current_member_idx + 1) / self.num_estimators * 100
            self.log("ensemble/training_progress", progress, prog_bar=False)
    
    def on_test_epoch_start(self) -> None:
        """Initialize inference tracking."""
        super().on_test_epoch_start()
        self.inference_times = []
    
    def on_test_epoch_end(self) -> None:
        """Log ensemble-specific test metrics including uncertainty."""
        super().on_test_epoch_end()
        
        # Log inference timing statistics
        if self.inference_times:
            mean_time = sum(self.inference_times) / len(self.inference_times)
            self.log("test/inference_time_mean", mean_time)
            logger.info(
                f"Ensemble inference: {mean_time*1000:.2f}ms per batch "
                f"({self.num_estimators} members)"
            )
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Test step with ensemble predictions and uncertainty quantification.
        """
        img_ids, x, targets, fold = batch
        
        # Get ensemble predictions with uncertainty
        if hasattr(self.net, 'get_predictive_uncertainty'):
            probs, uncertainty = self.net.get_predictive_uncertainty(
                x, 
                uncertainty_type=self.uncertainty_type
            )
            logits = torch.log(probs + 1e-10)  # Convert back to logits for metrics
        else:
            # Fallback if not a DeepEnsemble
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            uncertainty = torch.zeros(x.size(0), device=x.device)
        
        preds = torch.argmax(probs, dim=1)
        loss = self.criterion(logits, targets)
        
        # Update metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        
        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log average uncertainty
        mean_uncertainty = uncertainty.mean()
        self.log("test/uncertainty_mean", mean_uncertainty, on_step=False, on_epoch=True)
        
        # Store predictions and uncertainty for later analysis
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_targets = []
            self.test_uncertainties = []
            self.test_probs = []
        
        self.test_predictions.append(preds.cpu())
        self.test_targets.append(targets.cpu())
        self.test_uncertainties.append(uncertainty.cpu())
        self.test_probs.append(probs.cpu())
    
    def configure_optimizers(self):
        """
        Configure optimizer for ensemble training.
        
        For sequential training, only optimizes parameters of the active member.
        """
        if self.train_strategy == "sequential":
            # This will be called once, but during training we'll manage which
            # parameters to update through the forward pass
            optimizer = self.hparams.optimizer(params=self.net.parameters())
        else:
            # Train all members simultaneously
            optimizer = self.hparams.optimizer(params=self.net.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
