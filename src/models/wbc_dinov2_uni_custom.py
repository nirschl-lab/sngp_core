import os
import time
from typing import List, Optional
from typing import Tuple

import torch
from loguru import logger
from transformers import get_cosine_schedule_with_warmup

from src.losses.focal_loss import FocalLoss
from src.metrics.calibration_losses import CalibrationLossConfig, calibration_losses
from src.models.wbc_module_base2 import LitModuleBase
import timm
import torch.nn as nn

class BaselineClassificationLitModule(LitModuleBase):
    def __init__(
        self,
        compile: bool,
        class_indices: dict,
        num_classes: int = 13,
        log_csv: bool = False,
        csv_name: str = "test_predictions",
        csv_save_path: str = "csv/",
        log_metrics_per_class: bool = False,
        log_test_metrics: bool = True,
        class_freq: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0, # recommend avoiding with SNGP and calibration losses, if needed set alpha low [0.01, 0.05].
        loss_function = 'cross_entropy',
        freeze_backbone_epochs: int = 3,  # Number of epochs to freeze backbone
        max_epochs: int = 50,  # Total number of training epochs
        warmup_ratio: float = 0.1,  # Ratio of total steps for warmup
        **kwargs
    ) -> None:
        
        net = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=num_classes, dynamic_img_size=True,
            )
        cache_dir = "/data1/shared/models/pathology_fms/"
        output_model = os.path.join(cache_dir, "uni_dinov2_vit_L16.bin")
        net.load_state_dict(torch.load(output_model), strict=False)
        logger.info(f'using uni_dinov2_vit_L16.bin')

        d = net.num_features
        net.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(0.1),
            nn.Linear(d, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 13),
        )
        
        LitModuleBase.__init__(
            self,
            net=net,
            compile=compile,
            num_classes=num_classes,
            log_csv=log_csv,
            csv_save_path=csv_save_path,
            csv_name=csv_name,
            log_metrics_per_class=log_metrics_per_class,
            log_test_metrics=log_test_metrics,
            class_indices=class_indices,
            **kwargs
        )
        self.loss_function = loss_function
        self.class_freq = class_freq
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing  # recommend avoiding with SNGP and calibration losses
        self.num_classes = num_classes
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.max_epochs = max_epochs
        self.warmup_ratio = warmup_ratio
        
        # Before passing class_weights to FocalLoss
        if self.class_weights is not None:
            self.class_weights = torch.tensor(list(self.class_weights), dtype=torch.float32)

        # Initialize the loss criterion
        if self.loss_function == 'focal_loss':
            self.criterion = self._init_FL()
        else:
            self.criterion = self._init_CE()

        # Freeze backbone initially
        # self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all parameters except classifier."""
        for name, param in self.net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen - only classifier parameters will be updated")

    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for name, param in self.net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = True
        logger.info("Backbone unfrozen - all parameters will be updated")
    
    # In your configure_optimizers method, replace the entire method with:
    def configure_optimizers(self):
        # Always configure for all parameters, but with different learning rates
        # Use very small LR for backbone initially, then it will naturally increase
        backbone_params = [p for n, p in self.net.named_parameters() if "classifier" not in n]
        classifier_params = list(self.net.classifier.parameters())
        
        # # Calculate current learning rate based on epoch
        # if hasattr(self, 'current_epoch') and self.current_epoch < self.freeze_backbone_epochs:
        #     backbone_lr = 0.0  # Effectively frozen
        # else:
        #     backbone_lr = 1e-5

        backbone_lr = 1e-5
        optimizer = torch.optim.AdamW([
            {"params": classifier_params, "lr": 1e-3},
            {"params": backbone_params, "lr": backbone_lr},
        ], weight_decay=0.05)
        
        # Calculate total training steps
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            steps_per_epoch = 500  # reasonable default
            total_steps = self.max_epochs * steps_per_epoch
        
        warmup_steps = int(self.warmup_ratio * total_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # def on_train_epoch_start(self):
    #     """Called at the start of each training epoch."""
    #     # Call parent method
    #     super().on_train_epoch_start()
        
    #     # Unfreeze backbone after specified epochs
    #     if self.current_epoch == self.freeze_backbone_epochs:
    #         self._unfreeze_backbone()
    #         # Reconfigure optimizers to include backbone parameters
    #         optimizers, schedulers = self.configure_optimizers()
    #         self.trainer.optimizers = [optimizers] if not isinstance(optimizers, list) else optimizers
    #         if schedulers:
    #             self.trainer.lr_schedulers = [schedulers] if not isinstance(schedulers, list) else schedulers
    #         logger.info(f"Epoch {self.current_epoch}: Backbone unfrozen and optimizer reconfigured")

    # def configure_optimizers(self):
    #     # Check if backbone should be frozen based on current epoch
    #     if hasattr(self, 'current_epoch') and self.current_epoch < self.freeze_backbone_epochs:
    #         # Only optimize classifier parameters
    #         optimizer = torch.optim.AdamW([
    #             {"params": self.net.classifier.parameters(), "lr": 1e-3},
    #         ], weight_decay=0.05)
    #         logger.info(f"Optimizer configured for frozen backbone (epoch {self.current_epoch})")
    #     else:
    #         # Optimize all parameters with different learning rates
    #         optimizer = torch.optim.AdamW([
    #             {"params": self.net.classifier.parameters(), "lr": 1e-3},
    #             {"params": [p for n,p in self.net.named_parameters() if "classifier" not in n], "lr": 1e-5},
    #         ], weight_decay=0.05)
    #         logger.info("Optimizer configured for full model training")
        
    #     # Calculate total training steps for cosine schedule
    #     # Use Lightning's estimated stepping batches if available
    #     if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
    #         total_steps = self.trainer.estimated_stepping_batches
    #     else:
    #         # Fallback calculation if estimated stepping batches is not available
    #         # This might not be exact but provides a reasonable estimate
    #         steps_per_epoch = len(self.trainer.train_dataloader) if self.trainer.train_dataloader else 500
    #         total_steps = self.max_epochs * steps_per_epoch
        
    #     warmup_steps = int(self.warmup_ratio * total_steps)
        
    #     logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
    #     # Create cosine schedule with warmup
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=total_steps,
    #     )
        
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",  # Update scheduler every step (not epoch)
    #             "frequency": 1,
    #         },
    #     }
    
    def _init_FL(self):
        # Move class_weights to device if not None
        device_weights = self.class_weights.to(self.device) if self.class_weights is not None else None
        logger.info(f"Initializing focal loss with class weights: {self.class_weights}")
        return FocalLoss(alpha=device_weights, gamma=1.0)
    
    def _init_CE(self):
        """Initialize the loss criterion with class weights and label smoothing if provided."""

        # Set class weights, if provided
        if self.class_weights:
            assert len(self.class_weights) == self.num_classes, "Length of class_weights must match num_classes"
        elif self.class_freq:
            assert len(self.class_freq) == self.num_classes, "Length of class_freq must match num_classes"
            weights = torch.tensor([1.0 / self.class_freq[k] for k in self.class_freq], dtype=torch.float32)
            self.class_weights = weights / weights.sum()
        else:
            self.class_weights = None

        if self.class_weights is not None:
            logger.info(f"Using class weights for CrossEntropyLoss: {self.class_weights} and label smoothing: {self.label_smoothing}")
            class_weights_tensor = torch.tensor(self.class_weights, device=self.device)
            return torch.nn.CrossEntropyLoss(
                weight=class_weights_tensor, label_smoothing=self.label_smoothing
            )
        else:
            logger.info(f"No class weights provided, using unweighted CrossEntropyLoss and label smoothing: {self.label_smoothing}")
            return torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        



    # def model_step(
    #         self, batch: Tuple[torch.Tensor, torch.Tensor]
    #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Perform a single model step on a batch of data.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

    #     :return: A tuple containing (in order):
    #         - A tensor of losses.
    #         - A tensor of predictions.
    #         - A tensor of target labels.
    #     """
    #     img_ids, x, targets, fold = batch
    #     batch_size = x.size(0)


    #     # ---- Forward Pass with Optional Timing ----
    #     start_time = None
    #     if self.log_test_metrics:
    #         torch.cuda.synchronize() if torch.cuda.is_available() else None
    #         start_time = time.time()

    #     if self.use_mc:
    #         logger.info(
    #             f"Using Monte Carlo Dropout for inference for {self.mc_passes} passes"
    #         )
    #         logits, probs = self.net.mc_predict(x, T=self.mc_passes, return_std=False, apply_softmax=True)
    #     else:
    #         logits = self.forward(x)
    #         probs = torch.softmax(logits, dim=1)

    #     if self.log_test_metrics:
    #         torch.cuda.synchronize() if torch.cuda.is_available() else None
    #         end_time = time.time()
    #         inference_time_per_sample = (end_time - start_time) / batch_size
    #         self.inference_times.append(inference_time_per_sample)

    #     preds = torch.argmax(probs, dim=1)

    #     # ---- LOSS COMPUTATION (primary CE loss + secondary calibration) ----
    #     # when tested with ood data, class dims from dataloader may not match model output dims
    #     if not self.log_test_metrics:  
    #         loss = None
    #     else:
    #         ce = self.criterion(logits, targets)  # Cross-entropy classification loss
    #         cal_penalty = logits.new_tensor(0.0)
    #         cal_terms = {}

    #         # Apply calibration losses if enabled
    #         if ((self.training and self.cal_cfg) or
    #                 (self.compute_cal_on_val and (not self.training) and self.cal_cfg)):
    #             cal_penalty, cal_terms = calibration_losses(logits, targets, self.cal_cfg)

    #         # Total combined loss
    #         loss = ce 

    #         # ---- LOGGING ----
    #         mode = "train" if self.training else "val"

    #         # 1. Always log CE (both train and val)
    #         self.log(f"{mode}/ce", ce, on_step=False, on_epoch=True, prog_bar=True)

    #         # 2. Log calibration loss and sub-terms (train and val and enabled)
    #         if self.cal_cfg and self.log_calibration_terms:
    #             self.log(f"{mode}/cal_total", cal_penalty, on_step=False, on_epoch=True, prog_bar=True)

    #             # no need to log all sub-terms unless debugging
    #             if os.getenv("DEBUG", "0") == "1":
    #                 for k, v in cal_terms.items():
    #                     self.log(f"{mode}/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

    #             # 3. Log CE-to-Calibration ratio (monitor dominance)
    #             ratio = ce / (cal_penalty + 1e-8)
    #             self.log(f"{mode}/ce_to_cal_ratio", ratio, on_step=False, on_epoch=True, prog_bar=False)

    #             # # Optional sanity check: calibration relative magnitude (%)
    #             # cal_rel = (cal_penalty / (ce + 1e-8)) * 100
    #             # self.log(f"{mode}/calibration_pct_of_ce", cal_rel, on_step=False, on_epoch=True, prog_bar=False)

    #     return img_ids, loss, logits, probs, preds, targets, fold
