import os
import time
from typing import List, Optional
from typing import Tuple

import torch
from loguru import logger

from src.losses.focal_loss import FocalLoss
from src.metrics.calibration_losses import CalibrationLossConfig, calibration_losses
from src.models.wbc_module_base import LitModuleBase
import timm

class BaselineClassificationLitModule(LitModuleBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
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
        **kwargs
    ) -> None:
        
        net = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=num_classes, dynamic_img_size=True,
            )
        cache_dir = "/data1/shared/models/pathology_fms/"
        output_model = os.path.join(cache_dir, "uni_dinov2_vit_L16.bin")
        net.load_state_dict(torch.load(output_model), strict=False)
        logger.info(f'using uni_dinov2_vit_L16.bin')
        
        LitModuleBase.__init__(
            self,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
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
        # Before passing class_weights to FocalLoss
        if self.class_weights is not None:
            self.class_weights = torch.tensor(list(self.class_weights), dtype=torch.float32)

        # Initialize the loss criterion
        if self.loss_function == 'focal_loss':
            self.criterion = self._init_FL()
        else:
            self.criterion = self._init_CE()
    
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
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
