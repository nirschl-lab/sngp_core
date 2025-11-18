import torch
from typing import List, Optional
from src.models.lit_module_base import LitModuleBase
from loguru import logger
import time
import torch.nn.functional as F
import os
from typing import Tuple
from src.models.sngp.sngp_diagnostic_mixin import SNGPDiagnosticsMixin
from src.metrics.calibration_losses import CalibrationLossConfig, calibration_losses

class SNGPClassificationLitModule(SNGPDiagnosticsMixin, LitModuleBase):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        calibration_cfg: Optional[CalibrationLossConfig],
        compile: bool,
        num_classes: int = 8,
        hist_bins: int = 10, #for histogram plotting
        calibration_curve_bins: int =10, #for ece plot
        test_name: str = "test_predictions",
        log_csv: bool = False,
        csv_save_path: str = "csv/",
        log_metrics_per_class: bool = False,
        log_test_metrics: bool = True,
        log_calibration_terms: bool = True,
        compute_calibration_on_val: bool = False,
        class_freq: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0, # recommend avoiding with SNGP and calibration losses, if needed set alpha low [0.01, 0.05].
        **kwargs
    ) -> None:
        
        LitModuleBase.__init__(
                self,
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
                log_calibration_terms=log_calibration_terms,
                compute_calibration_on_val=compute_calibration_on_val,
                class_freq=class_freq,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                **kwargs
            )
        
        self.cal_cfg = calibration_cfg
        self.log_calibration_terms = log_calibration_terms
        self.compute_cal_on_val = compute_calibration_on_val

        self.inference_times = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if self.training:
            mf_logits, _, _ = self.net(x, update_cov=True)
        else:
            # logger.debug('eval mode update cov is false')
            mf_logits, _, _ = self.net(x, update_cov=False)

        return mf_logits
    
    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        img_ids, x, targets, fold = batch
        batch_size = x.size(0)

        # ---- Forward Pass with Optional Timing ----
        start_time = None
        if self.log_test_metrics:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
        
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)

        if self.log_test_metrics:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            inference_time_per_sample = (end_time - start_time) / batch_size
            self.inference_times.append(inference_time_per_sample)

        preds = torch.argmax(probs, dim=1)


        # ---- LOSS COMPUTATION (primary CE loss + secondary calibration) ----
        if not self.log_test_metrics: # when dim mismatch dont calculate loss only for testing purpose
            loss = None
        else:
            ce = self.criterion(logits, targets)  # Cross-entropy classification loss
            cal_penalty = logits.new_tensor(0.0)
            cal_terms = {}

            # Apply calibration losses if enabled
            if ((self.training and self.cal_cfg) or
                    (self.compute_cal_on_val and (not self.training) and self.cal_cfg)):
                cal_penalty, cal_terms = calibration_losses(logits, targets, self.cal_cfg)

            # Total combined loss
            loss = ce

            # ---- LOGGING ----
            mode = "train" if self.training else "val"

            # 1. Always log CE (both train and val)
            self.log(f"{mode}/ce", ce, on_step=False, on_epoch=True, prog_bar=True)

            # 2. Log calibration loss and sub-terms (train and val and enabled)
            if self.cal_cfg and self.log_calibration_terms:
                self.log(f"{mode}/cal_total", cal_penalty, on_step=False, on_epoch=True, prog_bar=True)

                # no need to log all sub-terms unless debugging
                if os.getenv("DEBUG", "0") == "1":
                    for k, v in cal_terms.items():
                        self.log(f"{mode}/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

                # 3. Log CE-to-Calibration ratio (monitor dominance)
                ratio = ce / (cal_penalty + 1e-8)
                self.log(f"{mode}/ce_to_cal_ratio", ratio, on_step=False, on_epoch=True, prog_bar=False)

                # # Optional sanity check: calibration relative magnitude (%)
                # cal_rel = (cal_penalty / (ce + 1e-8)) * 100
                # self.log(f"{mode}/calibration_pct_of_ce", cal_rel, on_step=False, on_epoch=True, prog_bar=False)

        return img_ids, loss, logits, probs, preds, targets, fold