import os
from typing import Any, Dict, Tuple, List, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from src.metrics.calibration_losses import CalibrationLossConfig, calibration_losses
from src.models.sngp.sngp_diagnostic_mixin import SNGPDiagnosticsMixin

import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy 
from torchmetrics.classification import \
                    MulticlassCalibrationError, \
                    MulticlassPrecision, \
                    MulticlassRecall, \
                    MulticlassF1Score
from src.visualization.multi_class_ROC import plot_roc_curve
from src.visualization.plot_prob_histograms import single_model_probability_histogram
from src.visualization.plot_ece import plot_calibration_curve
from src.visualization.dempster_shafer_uncertainity_plot import DempsterShaferUncertaintyPlot
from src.visualization.reliability import rel_diagram_smoothed, rel_diagram_binned
import matplotlib.pyplot as plt
import wandb
import pdb
from loguru import logger
import numpy as np
from src.utils import RankedLogger
import pandas as pd
from torch.nn.modules.dropout import _DropoutNd
from src.models.sngp.gaussian_process import mean_field_logits
import time

class BaselineClassificationLitModule(SNGPDiagnosticsMixin, LightningModule):
    
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
        reset_sngp_precision: bool =False,
        test_name: str = "test_predictions",
        log_csv: bool = False,
        csv_save_path: str = "csv/",
        log_metrics_per_class: bool = False,
        use_mc: bool = False,
        mc_passes: int = 25,
        use_mean_field_logits: bool = False,
        log_test_metrics: bool = True,
        log_calibration_terms: bool = True,
        compute_calibration_on_val: bool = False,
        class_freq: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0, # recommend avoiding with SNGP and calibration losses, if needed set alpha low [0.01, 0.05].
        **kwargs
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.num_classes = self.net.num_classes

        # set class weights, if provided
        if class_weights and len(class_weights) == num_classes:
            self.class_weights = class_weights
        elif class_freq  and len(class_freq) == num_classes:
            weights = torch.tensor([1.0 / class_freq[k] for k in class_freq], dtype=torch.float32)
            self.class_weights = weights / weights.sum()
        else:
            self.class_weights = None

        # set criterion with class weights if provided and optional label smoothing
        if self.class_weights is not None:
            # Ensure weights are a PyTorch tensor and on the correct device if necessary
            logger.info(f"Using class weights for CrossEntropyLoss: {self.class_weights}")
            class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32)
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
        else:
            logger.info("No class weights provided, using unweighted CrossEntropyLoss")
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.nll_loss = torch.nn.NLLLoss()

        # secondary losses
        if calibration_cfg is None:
            calibration_cfg = CalibrationLossConfig(
                sb_ece_label_weight=0.0,  # start disabled unless you enable
                soft_avuc_weight=0.0,
            )

        self.cal_cfg = calibration_cfg
        self.log_calibration_terms = log_calibration_terms
        self.compute_cal_on_val = compute_calibration_on_val

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # for calculating ece
        self.test_ece = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=10, norm='l1')
        self.val_ece = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=10, norm='l1')

        # Add precision, recall, and F1 metrics
        self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.test_recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        
        # Per-class metrics for detailed analysis
        self.test_precision_per_class = MulticlassPrecision(num_classes=self.num_classes, average=None)
        self.test_recall_per_class = MulticlassRecall(num_classes=self.num_classes, average=None)
        self.test_f1_per_class = MulticlassF1Score(num_classes=self.num_classes, average=None)

        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Add NLL loss metrics
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        self._test_logits: List[torch.Tensor] = []
        self._test_probs: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._test_image_ids: List[str] = []
        self._test_fold: List[str] = []

        # self._predict_probs: List[torch.Tensor] = []
        # self._predict_targets: List[torch.Tensor] = []

        #plotting
        self.hist_bins = hist_bins
        self.calibration_curve_bins = calibration_curve_bins
        self.log_ = RankedLogger(__name__, rank_zero_only=True)

        #sngp specifics
        self.reset_sngp_precision = reset_sngp_precision

        # csv name
        self.test_name = test_name
        self.log_csv = log_csv
        self.csv_save_path = csv_save_path
        self.log_test_metrics = log_test_metrics
        self.log_metrics_per_class = log_metrics_per_class

        #montae carlo
        self.use_mc = use_mc
        self.mc_passes = mc_passes
        self.use_mean_field_logits = use_mean_field_logits

        self.inference_times = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
    
    def _enable_mc_dropout(self):
        """Freeze everything (eval) but keep dropout stochastic."""
        self.net.eval()
        for m in self.net.modules():
            if isinstance(m, _DropoutNd):
                m.train()


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
        
        if self.use_mc:
            if self.log_test_metrics:
                logger.info("Using Monte Carlo Dropout for inference for {} passes".format(self.mc_passes))
            logits, probs = self.net.mc_predict(x, T=self.mc_passes, return_std=False, apply_softmax=True)
        else:
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
            nll_loss = self.nll_loss(F.log_softmax(logits, dim=-1), targets)
            cal_penalty = logits.new_tensor(0.0)
            cal_terms = {}

            # Apply calibration losses if enabled
            if ((self.training and self.cal_cfg) or
                    (self.compute_cal_on_val and (not self.training) and self.cal_cfg)):
                cal_penalty, cal_terms = calibration_losses(logits, targets, self.cal_cfg)

            # Total combined loss
            loss = ce #+ nll_loss 

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

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # self.log_.info('------------------->< * * ><-------------')
        img_ids, loss, logits, probs, preds, targets, _ = self.model_step(batch)

        # pdb.set_trace()

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        if self.reset_sngp_precision:
            self.net.sngp_classifier.gp_classifier.reset_precision()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # skip if batch and epoch are both 0
        if batch_idx == 0 and self.current_epoch == 0:
            logger.warning("Skipping validation step for batch 0 in epoch 0")
            return

        img_ids, loss, logits, probs, preds, targets, fold = self.model_step(batch)

        # Calculate NLL loss
        log_probs = torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
        nll_loss = F.nll_loss(log_probs, targets)

        # update and log metrics
        self.val_loss(loss)
        self.val_nll(nll_loss)
        self.val_acc(preds, targets)
        self.val_ece(probs, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/nll", self.val_nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ece", self.val_ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        acc = self.val_acc.compute()  # get current val acc
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        loss = self.val_loss.compute()
        nll = self.val_nll.compute()
        

        # log additional sngp diagnostics
        if not hasattr(self.net, "sngp_classifier"):
            self.log_sngp_diagnostics()

        # self.val_acc_best(acc)  # update best so far val acc
        # self.val_precision_best(precision)  # update best so far val precision
        # self.val_recall_best(recall)  # update best so far val recall
        # self.val_f1_best(f1)  # update best so far val f1

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        # self.log("val/precision_best", self.val_precision_best.compute(), sync_dist=True, prog_bar=True)
        # self.log("val/recall_best", self.val_recall_best.compute(), sync_dist=True, prog_bar=True)
        # self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

        self.log("val/acc", acc, sync_dist=True, prog_bar=True)
        self.log("val/precision", precision, sync_dist=True, prog_bar=True)
        self.log("val/recall", recall, sync_dist=True, prog_bar=True)
        self.log("val/f1", f1, sync_dist=True, prog_bar=True)
        self.log("val/nll", nll, sync_dist=True, prog_bar=True)
        self.log("val/loss", loss, sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        # pdb.set_trace()
        img_ids, loss, logits, probs, preds, targets, fold = self.model_step(batch)

        
        # update and log metrics
        self._test_logits.append(logits.detach().cpu())
        self._test_probs.append(probs.detach().cpu())
        self._test_targets.append(targets.detach().cpu())
        self._test_image_ids.extend(img_ids)  # Assuming img_ids is a list of strings
        self._test_fold.extend(fold)  # Assuming fold is a list of strings

        if self.log_test_metrics:
            # Calculate NLL loss
            log_probs = torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
            nll_loss = F.nll_loss(log_probs, targets)
            self.test_loss(loss)
            self.test_nll(nll_loss)
            self.test_acc(preds, targets)
            self.test_ece(probs, targets)

            # Update precision, recall, and F1 metrics
            self.test_precision(preds, targets)
            self.test_recall(preds, targets)
            self.test_f1(preds, targets)

        # self.test_precision_per_class(preds, targets)
        # self.test_recall_per_class(preds, targets)
        # self.test_f1_per_class(preds, targets)

        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/ece", self.test_ece, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        logits_all = torch.cat(self._test_logits).numpy() # n x C
        probs_all = torch.cat(self._test_probs).numpy() # n x C
        targets = torch.cat(self._test_targets).numpy() # N x 1 (0-C)
        prediction_prob_score = np.max(probs_all, axis=1)
        prediction = np.argmax(probs_all, axis=-1)
        true_bin_label = (np.argmax(probs_all, axis=-1) == targets)*1
        if self.log_test_metrics:

            # Compute final metrics
            precision_macro = self.test_precision.compute()
            recall_macro = self.test_recall.compute()
            f1_macro = self.test_f1.compute()
            loss = self.test_loss.compute()
            nll = self.test_nll.compute()
            acc = self.test_acc.compute()
            ece = self.test_ece.compute()

            # Log macro-averaged metrics
            self.log("test/precision_final", precision_macro, prog_bar=True)
            self.log("test/recall_final", recall_macro, prog_bar=True)
            self.log("test/f1_final", f1_macro, prog_bar=True)
            self.log("test/loss_final", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/nll_final", nll, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc_final", acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/ece_final", ece, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/inference_time_per_sample_avg", np.mean(self.inference_times), prog_bar=True)

            # Log per-class metrics
            if self.log_metrics_per_class:
                precision_per_class = self.test_precision_per_class.compute()
                recall_per_class = self.test_recall_per_class.compute()
                f1_per_class = self.test_f1_per_class.compute()
                if hasattr(self, 'test_idx_to_classes') and self.test_idx_to_classes:
                    for i, class_name in self.test_idx_to_classes.items():
                        self.log(f"test/precision_{class_name}", precision_per_class[i])
                        self.log(f"test/recall_{class_name}", recall_per_class[i])
                        self.log(f"test/f1_{class_name}", f1_per_class[i])
                else:
                    for i in range(self.num_classes):
                        self.log(f"test/precision_class_{i}", precision_per_class[i])
                        self.log(f"test/recall_class_{i}", recall_per_class[i])
                        self.log(f"test/f1_class_{i}", f1_per_class[i])


        if self.log_csv:
            # Create DataFrame with all the data
            data_dict = {
                'image_id': self._test_image_ids,
                'target': targets,
                'prediction': prediction,
                'prediction_prob_score': prediction_prob_score,
                'true_bin_label': true_bin_label,
                'class_logits': logits_all.tolist(),
                'class_probs': probs_all.tolist(),
                'fold': self._test_fold
            }
            self._log_csv_artifact(data_dict)

        # fig, ax = rel_diagram_smoothed(prediction_prob_score, true_bin_label, n_bootstrap=100, num_mesh=200)
        # self.logger.experiment.log({"test/smooth_ece_plot": wandb.Image(fig)})

        # fig, ax = rel_diagram_binned(prediction_prob_score, true_bin_label)
        # self.logger.experiment.log({"test/binned_ece_plot": wandb.Image(fig)})


        data_classes = len(self.test_idx_to_classes)
        if data_classes < self.num_classes:
            for i in range(self.num_classes - data_classes):
                self.test_idx_to_classes[data_classes + i] = 'No class ' + str(data_classes + i)
                self.log_.info("Class names not found, using numbers for plotting.")
        
        fig = plot_calibration_curve(preds=probs_all, \
                                    targets=targets, \
                                    num_classes=self.num_classes, \
                                    n_bins=self.calibration_curve_bins, \
                                    image_classes=self.test_idx_to_classes)
        
        self.logger.experiment.log({"test/ece_plot": wandb.Image(fig)})
        plt.close(fig)

        fig = plot_roc_curve(probs_all, targets, num_classes=self.num_classes, class_names=self.test_idx_to_classes)
        self.logger.experiment.log({"test/roc_curve": wandb.Image(fig)})
        plt.close(fig)
        
        fig = single_model_probability_histogram(prediction_prob_score, bins=self.hist_bins)
        self.logger.experiment.log({"test/logits_distribution": wandb.Image(fig)})
        plt.close(fig)

        fig = DempsterShaferUncertaintyPlot(logits_all)
        self.logger.experiment.log({"test/dempster_shafer_uncertainty": wandb.Image(fig)})
        plt.close(fig)

    def _log_csv_artifact(self, data_dict):

        # pdb.set_trace()
        dataset_name = self._trainer.datamodule.dataset_name if hasattr(self._trainer.datamodule, 'dataset_name') else None
        if dataset_name:
            dataset_name = dataset_name.split('/')[-1]
        else:
            dataset_name = 'test_predictions'

        df = pd.DataFrame(data_dict)
        os.makedirs(self.csv_save_path, exist_ok=True) 
        csv_path = os.path.join(self.csv_save_path, dataset_name + ".csv")
        df.to_csv(csv_path, index=False)
        # Create and log wandb artifact
        artifact = wandb.Artifact(
            name=dataset_name,
            type="predictions",
            description="Test set predictions with probabilities and metadata"
        )
        artifact.add_file(csv_path)
        self.logger.experiment.log_artifact(artifact)
    
        # Clean up local file if desired
        # import os
        # os.remove(csv_path)
    
        

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        
        #indexing classes
        self.log_.info(f'Trainer initialized - {self._trainer is not None}')
        if self._trainer is not None:
            # self.log_.info('------------------********-------------------')
            # self.train_classes_to_idx = self._trainer.train_classes_to_idx
            # self.train_idx_to_classes = self._trainer.train_idx_to_classes
            # self.val_classes_to_idx = self._trainer.val_classes_to_idx
            # self.val_idx_to_classes = self._trainer.val_idx_to_classes
            self.test_classes_to_idx = self._trainer.test_classes_to_idx
            self.test_idx_to_classes = self._trainer.test_idx_to_classes

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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

    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor and target labels.
        :param batch_idx: The index of the current batch.
        """
        img_ids, loss, logits, probs, preds, targets, fold = self.model_step(batch)

        
        return loss, probs, preds, targets

        # Optionally, return predictions for use in prediction dataloader
        # return {
        #     "probs": probs.detach().cpu(),
        #     "preds": preds.detach().cpu(),
        #     "targets": targets.detach().cpu(),
        #     "loss": loss.detach().cpu(),
        # }
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom state dict loading to handle mismatched criterion.weight"""
        # Create a copy to avoid modifying the original
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            # Skip criterion.weight if we don't have class weights
            if key == "criterion.weight" and self.class_weights is None:
                print(f"Skipping {key} from checkpoint as model has no class weights")
                continue
            filtered_state_dict[key] = value
        
        return super().load_state_dict(filtered_state_dict, strict=strict)

# cache_dir = 'timm_cache_dir'
# model = TimmBackboneWithProbe(
#     "resnet50",
#     proj_dim=512,
#     num_classes=10,
#     pretrained=True,
#     cache_dir=cache_dir,   # NEW
#     freeze_backbone=True            # NEW
# )
# # Test the model with random input
# batch_size = 2
# in_chans = 3
# height = width = 224
# x = torch.randn(batch_size, in_chans, height, width)
# out = model(x)
# print(f"Input shape: {x.shape}")
# print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    img_channels = 3
    img_len = 512
    img_width = 512
    input_dim = 384
    num_classes = 8
    batch_size = 4
    # net = BaselineModel(input_dim, num_classes)
    cache_dir = 'timm_cache_dir'
    net = TimmBackboneWithProbe(
        "resnet50",
        proj_dim=512,
        num_classes=num_classes,
        pretrained=True,
        cache_dir=cache_dir,   # NEW
        freeze_backbone=True            # NEW
    )
    module = TimmClassificationLitModule(net, None, None, False, num_classes=num_classes)

    # Create a synthetic batch
    x = torch.randn(batch_size, img_channels, img_len, img_width)
    y = torch.randint(0, num_classes, (batch_size,))
    batch = (x, y)

    # Forward pass
    logits = module.forward(x)
    if len(logits.shape)>2:
        logits = logits[0]
    print(f"Logits shape: {logits.shape}")

    # Training step
    loss = module.training_step(batch, 0)
    print(f"Training loss: {loss.item()}")

    # Validation step
    module.validation_step(batch, 0)
    print("Validation step completed.")
