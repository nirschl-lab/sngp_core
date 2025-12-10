import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from lightning import LightningModule
from loguru import logger
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.classification.accuracy import Accuracy

class LitModuleBase(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        class_indices: Dict[str, int],
        num_classes: int = 8,
        test_name: str = "test_predictions",
        log_csv: bool = False,
        csv_save_path: str = "csv/",
        log_metrics_per_class: bool = False,
        log_test_metrics: bool = True,
        class_freq: Optional[dict] = None,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,  # recommend avoiding with SNGP and calibration losses
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.num_classes = self.net.num_classes

        # Loss criterion parameters
        self.class_freq = class_freq
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.criterion = self._init_criterion()

        # Training metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        
        # Validation metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        
        # Per-class metrics for detailed analysis
        self.val_precision_per_class = MulticlassPrecision(num_classes=self.num_classes, average=None)
        self.val_recall_per_class = MulticlassRecall(num_classes=self.num_classes, average=None)
        self.val_f1_per_class = MulticlassF1Score(num_classes=self.num_classes, average=None)

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # For tracking best validation metrics
        self.val_acc_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        self._test_logits: List[torch.Tensor] = []
        self._test_probs: List[torch.Tensor] = []
        self._test_image_ids: List[str] = []
        self._test_fold: List[str] = []
        self._test_preds: List[torch.Tensor] = []

        # CSV logging configuration
        self.test_name = test_name
        self.log_csv = log_csv
        self.csv_save_path = csv_save_path
        self.log_test_metrics = log_test_metrics
        self.log_metrics_per_class = log_metrics_per_class

        # Prediction storage
        self._predict_logits: List[torch.Tensor] = []
        self._predict_probs: List[torch.Tensor] = []
        self._predict_image_ids: List[str] = []
        self._predict_fold: List[str] = []

        # Store class mappings
        self.classes_to_idx = class_indices
        self.idx_to_classes = {v: k for k, v in class_indices.items()} if class_indices else None


    def _init_criterion(self):
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
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

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
        img_ids, loss, logits, probs, preds, targets, _ = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # Skip if batch and epoch are both 0
        if batch_idx == 0 and self.current_epoch == 0:
            logger.warning("Skipping validation step for batch 0 in epoch 0")
            return

        img_ids, loss, logits, probs, preds, targets, fold = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""

        acc = self.val_acc.compute()  # Get current validation accuracy
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        loss = self.val_loss.compute()

        self.log("val/acc", acc, sync_dist=True, prog_bar=True)
        self.log("val/precision", precision, sync_dist=True, prog_bar=True)
        self.log("val/recall", recall, sync_dist=True, prog_bar=True)
        self.log("val/f1", f1, sync_dist=True, prog_bar=True)
        self.log("val/loss", loss, sync_dist=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        img_ids, x, targets, fold = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        self._test_probs.append(probs.detach().cpu())
        self._test_image_ids.extend(img_ids)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        probs_all = torch.cat(self._test_probs).numpy()
        prediction = np.argmax(probs_all, axis=-1)
        prediction = [self.idx_to_classes[idx] for idx in prediction]


        data_dict = {
            'ID': self._test_image_ids,
            'labels': prediction,
        }
        self._log_csv_artifact(data_dict)


    def _log_csv_artifact(self, data_dict):
        """Log CSV predictions as a wandb artifact."""
        dataset_name = self._trainer.datamodule.dataset_name if hasattr(self._trainer.datamodule, 'dataset_name') else None
        if dataset_name:
            dataset_name = dataset_name.split('/')[-1]
        else:
            dataset_name = 'submission'

        df = pd.DataFrame(data_dict)
        os.makedirs(self.csv_save_path, exist_ok=True)
        csv_path = os.path.join(self.csv_save_path, f"{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        
        # Create and log wandb artifact
        artifact = wandb.Artifact(
            name=dataset_name,
            type="predictions",
            description="Test set predictions with probabilities and metadata"
        )
        artifact.add_file(csv_path)
        self.logger.experiment.log_artifact(artifact)
    

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        
        # Set up class indexing for test stage
        # logger.info(f'Trainer initialized - {self._trainer is not None}')
        # # if self._trainer is not None and self._trainer.state.stage == "test":
        # self.test_classes_to_idx = self._trainer.test_classes_to_idx
        # self.test_idx_to_classes = self._trainer.test_idx_to_classes

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
        img_ids, x, targets, fold = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)

        # Store predictions and metadata
        self._predict_probs.append(probs.detach().cpu())
        self._predict_image_ids.extend(img_ids)
        
    
    def on_predict_end(self):
        """Lightning hook called at the end of prediction."""
        probs_all = torch.cat(self._predict_probs).numpy()  # n x C
        prediction = np.argmax(probs_all, axis=-1)
        # Map predicted indices to class names if available
        prediction = [self.idx_to_classes[idx] for idx in prediction]

        if self.log_csv:
            # Create DataFrame with all the data
            data_dict = {
                'ID': self._predict_image_ids,
                'labels': prediction,
            }
            self._log_csv_artifact(data_dict)

    
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