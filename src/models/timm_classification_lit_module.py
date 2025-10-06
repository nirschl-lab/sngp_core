import os
from typing import Any, Dict, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
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
from src.visualization.reliability import rel_diagram_smoothed, rel_diagram_binned
import matplotlib.pyplot as plt
import wandb
import pdb
import numpy as np
from src.utils import RankedLogger
import pandas as pd
from torch.nn.modules.dropout import _DropoutNd

class TimmClassificationLitModule(LightningModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int = 8,
        hist_bins = 10, #for histogram plotting
        calibration_curve_bins=10, #for ece plot
        reset_sngp_precision=False,
        test_name = "test_predictions",
        log_csv = False,
        log_metrics_per_class = False,
        use_mc = False,
        mc_passes = 25
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

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # for calculating ece
        self.test_ece = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=10, norm='l1')

        # Add precision, recall, and F1 metrics
        self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.test_recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        
        # Per-class metrics for detailed analysis
        self.test_precision_per_class = MulticlassPrecision(num_classes=self.num_classes, average=None)
        self.test_recall_per_class = MulticlassRecall(num_classes=self.num_classes, average=None)
        self.test_f1_per_class = MulticlassF1Score(num_classes=self.num_classes, average=None)

        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self._test_probs: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._test_image_ids: List[str] = []

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
        self.log_metrics_per_class = log_metrics_per_class

        #montae carlo
        self.use_mc = use_mc
        self.mc_passes = mc_passes

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

    @torch.no_grad()
    def _mc_forward(self, x: torch.Tensor, T: int):
        """
        Returns:
        mean_logits: [B, C]
        mean_probs:  [B, C]
        var_probs:   [B, C]
        pred_entropy: [B]
        mutual_info:  [B]
        """
        self._enable_mc_dropout()
        logits_list = []
        probs_list = []
        for _ in range(T):
            logits_t = self.net(x)
            # handle optional SNGP shape B, L, ...
            if logits_t.ndim > 2:
                logits_t = logits_t[0]
            logits_list.append(logits_t)
            probs_list.append(F.softmax(logits_t, dim=-1))

        logits = torch.stack(logits_list, dim=0)   # [T, B, C]
        probs  = torch.stack(probs_list,  dim=0)   # [T, B, C]

        mean_logits = logits.mean(dim=0)           # [B, C]
        mean_probs  = probs.mean(dim=0)            # [B, C]
        # var_probs   = probs.var(dim=0, unbiased=False)

        # Uncertainty (predictive entropy & BALD MI)
        # eps = 1e-12
        # pred_entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)
        # exp_entropy  = - (probs * (probs + eps).log()).sum(dim=-1).mean(dim=0)
        # mutual_info  = pred_entropy - exp_entropy
        # return mean_logits, mean_probs, var_probs, pred_entropy, mutual_info
        return mean_logits, mean_probs

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
        img_ids, x, y = batch

        if self.use_mc:
            self.log_.info('using monate carlo')
            mean_logits, mean_probs = self._mc_forward(x, T=self.mc_passes)
            logits = mean_logits
            probs = mean_probs
        else:
            logits = self.forward(x)
            if len(logits.shape) > 2: # for sngp output is B, L, Cov_matrix
                logits = logits[:1]
            probs = torch.softmax(logits, dim=1)
            
        preds = torch.argmax(probs, dim=1)
        loss = self.criterion(logits, y)

        return img_ids, loss, probs, preds, y

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
        img_ids, loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
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
        img_ids, loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        img_ids, loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self._test_probs.append(probs.detach().cpu())
        self._test_targets.append(targets.detach().cpu())
        self._test_image_ids.extend(img_ids)  # Assuming img_ids is a list of strings

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_ece(probs, targets)

        # Update precision, recall, and F1 metrics
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision_per_class(preds, targets)
        self.test_recall_per_class(preds, targets)
        self.test_f1_per_class(preds, targets)

        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/ece", self.test_ece, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        probs_all = torch.cat(self._test_probs).numpy() # n x C
        targets = torch.cat(self._test_targets).numpy() # N x 1 (0-C)
        # pdb.set_trace()
        prediction_prob_score = np.max(probs_all, axis=1)
        prediction = np.argmax(probs_all, axis=-1)
        true_bin_label = (np.argmax(probs_all, axis=-1) == targets)*1

        # Compute final metrics
        precision_macro = self.test_precision.compute()
        recall_macro = self.test_recall.compute()
        f1_macro = self.test_f1.compute()
        loss = self.test_loss.compute()
        acc = self.test_acc.compute()
        ece = self.test_ece.compute()

        # Log macro-averaged metrics
        self.log("test/precision_final", precision_macro, prog_bar=True)
        self.log("test/recall_final", recall_macro, prog_bar=True)
        self.log("test/f1_final", f1_macro, prog_bar=True)
        self.log("test/loss_final", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_final", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ece_final", ece, on_step=False, on_epoch=True, prog_bar=True)

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
                'class_probs': probs_all.tolist()
            }
            self._log_csv_artifact(data_dict)

        fig, ax = rel_diagram_smoothed(prediction_prob_score, true_bin_label, n_bootstrap=100, num_mesh=200)
        self.logger.experiment.log({"test/smooth_ece_plot": wandb.Image(fig)})

        fig, ax = rel_diagram_binned(prediction_prob_score, true_bin_label)
        self.logger.experiment.log({"test/binned_ece_plot": wandb.Image(fig)})


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

    def _log_csv_artifact(self, data_dict):

        df = pd.DataFrame(data_dict)
        csv_path = self.test_name + ".csv"
        df.to_csv(csv_path, index=False)
        # Create and log wandb artifact
        artifact = wandb.Artifact(
            name=self.test_name,
            type="predictions",
            description="Test set predictions with probabilities and metadata"
        )
        artifact.add_file(csv_path)
        self.logger.experiment.log_artifact(artifact)
    
        # Clean up local file if desired
        import os
        os.remove(csv_path)
    
        

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
            self.train_classes_to_idx = self._trainer.train_classes_to_idx
            self.train_idx_to_classes = self._trainer.train_idx_to_classes
            self.val_classes_to_idx = self._trainer.val_classes_to_idx
            self.val_idx_to_classes = self._trainer.val_idx_to_classes
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
        loss, probs, preds, targets = self.model_step(batch)

        
        return loss, probs, preds, targets

        # Optionally, return predictions for use in prediction dataloader
        # return {
        #     "probs": probs.detach().cpu(),
        #     "preds": preds.detach().cpu(),
        #     "targets": targets.detach().cpu(),
        #     "loss": loss.detach().cpu(),
        # }

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
