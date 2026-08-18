from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from loguru import logger
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import \
    MulticlassCalibrationError, \
    MulticlassPrecision, \
    MulticlassRecall, \
    MulticlassF1Score
from torchmetrics.classification.accuracy import Accuracy

from src.checkpointing.spec import FORMAT_VERSION, build_meta
from src.models.outputs import ModelOutput
from src.models.registry import build_net


class LitModuleBase(LightningModule):
    def __init__(
        self,
        net: Optional[torch.nn.Module] = None,
        net_spec: Optional[dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
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
        super().__init__()

        # `net` is a live nn.Module and must never be pickled into hparams -- that is
        # exactly what made checkpoints depend on the code structure they were saved
        # with. Instead we persist `net_spec` (plain data: registry name + ctor kwargs)
        # and rebuild `net` from it when one isn't passed directly (e.g. when Lightning
        # reconstructs this class from a checkpoint's saved hyperparameters).
        if net is None:
            if net_spec is None:
                raise ValueError(f"{type(self).__name__} requires either `net` or `net_spec`.")
            load_spec = dict(net_spec)
            if "pretrained" in load_spec:
                load_spec["pretrained"] = False  # weights are about to be overwritten by state_dict
            net = build_net(load_spec)
        net_spec = net.spec

        self.save_hyperparameters(
            ignore=["net", "optimizer", "scheduler", "calibration_cfg"], logger=False
        )
        self._optimizer_partial = optimizer
        self._scheduler_partial = scheduler

        self.net = net
        self.num_classes = self.net.num_classes

        #loss criterion parameters
        self.class_freq = class_freq
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.criterion = self._init_criterion()

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

        # Defaults so on_test_epoch_end / TestArtifactsCallback can rely on these
        # always existing, even for a subclass that never sets them or a `setup()`
        # call that never reaches the stage=="test" branch below.
        self.inference_times: List[float] = []
        self.test_idx_to_classes: Dict[int, str] = {}

        #plotting
        self.hist_bins = hist_bins
        self.calibration_curve_bins = calibration_curve_bins

        # csv logging
        self.test_name = test_name
        self.log_csv = log_csv
        self.csv_save_path = csv_save_path
        self.log_test_metrics = log_test_metrics
        self.log_metrics_per_class = log_metrics_per_class

    def _init_criterion(self):
        '''Initialize the loss criterion with class weights and label smoothing if provided.'''

        # set class weights, if provided
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
    
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A `ModelOutput` (see `src/models/outputs.py`); `.logits` is used for loss/argmax.
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
        logits = self.forward(x).logits
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

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_ece(probs, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
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
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends.

        CSV/figure artifact writing (and any wandb-specific logging of them) lives in
        `src.callbacks.test_artifacts_callback.TestArtifactsCallback` instead of here,
        so this module has no hard dependency on a particular logger backend -- it
        used to call `self.logger.experiment.log(...)` unconditionally, which crashed
        whenever no W&B logger was attached.
        """
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


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Attach a plain-data, format-versioned metadata block so checkpoints are
        self-describing without anyone needing to unpickle `hyper_parameters["net"]`."""
        idx_to_class = getattr(self, "test_idx_to_classes", None)
        dataset_name = None
        datamodule = getattr(self._trainer, "datamodule", None) if self._trainer is not None else None
        if datamodule is not None:
            dataset_name = getattr(datamodule, "dataset_name", None)

        checkpoint["sngp_core"] = build_meta(
            self,
            net_spec=self.net.spec,
            num_classes=self.num_classes,
            idx_to_class=idx_to_class,
            dataset_name=dataset_name,
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        sngp_core = checkpoint.get("sngp_core")
        if sngp_core is not None and sngp_core.get("format_version", 0) < FORMAT_VERSION:
            raise ValueError(
                f"Checkpoint format_version {sngp_core.get('format_version')} is older than "
                f"{FORMAT_VERSION}. Run scripts/checkpoints/migrate_checkpoints.py to migrate it."
            )

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
        logger.info(f'Trainer initialized - {self._trainer is not None}')
        if self._trainer is not None and self._trainer.state.stage == "test":
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
        if self._optimizer_partial is None:
            raise RuntimeError(
                f"{type(self).__name__} was built without an optimizer (e.g. reconstructed from a "
                "checkpoint via load_lit_module/load_from_checkpoint for inference). To resume "
                "training, instantiate it via Hydra with `optimizer=...`/`scheduler=...` and use "
                "trainer.fit(model, ckpt_path=...) instead."
            )
        optimizer = self._optimizer_partial(params=self.trainer.model.parameters())
        if self._scheduler_partial is not None:
            scheduler = self._scheduler_partial(optimizer=optimizer)
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

    