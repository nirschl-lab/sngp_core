from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from loguru import logger
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    Accuracy,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.checkpointing.spec import FORMAT_VERSION, build_meta
from src.models.components.losses import ClassBalancedFocalLoss
from src.models.outputs import ModelOutput
from src.models.registry import build_net


class LitModuleBase(LightningModule):
    """Shared train/val loop for classification model families.

    Deliberately lean: this class only trains on the train set and validates on the
    val set (loss/acc/precision/recall/F1, used for checkpointing and early stopping).
    Rich test-time analysis (per-class metrics, calibration, uncertainty, CSV/figure
    export) lives entirely in
    `src.callbacks.test_artifacts_callback.TestArtifactsCallback` -- `test_step` here
    only produces raw batch outputs for that callback to accumulate via
    `on_test_batch_end`.
    """

    def __init__(
        self,
        net: Optional[torch.nn.Module] = None,
        net_spec: Optional[dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        num_classes: int = 8,
        class_freq: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        cb_beta: float = 0.999,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        **kwargs,
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

        self.save_hyperparameters(ignore=["net", "optimizer", "scheduler"], logger=False)
        self._optimizer_partial = optimizer
        self._scheduler_partial = scheduler

        self.net = net
        self.num_classes = self.net.num_classes

        # loss criterion parameters
        self.class_freq = class_freq
        self.class_weights = class_weights
        self.cb_beta = cb_beta
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.criterion = self._init_criterion()

        # train/val metrics only -- test-time metrics live in TestArtifactsCallback
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=self.num_classes),
                "precision": MulticlassPrecision(num_classes=self.num_classes, average="macro"),
                "recall": MulticlassRecall(num_classes=self.num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
            },
            prefix="val/",
        )
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # Selection-metric family: probability-based, updated alongside val_metrics but
        # kept out of that MetricCollection deliberately -- val_metrics is updated with
        # hard `preds`, and mixing metric input types under one `.update()` call would
        # silently change what val/f1 measures. These never feed early stopping or
        # checkpointing during normal training; they exist so hparams_search sweeps can
        # select on macro-AUPRC (threshold-free, imbalance-robust) while NLL/ECE stay
        # visible as read-only calibration diagnostics -- never the selection axis.
        self.val_auprc = MulticlassAveragePrecision(num_classes=self.num_classes, average="macro")
        self.val_ece = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=10, norm="l1")
        self.val_nll = MeanMetric()
        self.val_auprc_best = MaxMetric()

    def _init_criterion(self):
        """Initialize the loss criterion.

        `class_freq` (per-class training-split sample counts) is the standard,
        data-driven path: builds a `ClassBalancedFocalLoss` so every model family
        gets identical imbalance handling for a given dataset. `class_weights` (an
        explicit weight vector, no `class_freq`) is a manual-override escape hatch
        that falls back to plain weighted `CrossEntropyLoss`. Neither given means
        unweighted `CrossEntropyLoss`.
        """
        if self.class_freq:
            assert len(self.class_freq) == self.num_classes, "Length of class_freq must match num_classes"
            logger.info(
                f"Using class-balanced focal loss: class_freq={self.class_freq}, "
                f"beta={self.cb_beta}, gamma={self.focal_gamma}, label_smoothing={self.label_smoothing}"
            )
            return ClassBalancedFocalLoss(
                class_freq=self.class_freq,
                beta=self.cb_beta,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
            )
        elif self.class_weights:
            assert len(self.class_weights) == self.num_classes, "Length of class_weights must match num_classes"
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

    def _predict_forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass used only by `test_step`/`predict_step`. Override this (not
        `forward`/`model_step`) for test-time-only behavior -- e.g. MC-Dropout
        averaging -- without affecting the train/val path."""
        return self.forward(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_auprc.reset()
        self.val_ece.reset()
        self.val_nll.reset()
        # val_auprc_best tracks a running max ACROSS epochs by design -- without this
        # reset, a sanity-check AUPRC computed on a barely-initialized model would
        # become a spurious early high-water mark that real training could never beat.
        self.val_auprc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a single train/val model step on a batch of data -- plain
        cross-entropy classification, shared by every family unless a subclass
        genuinely needs a different training-time loss.

        :param batch: A batch of data (img_ids, images, targets, fold).
        :return: (img_ids, loss, logits, probs, preds, targets, fold).
        """
        img_ids, x, targets, fold = batch
        logits = self.forward(x).logits
        probs = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(probs, dim=1)

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

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

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

        # update and log metrics. val_metrics (acc/precision/recall/f1) streams via
        # log_dict as before -- Lightning computes/resets it automatically at epoch
        # end. val_auprc/val_ece/val_nll are updated here too but computed explicitly
        # in on_validation_epoch_end instead (see there for why).
        self.val_loss(loss)
        self.val_metrics.update(preds, targets)
        self.val_auprc.update(probs, targets)
        self.val_ece.update(probs, targets)
        self.val_nll(F.cross_entropy(logits, targets))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Compute and log the probability-based validation metrics that back
        hparams-search selection (`val/auprc`, `val/auprc_best`) and read-only
        calibration diagnostics (`val/nll`, `val/ece`).

        Computed explicitly (rather than logged as streaming Metric objects the way
        `val_metrics` is) because `val_auprc_best` needs the already-*computed* epoch
        value to update against -- logging the Metric object directly and reading it
        back in the same hook would race with Lightning's own compute/reset cycle.

        `val/nll` is deliberately a plain (unweighted) cross-entropy, not `val/loss`
        (the class-balanced focal loss) -- NLL is only a proper scoring rule when it
        isn't reweighted. Neither `val/nll` nor `val/ece` ever feeds selection, early
        stopping, or checkpointing -- calibration/uncertainty are the evaluation axis
        for this project, never the training-selection axis.
        """
        if self.val_auprc.update_count == 0:
            # No validation batches were actually processed this epoch -- e.g.
            # `trainer.fast_dev_run=True` (exactly 1 val batch total, which is also
            # epoch 0's batch_idx 0, always skipped by validation_step above) or any
            # run whose only available val batch is that same skipped one.
            # MulticlassAveragePrecision/MulticlassCalibrationError accumulate raw
            # preds/targets and raise ValueError on an empty state; there is nothing
            # meaningful to log here, so skip rather than crash or log a misleading NaN.
            return

        auprc = self.val_auprc.compute()
        ece = self.val_ece.compute()
        nll = self.val_nll.compute()
        self.val_auprc_best.update(auprc)

        self.log("val/auprc", auprc, prog_bar=True)
        self.log("val/auprc_best", self.val_auprc_best.compute(), prog_bar=True)
        self.log("val/nll", nll, prog_bar=False)
        self.log("val/ece", ece, prog_bar=False)

        self.val_auprc.reset()
        self.val_ece.reset()
        self.val_nll.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Minimal test step: forward pass plus raw outputs for
        `TestArtifactsCallback` to accumulate and analyze via `on_test_batch_end`. No
        loss computation, no metric `.update()`, no CSV/figure logic here -- see
        `src.callbacks.test_artifacts_callback.TestArtifactsCallback`.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        img_ids, x, targets, fold = batch
        output = self._predict_forward(x)
        probs = torch.softmax(output.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return {
            "img_ids": list(img_ids),
            "fold": list(fold),
            "logits": output.logits.detach(),
            "probs": probs.detach(),
            "preds": preds.detach(),
            "targets": targets.detach(),
            "variance": output.variance.detach() if output.variance is not None else None,
            "member_logits": output.member_logits.detach() if output.member_logits is not None else None,
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Attach a plain-data, format-versioned metadata block so checkpoints are
        self-describing without anyone needing to unpickle `hyper_parameters["net"]`."""
        datamodule = getattr(self._trainer, "datamodule", None) if self._trainer is not None else None
        dataset_name = getattr(datamodule, "dataset_name", None) if datamodule is not None else None
        class_to_idx = getattr(datamodule, "class_to_idx", None) if datamodule is not None else None
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()} if class_to_idx else None

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
        img_ids, x, targets, fold = batch
        output = self._predict_forward(x)
        probs = torch.softmax(output.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return {
            "img_ids": img_ids,
            "probs": probs,
            "preds": preds,
            "targets": targets,
            "fold": fold,
            # Carried for the same reason `test_step` carries it: without this,
            # `trainer.predict()` silently drops all SNGP/ensemble uncertainty.
            "variance": output.variance.detach() if output.variance is not None else None,
        }

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
