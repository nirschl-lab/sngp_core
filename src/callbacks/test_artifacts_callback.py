"""All test-time analysis for a `trainer.test()` run: metrics, CSV, and figures.

`LitModuleBase.test_step` is deliberately minimal -- it returns raw batch outputs
(logits/probs/preds/targets/img_ids/fold, plus SNGP/ensemble uncertainty fields when
present) and nothing else. This callback owns all the state (accumulation buffers,
inference timing) and does all the analysis: per-class/macro metrics, calibration,
uncertainty, the prediction CSV, and diagnostic figures. Metric/state ownership lives
here, not on the LightningModule, so training stays a lean train/val loop and test-time
research analysis stays entirely opt-in and swappable.

Originally extracted out of `LitModuleBase.on_test_epoch_end`, which used to call
`self.logger.experiment.log(...)` unconditionally -- this crashed (`AttributeError:
'NoneType' object has no attribute 'experiment'`) whenever a test run had no W&B
logger attached (e.g. `logger: null` in a config, or a bare `trainer.test(...)` in a
test). This callback still generates and saves the CSV/figures either way, and only
skips the wandb-specific *logging* of them when no `WandbLogger` is active.
"""
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as log
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_average_precision,
    multiclass_calibration_error,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

from src.metrics.uncertainty import (
    confidence_margin,
    decompose_member_uncertainty,
    dempster_shafer,
    infer_uncertainty_kind,
    predictive_entropy,
)
from src.visualization.dempster_shafer_uncertainity_plot import DempsterShaferUncertaintyPlot
from src.visualization.multi_class_ROC import plot_roc_curve
from src.visualization.plot_ece import plot_calibration_curve
from src.visualization.plot_prob_histograms import single_model_probability_histogram


class TestArtifactsCallback(Callback):
    """Computes final test metrics and writes the per-run prediction CSV/diagnostic
    figures (calibration curve, ROC curve, confidence histogram, Dempster-Shafer
    uncertainty) at the end of a test epoch."""

    def __init__(
        self,
        log_csv: bool = False,
        csv_save_path: str = "csv/",
        log_metrics_per_class: bool = False,
        hist_bins: int = 10,
        calibration_curve_bins: int = 10,
    ) -> None:
        self.log_csv = log_csv
        self.csv_save_path = csv_save_path
        self.log_metrics_per_class = log_metrics_per_class
        self.hist_bins = hist_bins
        self.calibration_curve_bins = calibration_curve_bins

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._logits: List[torch.Tensor] = []
        self._probs: List[torch.Tensor] = []
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._img_ids: List[str] = []
        self._fold: List[str] = []
        self._variance: List[torch.Tensor] = []
        self._member_logits: List[torch.Tensor] = []
        self._raw_logits: List[torch.Tensor] = []
        self._inference_times: List[float] = []
        self._batch_start_time: Optional[float] = None

    def on_test_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start_time = time.time()

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _, x, _, _ = batch
        elapsed = time.time() - self._batch_start_time
        self._inference_times.append(elapsed / x.size(0))

        self._logits.append(outputs["logits"].cpu())
        self._probs.append(outputs["probs"].cpu())
        self._preds.append(outputs["preds"].cpu())
        self._targets.append(outputs["targets"].cpu())
        self._img_ids.extend(outputs["img_ids"])
        self._fold.extend(outputs["fold"])
        if outputs.get("variance") is not None:
            self._variance.append(outputs["variance"].cpu())
        if outputs.get("member_logits") is not None:
            self._member_logits.append(outputs["member_logits"].cpu())
        if outputs.get("raw_logits") is not None:
            self._raw_logits.append(outputs["raw_logits"].cpu())

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._logits:
            return  # nothing accumulated this epoch

        logits_all = torch.cat(self._logits)
        probs_all = torch.cat(self._probs)
        preds_all = torch.cat(self._preds)
        targets_all = torch.cat(self._targets)

        self._log_metrics(pl_module, logits_all, probs_all, preds_all, targets_all, trainer)

        probs_np = probs_all.numpy()
        targets_np = targets_all.numpy()
        prediction = np.argmax(probs_np, axis=-1)
        prediction_prob_score = np.max(probs_np, axis=1)
        true_bin_label = (prediction == targets_np) * 1

        if self.log_csv:
            self._write_csv(
                trainer, pl_module, logits_all, probs_all, targets_np, prediction, prediction_prob_score, true_bin_label
            )

        idx_to_class = self._padded_class_names(trainer, pl_module.num_classes)
        figures = {
            "test/ece_plot": plot_calibration_curve(
                preds=probs_np, targets=targets_np, num_classes=pl_module.num_classes,
                n_bins=self.calibration_curve_bins, image_classes=idx_to_class,
            ),
            "test/roc_curve": plot_roc_curve(
                probs_np, targets_np, num_classes=pl_module.num_classes, class_names=idx_to_class
            ),
            "test/logits_distribution": single_model_probability_histogram(
                prediction_prob_score, bins=self.hist_bins
            ),
            "test/dempster_shafer_uncertainty": DempsterShaferUncertaintyPlot(logits_all.numpy()),
        }
        self._log_figures(trainer, figures)

    def _log_metrics(
        self,
        pl_module: LightningModule,
        logits_all: torch.Tensor,
        probs_all: torch.Tensor,
        preds_all: torch.Tensor,
        targets_all: torch.Tensor,
        trainer: Trainer,
    ) -> None:
        nc = pl_module.num_classes
        acc = multiclass_accuracy(preds_all, targets_all, num_classes=nc, average="micro")
        auprc = multiclass_average_precision(probs_all, targets_all, num_classes=nc, average="macro")
        ece = multiclass_calibration_error(probs_all, targets_all, num_classes=nc, n_bins=10, norm="l1")
        precision = multiclass_precision(preds_all, targets_all, num_classes=nc, average="macro")
        recall = multiclass_recall(preds_all, targets_all, num_classes=nc, average="macro")
        f1 = multiclass_f1_score(preds_all, targets_all, num_classes=nc, average="macro")
        loss = torch.nn.functional.cross_entropy(logits_all, targets_all)
        nll = torch.nn.functional.nll_loss(torch.log(probs_all + 1e-8), targets_all)

        pl_module.log("test/acc_final", acc, prog_bar=True)
        pl_module.log("test/auprc_final", auprc, prog_bar=True)
        pl_module.log("test/precision_final", precision, prog_bar=True)
        pl_module.log("test/recall_final", recall, prog_bar=True)
        pl_module.log("test/f1_final", f1, prog_bar=True)
        pl_module.log("test/loss_final", loss, prog_bar=True)
        pl_module.log("test/nll_final", nll, prog_bar=True)
        pl_module.log("test/ece_final", ece, prog_bar=True)
        pl_module.log("test/inference_time_per_sample_avg", float(np.mean(self._inference_times)), prog_bar=True)

        if self._variance:
            pl_module.log("test/uncertainty_mean", torch.cat(self._variance).mean())

        if self.log_metrics_per_class:
            precision_pc = multiclass_precision(preds_all, targets_all, num_classes=nc, average=None)
            recall_pc = multiclass_recall(preds_all, targets_all, num_classes=nc, average=None)
            f1_pc = multiclass_f1_score(preds_all, targets_all, num_classes=nc, average=None)
            idx_to_class = self._padded_class_names(trainer, nc)
            for i in range(nc):
                name = idx_to_class.get(i, f"class_{i}")
                pl_module.log(f"test/precision_{name}", precision_pc[i])
                pl_module.log(f"test/recall_{name}", recall_pc[i])
                pl_module.log(f"test/f1_{name}", f1_pc[i])

    @staticmethod
    def _padded_class_names(trainer: Trainer, num_classes: int) -> Dict[int, str]:
        idx_to_class = dict(getattr(trainer, "test_idx_to_classes", None) or {})
        data_classes = len(idx_to_class)
        if data_classes < num_classes:
            for i in range(num_classes - data_classes):
                idx_to_class[data_classes + i] = f"No class {data_classes + i}"
            log.info("Class names not found, using numbers for plotting.")
        return idx_to_class

    def _uncertainty_columns(self, pl_module: LightningModule, n_rows: int) -> Dict[str, Any]:
        """The optional, family-dependent uncertainty columns for this run.

        Empty for a plain Baseline, which has no model-side variance at all -- the
        always-present `predictive_entropy`/`confidence_margin`/`dempster_shafer`
        columns are what make that run comparable with the others.

        Batches of `member_logits` concatenate along **dim 1**, not 0: the stack is
        `[M, B, C]`, so members are the leading axis and the batch axis is the middle
        one. Concatenating along 0 would silently produce `[M*n_batches, B, C]` and
        misalign every row.
        """
        columns: Dict[str, Any] = {}

        member_logits = torch.cat(self._member_logits, dim=1) if self._member_logits else None
        raw_logits = torch.cat(self._raw_logits) if self._raw_logits else None

        if member_logits is not None:
            decomposed = decompose_member_uncertainty(member_logits)
            columns["total_entropy"] = decomposed.total.tolist()
            columns["aleatoric_entropy"] = decomposed.aleatoric.tolist()
            columns["mutual_information"] = decomposed.epistemic.tolist()

        if raw_logits is not None:
            columns["raw_logits"] = raw_logits.tolist()

        if self._variance:
            variance = torch.cat(self._variance).reshape(-1)
            if variance.numel() == n_rows:
                columns["uncertainty"] = variance.tolist()
                kind = infer_uncertainty_kind(
                    variance=variance,
                    raw_logits=raw_logits,
                    member_logits=member_logits,
                    mc_dropout=bool(getattr(pl_module, "use_mc", False)),
                )
                if kind is not None:
                    columns["uncertainty_kind"] = [kind] * n_rows
            else:
                log.warning(
                    f"Skipping the uncertainty column: got {variance.numel()} variance values for "
                    f"{n_rows} rows. A per-class variance must be reduced to one scalar per sample."
                )

        return columns

    def _write_csv(
        self, trainer, pl_module, logits_all, probs_all, targets, prediction, prediction_prob_score, true_bin_label
    ) -> None:
        """Write the `trainer.test()` prediction CSV.

        Carries the same uncertainty columns as the inference schema
        (`src/inference/records.py`), computed through the same
        `src/metrics/uncertainty.py` helpers -- the two writers keep their own *column
        names* for the shared quantities (`prediction_prob_score` here vs `confidence`
        there) since consumers key on those to tell the schemas apart, but there is no
        reason for them to disagree on which uncertainty signals exist.
        """
        datamodule = getattr(trainer, "datamodule", None)
        dataset_name = getattr(datamodule, "dataset_name", None) if datamodule is not None else None
        dataset_name = dataset_name.split("/")[-1] if dataset_name else "test_predictions"

        df = pd.DataFrame({
            "image_id": self._img_ids,
            "target": targets,
            "prediction": prediction,
            "prediction_prob_score": prediction_prob_score,
            "true_bin_label": true_bin_label,
            "class_logits": logits_all.tolist(),
            "class_probs": probs_all.tolist(),
            "fold": self._fold,
            "predictive_entropy": predictive_entropy(probs_all).tolist(),
            "confidence_margin": confidence_margin(probs_all).tolist(),
            "dempster_shafer": dempster_shafer(logits_all).tolist(),
        })

        for name, values in self._uncertainty_columns(pl_module, len(df)).items():
            df[name] = values
        os.makedirs(self.csv_save_path, exist_ok=True)
        csv_path = os.path.join(self.csv_save_path, dataset_name + ".csv")
        df.to_csv(csv_path, index=False)

        wandb_logger = self._find_wandb_logger(trainer)
        if wandb_logger is not None:
            import wandb

            artifact = wandb.Artifact(
                name=dataset_name, type="predictions",
                description="Test set predictions with probabilities and metadata",
            )
            artifact.add_file(csv_path)
            wandb_logger.experiment.log_artifact(artifact)

    def _log_figures(self, trainer: Trainer, figures: Dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        wandb_logger = self._find_wandb_logger(trainer)
        if wandb_logger is None:
            log.debug("No W&B logger active; figures generated but not logged.")
        else:
            import wandb

            wandb_logger.experiment.log({name: wandb.Image(fig) for name, fig in figures.items()})
        for fig in figures.values():
            plt.close(fig)

    @staticmethod
    def _find_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
        loggers = trainer.loggers if getattr(trainer, "loggers", None) else ([trainer.logger] if trainer.logger else [])
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                return lg
        return None
