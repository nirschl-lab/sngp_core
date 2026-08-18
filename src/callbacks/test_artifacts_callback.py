"""Test-time CSV/figure artifact logging.

Extracted out of `LitModuleBase.on_test_epoch_end`, which used to call
`self.logger.experiment.log(...)` unconditionally -- this crashed (`AttributeError:
'NoneType' object has no attribute 'experiment'`) whenever a test run had no W&B
logger attached (e.g. `logger: null` in a config, or a bare `trainer.test(...)` in a
test). This callback still generates and saves the CSV/figures either way, and only
skips the wandb-specific *logging* of them when no `WandbLogger` is active.
"""
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as log

from src.visualization.dempster_shafer_uncertainity_plot import DempsterShaferUncertaintyPlot
from src.visualization.multi_class_ROC import plot_roc_curve
from src.visualization.plot_ece import plot_calibration_curve
from src.visualization.plot_prob_histograms import single_model_probability_histogram


class TestArtifactsCallback(Callback):
    """Writes the per-run prediction CSV (`pl_module.log_csv`) and diagnostic figures
    (calibration curve, ROC curve, confidence histogram, Dempster-Shafer uncertainty)
    at the end of a test epoch, reading the predictions `LitModuleBase.test_step`
    already accumulates on the module."""

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not pl_module._test_logits:
            return  # nothing accumulated this epoch

        logits_all = torch.cat(pl_module._test_logits).numpy()
        probs_all = torch.cat(pl_module._test_probs).numpy()
        targets = torch.cat(pl_module._test_targets).numpy()
        prediction = np.argmax(probs_all, axis=-1)
        prediction_prob_score = np.max(probs_all, axis=1)
        true_bin_label = (prediction == targets) * 1

        if getattr(pl_module, "log_csv", False):
            self._write_csv(trainer, pl_module, logits_all, probs_all, targets, prediction, prediction_prob_score, true_bin_label)

        idx_to_class = self._padded_class_names(pl_module)
        figures = {
            "test/ece_plot": plot_calibration_curve(
                preds=probs_all, targets=targets, num_classes=pl_module.num_classes,
                n_bins=pl_module.calibration_curve_bins, image_classes=idx_to_class,
            ),
            "test/roc_curve": plot_roc_curve(
                probs_all, targets, num_classes=pl_module.num_classes, class_names=idx_to_class
            ),
            "test/logits_distribution": single_model_probability_histogram(
                prediction_prob_score, bins=pl_module.hist_bins
            ),
            "test/dempster_shafer_uncertainty": DempsterShaferUncertaintyPlot(logits_all),
        }
        self._log_figures(trainer, figures)

    @staticmethod
    def _padded_class_names(pl_module: LightningModule) -> Dict[int, str]:
        idx_to_class = dict(getattr(pl_module, "test_idx_to_classes", None) or {})
        data_classes = len(idx_to_class)
        if data_classes < pl_module.num_classes:
            for i in range(pl_module.num_classes - data_classes):
                idx_to_class[data_classes + i] = f"No class {data_classes + i}"
            log.info("Class names not found, using numbers for plotting.")
        return idx_to_class

    def _write_csv(
        self, trainer, pl_module, logits_all, probs_all, targets, prediction, prediction_prob_score, true_bin_label
    ) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        dataset_name = getattr(datamodule, "dataset_name", None) if datamodule is not None else None
        dataset_name = dataset_name.split("/")[-1] if dataset_name else "test_predictions"

        df = pd.DataFrame({
            "image_id": pl_module._test_image_ids,
            "target": targets,
            "prediction": prediction,
            "prediction_prob_score": prediction_prob_score,
            "true_bin_label": true_bin_label,
            "class_logits": logits_all.tolist(),
            "class_probs": probs_all.tolist(),
            "fold": pl_module._test_fold,
        })
        os.makedirs(pl_module.csv_save_path, exist_ok=True)
        csv_path = os.path.join(pl_module.csv_save_path, dataset_name + ".csv")
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
