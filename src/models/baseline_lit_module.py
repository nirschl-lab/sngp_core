from typing import Optional

import torch
from loguru import logger

from src.models.lit_module_base import LitModuleBase
from src.models.outputs import ModelOutput


class BaselineLitModule(LitModuleBase):
    """Plain classifier training strategy. The only thing that differs from
    `LitModuleBase` is optional MC-Dropout at test/predict time -- training and
    validation are the shared CE loop."""

    def __init__(
        self,
        net: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        num_classes: int = 8,
        use_mc: bool = False,
        mc_passes: int = 25,
        class_freq: Optional[list] = None,
        class_weights: Optional[list] = None,
        cb_beta: float = 0.999,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            num_classes=num_classes,
            class_freq=class_freq,
            class_weights=class_weights,
            cb_beta=cb_beta,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            **kwargs,
        )
        self.use_mc = use_mc
        self.mc_passes = mc_passes

    def _predict_forward(self, x: torch.Tensor) -> ModelOutput:
        """Test/predict-only: use MC-Dropout averaging when enabled, otherwise a
        plain forward pass. Isolated here so training/validation never pay the cost
        of `mc_passes` extra forward passes."""
        if not self.use_mc:
            return self.forward(x)
        logger.info(f"Using Monte Carlo Dropout for inference for {self.mc_passes} passes")
        _, mean_probs = self.net.mc_predict(x, T=self.mc_passes, return_std=False, apply_softmax=True)
        # The true MC-Dropout predictive distribution is the mean of per-pass softmax
        # outputs, not softmax(mean logits) -- those differ. test_step/predict_step
        # always derive probs via softmax(output.logits), so we hand back log(mean_probs):
        # softmax(log(p)) == p exactly when p already sums to 1, which mean_probs does.
        return ModelOutput(logits=torch.log(mean_probs.clamp_min(1e-12)))
