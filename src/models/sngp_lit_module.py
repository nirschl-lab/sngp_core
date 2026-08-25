import torch

from src.models.lit_module_base import LitModuleBase
from src.models.outputs import ModelOutput


class SNGPLitModule(LitModuleBase):
    """SNGP training strategy. The only thing that differs from `LitModuleBase` is
    `forward`: the net's spectral-normed covariance update only runs while
    training."""

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A `ModelOutput` with mean-field-corrected `.logits`, raw `.raw_logits`,
            and predictive `.variance`.
        """
        return self.net(x, update_cov=self.training)
