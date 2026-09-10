import torch

from src.models.lit_module_base import LitModuleBase
from src.models.outputs import ModelOutput


class SNGPLitModule(LitModuleBase):
    """SNGP training strategy (Liu et al. 2020).

    Two things differ from `LitModuleBase`:

    * `forward` only lets the net accumulate the GP precision matrix while training.
    * `on_train_epoch_start` resets that accumulator.

    Why a per-epoch reset is the paper's "final epoch" step
    -------------------------------------------------------
    The paper writes the precision as one sum over the whole training set at converged
    weights, which reads like a separate post-training pass. The reference
    implementation makes no such pass: it resets the accumulator every epoch and
    accumulates during ordinary training forwards, so at any epoch boundary the
    accumulator holds exactly one full pass over the training data, and the last
    epoch's is the one kept. The reset is what makes it final-epoch-only.

    That also happens to be what makes this project's checkpointing self-consistent.
    With `ModelCheckpoint(monitor="val/f1", save_top_k=1)` and early stopping, the best
    epoch is usually not the last one; a per-epoch reset means whichever epoch gets
    checkpointed carries its own complete precision matrix. Accumulating only during
    the final epoch would write a `best.ckpt` from, say, epoch 12 with an empty one.

    Ordering is safe: `val_check_interval` defaults to 1.0 and is not overridden
    anywhere in `configs/`, so validation runs after all training batches and the
    precision is fully accumulated whenever validation and `ModelCheckpoint` run.
    """

    def on_train_epoch_start(self) -> None:
        """Reset the GP precision accumulator so it holds exactly this epoch's pass."""
        self.net.reset_precision()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A `ModelOutput`. In train mode `.logits` is the raw logits and
            `.variance` is `None` -- the mean-field correction is applied at inference
            only. In eval mode `.logits` is `raw_logits / sqrt(1 + lambda * variance)`.
        """
        return self.net(x, update_precision=self.training)
