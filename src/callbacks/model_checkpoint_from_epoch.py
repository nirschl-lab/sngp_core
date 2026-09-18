"""`ModelCheckpoint` that only *ranks* epochs from `start_epoch` on.

Motivation: a training schedule with an unregularized burn-in (see
`SNGPSpectralRegLitModule`) trains two different models in one run, and nothing
guarantees the monitored metric's minimum falls in the phase we actually want to keep.
This callback keeps `last.ckpt` rolling from epoch 0 (so the run stays resumable) but
does not consider any epoch before `start_epoch` for `best.ckpt`.

Implementation: while `trainer.current_epoch < start_epoch`, the public `save_top_k`
attribute is set to 0 around the parent hook -- Lightning's `_save_topk_checkpoint`
returns immediately at 0 while `_save_last_checkpoint` still runs -- and restored after.
Only public hooks and a public attribute are touched.
"""
from contextlib import contextmanager
from typing import Iterator

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class ModelCheckpointFromEpoch(ModelCheckpoint):
    """`ModelCheckpoint` whose top-k ranking ignores epochs before `start_epoch`.

    `start_epoch` is a 0-based epoch index: with `start_epoch=50` the validation run at
    the end of epoch index 50 is the first one that can produce `best.ckpt`. `save_last`
    behaves exactly as in the parent throughout.
    """

    def __init__(self, *args, start_epoch: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {start_epoch}")
        self.start_epoch = int(start_epoch)

    def ranking_enabled(self, trainer: "pl.Trainer") -> bool:
        return trainer.current_epoch >= self.start_epoch

    @contextmanager
    def _top_k_suspended_before_start(self, trainer: "pl.Trainer") -> Iterator[None]:
        if self.ranking_enabled(trainer):
            yield
            return
        save_top_k = self.save_top_k
        self.save_top_k = 0
        try:
            yield
        finally:
            self.save_top_k = save_top_k

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with self._top_k_suspended_before_start(trainer):
            super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with self._top_k_suspended_before_start(trainer):
            super().on_validation_end(trainer, pl_module)
