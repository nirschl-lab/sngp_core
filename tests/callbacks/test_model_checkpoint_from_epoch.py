"""`ModelCheckpointFromEpoch`: `best.ckpt` may only come from epochs >= `start_epoch`,
while `last.ckpt` rolls from epoch 0. A toy module logs a scripted `val/nll` whose global
minimum sits *before* `start_epoch`, so a stock `ModelCheckpoint` would pick the wrong
epoch. CPU, a few tiny epochs."""
import lightning as L
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.model_checkpoint_from_epoch import ModelCheckpointFromEpoch


class _ScriptedValNLL(L.LightningModule):
    def __init__(self, schedule):
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.schedule = list(schedule)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        self.log("val/nll", torch.tensor(self.schedule[self.current_epoch]))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loader():
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(8, 2), torch.randn(8, 1)), batch_size=4)


def _fit(tmp_path, schedule, start_epoch, **ckpt_kwargs):
    callback = ModelCheckpointFromEpoch(
        dirpath=str(tmp_path),
        filename="best",
        monitor="val/nll",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        start_epoch=start_epoch,
        **ckpt_kwargs,
    )
    trainer = L.Trainer(
        max_epochs=len(schedule),
        callbacks=[callback],
        logger=False,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(_ScriptedValNLL(schedule), _loader(), _loader())
    return callback


def _saved_epoch(path) -> int:
    return torch.load(str(path), map_location="cpu", weights_only=False)["epoch"]


def test_best_ignores_epochs_before_start_epoch(tmp_path):
    schedule = [0.1, 0.2, 0.9, 0.5, 0.6]  # global min at epoch 0; min from epoch 2 on is epoch 3
    callback = _fit(tmp_path, schedule, start_epoch=2)

    assert callback.best_model_score.item() == pytest.approx(0.5)
    assert _saved_epoch(tmp_path / "best.ckpt") == 3
    assert (tmp_path / "last.ckpt").exists()
    assert _saved_epoch(tmp_path / "last.ckpt") == 4


def test_last_ckpt_rolls_during_the_unranked_phase(tmp_path):
    """Nothing is ranked yet, but the run must stay resumable."""
    callback = _fit(tmp_path, schedule=[0.3, 0.2], start_epoch=10)

    assert not (tmp_path / "best.ckpt").exists()
    assert callback.best_model_path == ""
    assert _saved_epoch(tmp_path / "last.ckpt") == 1


def test_start_epoch_zero_is_the_stock_behaviour(tmp_path):
    callback = _fit(tmp_path, schedule=[0.1, 0.2, 0.9], start_epoch=0)
    assert callback.best_model_score.item() == pytest.approx(0.1)
    assert _saved_epoch(tmp_path / "best.ckpt") == 0


def test_save_top_k_is_restored_after_every_hook(tmp_path):
    callback = _fit(tmp_path, schedule=[0.3, 0.2, 0.1], start_epoch=1)
    assert callback.save_top_k == 1


def test_negative_start_epoch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="start_epoch"):
        ModelCheckpointFromEpoch(dirpath=str(tmp_path), monitor="val/nll", start_epoch=-1)
