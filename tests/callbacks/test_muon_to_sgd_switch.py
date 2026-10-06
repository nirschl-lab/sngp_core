"""`MuonToSGDSwitch`: the hidden convs train on Muon before `switch_epoch` and on SGD from it
on, the switch fires once, and it refuses an optimizer it cannot switch. CPU, a few tiny
epochs."""
import lightning as L
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.muon_to_sgd_switch import MuonToSGDSwitch
from src.models.components.optimizers import MuonWithAuxSGD


class _TinyConvNet(L.LightningModule):
    def __init__(self, optimizer_cls=MuonWithAuxSGD):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1),
                                 nn.Flatten(), nn.Linear(8 * 4 * 4, 2))
        self.optimizer_cls = optimizer_cls
        self.use_muon_by_epoch: dict[int, bool] = {}

    def training_step(self, batch, batch_idx):
        group = self.optimizers().optimizer.param_groups[0]
        self.use_muon_by_epoch[self.current_epoch] = group.get("use_muon", False)
        x, y = batch
        return F.cross_entropy(self.net(x), y)

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=0.02)


def _fit(model, switch_epoch, epochs=4):
    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 3, 4, 4), torch.randint(0, 2, (8,))),
                        batch_size=4)
    switch = MuonToSGDSwitch(switch_epoch=switch_epoch, weight_decay=6e-4)
    trainer = L.Trainer(max_epochs=epochs, accelerator="cpu", logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, callbacks=[switch])
    trainer.fit(model, loader)
    return trainer


def test_switches_hidden_convs_to_sgd_at_switch_epoch():
    model = _TinyConvNet()
    trainer = _fit(model, switch_epoch=2)
    assert model.use_muon_by_epoch == {0: True, 1: True, 2: False, 3: False}
    group = trainer.optimizers[0].param_groups[0]
    assert group["momentum"] == 0.9 and group["nesterov"] and group["weight_decay"] == 6e-4


def test_rejects_an_optimizer_it_cannot_switch():
    with pytest.raises(TypeError, match="MuonWithAuxSGD"):
        _fit(_TinyConvNet(optimizer_cls=torch.optim.SGD), switch_epoch=1, epochs=2)


def test_switch_epoch_zero_is_rejected():
    with pytest.raises(ValueError, match="switch_epoch"):
        MuonToSGDSwitch(switch_epoch=0)
