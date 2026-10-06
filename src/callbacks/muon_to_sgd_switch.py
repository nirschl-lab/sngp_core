"""Second stage of a Muon -> SGD two-stage schedule: hand the hidden convs to SGD mid-training.

After Shen et al. 2026 ("Towards Understanding the Power and Limits of the Muon Optimizer: A
River-Valley Perspective", arXiv:2606.21514): Muon's orthogonalized step moves fast along the
low-curvature "river" early in training, but oscillates near the optimum, so a GD-like optimizer
takes over for late refinement. The paper switches Muon -> AdamW on an LLM; here the hidden convs
move to SGD + Nesterov, the rule `MuonWithAuxSGD` already runs on the aux group.

The switch happens in place on the one optimizer `LitModuleBase.configure_optimizers` built
(`MuonWithAuxSGD.switch_muon_group_to_sgd`), so the LR scheduler, the param groups and the
checkpointed optimizer state all keep their layout. The LR is the scheduler's business
(`warmup_constant_cosine_lr`, whose `cosine_start_epoch` should equal `switch_epoch`).
"""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from loguru import logger as log

from src.models.components.optimizers import MuonWithAuxSGD


class MuonToSGDSwitch(Callback):
    """Switch `MuonWithAuxSGD`'s Muon group to SGD at the start of epoch `switch_epoch`.

    `switch_epoch` is Lightning's 0-based `current_epoch`: epoch `switch_epoch` is the first to
    train on SGD. The switch happens at an epoch boundary, so SNGP's per-epoch precision
    accumulation never mixes the two regimes. It fires once: a resumed stage-2 run restores
    `use_muon=False` from the optimizer state and the check below is a no-op.

    :param switch_epoch: First epoch (0-based) trained with SGD on the hidden convs.
    :param momentum: SGD momentum for the hidden convs after the switch.
    :param nesterov: Nesterov momentum for the hidden convs after the switch.
    :param weight_decay: Coupled L2 for the hidden convs after the switch.
    """

    def __init__(
        self,
        switch_epoch: int,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 6e-4,
    ) -> None:
        super().__init__()
        if switch_epoch < 1:
            raise ValueError(
                f"switch_epoch must be >= 1 (epoch 0 would leave no Muon stage), got {switch_epoch}"
            )
        self.switch_epoch = int(switch_epoch)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch < self.switch_epoch:
            return
        optimizer = trainer.optimizers[0]
        if not isinstance(optimizer, MuonWithAuxSGD):
            raise TypeError(
                f"MuonToSGDSwitch needs a MuonWithAuxSGD optimizer, got {type(optimizer).__name__}"
            )
        if not optimizer.param_groups[0]["use_muon"]:
            return
        optimizer.switch_muon_group_to_sgd(
            momentum=self.momentum, nesterov=self.nesterov, weight_decay=self.weight_decay
        )
        log.info(
            f"MuonToSGDSwitch: epoch {trainer.current_epoch}, hidden convs now on SGD "
            f"(momentum {self.momentum}, nesterov {self.nesterov}, L2 {self.weight_decay}, "
            f"lr {optimizer.param_groups[0]['lr']:.3g})"
        )
