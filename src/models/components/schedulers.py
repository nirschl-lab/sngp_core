"""Learning-rate schedule factories.

Every model family in this project uses `CosineAnnealingLR` (or
`CosineAnnealingWarmRestarts` for deep ensembles), declared inline in the model config as
a Hydra `_partial_`. The CIFAR-100 / WideResNet benchmark needs the reference recipe's
warmup-then-piecewise-decay schedule instead, which cannot be written as a bare
`_partial_`: `SequentialLR` takes *already-constructed* schedulers, and each of those
needs the optimizer that Hydra only supplies at the outermost call. Hence a factory.
"""
from typing import Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, MultiStepLR, SequentialLR


def warmup_piecewise_lr(
    optimizer: Optimizer,
    *,
    T_max: int,
    decay_epochs: Sequence[int] = (60, 120, 160),
    reference_epochs: int = 200,
    gamma: float = 0.2,
    warmup_epochs: int = 1,
    warmup_start_factor: float = 0.1,
    eta_min: float = 0.0,
) -> LRScheduler:
    """Linear warmup, then multiply the LR by `gamma` at each decay epoch.

    This is the schedule the SNGP CIFAR baselines train on
    (`ub.schedules.WarmUpPiecewiseConstantSchedule`), including its milestone rescaling:
    the reference fixes `lr_decay_epochs=[60, 120, 160]` against a 200-epoch budget and
    rescales them to the actual budget as `epoch * train_epochs // 200`. At the
    CIFAR-100 budget of 250 epochs that gives decays at 75 / 150 / 200.

    Taking the budget as `T_max` is also what makes this drop into an experiment config:
    the model configs declare `scheduler: {_target_: CosineAnnealingLR, T_max: ..., eta_min: ...}`,
    and Hydra *merges* rather than replaces that node, so an experiment that swaps
    `_target_` still inherits both keys. `T_max` is genuinely the parameter this schedule
    needs; `eta_min` is not (a piecewise schedule has no floor to anneal toward), so it is
    accepted only to absorb the inherited key and rejected if non-zero rather than
    silently ignored.

    One deliberate difference from the reference: `LitModuleBase.configure_optimizers`
    returns `interval: "epoch"`, so warmup steps once at the epoch boundary rather than
    ramping per optimizer step within epoch 0. Over 1 of 250 epochs that is immaterial.

    :param T_max: Total training epochs; the budget `decay_epochs` is rescaled against.
    :param decay_epochs: Decay epochs expressed against `reference_epochs`.
    :param reference_epochs: Budget `decay_epochs` is quoted against.
    :param gamma: Multiplicative decay applied at each milestone.
    :param warmup_epochs: Epochs spent warming up; `0` disables warmup.
    :param warmup_start_factor: Fraction of the base LR to start warmup from.
    :param eta_min: Must be 0.0. Present only to absorb the inherited config key.
    """
    if eta_min != 0.0:
        raise ValueError(
            "warmup_piecewise_lr has no LR floor to anneal toward, so eta_min must be 0.0 "
            f"(got {eta_min}). It exists in the signature only because Hydra merges the "
            "model config's CosineAnnealingLR node into this one."
        )
    if reference_epochs <= 0:
        raise ValueError(f"reference_epochs must be > 0, got {reference_epochs}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")

    milestones = [int(e) * int(T_max) // int(reference_epochs) for e in decay_epochs]
    if milestones != sorted(milestones) or len(set(milestones)) != len(milestones):
        raise ValueError(
            f"decay_epochs {list(decay_epochs)} rescaled to {milestones} at T_max={T_max}, "
            "which is not strictly increasing -- the budget is too short for this schedule."
        )
    if milestones and milestones[0] <= warmup_epochs:
        raise ValueError(
            f"first decay epoch ({milestones[0]}) must come after warmup ({warmup_epochs} "
            "epochs), otherwise the decay is swallowed by the warmup phase. "
            f"decay_epochs {list(decay_epochs)} rescaled against T_max={T_max}. If this is "
            "a short debug run, keep trainer.max_epochs at the real budget and use "
            "+trainer.limit_train_batches instead -- overriding max_epochs also rescales "
            "this schedule."
        )

    if warmup_epochs == 0:
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # `SequentialLR.step()` calls `scheduler.step(0)` at the handover, restarting the
    # decay scheduler's epoch counter from zero. Its milestones are therefore on a clock
    # offset by `warmup_epochs`, and have to be shifted back for them to land on the
    # absolute epochs above. Without this the reference's 75/150/200 decays silently
    # land at 76/151/201.
    decay = MultiStepLR(
        optimizer, milestones=[m - warmup_epochs for m in milestones], gamma=gamma
    )
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_epochs])
