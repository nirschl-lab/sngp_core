"""Learning-rate schedule factories.

Every model family in this project uses `CosineAnnealingLR` (or
`CosineAnnealingWarmRestarts` for deep ensembles), declared inline in the model config as
a Hydra `_partial_`. The CIFAR-100 / WideResNet benchmark needs the reference recipe's
warmup-then-piecewise-decay schedule instead, which cannot be written as a bare
`_partial_`: `SequentialLR` takes *already-constructed* schedulers, and each of those
needs the optimizer that Hydra only supplies at the outermost call. Hence a factory.
`warmup_stable_decay_lr` is the schedule for the Muon arms, whose update size is set by the
LR alone (see its docstring).
"""
from typing import Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR, LRScheduler, MultiStepLR, SequentialLR


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


def warmup_stable_decay_lr(
    optimizer: Optimizer,
    *,
    T_max: int,
    warmup_epochs: int = 1,
    warmup_start_factor: float = 0.1,
    decay_fraction: float = 0.3,
    eta_min: float = 0.0,
) -> LRScheduler:
    """Linear warmup, a constant plateau, then a linear decay toward zero (WSD).

    Meant for Muon (`src.models.components.optimizers.MuonWithAuxAdamW`): its orthogonalized
    step has spectral norm ~lr whatever the gradient's magnitude, so unlike SGD nothing
    shrinks the step as training converges -- the schedule is the only annealing. Step
    decay cuts the update 5x at a time and parks it at a non-zero floor; a linear decay to
    zero anneals smoothly (Bergsma et al. 2025, arXiv:2502.15938, found linear-to-zero best
    for AdamW, whose steps are likewise normalized).

    The LR factor at epoch `e`, with `D = round(decay_fraction * T_max)`:
      * `e < warmup_epochs`: linear from `warmup_start_factor` toward 1;
      * then 1 until epoch `T_max - D`;
      * then `(T_max - e) / D`, i.e. 1/D in the last epoch.
    `LitModuleBase.configure_optimizers` steps schedulers once per epoch, so the last epoch
    runs at 1/D rather than exactly 0 (a zero factor would waste that epoch). `LambdaLR`
    applies the factor to every param group's own base LR, so Muon's and AdamW's groups share
    the shape.

    :param T_max: Total training epochs.
    :param warmup_epochs: Epochs spent warming up; `0` disables warmup.
    :param warmup_start_factor: Fraction of the base LR to start warmup from.
    :param decay_fraction: Fraction of `T_max` spent in the final linear decay.
    :param eta_min: Must be 0.0. Present only to absorb the model config's inherited
        CosineAnnealingLR key (see `warmup_piecewise_lr`).
    """
    if eta_min != 0.0:
        raise ValueError(
            "warmup_stable_decay_lr decays to zero, so eta_min must be 0.0 "
            f"(got {eta_min}). It exists in the signature only because Hydra merges the "
            "model config's CosineAnnealingLR node into this one."
        )
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError(f"decay_fraction must be in (0, 1], got {decay_fraction}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    decay_epochs = round(decay_fraction * T_max)
    if decay_epochs < 1 or warmup_epochs + decay_epochs > T_max:
        raise ValueError(
            f"warmup ({warmup_epochs}) + decay ({decay_epochs} = round({decay_fraction} * "
            f"{T_max})) epochs must fit in T_max={T_max} with at least one decay epoch."
        )
    decay_start = T_max - decay_epochs

    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return warmup_start_factor + (1.0 - warmup_start_factor) * epoch / warmup_epochs
        if epoch < decay_start:
            return 1.0
        return max(T_max - epoch, 0) / decay_epochs

    return LambdaLR(optimizer, lr_lambda=factor)
