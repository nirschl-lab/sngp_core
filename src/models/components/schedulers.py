"""LR-schedule helpers that don't fit `torch.optim.lr_scheduler`'s built-ins directly.

Kept separate from `losses.py` since these are schedule *factories* (return a plain
callable for `LambdaLR`), not `nn.Module`s.
"""
from typing import Callable, Sequence


def cyclical_multistep_lr_lambda(
    epochs_per_member: int, milestones: Sequence[int], gamma: float = 0.1
) -> Callable[[int], float]:
    """Build a `lr_lambda` for `torch.optim.lr_scheduler.LambdaLR` that repeats a
    MultiStepLR-style decay pattern once per fixed-length window.

    Needed for `DeepEnsembleLitModule`: one optimizer/scheduler pair runs continuously
    across the whole `trainer.max_epochs`, but `set_active_member()` only routes the
    forward pass -- it does not reset the optimizer or scheduler when a new member
    starts training. Plain `MultiStepLR` decays *cumulatively* (multiplies by `gamma`
    at every milestone crossed, ever), so naively repeating `milestones` once per
    member window would compound to `gamma ** (len(milestones) * num_members)` by the
    last member -- an effectively frozen model. `LambdaLR` instead multiplies the
    *initial* LR by `lr_lambda(epoch)` fresh each call, so returning a value relative
    to `epoch % epochs_per_member` gives every member the same from-scratch decay
    curve within its own window.

    :param epochs_per_member: Length of each member's training window (epochs).
    :param milestones: Epoch offsets *within one window* at which LR drops by `gamma`
        (e.g. `[30, 70, 100]`, matching a single baseline model's `MultiStepLR`).
    :param gamma: Multiplicative decay factor applied per milestone crossed.
    :return: A function `epoch -> multiplicative factor on the base LR`, suitable for
        `LambdaLR(optimizer, lr_lambda=...)`.
    """
    milestones = sorted(milestones)

    def lr_lambda(epoch: int) -> float:
        offset = epoch % epochs_per_member
        num_crossed = sum(1 for m in milestones if offset >= m)
        return gamma**num_crossed

    return lr_lambda
