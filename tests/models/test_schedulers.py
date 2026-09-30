"""Guards for `src/models/components/schedulers.py`.

The CIFAR-100 benchmark's whole LR trajectory is defined by this one factory, and every
failure mode here is silent: a schedule that decays one epoch late, or that swallows a
decay inside warmup, still trains to completion and just produces slightly wrong numbers.
"""
import pytest
import torch

from src.models.components.schedulers import warmup_piecewise_lr, warmup_stable_decay_lr

BASE_LR = 0.04


def _lr_trace(total_epochs: int, **kwargs) -> list[float]:
    """Run the schedule for `total_epochs` and return the LR in force during each."""
    params = [torch.nn.Parameter(torch.zeros(1))]
    optimizer = torch.optim.SGD(params, lr=BASE_LR, momentum=0.9, nesterov=True)
    scheduler = warmup_piecewise_lr(optimizer, **kwargs)

    trace = []
    for _ in range(total_epochs):
        trace.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return trace


def _change_points(trace: list[float]) -> list[tuple[int, float]]:
    points, previous = [], None
    for epoch, lr in enumerate(trace):
        if previous is None or abs(lr - previous) > 1e-12:
            points.append((epoch, round(lr, 9)))
        previous = lr
    return points


class TestWarmupPiecewiseLR:
    def test_reference_cifar100_schedule(self):
        """250-epoch budget: 1 warmup epoch, then x0.2 at epochs 75 / 150 / 200.

        These are the reference's `[60, 120, 160]` rescaled by `T_max // 200`. Pinned
        exactly, because `SequentialLR` restarts the decay scheduler's epoch counter at
        the handover -- without the offset in the factory these land at 76 / 151 / 201.
        """
        assert _change_points(_lr_trace(250, T_max=250)) == [
            (0, BASE_LR * 0.1),
            (1, BASE_LR),
            (75, pytest.approx(BASE_LR * 0.2)),
            (150, pytest.approx(BASE_LR * 0.2**2)),
            (200, pytest.approx(BASE_LR * 0.2**3)),
        ]

    def test_milestones_rescale_with_the_budget(self):
        """At the reference's own 200-epoch budget the decays sit at [60, 120, 160]."""
        epochs = [epoch for epoch, _ in _change_points(_lr_trace(200, T_max=200))]
        assert epochs == [0, 1, 60, 120, 160]

    def test_short_pilot_budget_still_decays_three_times(self):
        """A 20-epoch pilot must exercise the same shape, not collapse to a flat LR."""
        epochs = [epoch for epoch, _ in _change_points(_lr_trace(20, T_max=20))]
        assert epochs == [0, 1, 6, 12, 16]

    def test_no_warmup_starts_at_base_lr(self):
        trace = _lr_trace(250, T_max=250, warmup_epochs=0)
        assert trace[0] == pytest.approx(BASE_LR)
        assert [epoch for epoch, _ in _change_points(trace)] == [0, 75, 150, 200]

    def test_nonzero_eta_min_is_rejected_not_ignored(self):
        """`eta_min` only exists to absorb the model config's inherited CosineAnnealingLR
        key. A caller who sets it meaningfully must hear about it."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
        with pytest.raises(ValueError, match="eta_min must be 0.0"):
            warmup_piecewise_lr(optimizer, T_max=250, eta_min=1e-6)

    def test_decay_inside_warmup_is_rejected(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
        with pytest.raises(ValueError, match="must come after warmup"):
            warmup_piecewise_lr(optimizer, T_max=200, decay_epochs=[1, 120], warmup_epochs=5)

    def test_budget_too_short_to_separate_milestones_is_rejected(self):
        """`[60, 120, 160] * 2 // 200` collapses to [0, 1, 1] -- fail loudly rather than
        silently decay twice in one epoch."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
        with pytest.raises(ValueError, match="not strictly increasing"):
            warmup_piecewise_lr(optimizer, T_max=2)


def _wsd_trace(total_epochs: int, **kwargs) -> list[float]:
    params = [torch.nn.Parameter(torch.zeros(1))]
    optimizer = torch.optim.SGD(params, lr=BASE_LR)
    scheduler = warmup_stable_decay_lr(optimizer, **kwargs)
    trace = []
    for _ in range(total_epochs):
        trace.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return trace


class TestWarmupStableDecayLR:
    def test_cifar100_muon_schedule(self):
        """250 epochs, 1 warmup epoch at 0.1x, flat through epoch 175, then linear to 1/75."""
        trace = _wsd_trace(250, T_max=250)
        assert trace[0] == pytest.approx(BASE_LR * 0.1)
        assert trace[1:176] == pytest.approx([BASE_LR] * 175)
        assert trace[176] == pytest.approx(BASE_LR * 74 / 75)
        assert trace[249] == pytest.approx(BASE_LR / 75)
        steps = [a - b for a, b in zip(trace[175:], trace[176:])]
        assert steps == pytest.approx([BASE_LR / 75] * 74)  # linear

    def test_non_increasing_after_warmup_and_never_negative(self):
        trace = _wsd_trace(300, T_max=250, warmup_epochs=3, decay_fraction=0.5)
        assert all(a >= b for a, b in zip(trace[3:], trace[4:]))
        assert min(trace) >= 0.0

    def test_scales_every_param_group_from_its_own_base_lr(self):
        """Muon and AdamW live in one optimizer; both must follow the same shape."""
        groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.02},
                  {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3}]
        optimizer = torch.optim.SGD(groups)
        scheduler = warmup_stable_decay_lr(optimizer, T_max=10)
        for _ in range(9):
            optimizer.step()
            scheduler.step()
            lrs = [g["lr"] for g in optimizer.param_groups]
            assert lrs[0] / lrs[1] == pytest.approx(20.0)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"T_max": 250, "eta_min": 1e-4}, "eta_min must be 0.0"),
            ({"T_max": 250, "decay_fraction": 0.0}, "decay_fraction"),
            ({"T_max": 250, "decay_fraction": 1.5}, "decay_fraction"),
            ({"T_max": 10, "warmup_epochs": 5, "decay_fraction": 0.8}, "must fit"),
        ],
    )
    def test_bad_arguments_are_rejected(self, kwargs, match):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
        with pytest.raises(ValueError, match=match):
            warmup_stable_decay_lr(optimizer, **kwargs)
