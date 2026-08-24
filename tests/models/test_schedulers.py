import pytest

from src.models.components.schedulers import cyclical_multistep_lr_lambda


class TestCyclicalMultistepLrLambda:
    def test_matches_plain_multistep_within_first_window(self):
        # For epochs inside member 0's window, this must reproduce exactly what
        # torch.optim.lr_scheduler.MultiStepLR(milestones=[30,70,100], gamma=0.1)
        # would give relative to the base LR (MultiStepLR's own cumulative-decay
        # factor at epoch e is gamma ** number_of_milestones_crossed).
        lr_lambda = cyclical_multistep_lr_lambda(epochs_per_member=150, milestones=[30, 70, 100], gamma=0.1)
        assert lr_lambda(0) == pytest.approx(1.0)
        assert lr_lambda(29) == pytest.approx(1.0)
        assert lr_lambda(30) == pytest.approx(0.1)
        assert lr_lambda(69) == pytest.approx(0.1)
        assert lr_lambda(70) == pytest.approx(0.01)
        assert lr_lambda(99) == pytest.approx(0.01)
        assert lr_lambda(100) == pytest.approx(0.001)
        assert lr_lambda(149) == pytest.approx(0.001)

    def test_resets_at_each_member_window(self):
        # This is the entire point of the helper: epoch 150 (the first epoch of
        # member 1's window) must land back at the base LR, not continue decaying
        # from member 0's final gamma**3 factor -- see the comment in
        # deep_ensemble_{tang,wong,acevedo,kather2018}.yaml's scheduler block for why
        # naively repeating MultiStepLR's milestones would compound catastrophically.
        lr_lambda = cyclical_multistep_lr_lambda(epochs_per_member=150, milestones=[30, 70, 100], gamma=0.1)
        assert lr_lambda(150) == pytest.approx(1.0)
        assert lr_lambda(180) == pytest.approx(0.1)
        assert lr_lambda(300) == pytest.approx(1.0)  # member 2's window starts fresh too

    def test_never_compounds_below_the_single_window_floor(self):
        # Sanity check against the actual bug found during implementation: with 5
        # members and 3 milestones each, a naive cumulative MultiStepLR repeat would
        # reach gamma**15. This must never go below gamma**len(milestones).
        milestones = [30, 70, 100]
        gamma = 0.1
        lr_lambda = cyclical_multistep_lr_lambda(epochs_per_member=150, milestones=milestones, gamma=gamma)
        floor = gamma ** len(milestones)
        for epoch in [0, 100, 149, 150, 400, 749]:
            assert lr_lambda(epoch) >= floor - 1e-12

    def test_unsorted_milestones_are_sorted(self):
        lr_lambda = cyclical_multistep_lr_lambda(epochs_per_member=150, milestones=[100, 30, 70], gamma=0.1)
        assert lr_lambda(30) == pytest.approx(0.1)
        assert lr_lambda(100) == pytest.approx(0.001)
