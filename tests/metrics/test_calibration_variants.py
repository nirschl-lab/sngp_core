import pytest
import torch
from torch_uncertainty.metrics.classification import CalibrationError

from src.metrics.calibration_variants import adaptive_ece, ece_minus, ece_plus, mce, smooth_ece


class TestEcePlusMinus:
    def test_sums_to_reference_l1_ece(self):
        torch.manual_seed(1)
        n, c = 200, 4
        probs = torch.softmax(torch.randn(n, c) * 2, dim=1)
        targets = torch.randint(0, c, (n,))

        ep = ece_plus(probs, targets)
        em = ece_minus(probs, targets)

        ref = CalibrationError(task="multiclass", num_classes=c, num_bins=10, norm="l1")
        ref.update(probs, targets)
        assert ep + em == pytest.approx(float(ref.compute()), abs=1e-6)

    def test_both_nonnegative(self):
        torch.manual_seed(2)
        n, c = 150, 3
        probs = torch.softmax(torch.randn(n, c), dim=1)
        targets = torch.randint(0, c, (n,))
        assert ece_plus(probs, targets) >= 0.0
        assert ece_minus(probs, targets) >= 0.0

    def test_perfectly_calibrated_bin_has_zero_over_and_under(self):
        probs = torch.tensor([[1.0, 0.0]] * 20)
        targets = torch.zeros(20, dtype=torch.long)
        assert ece_minus(probs, targets) == pytest.approx(0.0, abs=1e-6)
        assert ece_plus(probs, targets) == pytest.approx(0.0, abs=1e-6)

    def test_underconfident_bin_only_contributes_to_ece_minus(self):
        # confidence 0.90 < accuracy 1.0 (always correct) -> pure underconfidence.
        probs = torch.tensor([[0.90, 0.10]] * 20)
        targets = torch.zeros(20, dtype=torch.long)
        assert ece_plus(probs, targets) == pytest.approx(0.0, abs=1e-6)
        assert ece_minus(probs, targets) == pytest.approx(0.10, abs=1e-6)


class TestMce:
    def test_in_range(self):
        torch.manual_seed(3)
        n, c = 100, 4
        probs = torch.softmax(torch.randn(n, c), dim=1)
        targets = torch.randint(0, c, (n,))
        value = mce(probs, targets, num_classes=c)
        assert 0.0 <= value <= 1.0

    def test_perfect_prediction_is_zero(self):
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 10)
        targets = torch.tensor([0, 1] * 10)
        assert mce(probs, targets, num_classes=2) == pytest.approx(0.0, abs=1e-6)


class TestAdaptiveEce:
    def test_in_range(self):
        torch.manual_seed(4)
        n, c = 100, 3
        probs = torch.softmax(torch.randn(n, c), dim=1)
        targets = torch.randint(0, c, (n,))
        value = adaptive_ece(probs, targets, num_classes=c)
        assert 0.0 <= value <= 1.0


class TestSmoothEce:
    def test_in_range(self):
        torch.manual_seed(5)
        n, c = 100, 3
        probs = torch.softmax(torch.randn(n, c), dim=1)
        targets = torch.randint(0, c, (n,))
        value = smooth_ece(probs, targets)
        assert 0.0 <= value <= 1.0
