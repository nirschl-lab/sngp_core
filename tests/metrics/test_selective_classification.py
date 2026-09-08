import pytest
import torch

from src.metrics.selective_classification import augrc, aurc, coverage_at_5risk, risk_at_80cov


def _all_correct(n=50, num_classes=3):
    torch.manual_seed(0)
    targets = torch.randint(0, num_classes, (n,))
    probs = torch.rand(n, num_classes) * 0.3
    probs[torch.arange(n), targets] = 1.0
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs, targets


def _all_wrong(n=50, num_classes=3):
    torch.manual_seed(1)
    targets = torch.randint(0, num_classes, (n,))
    wrong = (targets + 1) % num_classes
    probs = torch.rand(n, num_classes) * 0.3
    probs[torch.arange(n), wrong] = 1.0
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs, targets


class TestAurc:
    def test_all_correct_is_zero(self):
        probs, targets = _all_correct()
        assert aurc(probs, targets) == pytest.approx(0.0, abs=1e-6)

    def test_all_wrong_is_one(self):
        probs, targets = _all_wrong()
        assert aurc(probs, targets) == pytest.approx(1.0, abs=1e-6)


class TestAugrc:
    def test_all_correct_is_zero(self):
        probs, targets = _all_correct()
        assert augrc(probs, targets) == pytest.approx(0.0, abs=1e-6)

    def test_all_wrong_matches_closed_form(self):
        # All ranks wrong -> R(k)=1 for every k, so AUGRC = (1/N^2) * sum_k k = (N+1)/(2N),
        # which is the finite-N form of the paper's asymptotic <= 0.5 bound.
        n = 50
        probs, targets = _all_wrong(n=n)
        expected = (n + 1) / (2 * n)
        assert augrc(probs, targets) == pytest.approx(expected, abs=1e-4)


class TestCoverageAt5Risk:
    def test_all_correct_reaches_full_coverage(self):
        probs, targets = _all_correct()
        assert coverage_at_5risk(probs, targets) == pytest.approx(1.0, abs=1e-6)


class TestRiskAt80Cov:
    def test_all_correct_is_zero(self):
        probs, targets = _all_correct()
        assert risk_at_80cov(probs, targets) == pytest.approx(0.0, abs=1e-6)

    def test_all_wrong_is_one(self):
        probs, targets = _all_wrong()
        assert risk_at_80cov(probs, targets) == pytest.approx(1.0, abs=1e-6)
