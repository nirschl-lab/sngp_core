import pytest
import torch

from src.metrics.brier import brier_score


class TestBrierScore:
    def test_perfect_prediction_is_zero(self):
        probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        targets = torch.tensor([0, 1])
        assert brier_score(probs, targets, num_classes=3) == 0.0

    def test_uniform_prediction_matches_closed_form(self):
        num_classes = 4
        probs = torch.full((5, num_classes), 1.0 / num_classes)
        targets = torch.randint(0, num_classes, (5,))
        expected = (num_classes - 1) / num_classes
        assert brier_score(probs, targets, num_classes=num_classes) == pytest.approx(expected)

    def test_confident_wrong_prediction_hits_upper_bound(self):
        probs = torch.tensor([[1.0, 0.0]])
        targets = torch.tensor([1])
        assert brier_score(probs, targets, num_classes=2) == pytest.approx(2.0)

    def test_binary_matches_manual_mse(self):
        probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        targets = torch.tensor([0, 1])
        one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        expected = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()
        assert brier_score(probs, targets, num_classes=2) == pytest.approx(expected)
