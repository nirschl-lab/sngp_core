import pytest
import torch
import torch.nn.functional as F

from src.models.components.losses import ClassBalancedFocalLoss


class TestClassBalancedFocalLoss:
    def test_beta_zero_gives_uniform_weights(self):
        loss = ClassBalancedFocalLoss(class_freq=[10, 100, 1000], beta=0.0, gamma=2.0)
        assert torch.allclose(loss.weight, torch.ones(3))

    def test_beta_upweights_rare_classes(self):
        loss = ClassBalancedFocalLoss(class_freq=[10, 100, 1000], beta=0.999, gamma=2.0)
        assert loss.weight[0] > loss.weight[1] > loss.weight[2]

    def test_gamma_zero_matches_class_balanced_ce(self):
        # nn.CrossEntropyLoss(weight=..., reduction="mean") divides by the sum of the
        # batch's per-sample weights, not by N -- ClassBalancedFocalLoss deliberately
        # divides by N instead (weights are normalized to mean 1 specifically so this
        # stays comparable to unweighted CE), so the expected value must use the same
        # per-sample-then-mean reduction, not plain F.cross_entropy(weight=...).
        torch.manual_seed(0)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))

        cb_focal = ClassBalancedFocalLoss(class_freq=[10, 100, 1000], beta=0.999, gamma=0.0)
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        expected = (cb_focal.weight[targets] * per_sample_ce).mean()

        assert torch.allclose(cb_focal(logits, targets), expected, atol=1e-5)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            ClassBalancedFocalLoss(class_freq=[10, 100], beta=1.0)

    def test_forward_backward_finite(self):
        torch.manual_seed(0)
        loss_fn = ClassBalancedFocalLoss(class_freq=[10, 100, 1000], beta=0.999, gamma=2.0)
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 3, (8,))

        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_weight_registered_as_buffer_named_weight(self):
        loss_fn = ClassBalancedFocalLoss(class_freq=[10, 100, 1000])
        state_dict = loss_fn.state_dict()
        assert "weight" in state_dict
