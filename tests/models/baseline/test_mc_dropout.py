import torch

from src.models.baseline.baseline_models import BaselineClassifier


def _model(num_classes: int = 3) -> BaselineClassifier:
    return BaselineClassifier(arch="resnet18", num_classes=num_classes, dropout_p=0.5, pretrained=False)


def test_mc_forward_samples_shape_is_t_b_c():
    model = _model(num_classes=3).eval()
    samples = model.mc_forward_samples(torch.randn(2, 3, 224, 224), T=4)
    assert samples.shape == (4, 2, 3)


def test_mc_forward_samples_restores_original_training_mode():
    model = _model().eval()
    model.mc_forward_samples(torch.randn(2, 3, 224, 224), T=2)
    assert model.training is False

    model.train()
    model.mc_forward_samples(torch.randn(2, 3, 224, 224), T=2)
    assert model.training is True


def test_mc_predict_is_a_pure_reduction_of_mc_forward_samples():
    """mc_predict was rewritten to call mc_forward_samples once and reduce over it --
    this pins that refactor as behavior-preserving."""
    model = _model(num_classes=3).eval()
    x = torch.randn(2, 3, 224, 224)

    torch.manual_seed(0)
    mean_logits, mean_probs, std = model.mc_predict(x, T=5, return_std=True, apply_softmax=True)

    torch.manual_seed(0)
    logits_stack = model.mc_forward_samples(x, T=5)
    expected_probs_stack = torch.softmax(logits_stack, dim=-1)

    assert torch.allclose(mean_logits, logits_stack.mean(0))
    assert torch.allclose(mean_probs, expected_probs_stack.mean(0))
    assert torch.allclose(std, logits_stack.std(0, unbiased=False))


def test_mc_predict_return_std_false_returns_a_two_tuple():
    """src/models/baseline_lit_module.py calls mc_predict with return_std=False."""
    model = _model(num_classes=2).eval()
    result = model.mc_predict(torch.randn(2, 3, 224, 224), T=3, return_std=False, apply_softmax=True)
    assert len(result) == 2


def test_mc_predict_apply_softmax_false_returns_logit_mean_twice():
    model = _model(num_classes=2).eval()
    x = torch.randn(2, 3, 224, 224)

    torch.manual_seed(0)
    mean_logits, mean_of_logits, _ = model.mc_predict(x, T=4, return_std=True, apply_softmax=False)

    assert torch.allclose(mean_logits, mean_of_logits)
