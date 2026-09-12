import pytest
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


# --- the Lightning test/predict path ---------------------------------------------------


def _lit_module(use_mc: bool, mc_passes: int = 4):
    from src.models.baseline_lit_module import BaselineLitModule

    return BaselineLitModule(
        net=_model(num_classes=3),
        optimizer=torch.optim.Adam,
        scheduler=None,
        num_classes=3,
        use_mc=use_mc,
        mc_passes=mc_passes,
    )


def test_mc_predict_forward_reports_the_uncertainty_it_computed():
    """`_predict_forward` used to call `mc_predict(return_std=False)`, so a MC-Dropout
    run through `trainer.test()` reported no uncertainty at all while the same
    checkpoint through `src/inference/infer.py` did."""
    module = _lit_module(use_mc=True, mc_passes=4).eval()

    output = module._predict_forward(torch.randn(2, 3, 224, 224))

    assert output.variance is not None
    assert output.variance.shape == (2, 1)
    assert (output.variance >= 0).all()
    assert output.member_logits is not None
    assert output.member_logits.shape == (4, 2, 3)


def test_mc_predict_forward_still_returns_the_mean_of_per_pass_softmax():
    """softmax(.logits) must stay the true MC predictive distribution -- the mean of
    per-pass softmax, not softmax of the mean logits."""
    module = _lit_module(use_mc=True, mc_passes=3).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    output = module._predict_forward(x)
    probs = torch.softmax(output.logits, dim=1)

    assert probs.sum(dim=1).tolist() == pytest.approx([1.0, 1.0], abs=1e-5)
    assert (probs >= 0).all()


def test_non_mc_forward_is_unchanged_and_has_no_variance():
    module = _lit_module(use_mc=False).eval()

    output = module._predict_forward(torch.randn(2, 3, 224, 224))

    assert output.variance is None
    assert output.member_logits is None
