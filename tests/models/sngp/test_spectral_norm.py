import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import SpectralNorm

from src.models.components.spectral_norm import (
    BoundedSpectralNorm,
    apply_spectral_norm,
    assert_spectral_norm_compatible,
)

class TestApplySpectralNorm:
    def test_apply_spectral_norm_conv2d(self):
        # Test with a container module
        container = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3))
        apply_spectral_norm(container)
        assert hasattr(container[0], 'weight_u')

    def test_apply_spectral_norm_linear(self):
        # Test with a container module
        container = nn.Sequential(nn.Linear(512, 10))
        apply_spectral_norm(container)
        assert hasattr(container[0], 'weight_u')

    def test_skip_batchnorm(self):
        container = nn.Sequential(nn.BatchNorm2d(64))
        apply_spectral_norm(container)
        assert not hasattr(container[0], 'weight_u')

    def test_avoid_double_wrapping(self):
        container = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3))
        apply_spectral_norm(container)
        apply_spectral_norm(container)  # Should not wrap twice
        assert hasattr(container[0], 'weight_u')

    def test_nested_modules(self):
        # Test with nested structure
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Sequential(
                nn.Conv2d(32, 64, 3),
                nn.BatchNorm2d(64)
            ),
            nn.Linear(64, 10)
        )
        apply_spectral_norm(model)

        # Check conv layers have spectral norm
        assert hasattr(model[0], 'weight_u')  # First conv
        assert hasattr(model[2][0], 'weight_u')  # Nested conv
        assert hasattr(model[3], 'weight_u')  # Linear layer

        # Check BatchNorm doesn't have spectral norm
        assert not hasattr(model[2][1], 'weight_u')


class TestSpectralNormCompatibility:
    @pytest.mark.parametrize("arch", ["resnet18", "resnet34", "resnet50"])
    def test_resnet_is_compatible(self, arch):
        assert_spectral_norm_compatible(arch)  # should not raise

    @pytest.mark.parametrize("arch", ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"])
    def test_vit_is_incompatible(self, arch):
        with pytest.raises(ValueError, match="not supported"):
            assert_spectral_norm_compatible(arch)

class TestPowerIterationWarmup:
    """`spectral_norm`'s power iteration only advances on train-mode forwards, so a
    freshly built net would otherwise normalize by `u^T W v` for random unit u, v. The
    error compounds across layers: an unwarmed spectral-normed resnet18 emits ~1e30 in
    eval mode. `cos()` in the GP head used to hide this; LayerNorm on the GP input turns
    it into NaN, and it corrupts Lightning's pre-training sanity-check validation either
    way.
    """

    def test_warmup_keeps_untrained_resnet_activations_sane(self):
        from src.models.backbones import build_backbone

        torch.manual_seed(0)
        backbone, _ = build_backbone("resnet18", False)
        apply_spectral_norm(backbone)
        backbone.eval()
        out = backbone(torch.randn(2, 3, 224, 224))
        assert torch.isfinite(out).all()
        assert out.abs().max() < 100

    def test_without_warmup_activations_explode(self):
        """Documents the failure mode the warmup exists to prevent."""
        from src.models.backbones import build_backbone

        torch.manual_seed(0)
        backbone, _ = build_backbone("resnet18", False)
        apply_spectral_norm(backbone, warmup_iterations=0)
        backbone.eval()
        assert backbone(torch.randn(2, 3, 224, 224)).abs().max() > 1e6

    def test_warmup_drives_the_normalized_weight_to_unit_spectral_norm(self):
        from src.models.components.spectral_norm import warm_up_spectral_norm

        def sigma_of(module):
            return torch.linalg.matrix_norm(module.weight.reshape(module.weight.shape[0], -1), ord=2).item()

        torch.manual_seed(0)
        wrapped = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(wrapped, warmup_iterations=0)
        unwarmed = sigma_of(wrapped[0])

        warm_up_spectral_norm(wrapped)
        warmed = sigma_of(wrapped[0])

        # Spectral norm targets sigma == 1; random u/v under-shoots badly.
        assert abs(warmed - 1.0) < abs(unwarmed - 1.0)
        assert warmed == pytest.approx(1.0, abs=0.05)

    def test_further_warmup_does_not_destabilize_sigma(self):
        """Power iteration keeps refining (and u may converge to -v), so u itself is not
        stable to compare -- what must hold is that sigma stays pinned near 1."""
        from src.models.components.spectral_norm import warm_up_spectral_norm

        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(model)
        for _ in range(3):
            warm_up_spectral_norm(model)
            sigma = torch.linalg.matrix_norm(model[0].weight.reshape(16, -1), ord=2).item()
            assert sigma == pytest.approx(1.0, abs=0.05)


def _sigma_of(module: nn.Module) -> float:
    return torch.linalg.matrix_norm(module.weight.reshape(module.weight.shape[0], -1), ord=2).item()


class TestBoundedSpectralNorm:
    """Liu et al. 2022 eq. 15: `W <- c * W / sigma_hat` only when `sigma_hat > c`. `bound=None`
    must stay the stock hard normalization (sigma == 1), byte-identical to before the bound
    existed, so old checkpoints rebuild the same net."""

    def test_none_bound_registers_stock_hook(self):
        model = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(model)
        hooks = list(model[0]._forward_pre_hooks.values())
        assert len(hooks) == 1
        assert type(hooks[0]) is SpectralNorm

    def test_bound_registers_subclass_with_same_state_dict_keys(self):
        stock = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(stock)
        bounded = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(bounded, bound=2.0)

        hook = next(iter(bounded[0]._forward_pre_hooks.values()))
        assert isinstance(hook, BoundedSpectralNorm)
        assert isinstance(hook, SpectralNorm)  # warm_up_spectral_norm relies on this
        assert hook.bound == 2.0
        assert set(bounded.state_dict()) == set(stock.state_dict()) == {
            "0.weight_orig", "0.weight_u", "0.weight_v", "0.bias",
        }

    def test_effective_sigma_is_at_most_bound_after_warmup(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(8, 16, 3))
        with torch.no_grad():
            model[0].weight.mul_(10.0)  # sigma(W_orig) well above the bound
        apply_spectral_norm(model, bound=2.0)

        sigma_orig = torch.linalg.matrix_norm(model[0].weight_orig.reshape(16, -1), ord=2).item()
        assert sigma_orig > 2.0
        assert _sigma_of(model[0]) == pytest.approx(2.0, abs=0.05)

    def test_weight_untouched_when_sigma_below_bound(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(model, bound=1e3)
        assert torch.equal(model[0].weight, model[0].weight_orig.detach())

        model.train()
        model(torch.randn(1, 8, 8, 8))  # the pre-hook recomputes the weight in train mode
        assert torch.equal(model[0].weight, model[0].weight_orig.detach())

    def test_bound_one_only_shrinks_never_inflates(self):
        """The one observable difference between `bound=1.0` and stock normalization:
        stock scales a small-sigma weight *up* to sigma == 1, the bound leaves it alone."""
        torch.manual_seed(0)
        stock = nn.Sequential(nn.Conv2d(8, 16, 3))
        with torch.no_grad():
            stock[0].weight.mul_(0.01)
        bounded = copy.deepcopy(stock)

        apply_spectral_norm(stock)
        apply_spectral_norm(bounded, bound=1.0)
        assert _sigma_of(stock[0]) == pytest.approx(1.0, abs=0.05)
        assert _sigma_of(bounded[0]) < 0.1

    def test_state_dict_round_trips_between_bounded_and_stock(self):
        torch.manual_seed(0)
        stock = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(stock)
        bounded = nn.Sequential(nn.Conv2d(8, 16, 3))
        apply_spectral_norm(bounded, bound=2.0)

        bounded.load_state_dict(stock.state_dict(), strict=True)
        stock.load_state_dict(bounded.state_dict(), strict=True)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_invalid_bound_raises(self, bad):
        with pytest.raises(ValueError, match="bound must be > 0"):
            apply_spectral_norm(nn.Sequential(nn.Linear(4, 4)), bound=bad)

    @pytest.mark.parametrize("bound", [None, 0.5, 100.0])
    def test_gradient_flows_in_both_regimes(self, bound):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 4))
        apply_spectral_norm(model, bound=bound)
        model(torch.randn(3, 6)).sum().backward()
        grad = model[0].weight_orig.grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    def test_eval_forward_uses_bounded_weight(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 4))
        with torch.no_grad():
            model[0].weight.mul_(20.0)
        apply_spectral_norm(model, bound=2.0)
        model.eval()

        x = torch.randn(3, 6)
        expected = x @ model[0].weight.t() + model[0].bias
        assert torch.allclose(model(x), expected, atol=1e-5)
        assert _sigma_of(model[0]) == pytest.approx(2.0, abs=0.05)


class TestSpectralBatchNorm:
    """`SpectralBatchNorm2d` / `apply_bn_spectral_norm`: DUE's BN gain cap
    max_i |gamma_i| / sqrt(running_var_i + eps) <= coeff (Gouk et al. eq. 28)."""

    @staticmethod
    def _gain(bn: nn.BatchNorm2d, weight: torch.Tensor) -> float:
        return (weight.abs() / (bn.running_var + bn.eps).sqrt()).max().item()

    @staticmethod
    def _effective_weight(bn) -> torch.Tensor:
        return bn.weight / torch.clamp(bn.lipschitz() / bn.coeff, min=1.0)

    def test_gain_is_capped_in_train_and_eval(self):
        from src.models.components.spectral_norm import SpectralBatchNorm2d

        torch.manual_seed(0)
        bn = SpectralBatchNorm2d(8, coeff=3.0)
        with torch.no_grad():
            bn.weight.copy_(torch.linspace(0.5, 4.0, 8))
        for _ in range(20):  # running_var -> ~0.04, so the raw gain is ~20 >> 3
            bn(torch.randn(16, 8, 4, 4) * 0.2)
        assert bn.lipschitz().item() > 10
        for mode in (bn.train, bn.eval):
            mode()
            assert self._gain(bn, self._effective_weight(bn)) == pytest.approx(3.0, rel=1e-5)
        # Eval output is exactly BN with the rescaled gamma.
        x = torch.randn(4, 8, 4, 4)
        ref = nn.functional.batch_norm(x, bn.running_mean, bn.running_var, self._effective_weight(bn), bn.bias, False, 0.0, bn.eps)
        torch.testing.assert_close(bn(x), ref)

    def test_no_op_below_the_cap(self):
        from src.models.components.spectral_norm import SpectralBatchNorm2d

        torch.manual_seed(0)
        plain = nn.BatchNorm2d(8)
        bn = SpectralBatchNorm2d(8, coeff=3.0)
        bn.load_state_dict(plain.state_dict())
        x = torch.randn(16, 8, 4, 4)
        torch.testing.assert_close(bn(x), plain(x))  # train: gamma = 1, running_var = 1 -> L = 1
        torch.testing.assert_close(bn.running_var, plain.running_var)
        assert bn.num_batches_tracked == plain.num_batches_tracked == 1
        bn.eval(), plain.eval()
        torch.testing.assert_close(bn(x), plain(x))

    def test_state_dict_keys_match_batchnorm_and_load_both_ways(self):
        from src.models.components.spectral_norm import SpectralBatchNorm2d

        plain, bn = nn.BatchNorm2d(8), SpectralBatchNorm2d(8, coeff=3.0)
        assert set(bn.state_dict()) == set(plain.state_dict())
        bn.load_state_dict(plain.state_dict(), strict=True)
        plain.load_state_dict(bn.state_dict(), strict=True)

    def test_gradient_reaches_gamma_and_beta(self):
        from src.models.components.spectral_norm import SpectralBatchNorm2d

        bn = SpectralBatchNorm2d(8, coeff=0.5)  # binding from the start
        bn(torch.randn(16, 8, 4, 4)).square().sum().backward()
        assert bn.weight.grad is not None and bn.weight.grad.abs().sum() > 0
        assert bn.bias.grad is not None and bn.bias.grad.abs().sum() > 0

    def test_invalid_coeff_raises(self):
        from src.models.components.spectral_norm import SpectralBatchNorm2d

        with pytest.raises(ValueError, match="coeff"):
            SpectralBatchNorm2d(8, coeff=0.0)

    def test_apply_replaces_every_wrn_batchnorm_and_is_idempotent(self):
        from src.models.backbones import build_backbone
        from src.models.components.spectral_norm import SpectralBatchNorm2d, apply_bn_spectral_norm

        torch.manual_seed(0)
        backbone, _ = build_backbone("wide_resnet28_10", False)
        before = {k: v.clone() for k, v in backbone.state_dict().items()}
        assert apply_bn_spectral_norm(backbone, coeff=3.0) == 25
        bns = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bns) == 25 and all(isinstance(m, SpectralBatchNorm2d) and m.coeff == 3.0 for m in bns)
        after = backbone.state_dict()
        assert set(after) == set(before) and all(torch.equal(before[k], after[k]) for k in before)
        assert apply_bn_spectral_norm(backbone, coeff=3.0) == 0
