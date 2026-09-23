"""Unit tests for the logit-Gaussian predictives (softmax mean-field vs the normalized
sigmoid / normCDF activations of arXiv:2502.03366). Synthetic data only.

The properties worth pinning are structural rather than numerical: softmax alone is
invariant to a global logit shift (so only the other two can read the degree of freedom
that cross-entropy training leaves unconstrained), and argmax is invariant across all three
in exact arithmetic (so accuracy cannot move between them) -- which survives floating point
for softmax and the sigmoid link but not for normCDF, whose saturation is pinned here as a
documented limit rather than asserted away.
"""
import math

import pytest
import torch

from src.metrics.gaussian_predictives import (
    LAMBDA_PROBIT,
    PREDICTIVE_NAMES,
    log_predictive,
    predictive,
    saturation_fraction,
)
from src.metrics.posthoc_calibration import mean_field_scale


def _synthetic(n: int = 256, num_classes: int = 100, seed: int = 0, scale: float = 3.0):
    """`(raw_logits [n, C], variance [n, 1])` at CIFAR-100's class count and logit scale.

    Defaults are sized to the real runs: `overnight_..._s1_sngp__cifar100` has raw logits
    spanning `[-6.4, +22.7]` and a GP variance averaging `0.024`.
    """
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, num_classes, generator=gen) * scale
    var = torch.rand(n, 1, generator=gen) * 0.05
    return raw, var


class TestArgmaxInvariance:
    """Accuracy cannot change between these predictives -- in exact arithmetic. It holds
    numerically for softmax and the sigmoid link; see `TestNormcdfSaturation` for where it
    stops holding for normCDF, which is a property of the link and not of the code."""

    @pytest.mark.parametrize("name", ["softmax", "normed_sigmoid"])
    @pytest.mark.parametrize("lam", [0.0, LAMBDA_PROBIT, 1.0, 7.5, 50.0])
    def test_argmax_matches_raw_logits(self, name, lam):
        raw, var = _synthetic()
        probs = predictive(name, raw, var, lam)
        assert torch.equal(probs.argmax(dim=1), raw.argmax(dim=1))

    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_argmax_holds_for_every_link_on_the_sensitive_range(self, name):
        """Within roughly |z| < 5 no link has saturated and all three agree with the logits."""
        raw, var = _synthetic(scale=1.0)
        probs = predictive(name, raw, var, 7.5)
        assert torch.equal(probs.argmax(dim=1), raw.argmax(dim=1))

    @pytest.mark.parametrize("name", ["softmax", "normed_sigmoid"])
    def test_argmax_survives_a_logit_shift(self, name):
        """A global shift rescales confidences but must not re-rank classes within a row."""
        raw, var = _synthetic()
        probs = predictive(name, raw, var, 7.5, logit_offset=-6.0)
        assert torch.equal(probs.argmax(dim=1), raw.argmax(dim=1))


class TestNormcdfSaturation:
    """`Phi` is within 1e-9 of 1 by x = 6, so the normalized normCDF cannot rank classes
    apart above that. These tests document the limit rather than assert it away -- it is
    the reason applying this link post-hoc to CE-trained logits changes accuracy."""

    def test_float64_is_required_but_not_sufficient(self):
        raw, var = _synthetic()
        log_p = log_predictive("normed_normcdf", raw, var, 0.0)
        assert log_p.dtype is torch.float64, "downcasting collapses the class ordering"

        n = raw.shape[0]
        drift64 = int((log_p.argmax(dim=1) != raw.argmax(dim=1)).sum())
        drift32 = int((log_p.to(torch.float32).argmax(dim=1) != raw.argmax(dim=1)).sum())
        # float64 removes most of the drift; what is left is the link saturating, and no
        # precision fixes that because the information is absent from Phi's output.
        assert 0 < drift64 < 0.05 * n
        assert drift32 > 10 * drift64

    def test_saturation_fraction_flags_the_affected_rows(self):
        raw, var = _synthetic()
        assert saturation_fraction(raw, var, 7.5) > 0.4
        # Cooling the logits into the link's sensitive range removes the saturation.
        assert saturation_fraction(raw, var, 7.5, temperature=10.0) == 0.0

    def test_sigmoid_link_does_not_saturate_over_the_same_range(self):
        raw, var = _synthetic()
        probs = predictive("normed_sigmoid", raw, var, 7.5)
        assert torch.equal(probs.argmax(dim=1), raw.argmax(dim=1))


class TestNormalization:
    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_rows_sum_to_one(self, name):
        raw, var = _synthetic()
        probs = predictive(name, raw, var, 7.5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0], dtype=probs.dtype), atol=1e-5)

    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_survives_extreme_negative_logits(self, name):
        """`Phi(x)` underflows to exactly 0.0 well before x = -38, so a naive
        `Phi(...) / sum(Phi(...))` returns 0/0 here. Log-space normalization does not."""
        raw = torch.full((4, 100), -60.0)
        raw[:, 0] = -55.0
        var = torch.full((4, 1), 0.01)
        probs = predictive(name, raw, var, 1.0)
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(dim=1), torch.ones(4, dtype=probs.dtype), atol=1e-5)
        assert torch.equal(probs.argmax(dim=1), torch.zeros(4, dtype=torch.long))

    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_survives_the_real_positive_logit_range(self, name):
        """Real CIFAR-100 SNGP logits reach +17.5 after mean-field scaling, with a median
        top logit of 11.4 -- past where `log Phi` underflows in float32. Whatever the link
        does with the ranking, the result must stay a finite distribution."""
        # Top class deliberately not at index 0: argmax ties resolve to the lowest index,
        # which would let a saturated link pass by accident.
        raw = torch.tensor([[-6.4] * 40 + [17.5, 14.0, 11.4, 6.0, -5.0] + [-6.4] * 55])
        var = torch.full((1, 1), 0.024)
        probs = predictive(name, raw, var, 7.5)
        assert torch.isfinite(probs).all()
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        if name != "normed_normcdf":
            assert int(probs.argmax(dim=1)) == 40

    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_log_predictive_is_the_log_of_predictive(self, name):
        raw, var = _synthetic()
        log_p = log_predictive(name, raw, var, 7.5)
        assert torch.allclose(log_p.exp(), predictive(name, raw, var, 7.5), atol=1e-6)


class TestShiftInvariance:
    """Softmax is shift-invariant; the normalized activations are not. This is the whole
    reason the post-hoc swap is not free -- CE training never pins the absolute level."""

    def test_softmax_is_invariant_to_a_global_offset(self):
        raw, var = _synthetic()
        base = predictive("softmax", raw, var, 7.5)
        for offset in (-8.0, -2.0, 2.0, 8.0):
            shifted = predictive("softmax", raw, var, 7.5, logit_offset=offset)
            assert torch.allclose(base, shifted, atol=1e-5)

    @pytest.mark.parametrize("name", ["normed_sigmoid", "normed_normcdf"])
    def test_normalized_activations_are_not_invariant(self, name):
        raw, var = _synthetic()
        base = predictive(name, raw, var, 7.5)
        shifted = predictive(name, raw, var, 7.5, logit_offset=-4.0)
        assert not torch.allclose(base, shifted, atol=1e-3)


class TestTemperature:
    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_identity_temperature_is_a_no_op(self, name):
        raw, var = _synthetic()
        assert torch.allclose(
            predictive(name, raw, var, 7.5), predictive(name, raw, var, 7.5, temperature=1.0)
        )

    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_raising_temperature_flattens_the_predictive(self, name):
        """Every link is monotone, so dividing the logits down must reduce top-1 confidence.

        normCDF is held to a looser bar: on saturated rows its top-1 probability is pinned
        at `1/(number of classes above the ceiling)` and cooling has to pull the row clear
        of the ceiling before it moves at all.
        """
        raw, var = _synthetic()
        cold = predictive(name, raw, var, 7.5, temperature=1.0).max(dim=1).values
        warm = predictive(name, raw, var, 7.5, temperature=5.0).max(dim=1).values
        floor = 0.95 if name == "normed_normcdf" else 0.99
        assert (warm < cold).double().mean() > floor


class TestMatchesTheLiveHead:
    def test_softmax_predictive_reproduces_the_gp_head(self):
        """The offline path must agree with what inference actually wrote to CSV."""
        from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess

        torch.manual_seed(0)
        head = RandomFeatureGaussianProcess(in_dim=16, num_classes=3, rff_dim=32, mean_field_factor=7.5)
        head.train()
        head(torch.randn(64, 16))  # populate the precision accumulator
        head.eval()
        logits, raw, var = head(torch.randn(8, 16))

        expected = torch.softmax(logits, dim=1).to(torch.float64)
        assert torch.allclose(predictive("softmax", raw, var, 7.5), expected, atol=1e-6)


class TestDegenerateCases:
    @pytest.mark.parametrize("name", PREDICTIVE_NAMES)
    def test_lambda_zero_drops_the_variance_correction(self, name):
        raw, var = _synthetic()
        zero_var = torch.zeros_like(var)
        assert torch.allclose(
            predictive(name, raw, var, 0.0), predictive(name, raw, zero_var, 7.5), atol=1e-6
        )

    def test_scaling_agrees_with_mean_field_scale(self):
        raw, var = _synthetic()
        expected = torch.log_softmax(mean_field_scale(raw, var, 7.5), dim=1).to(torch.float64)
        assert torch.allclose(log_predictive("softmax", raw, var, 7.5), expected, atol=1e-6)

    def test_sigmoid_approximates_normcdf_at_the_probit_constant(self):
        """`Phi(z) ~ rho(z / sqrt(pi/8))` -- the approximation the paper's eq. 16 trades on.
        Applied on the link's sensitive range, where the two actually differ."""
        raw, var = _synthetic(num_classes=10, seed=3, scale=1.0)
        sig = predictive("normed_sigmoid", raw / math.sqrt(LAMBDA_PROBIT), var, 0.0)
        cdf = predictive("normed_normcdf", raw, var, 0.0)
        assert (sig - cdf).abs().max() < 0.02

    def test_rejects_an_unknown_name(self):
        raw, var = _synthetic()
        with pytest.raises(ValueError, match="unknown predictive"):
            predictive("probit", raw, var, 1.0)
