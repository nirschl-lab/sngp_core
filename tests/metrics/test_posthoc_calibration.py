"""Unit tests for the shared post-hoc 1-D calibration fits (temperature, SNGP mean-field
factor) and the `CalibratedNLL` selection metric built on them. Synthetic data only."""
import math

import pytest
import torch

from src.metrics.posthoc_calibration import (
    DEFAULT_MEAN_FIELD_GRID,
    DEFAULT_TEMPERATURE_GRID,
    KNOB_MEAN_FIELD,
    KNOB_TEMPERATURE,
    CalibratedNLL,
    apply_knob,
    fit_knob,
    fit_mean_field_factor,
    fit_temperature,
    grid_for,
    mean_field_scale,
    minimize_scalar_log,
    nll,
    temperature_scale,
)


def _sample_targets(logits: torch.Tensor, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.multinomial(torch.softmax(logits, dim=1), 1, generator=gen).squeeze(1)


class TestMinimizer:
    def test_recovers_minimum_of_log_quadratic(self):
        x, fx = minimize_scalar_log(lambda x: (math.log(x) - math.log(2.0)) ** 2, 0.01, 100.0)
        assert x == pytest.approx(2.0, rel=1e-3)
        assert fx == pytest.approx(0.0, abs=1e-6)

    def test_rejects_bad_bracket(self):
        with pytest.raises(ValueError):
            minimize_scalar_log(lambda x: x, 0.0, 1.0)
        with pytest.raises(ValueError):
            minimize_scalar_log(lambda x: x, 2.0, 1.0)


class TestFitTemperature:
    def test_recovers_known_scale(self):
        torch.manual_seed(0)
        z = 2.0 * torch.randn(20_000, 5)
        targets = _sample_targets(z)
        # Logits that are 3x too sharp need T = 3 to be calibrated again.
        t, value = fit_temperature(3.0 * z, targets)
        assert t == pytest.approx(3.0, rel=0.15)
        assert value < nll(3.0 * z, targets)
        assert value == pytest.approx(nll(z, targets), rel=0.02)

    def test_identity_when_already_calibrated(self):
        torch.manual_seed(1)
        z = 2.0 * torch.randn(20_000, 5)
        targets = _sample_targets(z, seed=1)
        t, value = fit_temperature(z, targets)
        assert t == pytest.approx(1.0, rel=0.1)
        assert value <= nll(z, targets) + 1e-6

    def test_never_worse_than_uncalibrated(self):
        torch.manual_seed(2)
        z = torch.randn(64, 3)
        targets = torch.randint(0, 3, (64,))
        _, value = fit_temperature(z, targets)
        assert value <= nll(z, targets) + 1e-6


class TestFitMeanFieldFactor:
    @staticmethod
    def _synthetic(lam_true: float, n: int = 50_000, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        raw = 3.0 * torch.randn(n, 5, generator=gen)
        var = 0.5 + 4.5 * torch.rand(n, 1, generator=gen)
        targets = _sample_targets(mean_field_scale(raw, var, lam_true), seed=seed)
        return raw, var, targets

    def test_recovers_known_factor(self):
        raw, var, targets = self._synthetic(3.0)
        lam, value = fit_mean_field_factor(raw, var, targets)
        assert 2.0 <= lam <= 4.5, lam
        assert value <= nll(mean_field_scale(raw, var, 0.0), targets)
        assert value <= nll(mean_field_scale(raw, var, 1.0), targets)

    def test_zero_when_no_correction_needed(self):
        raw, var, targets = self._synthetic(0.0, seed=3)
        lam, value = fit_mean_field_factor(raw, var, targets)
        assert lam < 0.5, lam
        assert value <= nll(raw, targets) + 1e-6

    def test_accepts_flat_variance(self):
        raw, var, targets = self._synthetic(2.0, n=2_000, seed=4)
        lam_2d, value_2d = fit_mean_field_factor(raw, var, targets)
        lam_1d, value_1d = fit_mean_field_factor(raw, var.squeeze(1), targets)
        assert lam_1d == pytest.approx(lam_2d)
        assert value_1d == pytest.approx(value_2d)


class TestScaleFunctionsMatchTheNets:
    def test_mean_field_scale_matches_gp_head_eval_output(self):
        from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess

        torch.manual_seed(0)
        head = RandomFeatureGaussianProcess(in_dim=16, num_classes=3, rff_dim=32, mean_field_factor=2.0)
        head.train()
        head(torch.randn(64, 16))  # populate the precision accumulator
        head.eval()
        logits, raw, var = head(torch.randn(8, 16))
        assert torch.allclose(mean_field_scale(raw, var, 2.0), logits, atol=1e-6)

    def test_temperature_scale_is_division(self):
        z = torch.randn(4, 3)
        assert torch.equal(temperature_scale(z, 1.0), z)
        assert torch.allclose(temperature_scale(z, 2.0), z / 2.0)


class TestCalibratedNLLMetric:
    def test_auto_selects_temperature_without_variance(self):
        torch.manual_seed(0)
        metric = CalibratedNLL()
        z = torch.randn(32, 4)
        y = torch.randint(0, 4, (32,))
        metric.update(z[:16], y[:16])
        metric.update(z[16:], y[16:])
        value, knob_value, knob = metric.compute()
        assert knob == KNOB_TEMPERATURE
        assert value.item() <= nll(z, y) + 1e-6
        assert knob_value.item() > 0

    def test_auto_selects_mean_field_with_raw_logits_and_variance(self):
        torch.manual_seed(0)
        metric = CalibratedNLL()
        raw = torch.randn(32, 4)
        var = torch.rand(32, 1)
        y = torch.randint(0, 4, (32,))
        logits = mean_field_scale(raw, var, 1.0)
        metric.update(logits, y, raw_logits=raw, variance=var)
        value, knob_value, knob = metric.compute()
        assert knob == KNOB_MEAN_FIELD
        assert value.item() <= nll(raw, y) + 1e-6
        assert value.item() <= nll(logits, y) + 1e-6
        assert knob_value.item() >= 0

    def test_forced_temperature_on_sngp_like_inputs(self):
        metric = CalibratedNLL(knob="temperature")
        raw = torch.randn(16, 3)
        metric.update(raw, torch.randint(0, 3, (16,)), raw_logits=raw, variance=torch.rand(16, 1))
        _, _, knob = metric.compute()
        assert knob == KNOB_TEMPERATURE

    def test_forced_mean_field_without_inputs_raises(self):
        metric = CalibratedNLL(knob="mean_field")
        metric.update(torch.randn(16, 3), torch.randint(0, 3, (16,)))
        with pytest.raises(RuntimeError, match="raw_logits and variance"):
            metric.compute()

    def test_invalid_knob_rejected(self):
        with pytest.raises(ValueError):
            CalibratedNLL(knob="platt")

    def test_reset_clears_state(self):
        metric = CalibratedNLL()
        metric.update(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        metric.compute()
        metric.reset()
        with pytest.raises(RuntimeError, match="before any update"):
            metric.compute()


class TestDispatchHelpers:
    def test_fit_and_apply_round_trip_temperature(self):
        torch.manual_seed(0)
        z = 2.0 * torch.randn(5_000, 4)
        y = _sample_targets(z)
        t, value = fit_knob(KNOB_TEMPERATURE, 2.0 * z, y)
        assert torch.allclose(apply_knob(KNOB_TEMPERATURE, t, 2.0 * z), (2.0 * z) / t)
        assert value == pytest.approx(nll((2.0 * z) / t, y))

    def test_mean_field_dispatch_requires_inputs(self):
        with pytest.raises(ValueError):
            fit_knob(KNOB_MEAN_FIELD, torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
        with pytest.raises(ValueError):
            apply_knob(KNOB_MEAN_FIELD, 1.0, torch.randn(4, 2))

    def test_unknown_knob_rejected(self):
        with pytest.raises(ValueError):
            fit_knob("platt", torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
        with pytest.raises(ValueError):
            grid_for("platt")

    def test_grids(self):
        assert grid_for(KNOB_TEMPERATURE) == DEFAULT_TEMPERATURE_GRID
        assert grid_for(KNOB_MEAN_FIELD) == DEFAULT_MEAN_FIELD_GRID
        assert 1.0 in DEFAULT_TEMPERATURE_GRID
        assert 0.0 in DEFAULT_MEAN_FIELD_GRID and 1.0 in DEFAULT_MEAN_FIELD_GRID
