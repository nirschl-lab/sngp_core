import math

import pytest
import torch

from src.models.sngp.sngp_classifier import DEFAULT_MEAN_FIELD_FACTOR, RandomFeatureGaussianProcess


def make_gp(**overrides):
    kwargs = dict(
        in_dim=64,
        num_classes=10,
        rff_dim=128,
        length_scale=1.0,
        ridge_penalty=1e-3,
        cov_momentum=-1.0,
        mean_field=True,
    )
    kwargs.update(overrides)
    return RandomFeatureGaussianProcess(**kwargs)


def train_steps(gp, n, batch=8, in_dim=64, generator=None):
    """Run `n` training forwards so the precision accumulator is populated."""
    gp.train()
    for _ in range(n):
        gp(torch.randn(batch, in_dim, generator=generator))
    return gp


class TestShapes:
    @pytest.fixture
    def gp(self):
        return make_gp()

    def test_features_shape(self, gp):
        assert gp._features(torch.randn(4, 64)).shape == (4, 128)

    def test_eval_forward_shapes(self, gp):
        gp.eval()
        logits, raw_logits, pred_var = gp(torch.randn(4, 64))
        assert logits.shape == (4, 10)
        assert raw_logits.shape == (4, 10)
        assert pred_var.shape == (4, 1)

    def test_train_forward_returns_no_variance(self, gp):
        """Training computes no variance at all -- that is what removes the per-batch
        Cholesky from the training loop."""
        gp.train()
        logits, raw_logits, pred_var = gp(torch.randn(4, 64))
        assert pred_var is None
        assert logits is raw_logits


class TestMeanFieldIsInferenceOnly:
    """The correction must never touch the training loss (Liu et al. apply it at
    prediction time only). This is the regression guard for the whole change."""

    def test_train_mode_logits_are_raw(self):
        gp = train_steps(make_gp(), 5)
        gp.train()
        logits, raw_logits, _ = gp(torch.randn(4, 64))
        assert torch.equal(logits, raw_logits)

    def test_eval_mode_applies_probit_correction(self):
        gp = train_steps(make_gp(), 5)
        gp.eval()
        logits, raw_logits, pred_var = gp(torch.randn(4, 64))
        expected = raw_logits / torch.sqrt(1.0 + DEFAULT_MEAN_FIELD_FACTOR * pred_var)
        torch.testing.assert_close(logits, expected)
        assert not torch.allclose(logits, raw_logits)

    def test_default_factor_is_pi_over_eight(self):
        assert DEFAULT_MEAN_FIELD_FACTOR == pytest.approx(math.pi / 8)
        assert make_gp().mean_field_factor == pytest.approx(math.pi / 8)

    def test_custom_factor_is_honored(self):
        gp = train_steps(make_gp(mean_field_factor=2.0), 5)
        gp.eval()
        logits, raw_logits, pred_var = gp(torch.randn(4, 64))
        torch.testing.assert_close(logits, raw_logits / torch.sqrt(1.0 + 2.0 * pred_var))

    def test_without_mean_field(self):
        gp = train_steps(make_gp(mean_field=False), 5)
        gp.eval()
        logits, raw_logits, _ = gp(torch.randn(4, 64))
        assert torch.equal(logits, raw_logits)


class TestPredictiveVariance:
    def test_matches_explicit_inverse(self):
        """var(x) == phi^T (ridge*I + sum_i phi_i phi_i^T)^-1 phi -- the numerical
        check the suite previously lacked entirely.

        Tolerance is relative, not absolute: predictive variance is O(100) here, and the
        precision matrix has condition number ~3e3, so float32 round-off alone is worth
        ~1e-4 relative (eps_fp32 * cond). Seeded so suite ordering cannot shift it.
        """
        torch.manual_seed(0)
        gp = train_steps(make_gp(), 10)
        gp.eval()
        x = torch.randn(8, 64)
        _, _, pred_var = gp(x)

        phi = gp._features(x)
        precision = gp.ridge * torch.eye(gp.rff_dim) + gp.precision_accum
        expected = (phi @ torch.linalg.inv(precision) * phi).sum(dim=1, keepdim=True)
        torch.testing.assert_close(pred_var, expected, rtol=1e-3, atol=1e-3)

    def test_variance_is_non_negative(self):
        gp = train_steps(make_gp(), 10)
        gp.eval()
        _, _, pred_var = gp(torch.randn(16, 64))
        assert (pred_var >= 0).all()

    def test_untrained_variance_is_bounded_by_ridge(self):
        """With an empty accumulator, P = ridge*I so var = ||phi||^2 / ridge.

        Relative tolerance again: the values are O(100) and the quadratic form sums 128
        float32 terms, so exact equality is not on offer.
        """
        torch.manual_seed(0)
        gp = make_gp(ridge_penalty=1e-2)
        gp.eval()
        x = torch.randn(4, 64)
        _, _, pred_var = gp(x)
        phi = gp._features(x)
        torch.testing.assert_close(pred_var, phi.pow(2).sum(1, keepdim=True) / 1e-2, rtol=1e-3, atol=1e-3)

    def test_separates_in_distribution_from_far_ood(self):
        """The whole point of SNGP: distance-aware uncertainty."""
        torch.manual_seed(0)
        gp = make_gp(rff_dim=256)
        in_dist = torch.randn(32, 64) * 0.5
        gp.train()
        for _ in range(200):
            gp(in_dist + 0.01 * torch.randn(32, 64))

        gp.eval()
        _, _, var_id = gp(in_dist)
        _, _, var_ood = gp(torch.randn(16, 64) * 5.0)
        assert var_ood.mean() > 5 * var_id.mean()

    def test_variance_shrinks_as_precision_accumulates(self):
        """Canonical SNGP sums over the epoch, so variance falls roughly like 1/N --
        unlike the pre-correction EMA, whose scale was dataset-size-independent."""
        gp = make_gp()
        probe = torch.randn(8, 64)

        train_steps(gp, 5)
        gp.eval()
        _, _, var_early = gp(probe)

        train_steps(gp, 300)
        gp.eval()
        _, _, var_late = gp(probe)

        assert var_late.mean() < var_early.mean() / 10


class TestPrecisionLifecycle:
    def test_training_accumulates(self):
        gp = make_gp()
        assert torch.count_nonzero(gp.precision_accum) == 0
        train_steps(gp, 3)
        assert torch.count_nonzero(gp.precision_accum) > 0

    def test_reset_zeroes_accumulator(self):
        gp = train_steps(make_gp(), 5)
        gp.reset_precision()
        assert torch.count_nonzero(gp.precision_accum) == 0

    def test_eval_never_mutates_precision(self):
        """`src/inference/records.py` calls `model(x)` with the default
        `update_precision=True`, relying entirely on eval mode to freeze state."""
        gp = train_steps(make_gp(), 5)
        gp.eval()
        before = gp.precision_accum.clone()
        gp(torch.randn(4, 64), update_precision=True)
        assert torch.equal(gp.precision_accum, before)

    def test_update_precision_false_freezes_accumulator_in_train_mode(self):
        gp = train_steps(make_gp(), 5)
        gp.train()
        before = gp.precision_accum.clone()
        gp(torch.randn(4, 64), update_precision=False)
        assert torch.equal(gp.precision_accum, before)

    def test_covariance_cache_is_reused_until_precision_moves(self):
        gp = train_steps(make_gp(), 5)
        gp.eval()
        gp(torch.randn(4, 64))
        assert not bool(gp._cov_stale)

        cached = gp.covariance.clone()
        gp(torch.randn(4, 64))
        assert torch.equal(gp.covariance, cached)

        gp.train()
        gp(torch.randn(4, 64))
        assert bool(gp._cov_stale)

    def test_exact_sum_matches_manual_accumulation(self):
        """cov_momentum < 0 means an exact sum, the paper's scheme."""
        torch.manual_seed(0)
        gp = make_gp(cov_momentum=-1.0)
        batches = [torch.randn(8, 64) for _ in range(4)]
        gp.train()
        for batch in batches:
            gp(batch)

        expected = torch.zeros(gp.rff_dim, gp.rff_dim)
        with torch.no_grad():
            for batch in batches:
                phi = gp._features(batch)
                expected += phi.T @ phi
        torch.testing.assert_close(gp.precision_accum, expected, atol=1e-4, rtol=1e-4)

    def test_momentum_mode_is_an_ema(self):
        gp = make_gp(cov_momentum=0.9)
        train_steps(gp, 1)
        first = gp.precision_accum.clone()
        # An EMA discounts what is already there; an exact sum never would.
        train_steps(gp, 1)
        assert gp.precision_accum.abs().max() < first.abs().max() * 2


class TestLikelihood:
    def test_gaussian_is_the_default(self):
        assert make_gp().likelihood == "gaussian"

    def test_binary_logistic_downweights_relative_to_gaussian(self):
        """p(1-p) <= 0.25, so the Laplace-weighted accumulator is strictly smaller."""
        torch.manual_seed(0)
        x = torch.randn(8, 64)

        gaussian = make_gp(likelihood="gaussian")
        logistic = make_gp(likelihood="binary_logistic")
        logistic.load_state_dict(gaussian.state_dict())

        gaussian.train(), logistic.train()
        gaussian(x), logistic(x)
        assert logistic.precision_accum.diagonal().sum() < gaussian.precision_accum.diagonal().sum()

    def test_rejects_unknown_likelihood(self):
        with pytest.raises(ValueError, match="Unsupported likelihood"):
            make_gp(likelihood="poisson")

    def test_rejects_non_positive_ridge(self):
        with pytest.raises(ValueError, match="ridge_penalty must be > 0"):
            make_gp(ridge_penalty=0.0)


class TestBufferRoundTrip:
    def test_precision_survives_state_dict(self):
        gp = train_steps(make_gp(), 5)
        rebuilt = make_gp()
        rebuilt.load_state_dict(gp.state_dict())
        assert torch.equal(rebuilt.precision_accum, gp.precision_accum)

    def test_reloaded_model_reproduces_variance(self):
        """`_cov_stale` is non-persistent, so a reloaded net must recompute the
        covariance from `precision_accum` rather than trust a stale cache."""
        gp = train_steps(make_gp(), 5)
        gp.eval()
        x = torch.randn(4, 64)
        _, _, expected = gp(x)

        rebuilt = make_gp()
        rebuilt.load_state_dict(gp.state_dict())
        rebuilt.eval()
        _, _, actual = rebuilt(x)
        torch.testing.assert_close(actual, expected)

    def test_identity_is_not_persisted(self):
        """The old head shipped a constant `I` buffer -- 4 MB per checkpoint at
        rff_dim=1024."""
        assert "I" not in make_gp().state_dict()

    def test_legacy_checkpoint_raises_a_clear_error(self):
        gp = make_gp()
        legacy = gp.state_dict()
        legacy["cov_ema"] = torch.zeros(gp.rff_dim, gp.rff_dim)
        legacy["num_updates"] = torch.tensor(7)
        with pytest.raises(RuntimeError, match="pre-correction SNGP head"):
            make_gp().load_state_dict(legacy, strict=False)


class TestNormalizeInput:
    def test_layernorm_present_by_default(self):
        assert make_gp().input_norm is not None

    def test_can_be_disabled(self):
        gp = make_gp(normalize_input=False)
        assert gp.input_norm is None
        assert not any(k.startswith("input_norm") for k in gp.state_dict())

    def test_normalization_changes_features(self):
        torch.manual_seed(0)
        with_norm = make_gp(normalize_input=True)
        without = make_gp(normalize_input=False)
        without.load_state_dict({k: v for k, v in with_norm.state_dict().items() if not k.startswith("input_norm")})
        x = torch.randn(4, 64) * 10.0
        assert not torch.allclose(with_norm._features(x), without._features(x))
