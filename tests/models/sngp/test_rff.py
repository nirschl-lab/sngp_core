import math

import pytest
import torch

from src.models.sngp.sngp_classifier import (
    DEFAULT_MEAN_FIELD_FACTOR,
    PER_CLASS_LIKELIHOOD,
    PROBIT_MEAN_FIELD_FACTOR,
    SUPPORTED_LIKELIHOODS,
    RandomFeatureGaussianProcess,
)


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

    def test_default_factor_matches_the_imagenet_reference(self):
        """The reference treats this as tunable (1.0 ImageNet, 20.0 CIFAR), not as the
        textbook probit constant -- which is nearly inert at realistic dataset sizes."""
        assert DEFAULT_MEAN_FIELD_FACTOR == pytest.approx(1.0)
        assert make_gp().mean_field_factor == pytest.approx(1.0)
        assert PROBIT_MEAN_FIELD_FACTOR == pytest.approx(math.pi / 8)

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
        """p_max(1-p_max) <= 0.25, so the Laplace-weighted accumulator is strictly smaller."""
        torch.manual_seed(0)
        x = torch.randn(8, 64)

        gaussian = make_gp(likelihood="gaussian")
        logistic = make_gp(likelihood="binary_logistic")
        logistic.load_state_dict(gaussian.state_dict())

        gaussian.train(), logistic.train()
        gaussian(x), logistic(x)
        assert logistic.precision_accum.diagonal().sum() < gaussian.precision_accum.diagonal().sum()

    @pytest.mark.parametrize("likelihood", SUPPORTED_LIKELIHOODS)
    def test_accumulator_is_the_weighted_outer_product_sum(self, likelihood):
        """`precision_accum == sum_i w_i phi_i phi_i^T`, exactly.

        The regression guard for the weight being applied ONCE. The previous
        implementation multiplied `phi` by `sqrt(w)` on one side only and so accumulated
        `sum_i sqrt(w_i) phi_i phi_i^T` -- which every inequality-style test in this class
        still passes, because sqrt is monotone.
        """
        torch.manual_seed(0)
        gp = make_gp(likelihood=likelihood)
        batches = [torch.randn(8, 64) for _ in range(3)]
        gp.train()
        for batch in batches:
            gp(batch)

        expected = torch.zeros(gp.rff_dim, gp.rff_dim)
        with torch.no_grad():
            for batch in batches:
                phi = gp._features(batch)
                w = gp._laplace_weights(gp.classifier(phi), likelihood)
                weighted = phi if w is None else phi * w.unsqueeze(-1)
                expected += weighted.T @ phi
        torch.testing.assert_close(gp.precision_accum, expected, atol=1e-4, rtol=1e-4)

    def test_trace_logistic_sits_between_binary_logistic_and_gaussian(self):
        """`1 - ||p||^2 = sum_k p_k(1-p_k) >= p_max(1-p_max)`, and is at most `1 - 1/K`.

        So the trace reduction keeps more of the data than the max-probability one while
        still down-weighting confident examples against the unit-weight gaussian.
        """
        torch.manual_seed(0)
        x = torch.randn(16, 64)

        gaussian = make_gp(likelihood="gaussian")
        traced = make_gp(likelihood="trace_logistic")
        binary = make_gp(likelihood="binary_logistic")
        traced.load_state_dict(gaussian.state_dict())
        binary.load_state_dict(gaussian.state_dict())

        for gp in (gaussian, traced, binary):
            gp.train()
            gp(x)

        binary_mass = binary.precision_accum.diagonal().sum()
        trace_mass = traced.precision_accum.diagonal().sum()
        gaussian_mass = gaussian.precision_accum.diagonal().sum()
        assert binary_mass < trace_mass < gaussian_mass

    @pytest.mark.parametrize(
        "likelihood,expected_shape",
        [
            ("gaussian", None),
            ("binary_logistic", (4,)),
            ("trace_logistic", (4,)),
            (PER_CLASS_LIKELIHOOD, (4, 10)),
        ],
    )
    def test_laplace_weight_shapes(self, likelihood, expected_shape):
        """`_laplace_weights` is static, so the per-class branch stays testable even
        though no `RandomFeatureGaussianProcess` can be built with that mode."""
        logits = torch.randn(4, 10)
        w = RandomFeatureGaussianProcess._laplace_weights(logits, likelihood)
        if expected_shape is None:
            assert w is None
        else:
            assert tuple(w.shape) == expected_shape
            assert ((w >= 0) & (w <= 1)).all()

    def test_trace_logistic_weight_is_one_minus_squared_norm(self):
        logits = torch.randn(6, 10)
        prob = torch.softmax(logits, dim=-1)
        w = RandomFeatureGaussianProcess._laplace_weights(logits, "trace_logistic")
        torch.testing.assert_close(w, (prob * (1.0 - prob)).sum(dim=-1))

    def test_likelihood_does_not_touch_the_training_signal(self):
        """The mode feeds `precision_accum` and nothing else.

        This is what lets two runs differing only in `likelihood` be compared as the same
        trained model. If it ever breaks, the CIFAR-100 likelihood arms stop being paired.
        """
        torch.manual_seed(0)
        x = torch.randn(8, 64)

        gaussian = make_gp(likelihood="gaussian")
        traced = make_gp(likelihood="trace_logistic")
        traced.load_state_dict(gaussian.state_dict())

        grads = []
        outs = []
        for gp in (gaussian, traced):
            gp.train()
            logits, raw_logits, variance = gp(x)
            assert logits is raw_logits and variance is None
            raw_logits.square().sum().backward()
            outs.append(raw_logits.detach().clone())
            grads.append(gp.classifier.weight.grad.clone())

        torch.testing.assert_close(outs[0], outs[1])
        torch.testing.assert_close(grads[0], grads[1])
        # ... while the thing the mode *is* supposed to move has moved.
        assert not torch.allclose(gaussian.precision_accum, traced.precision_accum)

    def test_rejects_unknown_likelihood(self):
        with pytest.raises(ValueError, match="Unsupported likelihood"):
            make_gp(likelihood="poisson")

    def test_rejects_per_class_logistic_at_construction(self):
        """Implemented as a weight, refused as a mode -- it needs a [K, m, m] accumulator
        and a [B, K] variance that the ModelOutput/CSV contracts do not carry."""
        with pytest.raises(NotImplementedError, match="per_class_logistic"):
            make_gp(likelihood=PER_CLASS_LIKELIHOOD)

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


class TestRandomFeatureType:
    """ORF keeps the per-column marginal of plain RFF (uniform direction x chi norm) but
    removes redundancy between directions, lowering kernel-approximation variance."""

    def test_orf_is_the_default(self):
        assert make_gp().random_feature_type == "orf"

    def test_orf_columns_are_orthogonal_within_a_block(self):
        gp = make_gp(in_dim=32, rff_dim=32, random_feature_type="orf", length_scale=1.0)
        directions = gp.W / gp.W.norm(dim=0, keepdim=True)
        gram = directions.T @ directions
        off_diagonal = gram - torch.diag(torch.diagonal(gram))
        assert off_diagonal.abs().max() < 1e-5

    def test_rff_columns_are_not_orthogonal(self):
        gp = make_gp(in_dim=32, rff_dim=32, random_feature_type="rff", length_scale=1.0)
        directions = gp.W / gp.W.norm(dim=0, keepdim=True)
        gram = directions.T @ directions
        off_diagonal = gram - torch.diag(torch.diagonal(gram))
        assert off_diagonal.abs().max() > 0.1

    def test_orf_handles_rff_dim_larger_than_in_dim(self):
        gp = make_gp(in_dim=16, rff_dim=40, random_feature_type="orf")
        assert gp.W.shape == (16, 40)

    def test_orf_handles_rff_dim_smaller_than_in_dim(self):
        gp = make_gp(in_dim=64, rff_dim=8, random_feature_type="orf")
        assert gp.W.shape == (64, 8)

    def test_orf_column_norms_match_the_gaussian_marginal(self):
        """A standard Gaussian column has chi(in_dim)-distributed norm; ORF must too,
        or the kernel it approximates would differ from the RBF kernel."""
        torch.manual_seed(0)
        in_dim = 64
        orf = make_gp(in_dim=in_dim, rff_dim=2048, random_feature_type="orf", length_scale=1.0)
        rff = make_gp(in_dim=in_dim, rff_dim=2048, random_feature_type="rff", length_scale=1.0)
        assert orf.W.norm(dim=0).mean() == pytest.approx(rff.W.norm(dim=0).mean(), rel=0.05)

    def test_length_scale_still_divides_the_projection(self):
        torch.manual_seed(0)
        wide = make_gp(in_dim=32, rff_dim=64, length_scale=4.0)
        torch.manual_seed(0)
        unit = make_gp(in_dim=32, rff_dim=64, length_scale=1.0)
        torch.testing.assert_close(wide.W * 4.0, unit.W)

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unsupported random_feature_type"):
            make_gp(random_feature_type="sobol")


class TestScaleRandomFeatures:
    """`scale_random_features` controls the `sqrt(2/rff_dim)` factor on phi.

    It looks cosmetic -- a constant the learned readout could absorb -- but it also
    multiplies the gradient reaching the backbone. Adam renormalizes that away per
    parameter, so the project's AdamW protocol never noticed; plain SGD at a fixed lr
    trains the backbone ~22x too slowly at rff_dim=1024, which is what broke the first
    CIFAR-100 pilot (val/acc 0.17 vs a 0.59 deterministic baseline). The reference CIFAR
    baseline turns it off for exactly this reason. See docs/models/CIFAR100_BENCHMARK.md.
    """

    def test_default_keeps_the_sqrt_scaling(self):
        """Default must stay True: every checkpoint in this project was trained with it."""
        gp = make_gp(rff_dim=128)
        assert gp.scale_random_features is True
        assert gp.rff_scale == pytest.approx(math.sqrt(2.0 / 128))

    def test_disabling_drops_the_factor_entirely(self):
        gp = make_gp(rff_dim=128, scale_random_features=False)
        assert gp.rff_scale == 1.0

    def test_features_differ_by_exactly_the_constant(self):
        """The two heads must compute the same cos() and differ only by the scalar --
        i.e. this knob changes magnitude, never the kernel."""
        torch.manual_seed(0)
        scaled = make_gp(rff_dim=128)
        torch.manual_seed(0)
        unscaled = make_gp(rff_dim=128, scale_random_features=False)

        x = torch.randn(4, 64)
        phi_scaled = scaled._features(x)
        phi_unscaled = unscaled._features(x)
        assert torch.allclose(phi_scaled, phi_unscaled * math.sqrt(2.0 / 128), atol=1e-6)

    def test_gradient_into_the_input_scales_with_it(self):
        """The property that actually matters: the factor passes straight through to the
        gradient the backbone would receive."""
        torch.manual_seed(0)
        scaled = make_gp(rff_dim=128)
        torch.manual_seed(0)
        unscaled = make_gp(rff_dim=128, scale_random_features=False)

        sample = torch.randn(4, 64)  # one input, so only the knob differs
        grads = []
        for gp in (scaled, unscaled):
            x = sample.clone().requires_grad_(True)
            gp._features(x).sum().backward()
            grads.append(x.grad.norm().item())

        assert grads[1] == pytest.approx(grads[0] / math.sqrt(2.0 / 128), rel=1e-4)
