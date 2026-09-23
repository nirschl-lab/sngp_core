"""Predictives for a logit Gaussian: softmax mean-field vs normalized sigmoid / normCDF.

SNGP gives every example a Gaussian over logit space -- mean ``raw_logits``, variance
``phi^T Sigma phi`` -- and the predictive is that Gaussian pushed through an output
activation. The integral has no closed form under softmax, so this project uses the
mean-field approximation (Liu et al. 2022 eq. 19), which is what
``RandomFeatureGaussianProcess.forward`` applies in eval mode:

    softmax(mu / sqrt(1 + lambda * sigma^2))

Mucsanyi, Da Costa & Hennig (2025, arXiv:2502.03366) propose instead an *element-wise*
activation that is then normalized, which makes the pushforward available in closed form:

    normCDF:  Phi(mu / sqrt(1 + lambda * sigma^2))  / sum_c Phi(...)
    sigmoid:  rho(mu / sqrt(1 + lambda * sigma^2))  / sum_c rho(...)

with natural constants ``lambda = 1`` for normCDF and ``lambda = pi/8`` for sigmoid. Here
``lambda`` stays a free argument for all three, because this project's CIFAR-100 runs pin
it at 7.5: comparing our softmax at 7.5 against their sigmoid at pi/8 would confound the
link function with the variance scaling, so the two must be varied independently.

**These are not equivalent reparameterizations, and the difference is not cosmetic.**
Softmax is shift-invariant -- adding a constant to every logit of an example changes
nothing -- so a cross-entropy-trained network's absolute logit level is never constrained
by its loss, only the differences between logits are. The two normalized activations are
*not* shift-invariant. Applying them to CE-trained logits therefore reads a degree of
freedom that training left arbitrary, which is why ``logit_offset`` exists below: sweeping
it measures how much of any apparent gain is an accident of where training happened to
leave that level. In the source paper the question does not arise, because there the
activation is trained in (their ``NormedSigmoidNLLLoss`` / ``NormedNdtrNLLLoss`` optimize
``-log(normed_sigmoid(logit))[target]`` directly) and never swapped in post-hoc.

**Argmax is invariant across all three -- in exact arithmetic.** This project's GP variance
is one scalar per example (``(phi @ cov * phi).sum(dim=1, keepdim=True)`` -> ``[N, 1]``), so
the scale factor is shared across classes, every link here is strictly increasing
element-wise, and the normalizer is common to the row. Accuracy and macro-F1 therefore
cannot move between these predictives; only NLL, Brier, ECE and MSP-derived scores can.

That invariance survives contact with floating point for ``softmax`` and ``normed_sigmoid``,
and **does not** for ``normed_normcdf``. ``Phi`` is within ``1e-9`` of its ceiling by ``x = 6``,
so two classes whose logits are 7.7 and 10.2 differ in ``log Phi`` by about ``1e-15``
*relative* -- below float64's resolution, never mind float32's. On synthetic logits at this
project's CIFAR-100 scale, 172 of 256 rows get a different argmax in float32 and 5 of 256
still do in float64. The float32 number is a precision artifact that float64 fixes; the
residual is not fixable, because the information is genuinely absent from the link's output.
So applying normCDF post-hoc to cross-entropy-trained logits *does* change accuracy, by
arbitrary tie-breaking among saturated classes. Use :func:`saturation_fraction` to measure
how much of a given dataset falls in that regime before reading anything else.

**Numerics, which are load-bearing here.** Everything is computed in log-space, in float64,
and normalized with ``logsumexp``. Both choices are correctness requirements, not style:

  * ``Phi(x)`` underflows to exactly ``0.0`` well before ``x = -38``, so a naive
    ``Phi(z) / sum(Phi(z))`` over 100 classes returns ``0/0``. Log-space fixes that end.
  * At the other end ``log Phi(x)`` saturates to exactly ``-0.0`` -- above ``x ~ 20`` in
    float32, ``x ~ 38`` in float64 -- and every class past the cutoff then ties at the same
    probability, which silently destroys the argmax. This project's CIFAR-100 SNGP logits
    reach ``+17.5`` after mean-field scaling, with a median top logit of ``11.4``, so float32
    is genuinely not enough. The reference implementation casts to double for the same
    reason.

Note that float64 only rescues the *representation*. It does not rescue the statistics: at
``x = 6`` the normCDF link is already within ``1e-9`` of its ceiling, so on cross-entropy-
trained logits -- where 2247 of 10000 CIFAR-100 test rows have two or more classes above 6
-- the normalized normCDF spreads near-equal mass over every plausible class regardless of
how far apart their logits are. That is a property of applying the link off its sensitive
range, and it is what ``temperature`` below exists to probe.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from src.metrics.posthoc_calibration import mean_field_scale

# The textbook probit constant, i.e. the scaling under which a logistic sigmoid best
# approximates a normal CDF. `src/models/sngp/sngp_classifier.py` defines the same value as
# `PROBIT_MEAN_FIELD_FACTOR`; it is duplicated rather than imported so this module stays
# pure metrics code with no dependency on the model package.
LAMBDA_PROBIT: float = math.pi / 8

# The variance scaling each link is paired with in arXiv:2502.03366 -- eq. 14 applies
# normCDF at `1/sqrt(1 + sigma^2)`, eq. 16 applies sigmoid at `1/sqrt(1 + (pi/8) sigma^2)`.
# Reference points for the sweep, not defaults: nothing here reads them implicitly.
NATIVE_LAMBDA: Dict[str, float] = {
    "softmax": 1.0,
    "normed_sigmoid": LAMBDA_PROBIT,
    "normed_normcdf": 1.0,
}


def _scale(
    raw_logits: Tensor, variance: Tensor, lam: float, temperature: float, logit_offset: float
) -> Tensor:
    """``(raw_logits / T + offset) / sqrt(1 + lambda * variance)``, in float64.

    The three knobs are deliberately separate, because they correct different things:

      * ``lambda`` is the mean-field factor -- a *per-example* correction, the only one of
        the three that uses the GP variance at all.
      * ``temperature`` is the usual global scale (Guo et al. 2017). Cross-entropy does pin
        the logit scale, it just pins it badly; this is ordinary miscalibration.
      * ``logit_offset`` is the global level. Cross-entropy pins this *not at all* -- softmax
        is shift-invariant, so the loss is exactly constant along it. It is a free parameter
        of any softmax-trained network, and it only becomes observable once a
        non-shift-invariant activation is applied.

    Order is scale, then shift, then the variance correction: the first two describe the
    network's own output, the third is the inference-time correction applied on top.
    """
    z = raw_logits.to(torch.float64)
    if temperature != 1.0:
        z = z / temperature
    if logit_offset != 0.0:
        z = z + logit_offset
    return mean_field_scale(z, variance.to(torch.float64), lam)


def _normalize(log_act: Tensor) -> Tensor:
    """Normalize log-activations to log-probabilities, staying in float64.

    Casting the result down to float32 would undo the point of computing in float64: the
    normalized log-probability is ``~ -4`` for a 100-class problem, so a per-class term of
    ``1e-8`` -- which is what ``logsigmoid`` returns at a logit of 17.5 -- is below float32's
    resolution at that magnitude and the class ordering collapses. Every function here
    therefore returns float64 whatever it was given.
    """
    return log_act - torch.logsumexp(log_act, dim=-1, keepdim=True)


def log_softmax_predictive(
    raw_logits: Tensor,
    variance: Tensor,
    lam: float,
    *,
    temperature: float = 1.0,
    logit_offset: float = 0.0,
) -> Tensor:
    """Log of the SNGP mean-field predictive. ``[N, C], [N, 1] -> [N, C]``.

    Exactly reproduces what the live GP head emits at ``temperature=1, logit_offset=0``:
    ``mean_field_scale`` is the same function ``RandomFeatureGaussianProcess.forward``
    applies, and ``tests/metrics/test_posthoc_calibration.py`` already pins it against the
    head. ``logit_offset`` is accepted and has no effect, by construction -- that is the
    property under test in ``TestShiftInvariance``.
    """
    scaled = _scale(raw_logits, variance, lam, temperature, logit_offset)
    return F.log_softmax(scaled, dim=-1)


def log_normed_sigmoid_predictive(
    raw_logits: Tensor,
    variance: Tensor,
    lam: float,
    *,
    temperature: float = 1.0,
    logit_offset: float = 0.0,
) -> Tensor:
    """Log of the normalized element-wise sigmoid predictive (arXiv:2502.03366 eq. 16)."""
    scaled = _scale(raw_logits, variance, lam, temperature, logit_offset)
    return _normalize(F.logsigmoid(scaled))


def log_normed_normcdf_predictive(
    raw_logits: Tensor,
    variance: Tensor,
    lam: float,
    *,
    temperature: float = 1.0,
    logit_offset: float = 0.0,
) -> Tensor:
    """Log of the normalized element-wise normCDF predictive (arXiv:2502.03366 eq. 14)."""
    scaled = _scale(raw_logits, variance, lam, temperature, logit_offset)
    return _normalize(torch.special.log_ndtr(scaled))


def saturation_fraction(
    raw_logits: Tensor,
    variance: Tensor,
    lam: float,
    *,
    temperature: float = 1.0,
    logit_offset: float = 0.0,
    threshold: float = 6.0,
) -> float:
    """Fraction of rows where the normCDF link cannot rank the top classes apart.

    A row counts as saturated when two or more of its scaled logits exceed ``threshold``,
    past which ``Phi`` is within ``1e-9`` of 1 and the normalized normCDF gives every such
    class effectively the same probability. On this project's CIFAR-100 SNGP runs at the
    pinned ``lambda = 7.5`` this is about 22% of the test set, which is why the normCDF
    row of any comparison table needs reading with its argmax drift stated alongside.

    Reported for the normCDF link specifically; the sigmoid link's ``logsigmoid`` keeps
    usable resolution over this whole range in float64.
    """
    scaled = _scale(raw_logits, variance, lam, temperature, logit_offset)
    return float(((scaled > threshold).sum(dim=-1) >= 2).to(torch.float64).mean().item())


LOG_PREDICTIVES: Dict[str, Callable[..., Tensor]] = {
    "softmax": log_softmax_predictive,
    "normed_sigmoid": log_normed_sigmoid_predictive,
    "normed_normcdf": log_normed_normcdf_predictive,
}

PREDICTIVE_NAMES = tuple(LOG_PREDICTIVES)


def log_predictive(
    name: str,
    raw_logits: Tensor,
    variance: Tensor,
    lam: float,
    *,
    temperature: float = 1.0,
    logit_offset: float = 0.0,
) -> Tensor:
    """Dispatch to one of :data:`LOG_PREDICTIVES` by name. ``[N, C], [N, 1] -> [N, C]``."""
    try:
        fn = LOG_PREDICTIVES[name]
    except KeyError:
        raise ValueError(
            f"unknown predictive {name!r}; expected one of {', '.join(PREDICTIVE_NAMES)}"
        ) from None
    return fn(raw_logits, variance, lam, temperature=temperature, logit_offset=logit_offset)


def predictive(
    name: str,
    raw_logits: Tensor,
    variance: Tensor,
    lam: float,
    *,
    temperature: float = 1.0,
    logit_offset: float = 0.0,
) -> Tensor:
    """Probabilities from one of the three predictives. ``[N, C], [N, 1] -> [N, C]``.

    Rows sum to 1. Prefer :func:`log_predictive` when the consumer is a log-score (NLL),
    so the exponentiation and the subsequent log cancel instead of losing precision on
    the small probabilities that dominate a 100-class problem.
    """
    return log_predictive(
        name, raw_logits, variance, lam, temperature=temperature, logit_offset=logit_offset
    ).exp()
