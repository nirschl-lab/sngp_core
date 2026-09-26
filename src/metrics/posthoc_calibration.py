"""Post-hoc 1-D calibration fits on held-out logits.

Every model family in this project gets exactly one scalar calibration knob that is fit
*after* training on the validation split by minimizing NLL (a proper scoring rule):

  * Baseline / Deep Ensemble: a temperature ``T`` -- ``logits / T`` (Guo et al. 2017).
  * SNGP: the mean-field factor ``lambda`` -- ``raw_logits / sqrt(1 + lambda * variance)``
    (Liu et al. 2022 eq. 19). The reference implementation collapses this lambda and the
    paper's "kernel amplitude" sigma^2 into the single tunable ``gp_mean_field_factor``, and
    the paper recommends estimating it on held-out data by minimizing the log score.

Both knobs divide every logit of an example by one positive scalar, so accuracy and
macro-F1 are exactly invariant; only NLL/ECE respond, with an interior optimum.

This module is deliberately pure tensor math (CPU, no scipy) so that it can be shared by

  * :class:`CalibratedNLL` -- the per-epoch ``val/nll_cal`` selection metric logged by
    ``src/models/lit_module_base.py`` (the HPO objective is its running minimum), and
  * ``scripts/checkpoints/calibrate_checkpoint.py`` -- the post-hoc fit that writes the
    fitted knob back into a checkpoint's ``net_spec``.

Minimizer: a log-spaced grid seeds a golden-section refinement in log-space. NLL of
``softmax(z / T)`` is convex in ``1/T``, so the temperature fit is exact to tolerance;
NLL as a function of ``lambda`` is smooth and unimodal in practice, and the grid seed
guards against a bad bracket. The identity setting (``T = 1``; ``lambda`` in ``{0, 1}``) is
always evaluated as a candidate, so a fit can never come out worse than no calibration.
"""
import math
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

# Report-table grids (the post-hoc script prints one row per value before fitting).
DEFAULT_TEMPERATURE_GRID: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
# The SNGP grid the old tune_sngp_mean_field.py swept: 0 = no correction, 0.3927 = pi/8
# (the textbook probit constant), 1.0 = the ImageNet reference default, 20 = the CIFAR one.
DEFAULT_MEAN_FIELD_GRID: Tuple[float, ...] = (0.0, 0.3927, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

KNOB_TEMPERATURE = "temperature"
KNOB_MEAN_FIELD = "mean_field_factor"


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """``logits / T``."""
    return logits / temperature


def mean_field_scale(raw_logits: torch.Tensor, variance: torch.Tensor, factor: float) -> torch.Tensor:
    """``raw_logits / sqrt(1 + factor * variance)`` -- mirrors
    ``RandomFeatureGaussianProcess.forward`` in eval mode. ``variance`` is ``[N, 1]`` or ``[N]``."""
    var = variance if variance.dim() == 2 else variance.unsqueeze(1)
    return raw_logits / torch.sqrt(1.0 + factor * var)


def nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean negative log-likelihood (plain, unweighted cross-entropy)."""
    return F.cross_entropy(logits, targets).item()


def minimize_scalar_log(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    n_grid: int = 25,
    n_refine: int = 40,
) -> Tuple[float, float]:
    """Minimize ``f`` over ``[lo, hi]`` (``lo > 0``) in log-space.

    A log-spaced grid of ``n_grid`` points picks the best cell; golden-section search then
    refines within the neighbouring cells for ``n_refine`` iterations. Deterministic.
    Returns ``(x, f(x))``.
    """
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < lo < hi, got lo={lo}, hi={hi}")
    grid = torch.logspace(math.log10(lo), math.log10(hi), n_grid).tolist()
    values = [f(g) for g in grid]
    best = min(range(n_grid), key=values.__getitem__)
    a = math.log(grid[max(best - 1, 0)])
    b = math.log(grid[min(best + 1, n_grid - 1)])

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(math.exp(c)), f(math.exp(d))
    for _ in range(n_refine):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(math.exp(d))
    x = math.exp(0.5 * (a + b))
    fx = f(x)
    # The refinement bracket can, in a flat region, end marginally above the best grid
    # point -- never return something worse than a value we already evaluated.
    if values[best] < fx:
        return grid[best], values[best]
    return x, fx


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    t_min: float = 0.05,
    t_max: float = 20.0,
) -> Tuple[float, float]:
    """``argmin_T NLL(logits / T)`` over ``[t_min, t_max]``. Returns ``(T, nll_at_T)``.

    ``T = 1`` is always evaluated, so the result is never worse than uncalibrated.
    """
    logits = logits.detach().float()
    targets = targets.detach()

    def objective(t: float) -> float:
        return nll(temperature_scale(logits, t), targets)

    t, value = minimize_scalar_log(objective, t_min, t_max)
    at_identity = objective(1.0)
    if at_identity <= value:
        return 1.0, at_identity
    return t, value


def fit_mean_field_factor(
    raw_logits: torch.Tensor,
    variance: torch.Tensor,
    targets: torch.Tensor,
    *,
    lam_min: float = 1e-3,
    lam_max: float = 1e3,
) -> Tuple[float, float]:
    """``argmin_lambda NLL(raw / sqrt(1 + lambda * var))`` over ``{0, 1} U [lam_min, lam_max]``.

    Returns ``(lambda, nll_at_lambda)``. ``lambda = 0`` (no correction) and ``lambda = 1``
    (the training-time spec default) are always evaluated, so the result is never worse
    than either.
    """
    raw_logits = raw_logits.detach().float()
    variance = variance.detach().float()
    targets = targets.detach()

    def objective(lam: float) -> float:
        return nll(mean_field_scale(raw_logits, variance, lam), targets)

    lam, value = minimize_scalar_log(objective, lam_min, lam_max)
    for candidate in (0.0, 1.0):
        at_candidate = objective(candidate)
        if at_candidate <= value:
            lam, value = candidate, at_candidate
    return lam, value


class CalibratedNLL(Metric):
    """Validation NLL *after* fitting the family's single post-hoc calibration knob.

    Accumulates one validation epoch's logits/targets (plus ``raw_logits``/``variance``
    when the net reports them), fits the knob on that epoch's data, and returns the
    calibrated NLL together with the fitted value. This is what a post-hoc-calibrated
    final model will actually report, which is why it -- rather than the raw NLL -- is the
    hyperparameter-search objective (see docs/HPO_GUIDE.md).

    Knob choice is structural, not by family name, so ``LitModuleBase`` stays
    family-agnostic: if every update carried both ``raw_logits`` and ``variance`` (only
    SNGP's eval-mode ``ModelOutput`` does -- Deep Ensemble reports ``variance`` but no
    ``raw_logits``, Baseline reports neither) the knob is the SNGP mean-field factor;
    otherwise it is a temperature. ``knob="temperature"`` forces temperature scaling on
    SNGP for a like-for-like ablation; ``knob="mean_field"`` forces the mean-field fit and
    raises if the net did not report the inputs it needs.

    List states with ``dist_reduce_fx="cat"`` are gathered across DDP ranks inside
    ``compute()`` -- the same mechanism ``MulticlassAveragePrecision`` already relies on in
    ``LitModuleBase`` -- so every rank fits identical values. Memory is ``N_val x C``
    floats: thousands x <= 9 classes here, i.e. negligible.

    ``compute()`` returns ``(nll_cal, fitted_value, knob_name)`` with
    ``knob_name in {"temperature", "mean_field_factor"}``.
    """

    full_state_update = False
    KNOBS = ("auto", "temperature", "mean_field")

    def __init__(self, knob: str = "auto", **kwargs) -> None:
        super().__init__(**kwargs)
        if knob not in self.KNOBS:
            raise ValueError(f"knob must be one of {self.KNOBS}, got {knob!r}")
        self.knob = knob
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("raw_logits", default=[], dist_reduce_fx="cat")
        self.add_state("variance", default=[], dist_reduce_fx="cat")

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        raw_logits: Optional[torch.Tensor] = None,
        variance: Optional[torch.Tensor] = None,
    ) -> None:
        self.logits.append(logits.detach().float())
        self.targets.append(targets.detach())
        if raw_logits is not None and variance is not None:
            self.raw_logits.append(raw_logits.detach().float())
            var = variance.detach().float()
            self.variance.append(var if var.dim() == 2 else var.unsqueeze(1))

    def _has_mean_field_inputs(self) -> bool:
        return len(self.raw_logits) > 0 and len(self.raw_logits) == len(self.logits)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if len(self.logits) == 0:
            raise RuntimeError("CalibratedNLL.compute() called before any update()")
        logits = dim_zero_cat(self.logits).cpu()
        targets = dim_zero_cat(self.targets).cpu()

        has_mf = self._has_mean_field_inputs()
        use_mf = self.knob == "mean_field" or (self.knob == "auto" and has_mf)
        if use_mf:
            if not has_mf:
                raise RuntimeError(
                    "CalibratedNLL(knob='mean_field') needs raw_logits and variance on every "
                    "update, but the net did not report them -- is this an SNGP net in eval mode?"
                )
            raw = dim_zero_cat(self.raw_logits).cpu()
            var = dim_zero_cat(self.variance).cpu()
            lam, value = fit_mean_field_factor(raw, var, targets)
            return torch.tensor(value), torch.tensor(lam), KNOB_MEAN_FIELD

        t, value = fit_temperature(logits, targets)
        return torch.tensor(value), torch.tensor(t), KNOB_TEMPERATURE


def fit_knob(
    knob: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    raw_logits: Optional[torch.Tensor] = None,
    variance: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Dispatch on knob name (``"temperature"`` | ``"mean_field_factor"``). Returns
    ``(fitted_value, nll)``. Convenience for callers that already know the family."""
    if knob == KNOB_TEMPERATURE:
        return fit_temperature(logits, targets)
    if knob == KNOB_MEAN_FIELD:
        if raw_logits is None or variance is None:
            raise ValueError("mean_field_factor fit needs raw_logits and variance")
        return fit_mean_field_factor(raw_logits, variance, targets)
    raise ValueError(f"Unknown calibration knob {knob!r}; expected {KNOB_TEMPERATURE!r} or {KNOB_MEAN_FIELD!r}")


def apply_knob(
    knob: str,
    value: float,
    logits: torch.Tensor,
    *,
    raw_logits: Optional[torch.Tensor] = None,
    variance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a knob value to knob-free base logits. For temperature, ``logits`` are the
    pre-temperature logits; for the mean-field factor, ``raw_logits``/``variance`` are used."""
    if knob == KNOB_TEMPERATURE:
        return temperature_scale(logits, value)
    if knob == KNOB_MEAN_FIELD:
        if raw_logits is None or variance is None:
            raise ValueError("mean_field_factor needs raw_logits and variance")
        return mean_field_scale(raw_logits, variance, value)
    raise ValueError(f"Unknown calibration knob {knob!r}")


def grid_for(knob: str) -> Sequence[float]:
    """The report-table grid for a knob."""
    if knob == KNOB_TEMPERATURE:
        return DEFAULT_TEMPERATURE_GRID
    if knob == KNOB_MEAN_FIELD:
        return DEFAULT_MEAN_FIELD_GRID
    raise ValueError(f"Unknown calibration knob {knob!r}")
