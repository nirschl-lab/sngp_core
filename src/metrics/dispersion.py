"""Standard deviations for the metrics that are means over per-sample values.

`metrics.json` records point estimates, which say nothing about how concentrated a
number is. A metric that is literally `mean(per-sample value)` -- NLL, Brier,
accuracy -- carries that information for free: the per-sample values it averaged are
already a sample, so its spread is exact, not estimated. No resampling is involved and
nothing here is a bootstrap.

This deliberately covers only that class of metric. AUROC/AUPRC are rank statistics
over the whole set, ECE is computed over bins, and macro precision/recall/F1 are ratios
of aggregate counts -- none decomposes into per-sample terms, so none gets a `_std`
here. For those, the project's existing dispersion estimate is the fixed-seed
subsampling in `src/metrics/calculate_ood_metrics.py::_auroc_mean_std_parts`, which is
frozen for reproducibility against published paper numbers and must not be reworked
into this shape.

**`_std` and `_sem` answer different questions.** `_std` is the spread *across samples*
-- how much per-image NLL varies within the dataset, a property of the data and model.
`_sem = _std / sqrt(n)` is the uncertainty *of the mean itself*, which shrinks with
dataset size. An error bar on a reported metric is `_sem`; `_std` describes the
distribution. Both are written because they are not interchangeable and the cost is
one extra float each.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

import torch
from torch import Tensor

# Matches the clamp in `src/inference/infer.py::_finalize`'s NLL, so the mean of
# `per_sample_nll` reproduces the `nll` value written alongside it.
NLL_EPS = 1e-8


class Dispersion(NamedTuple):
    """A metric's mean and how concentrated the per-sample values behind it are."""

    mean: float
    std: float
    sem: float
    n: int


def mean_std_sem(values: Tensor) -> Dispersion:
    """Summarize per-sample metric values. `[N] -> (mean, std, sem, n)`.

    `std` is the sample standard deviation (Bessel-corrected, `ddof=1`), the estimate
    of the population spread. A single sample has no defined spread, so `std`/`sem` are
    `0.0` there rather than NaN -- a NaN would propagate into `metrics.json` and make
    the whole file fail a strict JSON round-trip.
    """
    v = values.detach().to(torch.float64).reshape(-1)
    n = int(v.numel())
    if n == 0:
        raise ValueError("mean_std_sem requires at least one value")
    if n == 1:
        return Dispersion(mean=float(v.item()), std=0.0, sem=0.0, n=1)

    std = float(v.std(unbiased=True).item())
    return Dispersion(mean=float(v.mean().item()), std=std, sem=std / (n**0.5), n=n)


def per_sample_nll(probs: Tensor, targets: Tensor) -> Tensor:
    """Negative log-likelihood of the true class, per sample. `[N, C], [N] -> [N]`.

    Takes probabilities rather than logits because that is what the inference write
    path has: for MC-Dropout `class_probs` is the mean of per-pass softmax, which is
    the predictive distribution whose likelihood is meaningful, while
    `softmax(class_logits)` is not.
    """
    return torch.nn.functional.nll_loss(torch.log(probs + NLL_EPS), targets, reduction="none")


def per_sample_brier(probs: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """Multiclass Brier score per sample. `[N, C], [N] -> [N]`.

    0 is a perfect prediction; the multiclass upper bound is 2 (maximally confident and
    wrong). `src.metrics.brier.brier_score` is the mean of this.
    """
    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).to(probs.dtype)
    return torch.sum((probs - one_hot) ** 2, dim=1)


def summarize_metric(name: str, values: Tensor) -> Dict[str, Any]:
    """`metrics.json` keys for one mean-decomposable metric.

    Returns `{name, name_std, name_sem, n_samples}` -- flat sibling keys rather than a
    nested object, because `metrics.json` already has readers that index it directly
    (`src/metrics/artifact_quantification.py::_load_nll_brier` reads
    `metrics.json["artifact.nll"]`), and nesting would break every one of them for no
    gain. `n_samples` is written once per metric and is identical across them, so
    repeated `dict.update` calls simply overwrite it with the same value; in the
    artifact runner, where keys are `<stream>.`-prefixed, it correctly becomes one
    count per stream.
    """
    summary = mean_std_sem(values)
    return {
        name: summary.mean,
        f"{name}_std": summary.std,
        f"{name}_sem": summary.sem,
        "n_samples": summary.n,
    }


def per_sample_correct(preds: Tensor, targets: Tensor) -> Tensor:
    """1.0 where the prediction is right, 0.0 where it is not. `[N], [N] -> [N]`.

    The mean of this is micro-accuracy, so its `_std` is the Bernoulli spread
    `sqrt(p(1-p))` up to the Bessel correction -- redundant with accuracy itself, but
    written for consistency so every mean-decomposable metric in `metrics.json` has the
    same three keys rather than accuracy being a silent exception.
    """
    return (preds == targets).to(torch.float64)
