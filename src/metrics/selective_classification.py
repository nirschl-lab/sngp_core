"""Selective-classification / risk-coverage metrics: AURC, AUGRC, coverage-at-5%-risk,
risk-at-80%-coverage.

Thin wrappers around `torch_uncertainty.metrics.classification`'s tested
`torchmetrics.Metric` implementations rather than reimplemented formulas -- AUGRC in
particular has a known, easy-to-get-subtly-wrong-by-hand low-coverage weighting fix over
plain AURC (Traub et al., NeurIPS 2024, "Overcoming Common Flaws in the Evaluation of
Selective Classification Systems"), so this reuses the library's own implementation.
"""
import torch
from torch_uncertainty.metrics.classification import AUGRC, AURC, CovAt5Risk, RiskAt80Cov


def aurc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Area Under the Risk-Coverage curve (Geifman & El-Yaniv 2017)."""
    metric = AURC()
    metric.update(probs, targets)
    return float(metric.compute())


def augrc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Area Under the Generalized Risk-Coverage curve, bounded in [0, 0.5]."""
    metric = AUGRC()
    metric.update(probs, targets)
    return float(metric.compute())


def coverage_at_5risk(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Maximum coverage achievable while keeping selective risk <= 5%."""
    metric = CovAt5Risk()
    metric.update(probs, targets)
    return float(metric.compute())


def risk_at_80cov(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Selective risk (error rate) among the 80% most-confident predictions."""
    metric = RiskAt80Cov()
    metric.update(probs, targets)
    return float(metric.compute())
