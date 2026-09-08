"""Calibration-error variants beyond plain ECE: ECE+/ECE- (over/under-confidence
decomposition), MCE, adaptive-binned ECE (aECE), and smooth ECE (SmECE).

MCE/aECE/SmECE are thin wrappers around `torch_uncertainty.metrics.classification`'s
tested `torchmetrics.Metric` implementations rather than reimplemented formulas. ECE+/
ECE- have no equivalent in the installed torch-uncertainty release (0.13.0's
`CalibrationError` factory does not expose a signed/`direction` variant), so they're
computed here directly from `torchmetrics`' own top-label binning primitive
(`_binning_bucketize`) -- the same primitive `MulticlassCalibrationError` (the source
of this project's existing "ECE" numbers) and torch-uncertainty's own `CalibrationError`
wrapper both use internally, which is what guarantees `ece_plus(...) + ece_minus(...)`
sums exactly to the standard L1 ECE.
"""
import torch
from torchmetrics.functional.classification.calibration_error import _binning_bucketize
from torch_uncertainty.metrics.classification import AdaptiveCalibrationError, CalibrationError, SmoothCalibrationError


def _top_label_confidence_correct(probs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    confidence, preds = torch.max(probs, dim=1)
    correct = (preds == targets).float()
    return confidence, correct


def ece_plus(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Over-confidence component of top-label ECE: mean(max(confidence - accuracy, 0))
    weighted by bin occupancy, over `n_bins` equal-width bins in [0, 1].

    `ece_plus(...) + ece_minus(...) == ECE` exactly, since `|x| = max(x, 0) + max(-x, 0)`.
    """
    confidence, correct = _top_label_confidence_correct(probs, targets)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    acc_bin, conf_bin, prop_bin = _binning_bucketize(confidence, correct, bin_boundaries)
    return float(torch.sum(torch.clamp(conf_bin - acc_bin, min=0.0) * prop_bin).item())


def ece_minus(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Under-confidence component of top-label ECE: mean(max(accuracy - confidence, 0))
    weighted by bin occupancy, over `n_bins` equal-width bins in [0, 1].

    `ece_plus(...) + ece_minus(...) == ECE` exactly, since `|x| = max(x, 0) + max(-x, 0)`.
    """
    confidence, correct = _top_label_confidence_correct(probs, targets)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    acc_bin, conf_bin, prop_bin = _binning_bucketize(confidence, correct, bin_boundaries)
    return float(torch.sum(torch.clamp(acc_bin - conf_bin, min=0.0) * prop_bin).item())


def mce(probs: torch.Tensor, targets: torch.Tensor, num_classes: int, n_bins: int = 10) -> float:
    """Maximum Calibration Error: worst-bin |accuracy - confidence| gap (Naeini et al. 2015)."""
    metric = CalibrationError(task="multiclass", num_classes=num_classes, num_bins=n_bins, norm="max")
    metric.update(probs, targets)
    return float(metric.compute())


def adaptive_ece(probs: torch.Tensor, targets: torch.Tensor, num_classes: int, n_bins: int = 10) -> float:
    """Adaptive/equal-frequency-binned ECE (Nguyen & O'Connor 2015 / Mukhoti et al. 2020)."""
    metric = AdaptiveCalibrationError(task="multiclass", num_classes=num_classes, num_bins=n_bins)
    metric.update(probs, targets)
    return float(metric.compute())


def smooth_ece(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Smooth/kernel-density ECE (Blasiok & Nakkiran 2023) -- library-native alternative to
    this project's own `src/metrics/smooth_ece.py::smECE_fast_compat`."""
    metric = SmoothCalibrationError()
    metric.update(probs, targets)
    return float(metric.compute())
