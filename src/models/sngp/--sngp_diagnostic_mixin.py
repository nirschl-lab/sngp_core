#!/usr/bin/env python3
"""sngp_diagnostic_mixin.py in src/metrics."""

import torch
import math
from loguru import logger

from src.models.sngp.gaussian_process import mean_field_logits

class SNGPDiagnosticsMixin:
    """A mixin that adds SNGP-specific diagnostics logging during validation epochs.

    It automatically computes and logs:
      - GP covariance statistics
      - GP precision conditioning
      - Mean-field scaling diagnostics
      - Uncertainty calibration metrics
    """

    @torch.no_grad()
    def log_gp_covariance_stats(self):
        """Logs Gaussian Process covariance diagnostics."""
        # Only run if model has an SNGP classifier
        if not hasattr(self.net, "sngp_classifier"):
            logger.debug("Skipping GP covariance logging: model has no sngp_classifier.")
            return
        gp = getattr(self.net.sngp_classifier, "gp_classifier", None)
        if gp is None or not hasattr(gp, "covariance_layer"):
            logger.debug("Skipping GP covariance logging: covariance_layer missing.")
            return

        cov_layer = gp.covariance_layer
        cov = cov_layer.covariance
        cov_diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        mean_diag = cov_diag.mean().item()
        std_diag = cov_diag.std().item()
        max_diag = cov_diag.max().item()
        min_diag = cov_diag.min().item()

        self.log("val/gp_cov_mu_diag", mean_diag, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/gp_cov_std_diag", std_diag, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/gp_cov_min_diag", min_diag, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/gp_cov_max_diag", max_diag, on_epoch=True, prog_bar=False, sync_dist=True)

        # Check condition number of precision for stability
        prec = cov_layer.precision
        try:
            cond_num = torch.linalg.cond(
                prec + cov_layer.ridge_penalty * torch.eye(prec.shape[0], device=prec.device)
            )
            self.log("val/gp_precision_cond_num", cond_num.item(), on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception as e:
            logger.warning(f"Failed to compute precision condition number: {e}")

    @torch.no_grad()
    def log_mean_field_scale_stats(self, logits: torch.Tensor, cov: torch.Tensor | None):
        """Logs mean-field scaling statistics using actual mean_field_logits() behavior."""
        if logits is None or cov is None or not isinstance(cov, torch.Tensor):
            # logger.debug("Missing logits or covariance for mean-field stats.")
            return

        # Compute adjusted logits using your actual mean-field correction
        logits_adjusted = mean_field_logits(logits, cov)

        # Empirical scaling factor applied elementwise
        # Avoid divide-by-zero errors for zero logits
        safe_logits = logits.clone()
        safe_logits[safe_logits == 0] = 1e-8
        scale = (logits_adjusted / safe_logits).abs()

        self.log("val/mean_field_scale_mean", scale.mean().item(), on_epoch=True, sync_dist=True)
        self.log("val/mean_field_scale_std", scale.std().item(), on_epoch=True, sync_dist=True)

        # Optional sanity check: average shrinkage per logit dimension
        if scale.ndim == 2:
            per_dim = scale.mean(dim=0)
            self.log("val/mean_field_scale_max", per_dim.max().item(), on_epoch=True)
            self.log("val/mean_field_scale_min", per_dim.min().item(), on_epoch=True)

    @torch.no_grad()
    def log_uncertainty_metrics(self, probs: torch.Tensor, targets: torch.Tensor):
        """Logs confidence and predictive entropy statistics for uncertainty sanity checks."""
        if probs is None or probs.ndim < 2:
            logger.debug("Invalid probs tensor for uncertainty metrics.")
            return

        confidences, preds = torch.max(probs, dim=-1)
        mean_conf = confidences.mean().item()
        entropy = (-probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
        accuracy = (preds == targets).float().mean().item()

        self.log("val/mean_confidence", mean_conf, on_epoch=True, sync_dist=True)
        self.log("val/predictive_entropy", entropy, on_epoch=True, sync_dist=True)
        self.log("val/conf_acc_gap", mean_conf - accuracy, on_epoch=True, sync_dist=True)

    # === Optional master hook to call all metrics at once ===
    def log_sngp_diagnostics(self, logits=None, probs=None, targets=None):
        """Convenience function to log all SNGP diagnostics in one call."""
        try:
            # Log GP covariance health
            self.log_gp_covariance_stats()

            # Log mean-field scaling stats (if possible)
            cov = None
            if hasattr(self.net, "sngp_classifier"):
                gp = getattr(self.net.sngp_classifier, "gp_classifier", None)
                if gp and hasattr(gp, "covariance_layer"):
                    cov = gp.covariance_layer.covariance
            if logits is not None and cov is not None:
                self.log_mean_field_scale_stats(logits, cov)

            # Log uncertainty confidence/entropy calibration
            if probs is not None and targets is not None:
                self.log_uncertainty_metrics(probs, targets)

        except Exception as e:
            logger.warning(f"Error logging SNGP diagnostics: {e}")