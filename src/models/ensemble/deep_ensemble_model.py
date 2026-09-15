"""
Deep Ensemble implementation for uncertainty quantification.

Deep ensembles train multiple neural networks with different random initializations
and average their predictions at inference time. This provides both improved accuracy
and uncertainty estimates through prediction variance.

Reference:
    Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation 
    using Deep Ensembles" (NeurIPS 2017)
"""
from typing import Tuple
import torch
import torch.nn as nn

from src.metrics.uncertainty import decompose_member_uncertainty
from src.models.outputs import ModelOutput
from src.models.registry import build_net, register_net


@register_net("deep_ensemble")
class DeepEnsemble(nn.Module):
    """
    Deep Ensemble wrapper that manages multiple models.

    During training, only one model is active (controlled by active_member_idx).
    During inference, all models make predictions and outputs are averaged.

    Args:
        base_model_spec: Plain-data `spec` dict (registry `name` + ctor kwargs) used to
            build each ensemble member via `src.models.registry.build_net`.
        num_estimators: Number of ensemble members
        task: Task type ("classification" or "regression")
        temperature: Ensemble-level post-hoc calibration knob applied to the pooled
            (mean) logits, `logits / T`. Never trained or swept; fit on validation NLL
            by `scripts/checkpoints/calibrate_checkpoint.py`, which writes it into the
            checkpoint's `net_spec`. Members keep their own (default 1.0) temperature and
            `member_logits`/`variance` are reported unscaled, in raw member-logit units.
    """

    def __init__(
        self,
        base_model_spec: dict,
        num_estimators: int = 5,
        task: str = "classification",
        temperature: float = 1.0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.base_model_spec = dict(base_model_spec)
        self.num_classes = self.base_model_spec.get('num_classes', None)
        self.num_estimators = num_estimators
        self.task = task
        self.temperature = float(temperature)
        self.active_member_idx = None  # Used during training

        # Create ensemble members
        self.ensemble_members = nn.ModuleList([
            build_net(self.base_model_spec)
            for _ in range(num_estimators)
        ])

        # Initialize each member with different random weights
        for i, member in enumerate(self.ensemble_members):
            self._reset_parameters(member, seed=i)

    @property
    def spec(self) -> dict:
        """Plain-data description of this net, sufficient to rebuild it via `build_net`."""
        return {
            "name": self.registry_name,
            "base_model_spec": self.base_model_spec,
            "num_estimators": self.num_estimators,
            "task": self.task,
            "temperature": self.temperature,
        }
    
    def _reset_parameters(self, model: nn.Module, seed: int):
        """Reset model parameters with a specific seed for diversity."""
        torch.manual_seed(seed)
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def set_active_member(self, idx: int):
        """Set which ensemble member is active during training."""
        assert 0 <= idx < self.num_estimators, f"Invalid member index: {idx}"
        self.active_member_idx = idx
    
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass through the ensemble.

        Training mode: Only the active member makes predictions
        Eval mode: All members predict and outputs are averaged

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            ModelOutput with `logits` set; in eval mode `member_logits` and `variance`
            (prediction variance across members) are set too.
        """
        if self.training:
            # During training, use only the active member
            if self.active_member_idx is None:
                raise RuntimeError(
                    "active_member_idx must be set during training. "
                    "Call set_active_member(idx) before forward pass."
                )
            return ModelOutput(logits=self.ensemble_members[self.active_member_idx](x).logits / self.temperature)
        else:
            # During inference, average predictions from all members
            mean_logits, individual_logits = self.ensemble_predict(x, return_individual=True)
            variance = individual_logits.var(dim=0).mean(dim=-1, keepdim=True)
            # Temperature applies to the pooled predictive only; member_logits/variance
            # stay in raw member-logit units (see the class docstring).
            return ModelOutput(
                logits=mean_logits / self.temperature, member_logits=individual_logits, variance=variance
            )

    def ensemble_predict(self, x: torch.Tensor, return_individual: bool = False) -> torch.Tensor:
        """
        Get predictions from all ensemble members.

        Args:
            x: Input tensor [batch_size, ...]
            return_individual: If True, return individual predictions

        Returns:
            If return_individual=False:
                Mean logits across ensemble [batch_size, num_classes]
            If return_individual=True:
                Tuple of (mean_logits, individual_logits)
                where individual_logits is [num_estimators, batch_size, num_classes]
        """
        individual_outputs = []

        for member in self.ensemble_members:
            member.eval()
            with torch.no_grad():
                output = member(x).logits
                individual_outputs.append(output)

        # Stack: [num_estimators, batch_size, num_classes]
        individual_outputs = torch.stack(individual_outputs)

        # Average across ensemble members
        mean_output = individual_outputs.mean(dim=0)

        if return_individual:
            return mean_output, individual_outputs
        return mean_output
    
    def get_predictive_uncertainty(
        self, 
        x: torch.Tensor,
        uncertainty_type: str = "variance"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predictive uncertainty from ensemble predictions.
        
        Args:
            x: Input tensor [batch_size, ...]
            uncertainty_type: Type of uncertainty metric
                - "variance": Variance of predicted probabilities
                - "entropy": Mean entropy across ensemble
                - "mutual_info": Mutual information (total - aleatoric)
        
        Returns:
            probs: Mean predicted probabilities [batch_size, num_classes]
            uncertainty: Uncertainty values [batch_size]

        The "entropy"/"mutual_info" branches delegate to `src/metrics/uncertainty.py`,
        which is the same code the `predictions.csv` write path uses -- so this offline
        API and the `total_entropy`/`mutual_information` columns cannot drift apart.
        They now use that module's `1e-12` epsilon rather than the `1e-10` this method
        used to hardcode; the difference is ~1e-10 absolute, far below any reported
        precision.
        """
        _, individual_logits = self.ensemble_predict(x, return_individual=True)

        # Convert to probabilities: [num_estimators, batch_size, num_classes]
        individual_probs = torch.softmax(individual_logits, dim=-1)

        # Mean probabilities: [batch_size, num_classes]
        mean_probs = individual_probs.mean(dim=0)

        if uncertainty_type == "variance":
            # Variance of predicted class probabilities
            # Average variance across classes
            variance = individual_probs.var(dim=0).mean(dim=-1)  # [batch_size]
            return mean_probs, variance

        if uncertainty_type in ("entropy", "mutual_info"):
            decomposed = decompose_member_uncertainty(individual_logits)
            uncertainty = decomposed.total if uncertainty_type == "entropy" else decomposed.epistemic
            return mean_probs, uncertainty.to(mean_probs.dtype)

        raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")
    
    def get_member(self, idx: int) -> nn.Module:
        """Get a specific ensemble member."""
        return self.ensemble_members[idx]
    
    def __repr__(self):
        return (
            f"DeepEnsemble(\n"
            f"  num_estimators={self.num_estimators},\n"
            f"  task={self.task},\n"
            f"  base_model={self.ensemble_members[0].__class__.__name__}\n"
            f")"
        )
