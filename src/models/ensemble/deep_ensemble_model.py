"""
Deep Ensemble implementation for uncertainty quantification.

Deep ensembles train multiple neural networks with different random initializations
and average their predictions at inference time. This provides both improved accuracy
and uncertainty estimates through prediction variance.

Reference:
    Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation 
    using Deep Ensembles" (NeurIPS 2017)
"""
from typing import Tuple, Optional, List
import torch
import torch.nn as nn


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble wrapper that manages multiple models.
    
    During training, only one model is active (controlled by active_member_idx).
    During inference, all models make predictions and outputs are averaged.
    
    Args:
        base_model_class: Class to instantiate for each ensemble member
        base_model_kwargs: Arguments to pass to base model constructor
        num_estimators: Number of ensemble members
        task: Task type ("classification" or "regression")
    """
    
    def __init__(
        self,
        base_model_class: type,
        base_model_kwargs: dict,
        num_estimators: int = 5,
        task: str = "classification",
    ):
        super().__init__()
        
        self.num_classes = base_model_kwargs.get('num_classes', None)
        self.num_estimators = num_estimators
        self.task = task
        self.active_member_idx = None  # Used during training
        
        # Create ensemble members
        self.ensemble_members = nn.ModuleList([
            base_model_class(**base_model_kwargs) 
            for _ in range(num_estimators)
        ])
        
        # Initialize each member with different random weights
        for i, member in enumerate(self.ensemble_members):
            self._reset_parameters(member, seed=i)
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Training mode: Only the active member makes predictions
        Eval mode: All members predict and outputs are averaged
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            logits: [batch_size, num_classes] for classification
        """
        if self.training:
            # During training, use only the active member
            if self.active_member_idx is None:
                raise RuntimeError(
                    "active_member_idx must be set during training. "
                    "Call set_active_member(idx) before forward pass."
                )
            return self.ensemble_members[self.active_member_idx](x)
        else:
            # During inference, average predictions from all members
            return self.ensemble_predict(x)
    
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
                output = member(x)
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
        """
        mean_logits, individual_logits = self.ensemble_predict(x, return_individual=True)
        
        # Convert to probabilities: [num_estimators, batch_size, num_classes]
        individual_probs = torch.softmax(individual_logits, dim=-1)
        
        # Mean probabilities: [batch_size, num_classes]
        mean_probs = individual_probs.mean(dim=0)
        
        if uncertainty_type == "variance":
            # Variance of predicted class probabilities
            # Average variance across classes
            variance = individual_probs.var(dim=0).mean(dim=-1)  # [batch_size]
            return mean_probs, variance
        
        elif uncertainty_type == "entropy":
            # Mean entropy of ensemble predictions
            epsilon = 1e-10
            entropy = -(mean_probs * torch.log(mean_probs + epsilon)).sum(dim=-1)
            return mean_probs, entropy
        
        elif uncertainty_type == "mutual_info":
            # Mutual information = Total uncertainty - Aleatoric uncertainty
            # Total uncertainty: entropy of mean predictions
            epsilon = 1e-10
            total_entropy = -(mean_probs * torch.log(mean_probs + epsilon)).sum(dim=-1)
            
            # Aleatoric (data) uncertainty: mean of individual entropies
            individual_entropy = -(individual_probs * torch.log(individual_probs + epsilon)).sum(dim=-1)
            aleatoric_entropy = individual_entropy.mean(dim=0)
            
            # Mutual information (epistemic uncertainty)
            mutual_info = total_entropy - aleatoric_entropy
            return mean_probs, mutual_info
        
        else:
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
