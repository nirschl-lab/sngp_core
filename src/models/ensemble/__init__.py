"""Deep Ensemble net. The training-strategy LightningModule lives at
`src.models.deep_ensemble_lit_module.DeepEnsembleLitModule`, alongside the other
family lit modules -- not nested here, for consistency with baseline/sngp."""

from src.models.ensemble.deep_ensemble_model import DeepEnsemble

__all__ = ["DeepEnsemble"]
