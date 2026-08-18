# Deep Ensembles Implementation Guide

This guide explains how to use the Deep Ensemble implementation for uncertainty quantification in your experiments.

## 📚 Background

Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017) is a simple yet effective method for uncertainty estimation that:
- Trains multiple neural networks with different random initializations
- Averages predictions at inference time for improved accuracy
- Provides uncertainty estimates through prediction variance
- Requires no architectural changes to your base model

## 🏗️ Architecture Overview

The implementation consists of three main components:

1. **`DeepEnsemble`** (`src/models/ensemble/deep_ensemble_model.py`)
   - Wraps multiple base model instances
   - Handles ensemble predictions and uncertainty quantification
   - Supports different uncertainty metrics (variance, entropy, mutual information)

2. **`DeepEnsembleLitModule`** (`src/models/ensemble/deep_ensemble_lit_module.py`)
   - Lightning module for training and evaluation
   - Manages sequential training of ensemble members
   - Integrates with existing metrics and logging

3. **Configuration** (`configs/model/deep_ensemble_classifier.yaml`)
   - Hydra config for easy experimentation
   - Compatible with existing training infrastructure

## 🚀 Quick Start

### 1. Basic Training

Train a deep ensemble with 5 members using ResNet18:

```bash
python src/train.py model=deep_ensemble_classifier
```

### 2. Customize Number of Ensemble Members

```bash
python src/train.py model=deep_ensemble_classifier model.num_estimators=10
```

### 3. Use Different Backbone

```bash
python src/train.py model=deep_ensemble_classifier \
    model.net.base_model_spec.arch=resnet50
```

### 4. Change Uncertainty Type

```bash
# Use entropy-based uncertainty
python src/train.py model=deep_ensemble_classifier \
    model.uncertainty_type=entropy

# Or mutual information (epistemic uncertainty)
python src/train.py model=deep_ensemble_classifier \
    model.uncertainty_type=mutual_info
```

## ⚙️ Configuration Options

### Key Parameters in `deep_ensemble_classifier.yaml`:

```yaml
num_estimators: 5              # Number of ensemble members
train_strategy: sequential     # How to train: "sequential" or "all"
uncertainty_type: variance     # Uncertainty metric: "variance", "entropy", "mutual_info"

net:
  base_model_spec:
    name: baseline_classifier # Net registry key (see src/models/registry.py)
    arch: resnet18            # Backbone architecture
    num_classes: 8            # Number of output classes
    dropout_p: 0.2            # Dropout probability
    pretrained: true          # Use ImageNet pretrained weights
```

### Training Strategies:

- **`sequential`** (recommended): Trains one ensemble member at a time across epochs
  - More memory efficient
  - Better for limited GPU memory
  - Epochs are divided equally among members
  
- **`all`** (experimental): Trains all members simultaneously
  - Faster if you have enough memory
  - May require gradient accumulation

### Uncertainty Types:

1. **`variance`**: Variance of predicted probabilities across ensemble
   - Captures disagreement between models
   - Higher variance = more uncertainty

2. **`entropy`**: Entropy of mean prediction
   - Total uncertainty (aleatoric + epistemic)
   - Standard information-theoretic measure

3. **`mutual_info`**: Mutual information (Total entropy - Aleatoric entropy)
   - Epistemic (model) uncertainty only
   - Useful for active learning

## 📊 Training Schedule

### Sequential Training (Recommended)

With 5 ensemble members and 150 epochs total:

- Epochs 0-29: Train member 1
- Epochs 30-59: Train member 2
- Epochs 60-89: Train member 3
- Epochs 90-119: Train member 4
- Epochs 120-149: Train member 5

Adjust `trainer.max_epochs` to ensure each member gets adequate training:

```bash
# For 5 members, 30 epochs each = 150 total epochs
python src/train.py model=deep_ensemble_classifier trainer.max_epochs=150

# For 10 members, 20 epochs each = 200 total epochs
python src/train.py model=deep_ensemble_classifier \
    model.num_estimators=10 \
    trainer.max_epochs=200
```

## 🧪 Evaluation & Testing

### Run Evaluation

```bash
python src/eval.py model=deep_ensemble_classifier \
    ckpt_path=path/to/checkpoint.ckpt
```

### Access Uncertainty Estimates

During testing, the model automatically computes and logs:
- Mean predictions across ensemble
- Uncertainty scores per sample
- Average uncertainty across test set

Logged metrics:
- `test/acc`: Ensemble accuracy
- `test/loss`: Ensemble loss
- `test/uncertainty_mean`: Average uncertainty

## 📈 Comparing with Other Methods

### Example: Compare Baseline, SNGP, and Deep Ensemble

```bash
# Baseline
python src/train.py model=baseline_classifier experiment=my_baseline

# SNGP
python src/train.py model=sngp_classifier experiment=my_sngp

# Deep Ensemble
python src/train.py model=deep_ensemble_classifier experiment=my_ensemble
```

Use your existing evaluation scripts to compare:
- Accuracy
- Calibration (ECE, Brier score)
- Uncertainty quality (AUROC, AUPR)
- Out-of-distribution detection

## 💡 Tips & Best Practices

### 1. Training Time
- Deep Ensembles take ~N times longer to train (where N = num_estimators)
- Plan accordingly: 5 members ≈ 5× training time
- Consider training members in parallel on different GPUs (requires custom script)

### 2. Memory Usage
- Sequential training uses similar memory to single model
- Each member needs to be loaded during inference
- For large models, consider keeping members on CPU and moving to GPU one at a time

### 3. Checkpoint Management
- The checkpoint contains all ensemble members
- Checkpoint size ≈ N × single model size
- Consider saving best performing members only

### 4. Number of Members
- 5 members is a good default (original paper uses 5-10)
- More members = better uncertainty, but diminishing returns
- 3 members can work well for quick experiments
- 10+ members for critical applications

### 5. Optimization
- Use same hyperparameters (LR, weight decay) as baseline model
- No special tricks needed - simplicity is the strength of deep ensembles
- Each member should converge to similar accuracy

## 🔍 Advanced Usage

### Custom Base Model

Ensemble members are built through the shared net registry (`src/models/registry.py`),
the same mechanism every net family uses — see the `add-model` skill. Register your
model, then reference it by name from the ensemble config:

```python
# my_custom_model.py
from src.models.outputs import ModelOutput
from src.models.registry import register_net

@register_net("my_custom_model")
class MyCustomModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Your architecture here

    def forward(self, x) -> ModelOutput:
        # Your forward pass
        return ModelOutput(logits=logits)
```

Update config:
```yaml
net:
  base_model_spec:
    name: my_custom_model
    num_classes: 8
```

### Programmatic Usage

```python
from src.models.ensemble import DeepEnsemble

# Create ensemble
ensemble = DeepEnsemble(
    base_model_spec={
        'name': 'baseline_classifier',
        'arch': 'resnet18',
        'num_classes': 8,
        'dropout_p': 0.2,
        'pretrained': True
    },
    num_estimators=5,
    task='classification'
)

# Training loop
for member_idx in range(5):
    ensemble.set_active_member(member_idx)
    # Train this member...
    
# Inference
ensemble.eval()
with torch.no_grad():
    probs, uncertainty = ensemble.get_predictive_uncertainty(images)
```

## 📚 References

```bibtex
@inproceedings{lakshminarayanan2017simple,
  title={Simple and scalable predictive uncertainty estimation using deep ensembles},
  author={Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6402--6413},
  year={2017}
}
```

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Ensure `train_strategy: sequential` is set, which trains one member at a time.

### Issue: Low Diversity in Predictions
**Solution**: 
- Verify each member is initialized with different random seeds (automatic)
- Ensure sufficient training per member
- Try different dropout values or augmentation strategies

### Issue: Checkpoint Loading Fails
**Solution**: Make sure the checkpoint was saved with the same `num_estimators` configuration.

### Issue: Slow Inference
**Solution**: Ensemble inference requires N forward passes. Consider:
- Reducing `num_estimators` for deployment
- Implementing parallel inference across multiple GPUs
- Using knowledge distillation to create a single model

## 📞 Support

For questions or issues:
1. Check the existing baseline and SNGP implementations for reference
2. Review Lightning and Hydra documentation
3. Open an issue in the repository

Happy ensembling! 🎉
