# 🚀 Quick Start: Deep Ensembles

Get started with Deep Ensembles in 5 minutes!

## Prerequisites

Ensure your environment is set up and dependencies are installed (this project uses [uv](https://docs.astral.sh/uv/)):
```bash
uv sync
```

## Step 1: Train Your First Deep Ensemble (2 minutes)

### Option A: Using Default Settings

Train a 5-member ensemble with ResNet18:

```bash
uv run src/train.py model=deep_ensemble_classifier
```

This will:
- Create 5 ResNet18 models with different initializations
- Train them sequentially (each gets ~1/5 of total epochs)
- Save checkpoint with all ensemble members
- Log metrics to your configured logger (wandb, tensorboard, etc.)

### Option B: Quick Experiment (Fewer Epochs)

For testing, use fewer epochs:

```bash
uv run src/train.py \
    model=deep_ensemble_classifier \
    trainer.max_epochs=50 \
    model.num_estimators=3
```

This trains 3 members with ~17 epochs each.

## Step 2: Run Inference (1 minute)

After training, run inference on a test image:

```bash
uv run examples/03_deep_ensemble_inference.py \
    --checkpoint logs/train/runs/<your-run>/checkpoints/last.ckpt \
    --image path/to/test/image.jpg \
    --class-names class1 class2 class3 class4 class5 class6 class7 class8
```

You'll get:
- Predicted class with confidence
- Uncertainty score
- Predictions from individual ensemble members
- Visualization saved automatically

## Step 3: Compare with Other Methods (2 minutes)

Compare your deep ensemble with baseline and SNGP:

```bash
# First, train baseline and SNGP if you haven't already
uv run src/train.py model=baseline_classifier experiment=baseline_exp
uv run src/train.py model=sngp_classifier experiment=sngp_exp

# Then compare all three
uv run scripts/compare_methods.py \
    --baseline-ckpt checkpoints/baseline.ckpt \
    --sngp-ckpt checkpoints/sngp.ckpt \
    --ensemble-ckpt checkpoints/ensemble.ckpt \
    --data-config configs/data/your_data.yaml \
    --output-dir comparison_results/
```

This generates:
- Accuracy, ECE, and Brier score comparison
- Calibration curves for each method
- Uncertainty distribution plots
- CSV summary of results

## Common Customizations

### Change Backbone Architecture

```bash
uv run src/train.py \
    model=deep_ensemble_classifier \
    model.net.base_model_kwargs.arch=resnet50
```

Supported architectures: `resnet18`, `resnet34`, `resnet50`, `vit_b_16`, `vit_l_16`, etc.

### Adjust Number of Ensemble Members

```bash
uv run src/train.py \
    model=deep_ensemble_classifier \
    model.num_estimators=10 \
    trainer.max_epochs=200
```

More members = better uncertainty estimates but longer training.

### Change Uncertainty Type

```bash
uv run src/train.py \
    model=deep_ensemble_classifier \
    model.uncertainty_type=mutual_info
```

Options: `variance`, `entropy`, `mutual_info`

### Use Different Dataset

```bash
uv run src/train.py \
    model=deep_ensemble_classifier \
    data=your_dataset_config
```

## Expected Training Times

On a single GPU (e.g., NVIDIA RTX 3090):

| Configuration | Approximate Time |
|--------------|------------------|
| 5 members, ResNet18, 150 epochs | ~3-4 hours |
| 3 members, ResNet18, 90 epochs | ~1.5-2 hours |
| 5 members, ResNet50, 150 epochs | ~6-8 hours |
| 10 members, ResNet18, 200 epochs | ~8-10 hours |

*Times vary based on dataset size and hardware*

## Troubleshooting

### "Out of memory" error
- Use `train_strategy: sequential` in config (should be default)
- Reduce batch size: `data.datamodule.batch_size=16`
- Use smaller backbone: `model.net.base_model_kwargs.arch=resnet18`

### "Model not converging"
- Ensure each member gets enough epochs (total_epochs / num_estimators ≥ 20)
- Check learning rate: `model.optimizer.lr=0.0001`
- Verify data augmentations are appropriate

### "Checkpoint not loading"
- Ensure you're using `DeepEnsembleLitModule.load_from_checkpoint()`
- Check that `num_estimators` matches between training and loading

## Next Steps

📖 **Read the full guide**: [docs/DEEP_ENSEMBLES_GUIDE.md](DEEP_ENSEMBLES_GUIDE.md)

🔬 **Run experiments**: Create experiment configs in `configs/experiment/`

📊 **Analyze results**: Use notebooks for detailed uncertainty analysis

🔧 **Customize**: Extend the ensemble module for your specific needs

## Example Complete Workflow

```bash
# 1. Train deep ensemble
uv run src/train.py \
    model=deep_ensemble_classifier \
    data=acevedo_data \
    experiment=acevedo_ensemble \
    trainer.max_epochs=150

# 2. Evaluate on test set
uv run src/eval.py \
    model=deep_ensemble_classifier \
    data=acevedo_data \
    ckpt_path=logs/train/runs/acevedo_ensemble/checkpoints/best.ckpt

# 3. Run inference on single image
uv run examples/03_deep_ensemble_inference.py \
    --checkpoint logs/train/runs/acevedo_ensemble/checkpoints/best.ckpt \
    --image data/test/sample.jpg \
    --uncertainty-type variance \
    --save-viz results/prediction.png

# 4. Compare with other methods
uv run scripts/compare_methods.py \
    --baseline-ckpt checkpoints/baseline_best.ckpt \
    --sngp-ckpt checkpoints/sngp_best.ckpt \
    --ensemble-ckpt logs/train/runs/acevedo_ensemble/checkpoints/best.ckpt \
    --data-config configs/data/acevedo_data.yaml \
    --output-dir comparison_results/acevedo/
```

## Questions?

- 📖 Full documentation: [docs/DEEP_ENSEMBLES_GUIDE.md](DEEP_ENSEMBLES_GUIDE.md)
- 💬 Check existing issues in the repository
- 🔬 Explore example scripts in `examples/`

Happy ensembling! 🎉
