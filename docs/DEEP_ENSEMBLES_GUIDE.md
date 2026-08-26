# Deep Ensembles Guide

How to train, tune, and evaluate the Deep Ensemble model family. For where Deep
Ensembles fit relative to Baseline/SNGP (config, LightningModule, shared contracts),
see [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md).

## Background

Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017) estimate predictive
uncertainty by training multiple networks from different random initializations and
combining their predictions:

- Trains multiple neural networks with different random initializations.
- Averages predictions at inference time for improved accuracy.
- Provides uncertainty estimates through prediction/logit variance across members.
- Requires no architectural changes to the base model.

## Architecture Overview

Three components:

1. **`DeepEnsemble`** (`src/models/ensemble/deep_ensemble_model.py`) — wraps
   `num_estimators` independently-initialized member nets (any registered net as
   `base_model_spec`), handles ensemble forward/predictions and uncertainty
   quantification.
2. **`DeepEnsembleLitModule`** (`src/models/deep_ensemble_lit_module.py`) —
   LightningModule managing sequential training of ensemble members, integrated with
   the project's shared metrics/logging/checkpoint hooks (`LitModuleBase`).
3. **Configuration** (`configs/model/deep_ensemble_classifier.yaml`) — the Hydra
   config for `model=deep_ensemble_classifier`.

All three nets (Baseline, SNGP, Deep Ensemble) return the same `ModelOutput` shape
and register into the same `NET_REGISTRY` — an ensemble member is just another net
built from a `base_model_spec`, so an ensemble of SNGP members, for example, needs no
new code, only `base_model_spec.name: sngp_classifier`.

## Quick Start

Train a 5-member ensemble with ResNet18 (bare `model=` composition, no experiment
preset):

```bash
uv run src/train.py model=deep_ensemble_classifier
```

Or use one of the existing experiment presets, which also set a sane
`trainer.max_epochs` for the member count and dataset-appropriate hyperparameters:

```bash
uv run src/train.py experiment=deep_ensemble_acevedo
# also: deep_ensemble_tang, deep_ensemble_wong, deep_ensemble_kather2018
```

Customize member count or backbone:

```bash
uv run src/train.py model=deep_ensemble_classifier model.num_estimators=10
uv run src/train.py model=deep_ensemble_classifier model.net.base_model_spec.arch=resnet50
```

## Configuration Options

Key parameters in `deep_ensemble_classifier.yaml`:

```yaml
num_estimators: 5              # Number of ensemble members
train_strategy: sequential     # Only "sequential" is implemented; anything else
                                # raises NotImplementedError at construction time

net:
  base_model_spec:
    name: baseline_classifier # Net registry key (see src/models/registry.py)
    arch: resnet18            # Backbone architecture
    num_classes: 8            # Number of output classes
    dropout_p: 0.2            # Dropout probability
    pretrained: true          # Use ImageNet pretrained weights
```

- **`sequential`** (the only implemented strategy): trains one ensemble member at a
  time, cycling across epochs — epochs are divided equally among members. More
  memory-efficient than training all members at once.
- **`all`**: not implemented — `DeepEnsembleLitModule` raises `NotImplementedError`
  at construction time if you set this. Simultaneous multi-member training within one
  run is a future TODO; for true parallelism today, see [Parallel Multi-GPU
  Training](#parallel-multi-gpu-training) below.

### Uncertainty types

The live training/test path reports ensemble disagreement generically via
`ModelOutput.variance` (variance of per-member logits) as `test/uncertainty_mean` —
the same field every uncertainty-aware family (SNGP included) populates. No
per-run configuration is needed or available for this.

For offline/notebook analysis with a specific uncertainty formulation, call
`DeepEnsemble.get_predictive_uncertainty(x, uncertainty_type=...)` directly on the
net (see [Programmatic Usage](#programmatic-usage) below) — it supports:

1. **`variance`** — variance of predicted probabilities across the ensemble.
   Captures disagreement between members; higher variance = more uncertainty.
2. **`entropy`** — entropy of the mean prediction. Total uncertainty (aleatoric +
   epistemic).
3. **`mutual_info`** — mutual information (total entropy − aleatoric entropy).
   Epistemic (model) uncertainty only; useful for active learning.

## Training Schedule

### Sequential (single-run, default)

With 5 members and 150 total epochs:

- Epochs 0–29: member 1
- Epochs 30–59: member 2
- Epochs 60–89: member 3
- Epochs 90–119: member 4
- Epochs 120–149: member 5

Set `trainer.max_epochs` so each member gets adequate training:

```bash
# 5 members, 30 epochs each = 150 total epochs
uv run src/train.py model=deep_ensemble_classifier trainer.max_epochs=150

# 10 members, 20 epochs each = 200 total epochs
uv run src/train.py model=deep_ensemble_classifier \
    model.num_estimators=10 \
    trainer.max_epochs=200
```

Approximate single-GPU wall-clock time (varies with dataset size/hardware):

| Configuration | Approximate time |
|---|---|
| 5 members, ResNet18, 150 epochs | ~3–4 hours |
| 3 members, ResNet18, 90 epochs | ~1.5–2 hours |
| 5 members, ResNet50, 150 epochs | ~6–8 hours |
| 10 members, ResNet18, 200 epochs | ~8–10 hours |

Since this path trains members one at a time in a single run, total time scales
roughly linearly with `num_estimators` — see [Parallel Multi-GPU
Training](#parallel-multi-gpu-training) to train members concurrently instead.

## Parallel Multi-GPU Training

An alternative to sequential single-run cycling: train all N members as fully
independent, concurrent `train.py` runs across one or more GPUs, then assemble their
checkpoints into one ensemble checkpoint afterward.

```bash
# 5 members of the baseline_acevedo experiment, auto-detecting/using available GPUs
scripts/ensemble/train_members_parallel.sh baseline_acevedo 5

# pin to specific GPUs
scripts/ensemble/train_members_parallel.sh baseline_acevedo 3 0 2 3

# pack 6 members onto 2 GPUs (round-robin: member i -> GPU_IDS[i % len(GPU_IDS)])
scripts/ensemble/train_members_parallel.sh baseline_acevedo 6 0 1
```

Each member is trained as an ordinary `baseline_*` experiment run, pinned to one GPU
via `CUDA_VISIBLE_DEVICES` and given a distinct `seed=<member index>` for diversity
(mirroring `DeepEnsemble`'s own per-member seeding convention). With no GPU ids
given, it uses `CUDA_VISIBLE_DEVICES` if already set (e.g. a SLURM allocation) or
auto-detects via `nvidia-smi -L`.

Useful env vars:
- `MEMBER_EXTRA_OVERRIDES="<hydra overrides>"` — forwarded to every member's
  `train.py` call, e.g. to shrink per-member dataloader workers when packing
  multiple members per GPU: `MEMBER_EXTRA_OVERRIDES="data.datamodule.num_workers=8"`.
- `DRY_RUN=1` — print the GPU-packing plan and exit without launching anything.

Members land under
`train/<model.name>_<data.name>/ensemble_members/<run_id>/member_<i>/` (each an
ordinary Hydra run dir, plus a `member_<i>.log` capturing stdout/stderr) — see
[OUTPUT_LAYOUT.md §6](OUTPUT_LAYOUT.md#6-ensemble-member-parallel-training-outputs)
for the full path derivation.

Once all members finish, combine their checkpoints into one `DeepEnsemble`
checkpoint:

```bash
uv run scripts/ensemble/assemble_ensemble_checkpoint.py \
    --members-dir <members-root-printed-above> \
    --out <members-root>/ensemble.ckpt

# or list member checkpoints explicitly, in member order:
uv run scripts/ensemble/assemble_ensemble_checkpoint.py \
    --ckpts m0/checkpoints/best.ckpt m1/checkpoints/best.ckpt m2/checkpoints/best.ckpt \
    --out ensemble.ckpt
```

This validates that every member checkpoint is a `baseline_classifier` run with
matching architecture and source dataset, then writes a checkpoint satisfying the
same contract (CLAUDE.md §3) as one produced by a real `DeepEnsembleLitModule`
training run — it loads through `src/checkpointing/io.py` and works with
`src/eval.py`/`src/inference/infer.py` unmodified, with no separate code path needed.

On a SLURM cluster, `scripts/slurm/train_ensemble_members.sh
<baseline_experiment> <num_estimators>` is the `sbatch` wrapper around the same
script (default `--gres=gpu:4`, override on the `sbatch` command line for a
different allocation).

## Evaluation & Testing

```bash
uv run src/eval.py model=deep_ensemble_classifier ckpt_path=path/to/checkpoint.ckpt
```

`src/eval.py`'s `trainer.test()` run computes final metrics via
`TestArtifactsCallback` (`src/callbacks/test_artifacts_callback.py`), not the
LightningModule itself. Logged metrics include:
- `test/acc_final`, `test/precision_final`, `test/recall_final`, `test/f1_final`
- `test/uncertainty_mean`: mean of `ModelOutput.variance` (per-member logit variance)
  across the test set
- Per-sample logits/probs/predictions/variance in the prediction CSV, when
  `callbacks.test_artifacts.log_csv=true`

## Run Inference on a Single Image

`examples/03_deep_ensemble_inference.py` loads a checkpoint through
`src/checkpointing/io.py`'s `load_lit_module` and runs a single-image prediction with
per-member breakdown and an uncertainty estimate:

```bash
uv run examples/03_deep_ensemble_inference.py \
    --checkpoint <log_dir>/train/deep_ensemble_classifier_acevedo/runs/<run_id>/checkpoints/best.ckpt \
    --image path/to/test/image.jpg \
    --class-names class1 class2 class3 class4 class5 class6 class7 class8 \
    --uncertainty-type variance \
    --save-viz results/prediction.png
```

`--uncertainty-type` accepts `variance`, `entropy`, or `mutual_info` (see
[Uncertainty types](#uncertainty-types) above); `--save-viz` is optional. Output
includes the ensemble-averaged prediction with confidence, the uncertainty score,
and each individual member's prediction.

For the project's canonical, dataset/artifact-aware inference entrypoint (sweeps,
batch inference, metrics), use `src/inference/infer.py` per the `inference` skill /
[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) instead — this example script is a minimal,
single-image reference, not a second inference entrypoint.

## Comparing with Other Methods

```bash
# Baseline
uv run src/train.py experiment=baseline_acevedo

# SNGP
uv run src/train.py experiment=sngp_acevedo

# Deep Ensemble
uv run src/train.py experiment=deep_ensemble_acevedo
```

`scripts/compare_methods.py` evaluates all three on the same test set and compares
accuracy, calibration (ECE, Brier score), and uncertainty quality:

```bash
uv run scripts/compare_methods.py \
    --baseline-ckpt path/to/baseline/checkpoints/best.ckpt \
    --sngp-ckpt path/to/sngp/checkpoints/best.ckpt \
    --ensemble-ckpt path/to/ensemble/checkpoints/best.ckpt \
    --data-config configs/data/acevedo.yaml \
    --output-dir comparison_results/
```

This generates a CSV summary plus calibration-curve and uncertainty-distribution
plots. For the project's standard offline/research metrics (cross-dataset OOD AUROC,
smooth-ECE, Dempster-Shafer uncertainty) driven from prediction CSVs, use the
`metrics` skill instead.

## Tips & Best Practices

### Training time
- Sequential (single-run) training takes ~N times longer than one model, where N =
  `num_estimators`. Use [Parallel Multi-GPU Training](#parallel-multi-gpu-training)
  to train members concurrently instead when multiple GPUs are available.

### Memory usage
- Sequential training uses similar memory to a single model (one member active at a
  time).
- All members are loaded simultaneously during inference/eval.

### Checkpoint management
- A sequentially-trained checkpoint contains all ensemble members; checkpoint size ≈
  N × single-model size.
- An assembled checkpoint (see Parallel Multi-GPU Training) is built directly from
  each member's own `best.ckpt`, so no extra retention decision is needed there.

### Number of members
- 5 members is a good default (the original paper uses 5–10).
- More members generally improve uncertainty quality with diminishing returns.
- 3 members can work for quick experiments; 10+ for critical applications.

### Optimization
- Use the same hyperparameters (LR, weight decay) as the baseline model — no special
  tricks are needed. Each member should converge to similar accuracy independently.

## Advanced Usage

### Custom base model

Ensemble members are built through the shared net registry
(`src/models/registry.py`), the same mechanism every net family uses — see the
`add-model` skill. Register a model, then reference it by name from the ensemble
config:

```python
# my_custom_model.py
from src.models.outputs import ModelOutput
from src.models.registry import register_net

@register_net("my_custom_model")
class MyCustomModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # architecture here

    def forward(self, x) -> ModelOutput:
        # forward pass here
        return ModelOutput(logits=logits)
```

```yaml
net:
  base_model_spec:
    name: my_custom_model
    num_classes: 8
```

### Programmatic usage

```python
from src.models.ensemble import DeepEnsemble

ensemble = DeepEnsemble(
    base_model_spec={
        'name': 'baseline_classifier',
        'arch': 'resnet18',
        'num_classes': 8,
        'dropout_p': 0.2,
        'pretrained': True,
    },
    num_estimators=5,
    task='classification',
)

# Inference
ensemble.eval()
with torch.no_grad():
    probs, uncertainty = ensemble.get_predictive_uncertainty(images)
```

## References

```bibtex
@inproceedings{lakshminarayanan2017simple,
  title={Simple and scalable predictive uncertainty estimation using deep ensembles},
  author={Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6402--6413},
  year={2017}
}
```

## Troubleshooting

**Out of memory**: confirm `train_strategy: sequential` is set (trains one member at
a time), or reduce `data.datamodule.batch_size`/switch to a smaller backbone.

**Low diversity in predictions**: verify each member gets sufficient training epochs;
try different dropout values or augmentation strategies. Member seeding is automatic
and already differs per member.

**Checkpoint won't load**: all checkpoint I/O goes through
`src/checkpointing/io.py` (`load_lit_module`/`load_net`/`read_meta`) — never a bare
`torch.load` or `LightningModule.load_from_checkpoint()`, per CLAUDE.md's checkpoint
contract. If loading a sequentially-trained checkpoint, confirm `num_estimators`
matches between training and loading; if loading an assembled checkpoint, confirm it
was built by `assemble_ensemble_checkpoint.py` from checkpoints with matching
architecture and dataset.

**Slow inference**: ensemble inference requires N forward passes. Consider reducing
`num_estimators` for deployment, or distributing member forward passes across GPUs.
