# CIFAR-100 benchmark checkpoints

Checkpoints for the CIFAR-100 / WideResNet-28-10 reproduction of the SNGP benchmark.
What the runs are and how to reproduce them:
[../models/CIFAR100_BENCHMARK.md](../models/CIFAR100_BENCHMARK.md). Entry point for every
other dataset's checkpoints: [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md).

> **Convention for this page, which differs from the master doc.** These runs have
> **no `best.calibrated.ckpt`** — the CIFAR arms skip the post-hoc calibration pass and
> pin `mean_field_factor` at the reference's CIFAR-100 value of 7.5 instead. Record two
> checkpoints per run and report both:
>
> - **`best.ckpt`** — lowest `val/loss` (for the spectral-reg arms, ranked only over
>   epochs ≥ the burn-in, via `ModelCheckpointFromEpoch`). The primary.
> - **`last.ckpt`** — epoch 249, which is what the reference implementation reports.
>
> Everything on this page is **off the biomedical protocol** (SGD + piecewise schedule,
> `val/loss` selection, no early stopping, reference GP constants). Do not report these
> numbers next to the protocol runs without saying so.

Run directories follow the standard layout
([../OUTPUT_LAYOUT.md](../OUTPUT_LAYOUT.md)):

```
${EXPERIMENTS_HOME}/${PROJECT_NAME}/train/<model.name>_cifar100/runs/<timestamp>/checkpoints/
```

so the four arms land under `train/baseline_classifier_cifar100`,
`train/sngp_classifier_cifar100` and `train/sngp_specreg_classifier_cifar100` (both
spectral-reg arms share that last one — tell them apart by timestamp and by the W&B run
name, `cifar100_sngp_specreg_wrn28x10` vs `cifar100_sngp_specreg_literal_wrn28x10`).

Verify any entry before using it:

```bash
uv run python -c "from src.checkpointing.io import read_meta; print(read_meta('<path>'))"
# net_spec.arch must be 'wide_resnet28_10'; num_classes 100
```

---

## baseline_cifar100 — deterministic WRN-28-10

_pending — run not yet launched_

```bash
# best.ckpt:
# last.ckpt:
# W&B run:
```

## sngp_cifar100 — SNGP, reference recipe (`spectral_norm_bound` 6.0)

_pending — run not yet launched_

```bash
# best.ckpt:
# last.ckpt:
# W&B run:
```

## sngp_specreg_cifar100 — spectral regularization, matched to the SNGP arm

_pending — run not yet launched. `spec_reg_coef` is set from the 20-epoch pilot; record
the value used here alongside the paths, since the config's 0.01 is a placeholder
inherited from the rep-spectral paper's ResNet18 setting._

```bash
# best.ckpt:
# last.ckpt:
# spec_reg_coef:
# W&B run:
```

## sngp_specreg_cifar100_literal — rep-spectral paper-literal (burn-in 200, wd 0)

_pending — run not yet launched_

```bash
# best.ckpt:
# last.ckpt:
# W&B run:
```
