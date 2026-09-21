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

All four arms were trained 2026-09-20 in parallel, one per L40S, 250 epochs, seed 12345,
W&B group `CIFAR100`. Results: [../results/CIFAR100_RESULTS.md](../results/CIFAR100_RESULTS.md).

> **Run-directory collision — read before using the spectral-reg checkpoints.** Both
> spectral-reg arms had `model.name: sngp_specreg_classifier`, so `task_name` was
> identical, and launching them in the same second gave them the *same* Hydra run
> directory. Two trainers wrote into one `checkpoints/`: Lightning suffixed the second
> `best` as `best-v1.ckpt`, and **the literal arm's `last.ckpt` was overwritten** by the
> matched arm's and is gone. The files below are identified by their `hyper_parameters`
> (`spec_reg_burnin_epochs`), not by filename order. Fixed for future runs by
> `model.name: sngp_specreg_literal_classifier` in
> `configs/experiment/sngp_specreg_cifar100_literal.yaml`; re-running the literal arm
> would give it a clean directory and recover its `last.ckpt`.

## baseline_cifar100 — deterministic WRN-28-10

```bash
# best.ckpt  (epoch 75, min val/loss)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best.ckpt
# last.ckpt  (epoch 249 -- the reference reporting point)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt
```

## sngp_cifar100 — SNGP, reference recipe (`spectral_norm_bound` 6.0)

```bash
# best.ckpt  (epoch 76, min val/loss)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best.ckpt
# last.ckpt  (epoch 249)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt
```

## sngp_specreg_cifar100 — spectral regularization, matched to the SNGP arm

`spec_reg_coef` 0.01, burn-in 1, weight decay 6e-4.

```bash
# best.ckpt  (epoch 246, min val/loss over epochs >= burn-in)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best.ckpt
# last.ckpt  (epoch 249)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt
```

## sngp_specreg_cifar100_literal — rep-spectral paper-literal (burn-in 200, wd 0)

Shares the directory above; **its checkpoint is the `-v1` one** (`spec_reg_burnin_epochs:
200`). Its `last.ckpt` does not exist — see the collision note.

```bash
# best.ckpt  (epoch 200, the first rankable epoch)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best-v1.ckpt
```

Verify any entry before use:

```bash
uv run python -c "import torch; ck=torch.load('<path>', map_location='cpu', weights_only=False); \
print(ck['epoch'], ck['hyper_parameters'].get('spec_reg_burnin_epochs'))"
```
