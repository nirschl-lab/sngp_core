# CIFAR-100 / WideResNet-28-10 — SNGP vs. spectral regularization (uncalibrated)

Reproduction of the benchmark SNGP was published on (Liu et al. 2022,
[arXiv 2205.00403](https://arxiv.org/abs/2205.00403)), run so that the rep-spectral
variant can be judged against a *reproduced* SNGP number rather than only against the
Acevedo pilot ([ACEVEDO_SPECREG_RESULTS.md](ACEVEDO_SPECREG_RESULTS.md)). Recipe, every
deviation, and the two GP-head bugs the pilot caught:
[../models/CIFAR100_BENCHMARK.md](../models/CIFAR100_BENCHMARK.md).

Trained 2026-09-20 on branch `sngp-spectral-reg`, W&B group `CIFAR100`, four arms in
parallel on one L40S each, 250 epochs, **seed 12345 only**. Checkpoints:
[../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md). Inference
directories and every reproduce command:
[../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md).

**Off the fair-comparison protocol, deliberately** — SGD+Nesterov with a
warmup/piecewise schedule instead of AdamW+cosine, a fixed 250-epoch budget with no
early stopping, `val/loss` selection instead of `val/nll_cal`, the reference's CIFAR GP
constants, and **no post-hoc calibration** (`mean_field_factor` pinned at the
reference's 7.5, never fitted). Do not read these numbers next to the biomedical
protocol runs.

**Read the `last.ckpt` tables, not `best.ckpt`.** `val/loss` selected epoch 75 for the
baseline and 76 for SNGP — CE validation loss degrades after the first LR drop while
accuracy keeps climbing — but epoch 246 for spectral regularization, whose penalty
suppresses that degradation. The arms are therefore *not* comparable at `best.ckpt`, and
epoch 249 is the reference's own reporting point besides. Both are given; the gap
between them is itself a finding.

Single seed, so no error bars on any between-arm difference below. The `± ` on the AUROC
rows is the spread over the 10 bootstrap resamples of a *single* run, not run-to-run
variation.

---

## In-distribution — CIFAR-100 test, epoch 249 (`last.ckpt`)

| Arm | Accuracy | NLL | smECE | mean DS |
|---|---:|---:|---:|---:|
| Baseline (deterministic) | 0.8045 | 0.8067 | 0.0800 | 0.018 |
| SNGP (`c = 6.0`) | 0.7973 | 0.7995 | 0.0734 | 0.021 |
| **SpecReg (matched)** | **0.8047** | **0.7771** | **0.0613** | 0.023 |
| SpecReg (rep-spectral literal) | — | — | — | — |

*Published reference points: deterministic ~0.798 accuracy / 0.875 NLL, SNGP ~0.791.
Both reproduce within ~0.7 pt on accuracy and beat the published NLL, on 45k training
images instead of 50k. smECE is top-label (confidence vs. correct); the multiclass macro
one-vs-rest average reads ~0.003 for every arm at 100 classes and is not informative.
`—` on the literal arm means no epoch-249 checkpoint exists, not that it went unmeasured:
its `last.ckpt` was overwritten by a run-directory collision and is unrecoverable without
retraining — see [../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md).*

**At equal epoch the three arms are within 0.7 pt on accuracy.** Spectral regularization
separates on NLL (−0.030 vs SNGP) and calibration (smECE −0.012 vs SNGP, −0.019 vs
baseline), not on accuracy.

## OOD detection — Dempster-Shafer AUROC, epoch 249 (`last.ckpt`)

| Arm | vs CIFAR-10 (near) | vs SVHN (far) |
|---|---:|---:|
| Baseline (deterministic) | 0.8155 ± 0.0158 | 0.7076 ± 0.0130 |
| SNGP (`c = 6.0`) | 0.8152 ± 0.0115 | 0.7933 ± 0.0110 |
| **SpecReg (matched)** | **0.8239 ± 0.0134** | **0.8192 ± 0.0100** |
| SpecReg (rep-spectral literal) | — | — |

*`—` is "no epoch-249 checkpoint exists", not "not measured": the literal arm's
`last.ckpt` was overwritten by a run-directory collision and is unrecoverable without
retraining. See [../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md).
Its `best.ckpt` (epoch 200) is in the table further down.*

*Scored with Dempster-Shafer uncertainty `K / (K + Σ exp(logit))`, the score the
reference itself uses for CIFAR OOD (`dempster_shafer_ood` in `baselines/cifar/sngp.py`):
softmax is shift-invariant, so MSP and entropy discard exactly the logit magnitude an
SNGP head is trained to modulate. Mean ± std over 10 bootstrap resamples of 1000 ID and
1000 OOD rows.*

**SNGP's central claim reproduces**: +8.6 pts over the deterministic baseline on SVHN
(0.793 vs 0.708), landing on the paper's reported ~0.80. The baseline collapses on
far-OOD at the final epoch while both regularized models hold. Spectral regularization is
ahead of SNGP on both axes, by +2.6 pts (SVHN) and +0.9 pts (CIFAR-10).

---

## `best.ckpt` — for reference only, not a like-for-like comparison

| Arm | Epoch | Accuracy | NLL | smECE | CIFAR-10 AUROC | SVHN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 75 | 0.7800 | 0.7974 | 0.0625 | 0.7952 ± 0.0140 | 0.8659 ± 0.0097 |
| SNGP | 76 | 0.7777 | 0.7927 | 0.0604 | 0.7651 ± 0.0133 | 0.8502 ± 0.0090 |
| SpecReg (matched) | 246 | 0.8034 | 0.7753 | 0.0612 | 0.8232 ± 0.0134 | 0.8316 ± 0.0095 |
| SpecReg (literal) | 200 | 0.7404 | 1.2395 | 0.1370 | 0.7318 ± 0.0103 | 0.8600 ± 0.0075 |

*Spectral regularization's apparent +2.6 pt accuracy lead here is mostly the 246-vs-75
epoch gap, not the regularizer: it vanishes at equal epoch above. Conversely SNGP looks
*worse* than the baseline on OOD here only because epoch 76 precedes the distance-aware
behaviour the method depends on. That the penalty is what let this arm keep improving to
epoch 246 is a real property — but it is a different claim from "better at equal budget",
and this table cannot support either one on its own.*

## The rep-spectral-literal arm

`sngp_specreg_cifar100_literal` follows the rep-spectral paper's own recipe — 200 of 250
epochs unregularized, weight decay 0 so the penalty is the only weight regularizer — and
loses badly: 0.7404 accuracy, 1.2395 NLL, 0.1370 smECE at its selected epoch 200. Both
causes were predicted in the config header: 200 epochs with no weight decay overfits a
36M-parameter WRN, and by epoch 200 the LR has decayed to `0.04 × 0.2³`, so the penalty
arrives with almost no learning rate left to act through.

**The published rep-spectral recipe does not transfer to CIFAR-100 as written.** The
matched arm — same penalty, active from epoch 1, weight decay kept at the SNGP arm's
6e-4 — is where the method's benefit shows up.

---

## What these numbers do not establish

- **One seed per arm.** Every between-arm difference above is a single draw. The NLL and
  OOD gaps favouring spectral regularization are consistent across four independent
  measures (NLL, smECE, two OOD pairs), which is suggestive, but nothing here is an
  error bar on the method.
- **`spectral_norm_bound = 6.0` is not the reference's `c = 6.0`.** Ours constrains the
  reshaped Miyato norm; the reference constrains the true conv operator norm, measured
  1.46× larger on this backbone — so the SNGP arm is under a materially weaker
  constraint than the paper's. A `c ≈ 4.1` arm is the untested comparison.
- **No post-hoc calibration.** `mean_field_factor` is pinned at 7.5, not fitted, so the
  calibration columns are uncalibrated for every arm.
- **No figures yet** — reliability curves, DS histograms and the OOD-AUROC comparison
  plot still need generating via `src/visualization/`.
