# CIFAR-100 / WideResNet-28-10 — SNGP vs. spectral regularization (uncalibrated)

Reproduction of the benchmark SNGP was published on (Liu et al. 2022,
[arXiv 2205.00403](https://arxiv.org/abs/2205.00403)), run so that the rep-spectral
variant can be judged against a *reproduced* SNGP number rather than only against the
Acevedo pilot ([ACEVEDO_SPECREG_RESULTS.md](ACEVEDO_SPECREG_RESULTS.md)). Recipe, every
deviation, and the two GP-head bugs the pilot caught:
[../models/CIFAR100_BENCHMARK.md](../models/CIFAR100_BENCHMARK.md).

Trained 2026-09-20/21 on branch `sngp-spectral-reg`, 250 epochs per run on one L40S
each. **Three seeds (12345, 1, 2)** for the three healthy arms; one seed for the
`c = 4.1` control and the rep-spectral-literal arm. W&B groups `CIFAR100` and
`CIFAR100_overnight_2026-09-20_21-38-42`. Checkpoints:
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

`±` means **across seeds** in the headline table below, and across bootstrap resamples
of one run in the single-seed tables further down. The two are not interchangeable.

---

## Headline — three seeds, epoch 249 (`last.ckpt`)

| Arm | seeds | Accuracy | NLL | smECE | AUROC vs CIFAR-10 | AUROC vs SVHN |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (deterministic) | 3 | **0.8061 ± 0.0027** | 0.8074 ± 0.0096 | 0.0779 ± 0.0038 | 0.8129 ± 0.0023 | 0.7503 ± 0.0388 |
| SNGP (`c = 6.0`) | 3 | 0.8025 ± 0.0045 | 0.7912 ± 0.0073 | 0.0685 ± 0.0042 | 0.8183 ± 0.0037 | 0.7961 ± 0.0057 |
| **SpecReg (matched)** | 3 | 0.8048 ± 0.0036 | **0.7739 ± 0.0107** | **0.0609 ± 0.0013** | **0.8206 ± 0.0029** | **0.8285 ± 0.0119** |
| SNGP (`c = 4.1`) | 1 | 0.8019 | 0.7965 | 0.0677 | 0.8089 | 0.7888 |
| SpecReg (rep-spectral literal) | 1 | 0.7372 | 1.2651 | 0.1363 | 0.7273 | 0.8564 |

*Mean ± std across seeds. AUROC is Dempster-Shafer, itself averaged over 10 bootstrap
resamples per run before the across-seed statistics.*

### Paired SpecReg(matched) − SNGP(`c = 6.0`), per seed

Paired rather than comparing the marginal means: the seeds are shared, so a per-seed
difference removes the run-to-run variation that dominates the `±` columns above.

| seed | Accuracy | NLL | smECE | AUROC C-10 | AUROC SVHN |
|---|---:|---:|---:|---:|---:|
| 12345 | +0.0074 | −0.0224 | −0.0121 | +0.0088 | +0.0259 |
| 1 | +0.0029 | −0.0264 | −0.0070 | −0.0037 | +0.0322 |
| 2 | −0.0034 | −0.0029 | −0.0038 | +0.0018 | +0.0393 |
| **mean** | +0.0023 | **−0.0173** | **−0.0076** | +0.0023 | **+0.0325** |
| sign-consistent | mixed | **3/3** | **3/3** | mixed | **3/3** |

**What survives three seeds.** Spectral regularization beats SNGP on **far-OOD detection
(SVHN, +0.033)**, **NLL (−0.017)** and **calibration (smECE −0.008)**, sign-consistent
across all three seeds; the SVHN per-seed gaps (+0.026, +0.032, +0.039) each exceed
SNGP's own across-seed std of 0.006.

**What does not.** Accuracy and near-OOD (CIFAR-10) are **mixed in sign** and within
noise — the three arms are indistinguishable on both. The first run's single-seed
CIFAR-10 lead for spectral regularization (+0.009) was an artifact.

**SNGP's own claim reproduces, and is about stability as much as level.** +0.046 over the
deterministic baseline on SVHN — and the baseline's across-seed std there is 0.0388
against SNGP's 0.0057, i.e. the deterministic model's far-OOD behaviour is erratic
(0.708 to 0.769 across seeds) while SNGP's is not.

**`c = 4.1` does not help** (SVHN 0.7888, CIFAR-10 0.8089 — both slightly *below*
`c = 6.0`), so the 1.46× reshaped-vs-operator estimator mismatch was **not** leaving the
SNGP arm under-constrained. That caveat from the first run is closed. One seed, so treat
it as a check that nothing was badly wrong rather than a tuning result.

**Ignore the literal arm's SVHN 0.8564.** It is the best OOD number in the table attached
to the worst model in the table (0.737 accuracy, 1.265 NLL, 0.136 smECE): a badly-fit
network whose logits are uniformly small separates ID from OOD on total evidence without
being useful for anything. A caution about reading OOD AUROC alone, not a result.

---

## Single-seed detail (seed 12345) — in-distribution, epoch 249

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

## Single-seed detail (seed 12345) — OOD detection, epoch 249

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

*Superseded by the three-seed table above; kept for provenance.* On this seed the
baseline's SVHN AUROC is 0.708, its worst of the three seeds — so the +8.6 pt SNGP margin
read here overstates the three-seed figure of +4.6. The +0.9 pt CIFAR-10 lead for
spectral regularization does not survive additional seeds at all.*

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

- **Three seeds, not more.** The surviving claims (SVHN OOD, NLL, smECE) are
  sign-consistent across all three, but three runs is a weak basis for a std. Accuracy
  and CIFAR-10 OOD are *not* established in either direction.
- **`c = 4.1` and the literal arm are single-seed.** Both are checks, not measurements.
- **The `c = 6.0` estimator mismatch is checked, not eliminated.** `c = 4.1` (the
  operator-norm equivalent of the reference's 6.0, given the measured 1.46× ratio) came
  out slightly *worse*, so the bound was not the limiting factor — but that is one run,
  and no intermediate value was tried.
- **No post-hoc calibration.** `mean_field_factor` is pinned at 7.5, not fitted, so the
  calibration columns are uncalibrated for every arm.
- **No figures yet** — reliability curves, DS histograms and the OOD-AUROC comparison
  plot still need generating via `src/visualization/`.
