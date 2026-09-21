# CIFAR-100 / WideResNet-28-10 — SNGP vs. spectral regularization (uncalibrated)

Reproduction of the benchmark SNGP was published on (Liu et al. 2022,
[arXiv 2205.00403](https://arxiv.org/abs/2205.00403)), run so that the rep-spectral
variant can be judged against a *reproduced* SNGP number rather than only against the
Acevedo pilot ([ACEVEDO_SPECREG_RESULTS.md](ACEVEDO_SPECREG_RESULTS.md)).

| | |
|---|---|
| Runs | 250 epochs, one L40S per arm, 2026-09-20/21, branch `sngp-spectral-reg` |
| Seeds | 12345 / 1 / 2 — baseline, SNGP `c = 6.0`, SpecReg (matched)<br>12345 only — SNGP `c = 4.1`, rep-spectral literal |
| W&B | groups `CIFAR100` and `CIFAR100_overnight_2026-09-20_21-38-42` |
| Recipe, deviations, GP-head bugs | [../models/CIFAR100_BENCHMARK.md](../models/CIFAR100_BENCHMARK.md) |
| Checkpoints | [../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md) |
| Inference dirs, reproduce commands | [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md) |

## Read this first

1. **Off the fair-comparison protocol, deliberately** — SGD+Nesterov with a
   warmup/piecewise schedule instead of AdamW+cosine; a fixed 250-epoch budget, no early
   stopping; `val/loss` selection instead of `val/nll_cal`; the reference's CIFAR GP
   constants; **no post-hoc calibration** (`mean_field_factor` pinned at the reference's
   7.5, never fitted). Do not read these numbers next to the biomedical protocol runs.
2. **Compare the arms at epoch 249 (`last.ckpt`), never at `best.ckpt`.** `val/loss`
   selects epoch 75 / 76 / 246 for baseline / SNGP / SpecReg: CE validation loss degrades
   after the first LR drop while accuracy keeps climbing, and the spectral penalty
   suppresses exactly that degradation. Epoch 249 is also the reference's own reporting
   point. The `best.ckpt` table below is kept for reference only.
3. **`±` means two different things** — across seeds in the headline table, across 10
   bootstrap resamples of one run in the seed-12345 tables. Not interchangeable.
4. **Metrics.** smECE is top-label (confidence vs. correct); the macro one-vs-rest
   average reads ~0.003 for every arm at 100 classes and says nothing. AUROC uses
   Dempster-Shafer uncertainty `K / (K + Σ exp(logit))`, the reference's own OOD score
   (`dempster_shafer_ood` in `baselines/cifar/sngp.py`) — softmax is shift-invariant, so
   MSP and entropy discard the logit magnitude an SNGP head is trained to modulate.

---

## Headline — three seeds, epoch 249 (`last.ckpt`)

| Arm | seeds | Accuracy | NLL | smECE | AUROC vs CIFAR-10 | AUROC vs SVHN |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (deterministic) | 3 | **0.8061 ± 0.0027** | 0.8074 ± 0.0096 | 0.0779 ± 0.0038 | 0.8129 ± 0.0023 | 0.7503 ± 0.0388 |
| SNGP (`c = 6.0`) | 3 | 0.8025 ± 0.0045 | 0.7912 ± 0.0073 | 0.0685 ± 0.0042 | 0.8183 ± 0.0037 | 0.7961 ± 0.0057 |
| **SpecReg (matched)** | 3 | 0.8048 ± 0.0036 | **0.7739 ± 0.0107** | **0.0609 ± 0.0013** | **0.8206 ± 0.0029** | **0.8285 ± 0.0119** |
| SNGP (`c = 4.1`) | 1 | 0.8019 | 0.7965 | 0.0677 | 0.8089 | 0.7888 |
| SpecReg (rep-spectral literal) | 1 | 0.7372 | 1.2651 | 0.1363 | 0.7273 | 0.8564 |

*Mean ± std across seeds; each run's AUROC is itself averaged over 10 bootstrap resamples
first. The literal arm's row comes from its clean re-run, not from the seed-12345 run
detailed below.*

### Paired SpecReg (matched) − SNGP (`c = 6.0`), per seed

Paired rather than differencing the marginal means: the seeds are shared, so a per-seed
difference removes the run-to-run variation that dominates the `±` columns above.

| seed | Accuracy | NLL | smECE | AUROC C-10 | AUROC SVHN |
|---|---:|---:|---:|---:|---:|
| 12345 | +0.0074 | −0.0224 | −0.0121 | +0.0088 | +0.0259 |
| 1 | +0.0029 | −0.0264 | −0.0070 | −0.0037 | +0.0322 |
| 2 | −0.0034 | −0.0029 | −0.0038 | +0.0018 | +0.0393 |
| **mean** | +0.0023 | **−0.0173** | **−0.0076** | +0.0023 | **+0.0325** |
| sign-consistent | mixed | **3/3** | **3/3** | mixed | **3/3** |

### What three seeds support

- **Survives.** SpecReg beats SNGP on **far-OOD (SVHN, +0.033)**, **NLL (−0.017)** and
  **calibration (smECE −0.008)**, sign-consistent 3/3. The per-seed SVHN gaps (+0.026,
  +0.032, +0.039) each exceed SNGP's own across-seed std of 0.006.
- **Does not survive.** Accuracy and near-OOD (CIFAR-10) are mixed in sign and within
  noise — the three arms are indistinguishable on both. The first run's single-seed
  CIFAR-10 lead for SpecReg (+0.009) was an artifact.
- **The reproduction holds.** Baseline 0.806 and SNGP 0.803 against published ~0.798 /
  ~0.791, on 45k training images instead of 50k.
- **SNGP's own claim reproduces — stability as much as level.** +0.046 over the baseline
  on SVHN, and the baseline's across-seed std there is 0.0388 against SNGP's 0.0057: the
  deterministic model swings 0.708–0.769 across seeds, SNGP does not.
- **`c = 4.1` does not help** (SVHN 0.7888, CIFAR-10 0.8089 — both slightly *below*
  `c = 6.0`), so the 1.46× reshaped-vs-operator estimator mismatch was **not** leaving
  the SNGP arm under-constrained. That caveat from the first run is closed. One seed — a
  check, not a tuning result.
- **Ignore the literal arm's SVHN 0.8564.** The best OOD number in the table belongs to
  its worst model (0.737 accuracy, 1.265 NLL, 0.136 smECE): a badly-fit network with
  uniformly small logits separates ID from OOD on total evidence while being useless for
  anything else. A caution about reading OOD AUROC alone, not a result.

---

## `best.ckpt` — reference only, not a like-for-like comparison

| Arm | Epoch | Accuracy | NLL | smECE | CIFAR-10 AUROC | SVHN AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 75 | 0.7800 | 0.7974 | 0.0625 | 0.7952 ± 0.0140 | 0.8659 ± 0.0097 |
| SNGP | 76 | 0.7777 | 0.7927 | 0.0604 | 0.7651 ± 0.0133 | 0.8502 ± 0.0090 |
| SpecReg (matched) | 246 | 0.8034 | 0.7753 | 0.0612 | 0.8232 ± 0.0134 | 0.8316 ± 0.0095 |
| SpecReg (literal) | 200 | 0.7404 | 1.2395 | 0.1370 | 0.7318 ± 0.0103 | 0.8600 ± 0.0075 |

*SpecReg's apparent +2.6 pt accuracy lead here is mostly the 246-vs-75 epoch gap, not the
regularizer — it vanishes at equal epoch. SNGP looks worse than the baseline on OOD only
because epoch 76 precedes the distance-aware behaviour the method depends on. That the
penalty let the matched arm keep improving to epoch 246 is a real property, but it is a
different claim from "better at equal budget", and this table supports neither on its
own.*

## The rep-spectral-literal arm

**The published rep-spectral recipe does not transfer to CIFAR-100 as written.**
`sngp_specreg_cifar100_literal` keeps that recipe — 200 of 250 epochs unregularized,
weight decay 0 so the penalty is the only weight regularizer — and loses badly: 0.7404
accuracy, 1.2395 NLL, 0.1370 smECE at its selected epoch 200. Both causes were predicted
in the config header: 200 epochs with no weight decay overfits a 36M-parameter WRN, and
by epoch 200 the LR has decayed to `0.04 × 0.2³`, leaving the penalty almost no learning
rate to act through. The matched arm — same penalty, active from epoch 1, weight decay
kept at the SNGP arm's 6e-4 — is where the method's benefit shows up.

---

## Seed 12345 in detail — superseded by the three-seed tables, kept for provenance

In-distribution, epoch 249:

| Arm | Accuracy | NLL | smECE | mean DS |
|---|---:|---:|---:|---:|
| Baseline (deterministic) | 0.8045 | 0.8067 | 0.0800 | 0.018 |
| SNGP (`c = 6.0`) | 0.7973 | 0.7995 | 0.0734 | 0.021 |
| **SpecReg (matched)** | **0.8047** | **0.7771** | **0.0613** | 0.023 |
| SpecReg (rep-spectral literal) | — | — | — | — |

*Published reference points: deterministic ~0.798 accuracy / 0.875 NLL, SNGP ~0.791. This
seed reproduces both within ~0.7 pt on accuracy and beats the published NLL, on 45k
training images instead of 50k.*

OOD detection, epoch 249:

| Arm | vs CIFAR-10 (near) | vs SVHN (far) |
|---|---:|---:|
| Baseline (deterministic) | 0.8155 ± 0.0158 | 0.7076 ± 0.0130 |
| SNGP (`c = 6.0`) | 0.8152 ± 0.0115 | 0.7933 ± 0.0110 |
| **SpecReg (matched)** | **0.8239 ± 0.0134** | **0.8192 ± 0.0100** |
| SpecReg (rep-spectral literal) | — | — |

*± is over 10 bootstrap resamples of 1000 ID and 1000 OOD rows from this one run.*

- **At equal epoch the three arms are within 0.7 pt on accuracy.** SpecReg separates on
  NLL (−0.030 vs SNGP) and calibration (smECE −0.012 vs SNGP, −0.019 vs baseline), not on
  accuracy.
- **This seed flatters SNGP on OOD.** Its baseline SVHN AUROC of 0.708 is the worst of
  the three seeds, so the +8.6 pt SNGP margin read here overstates the three-seed +4.6.
  The +0.9 pt CIFAR-10 lead for SpecReg does not survive more seeds at all.
- **`—` is "no epoch-249 checkpoint", not "not measured".** This run's literal arm lost
  its `last.ckpt` to a run-directory collision
  ([../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md)); its
  `best.ckpt` (epoch 200) is in the table above, and the headline's epoch-249 row for
  this arm comes from the clean re-run.

---

## What these numbers do not establish

- **Three seeds, not more.** The surviving claims (SVHN OOD, NLL, smECE) are
  sign-consistent across all three, but three runs is a weak basis for a std. Accuracy
  and CIFAR-10 OOD are *not* established in either direction.
- **`c = 4.1` and the literal arm are single-seed.** Both are checks, not measurements.
- **The `c = 6.0` estimator mismatch is checked, not eliminated.** `c = 4.1` (the
  operator-norm equivalent of 6.0, given the measured 1.46× ratio) came out slightly
  *worse*, so the bound was not the limiting factor — but that is one run, and no
  intermediate value was tried.
- **No post-hoc calibration.** `mean_field_factor` is pinned at 7.5, not fitted, so the
  calibration columns are uncalibrated for every arm.
- **No figures yet** — reliability curves, DS histograms and the OOD-AUROC comparison
  plot still need generating via `src/visualization/`.
