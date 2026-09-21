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
3. **`±` is across training seeds, always.** Nothing here is resampled. Single-seed
   arms get a bare number.
4. **Metrics.** smECE is top-label (confidence vs. correct); the macro one-vs-rest
   average reads ~0.003 for every arm at 100 classes and says nothing.
5. **OOD AUROC follows the paper's protocol** (arXiv 2205.00403 §6.2.1, appendix C.1):
   the **full** CIFAR-100 test set (10,000 rows) against the **full** OOD test set
   (CIFAR-10 10,000, SVHN 26,032), no subsampling, one deterministic number per run.
   Unequal group sizes are fine — AUROC is a rank statistic estimating
   `P(score_OOD > score_ID)`. Two scores are reported:
   - **MSP**, `max_k softmax(g_k(x))` — what the paper's own tables use, so these are
     the rows to read against its published numbers. For SNGP arms this is MSP of the
     mean-field-adjusted posterior, which is what the paper reports, but
     `mean_field_factor` is pinned at 7.5 and never fitted (see caveat 1).
   - **Dempster-Shafer**, `K / (K + Σ exp(logit))` — what the reference
     *implementation* uses (`dempster_shafer_ood` in `baselines/cifar/sngp.py`). Kept
     because softmax is shift-invariant, so MSP is blind to exactly the logit magnitude
     an SNGP head is trained to modulate.

   Earlier versions of this page reported DS over 10 fixed-seed subsamples of 1000
   rows per frame. Those numbers are within ~0.003 of the full-population ones, so
   nothing here rests on the change — but they are not the paper's protocol, and the
   `±` they carried described resampling noise rather than run-to-run variation.

---

## Headline — three seeds, epoch 249 (`last.ckpt`)

MSP, the paper's own score:

| Arm | seeds | Accuracy | NLL | smECE | MSP AUROC vs CIFAR-10 | MSP AUROC vs SVHN |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (deterministic) | 3 | **0.8061 ± 0.0027** | 0.8074 ± 0.0096 | 0.0779 ± 0.0038 | 0.8056 ± 0.0024 | 0.7235 ± 0.0378 |
| SNGP (`c = 6.0`) | 3 | 0.8025 ± 0.0045 | 0.7912 ± 0.0073 | 0.0685 ± 0.0042 | 0.8070 ± 0.0033 | 0.7483 ± 0.0057 |
| **SpecReg (matched)** | 3 | 0.8048 ± 0.0036 | **0.7739 ± 0.0107** | **0.0609 ± 0.0013** | **0.8111 ± 0.0014** | **0.7857 ± 0.0080** |
| SNGP (`c = 4.1`) | 1 | 0.8019 | 0.7965 | 0.0677 | 0.8026 | 0.7374 |
| SpecReg (rep-spectral literal) | 1 | 0.7372 | 1.2651 | 0.1363 | 0.7332 | 0.7490 |

Dempster-Shafer companion:

| Arm | seeds | DS AUROC vs CIFAR-10 | DS AUROC vs SVHN |
|---|---:|---:|---:|
| Baseline (deterministic) | 3 | 0.8129 ± 0.0020 | 0.7487 ± 0.0382 |
| SNGP (`c = 6.0`) | 3 | 0.8162 ± 0.0037 | 0.7932 ± 0.0045 |
| **SpecReg (matched)** | 3 | **0.8198 ± 0.0019** | **0.8267 ± 0.0114** |
| SNGP (`c = 4.1`) | 1 | 0.8097 | 0.7891 |
| SpecReg (rep-spectral literal) | 1 | 0.7282 | 0.8549 |

*Mean ± std across the three training seeds (12345 / 1 / 2). The literal arm's row comes
from its clean re-run, not from the seed-12345 run detailed below. Regenerate both with
`scripts/metrics/cifar100_overnight_report.py --include-seed-12345`.*

### Paired SpecReg (matched) − SNGP (`c = 6.0`), per seed

Paired rather than differencing the marginal means: the seeds are shared, so a per-seed
difference removes the run-to-run variation that dominates the `±` columns above.

| seed | Accuracy | NLL | smECE | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12345 | +0.0074 | −0.0224 | −0.0121 | +0.0082 | +0.0263 | +0.0086 | +0.0249 |
| 1 | +0.0029 | −0.0264 | −0.0070 | −0.0001 | +0.0427 | −0.0020 | +0.0346 |
| 2 | −0.0034 | −0.0029 | −0.0038 | +0.0043 | +0.0432 | +0.0040 | +0.0410 |
| **mean** | +0.0023 | **−0.0173** | **−0.0076** | +0.0041 | **+0.0374** | +0.0035 | **+0.0335** |
| sign-consistent | mixed | **3/3** | **3/3** | mixed | **3/3** | mixed | **3/3** |

### What three seeds support

- **Survives, and is larger under the paper's own score.** SpecReg beats SNGP on
  **far-OOD (SVHN)** by **+0.037 under MSP** and +0.033 under DS, sign-consistent 3/3
  under both, alongside **NLL (−0.017)** and **calibration (smECE −0.008)**. The
  per-seed MSP SVHN gaps (+0.026, +0.043, +0.043) each exceed SNGP's own across-seed
  std of 0.006. This is the one claim the score choice cannot be accused of carrying.
- **Does not survive.** Accuracy and near-OOD (CIFAR-10) are mixed in sign and within
  noise under both scores — the three arms are indistinguishable on both. Under MSP the
  seed-1 near-OOD delta is −0.0001, i.e. exactly nothing.
- **Near-OOD reproduces; far-OOD does not.** Under MSP our near-OOD numbers (baseline
  0.806, SNGP 0.807) sit ~1 pt above the published 0.795 / 0.798 — a good reproduction.
  Far-OOD is 7–10 pt *below* published (ours 0.724 / 0.748 vs 0.799 / 0.846). Accuracy
  reproduces (0.806 / 0.803 vs ~0.798 / ~0.791) on 45k training images instead of 50k,
  so this is not a broken model; SVHN AUROC is simply the high-variance axis here, and
  the published figure averages 10 seeds against our 3. It is **not** the OOD sample
  count — that is ruled out below. **Treat the absolute far-OOD level as unreproduced
  and read only the between-arm gaps.**
- **SNGP's own claim reproduces on stability, weakly on level.** Its SVHN margin over
  the baseline is **+0.025 under MSP** but +0.044 under DS — so the headline benefit is
  partly a property of the score, not only of the method. What holds under both is the
  *spread*: the baseline's across-seed std on SVHN is ~0.038 against SNGP's ~0.005, so
  the deterministic model swings across seeds where SNGP does not.
- **`c = 4.1` does not help** (MSP SVHN 0.7374, CIFAR-10 0.8026 — both *below*
  `c = 6.0`), so the 1.46× reshaped-vs-operator estimator mismatch was **not** leaving
  the SNGP arm under-constrained. That caveat from the first run is closed. One seed — a
  check, not a tuning result.
- **The literal arm's DS SVHN 0.8549 is an artifact of the score, and MSP shows it.**
  Under DS it posts the best far-OOD number in the table; under MSP it drops to 0.7490,
  mid-pack. The model behind it is the worst one here (0.737 accuracy, 1.265 NLL, 0.136
  smECE): a badly-fit network with uniformly small logits separates ID from OOD on
  *total evidence* while being useless for anything else, and DS reads total evidence
  where MSP does not. The clearest case on this page for reporting both scores.

### Does the SVHN sample count explain the gap to published? No.

SVHN's test split is 26,032 rows against the 10,000 of the CIFAR splits the paper's other
columns use, which is the obvious suspect for the far-OOD shortfall above. Capping SVHN at
10,000 rules it out — across all three arms, three seeds and both scores, ten independent
draws each:

| Arm | Score | SVHN full (26,032) | 10K draws (mean) | Δ |
|---|---|---:|---:|---:|
| Baseline | MSP | 0.7235 ± 0.0378 | 0.7234 ± 0.0379 | −0.0001 |
| SNGP (`c = 6.0`) | MSP | 0.7483 ± 0.0057 | 0.7480 ± 0.0056 | −0.0003 |
| SpecReg (matched) | MSP | 0.7857 ± 0.0080 | 0.7855 ± 0.0080 | −0.0002 |
| Baseline | DS | 0.7487 ± 0.0382 | 0.7484 ± 0.0383 | −0.0002 |
| SNGP (`c = 6.0`) | DS | 0.7932 ± 0.0045 | 0.7929 ± 0.0044 | −0.0003 |
| SpecReg (matched) | DS | 0.8267 ± 0.0114 | 0.8264 ± 0.0116 | −0.0002 |

Every arm moves by ≤ 0.0003; per-run spread across the ten draws is 0.0026–0.0066, so even
a single unlucky draw lands within ~0.004 of the full-population value. This is what the
estimator predicts rather than a surprise: AUROC is a rank statistic estimating
`P(score_OOD > score_ID)`, so discarding 60% of the OOD rows at random costs precision,
not position — the same reason unequal group sizes are fine in the first place.

The full split stays the reported number: same answer, lower variance, no seed.
Reproduce with `scripts/metrics/cifar100_svhn_subsample_check.py`.

---

## `best.ckpt` — reference only, not a like-for-like comparison

Seed 12345 only, so no `±`.

| Arm | Epoch | Accuracy | NLL | smECE | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 75 | 0.7800 | 0.7974 | 0.0625 | 0.7789 | 0.7780 | 0.7963 | 0.8638 |
| SNGP | 76 | 0.7777 | 0.7927 | 0.0604 | 0.7663 | 0.7160 | 0.7656 | 0.8476 |
| SpecReg (matched) | 246 | 0.8034 | 0.7753 | 0.0612 | 0.8106 | 0.7861 | 0.8212 | 0.8292 |
| SpecReg (literal) | 200 | 0.7404 | 1.2395 | 0.1370 | 0.7380 | 0.7425 | 0.7319 | 0.8573 |

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

| Arm | MSP C-10 (near) | MSP SVHN (far) | DS C-10 | DS SVHN |
|---|---:|---:|---:|---:|
| Baseline (deterministic) | 0.8071 | 0.6804 | 0.8150 | 0.7059 |
| SNGP (`c = 6.0`) | 0.8045 | 0.7521 | 0.8134 | 0.7919 |
| **SpecReg (matched)** | **0.8127** | **0.7784** | **0.8220** | **0.8167** |
| SpecReg (rep-spectral literal) | — | — | — | — |

*One run, so no `±`.*

- **At equal epoch the three arms are within 0.7 pt on accuracy.** SpecReg separates on
  NLL (−0.030 vs SNGP) and calibration (smECE −0.012 vs SNGP, −0.019 vs baseline), not on
  accuracy.
- **This seed flatters SNGP on OOD.** Its baseline SVHN AUROC is the worst of the three
  seeds under either score (MSP 0.680, DS 0.706), so the SNGP margin read here (+7.2 pt
  MSP, +8.6 pt DS) overstates the three-seed +2.5 / +4.4. The CIFAR-10 lead for SpecReg
  does not survive more seeds at all.
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
- **The absolute far-OOD level is not reproduced.** MSP SVHN comes out 7–10 pt below the
  published numbers for both the baseline and SNGP, on 45k training images against 50k
  and 3 seeds against 10. The OOD sample count is *not* the cause — capping SVHN at
  10,000 moves every arm by ≤ 0.0003 (see above) — so the remaining candidates are the
  training-set size, the seed count, or a real difference in the arm. Between-arm gaps on
  this axis are consistent and survive both scores; the level does not, so do not quote
  it as a reproduction of the paper.
- **`c = 4.1` and the literal arm are single-seed.** Both are checks, not measurements.
- **The `c = 6.0` estimator mismatch is checked, not eliminated.** `c = 4.1` (the
  operator-norm equivalent of 6.0, given the measured 1.46× ratio) came out slightly
  *worse*, so the bound was not the limiting factor — but that is one run, and no
  intermediate value was tried.
- **No post-hoc calibration.** `mean_field_factor` is pinned at 7.5, not fitted, so the
  calibration columns are uncalibrated for every arm.
- **No figures yet** — reliability curves, DS histograms and the OOD-AUROC comparison
  plot still need generating via `src/visualization/`.
