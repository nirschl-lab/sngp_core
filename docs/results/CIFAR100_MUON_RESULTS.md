# CIFAR-100 / WideResNet-28-10: GP head on an unconstrained backbone, trained with Muon

The CIFAR-100 follow-up of [ACEVEDO_MUON_RESULTS.md](ACEVEDO_MUON_RESULTS.md). The SNGP GP head
sits on a WRN-28-10 with **no spectral normalization and no spectral penalty**, trained with Muon
(`src/models/components/optimizers.py`) at the evidence-picked ℓ = 7.

| | |
|---|---|
| Arms | `sngp_muon_cifar100` (piecewise LR, the benchmark's schedule) and `sngp_muon_cifar100_wsd` (warmup-stable-decay: 1 warmup epoch, flat to epoch 175, linear to 1/75 at 249), each at Muon weight decay 0 and 0.1. Muon lr 0.02 on the 27 hidden convs; AdamW (lr 1e-3, wd 0.01) on the stem, BatchNorm and GP output layer. Head: ℓ 7, σ² 1, ridge 1.0, unscaled features, `gaussian` |
| Runs | seed 12345 only, 250 epochs, `last.ckpt`; [../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md), section "muon". The other 5 rows are the [online ℓ page](CIFAR100_ONLINE_LS_RESULTS.md)'s runs (3 seeds) |
| Protocol | One post-hoc knob per row, fit on **val** NLL (λ for SNGP arms, T for the baseline); metrics on **test**. `±` is the std across 3 seeds; the Muon rows are one seed and carry none |
| Data, commands | [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md), section "muon" |

## Results

| Arm | Acc | NLL | smECE | knob\* (val) |
|---|---:|---:|---:|---:|
| Baseline | **0.8061 ± 0.0027** | 0.7574 ± 0.0072 | 0.0364 ± 0.0007 | T 1.31 |
| SNGP (`c = 6.0`) | 0.8025 ± 0.0045 | 0.7600 ± 0.0065 | 0.0348 ± 0.0022 | λ 32.5 |
| SNGP + SpecReg | 0.8048 ± 0.0036 | **0.7472 ± 0.0100** | 0.0299 ± 0.0041 | λ 35.3 |
| SNGP + SpecReg + evidence ℓ = 7 | 0.7978 ± 0.0017 | 0.7513 ± 0.0062 | 0.0253 ± 0.0023 | λ 43.1 |
| SNGP + SpecReg + online evidence ℓ | 0.7999 ± 0.0003 | 0.7566 ± 0.0039 | 0.0274 ± 0.0011 | λ 49.1 |
| GP head, no SN, Muon wd 0 (piecewise) | 0.7626 | 0.8855 | 0.0173 | λ 191.1 |
| GP head, no SN, Muon wd 0.1 (piecewise) | 0.7729 | 0.8282 | 0.0157 | λ 113.5 |
| GP head, no SN, Muon wd 0 (WSD) | 0.7537 | 0.9129 | 0.0168 | λ 240.4 |
| GP head, no SN, Muon wd 0.1 (WSD) | 0.7462 | 0.9003 | **0.0127** | λ 63.5 |

| Arm | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN | Var C-10 | Var SVHN | FPR95 MSP SVHN ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | **0.8101 ± 0.0023** | 0.7309 ± 0.0373 | **0.8121 ± 0.0020** | 0.7496 ± 0.0383 | — | — | 0.855 ± 0.025 |
| SNGP (`c = 6.0`) | 0.8059 ± 0.0031 | 0.7620 ± 0.0053 | 0.8082 ± 0.0029 | 0.8047 ± 0.0032 | **0.437 ± 0.012** | **0.580 ± 0.018** | 0.828 ± 0.011 |
| SNGP + SpecReg | 0.8092 ± 0.0016 | 0.7948 ± 0.0060 | 0.8110 ± 0.0020 | 0.8330 ± 0.0068 | 0.314 ± 0.001 | 0.410 ± 0.046 | 0.794 ± 0.023 |
| SNGP + SpecReg + evidence ℓ = 7 | 0.7948 ± 0.0014 | 0.8016 ± 0.0269 | 0.7846 ± 0.0036 | 0.8571 ± 0.0230 | 0.336 ± 0.007 | 0.449 ± 0.030 | 0.778 ± 0.036 |
| SNGP + SpecReg + online evidence ℓ | 0.7866 ± 0.0066 | 0.7667 ± 0.0113 | 0.7618 ± 0.0127 | 0.8012 ± 0.0143 | 0.294 ± 0.010 | 0.297 ± 0.044 | 0.824 ± 0.017 |
| GP head, no SN, Muon wd 0 (piecewise) | 0.7615 | 0.8120 | 0.7135 | 0.8777 | 0.370 | 0.427 | 0.757 |
| GP head, no SN, Muon wd 0.1 (piecewise) | 0.7685 | 0.7507 | 0.7321 | 0.8441 | 0.410 | 0.416 | 0.796 |
| GP head, no SN, Muon wd 0 (WSD) | 0.7624 | 0.8249 | 0.7229 | 0.8853 | 0.388 | 0.567 | 0.729 |
| GP head, no SN, Muon wd 0.1 (WSD) | 0.7601 | **0.8301** | 0.7205 | **0.8919** | 0.431 | 0.470 | **0.699** |

Paired at seed 12345, each Muon row minus SNGP + SpecReg + evidence ℓ = 7 (same ℓ), each at its
own λ\*:

| | Acc | NLL | smECE | MSP C-10 | MSP SVHN | DS SVHN | Var C-10 | Var SVHN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Muon wd 0 (piecewise) | −0.0372 | +0.1383 | −0.0095 | −0.0349 | +0.0382 | +0.0467 | +0.0420 | −0.0546 |
| Muon wd 0.1 (piecewise) | −0.0269 | +0.0811 | −0.0110 | −0.0279 | −0.0231 | +0.0131 | +0.0819 | −0.0658 |
| Muon wd 0 (WSD) | −0.0461 | +0.1657 | −0.0099 | −0.0340 | +0.0512 | +0.0544 | +0.0595 | +0.0852 |
| Muon wd 0.1 (WSD) | −0.0536 | +0.1531 | −0.0141 | −0.0363 | +0.0563 | +0.0610 | +0.1027 | −0.0115 |

WSD minus piecewise, same Muon wd, seed 12345:

| | Acc | NLL | smECE | MSP C-10 | MSP SVHN | DS SVHN | Var C-10 | Var SVHN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wd 0 | −0.0089 | +0.0274 | −0.0004 | +0.0009 | +0.0130 | +0.0076 | +0.0175 | +0.1398 |
| wd 0.1 | −0.0267 | +0.0721 | −0.0030 | −0.0084 | +0.0793 | +0.0479 | +0.0208 | +0.0543 |

- **In-distribution: worse than every SGD row.** Accuracy is down 2.7–5.4 points and NLL up
  0.08–0.17 against evidence ℓ = 7. The val-fitted λ\* is 63–240, against 33–49 for the SGD
  rows, so the trained logits are far more overconfident. smECE is the lowest of any row only
  after that large correction.
- **Near-OOD (CIFAR-10): worse on every row.** Against evidence ℓ = 7, MSP is −0.03 and DS
  −0.05 to −0.07.
- **Far-OOD (SVHN): three of the four Muon rows beat every SGD row.** Their DS SVHN is 0.878–0.892
  and MSP SVHN 0.812–0.830, against 0.857 ± 0.023 and 0.802 ± 0.027 for evidence ℓ = 7. The
  exception is piecewise wd 0.1 (DS 0.844, MSP 0.751). The best FPR95 is 0.70 (WSD wd 0.1),
  against 0.78. The GP variance is not the source: Var SVHN is 0.42–0.57, no better than
  spectral-norm SNGP's 0.58.
- **WSD did not help.** At both weight decays it lowered accuracy and raised NLL against
  piecewise, while raising SVHN AUROC.

Caveats: one seed per Muon row. The seed spread of DS SVHN is ±0.02–0.04 for the SGD rows. The
Muon recipe was not tuned: lr and momentum are the reference defaults, and the AdamW group's
weight decay (lr × wd = 1e-5) is effectively off, unlike the SGD recipe's L2 6e-4 on every
weight.
