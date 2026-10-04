# CIFAR-100 / WideResNet-28-10: GP head on an unconstrained backbone, trained with Muon

The CIFAR-100 follow-up of [ACEVEDO_MUON_RESULTS.md](ACEVEDO_MUON_RESULTS.md). The SNGP GP head
sits on a WRN-28-10 with **no spectral normalization and no spectral penalty on the convs**. It is trained
with Muon on the 27 hidden convs (`src/models/components/optimizers.py`), at the evidence-picked ℓ = 7.
The rows below vary:
- the LR schedule;
- the optimizer on the remaining parameters (the stem, BatchNorm and GP output layer);
- a cap on BatchNorm's gain;
- a LayerNorm on the GP input;
- the head: a linear-head Baseline per optimizer, with no GP;
- the GP length scale ℓ, without the LayerNorm ([section below](#length-scale-sweep-no-layernorm)).

All are set against the SGD-trained SNGP / SpecReg references.

| | |
|---|---|
| Optimizer | `[SGD]`: SGD-Nesterov (0.04, L2 6e-4) on every weight. `[Muon + AdamW]` / `[Muon + SGD]`: Muon (lr 0.02, momentum 0.95, decoupled wd 0 or 0.1) on the hidden convs, with AdamW (lr 1e-3, wd 0.01) or the `[SGD]` recipe on the stem / BN / GP head (`MuonWithAuxAdamW` / `MuonWithAuxSGD`) |
| Schedule | piecewise (the benchmark's ×0.2 at epochs 75 / 150 / 200, 1 warmup epoch); WSD (1 warmup epoch, flat to 175, linear to 1/75 at 249); cosine (`CosineAnnealingLR` to 0, no warmup — the Acevedo schedule) |
| BN spectral norm | `SpectralBatchNorm2d` (DUE, van Amersfoort et al. 2021): every BN's gain max_i \|γ_i\|/√(running_var_i+ε) capped at 3. A BN spectral-*regularization* run (loss + 0.01·Σ_l gain²) trained unstably and is not evaluated |
| Head | σ² 1, ridge 1.0, unscaled features, `gaussian`; ℓ as in the ℓ column (20 is the benchmark recipe; "online" is the [online ℓ page](CIFAR100_ONLINE_LS_RESULTS.md)'s in-training evidence ℓ). GP input not normalized (the reference's `gp_input_normalization=False`), except the GP-input LayerNorm row (`normalize_input: true`, ℓ = 20 so that ‖h‖/ℓ ≈ √640/20 ≈ 1.26 at init) |
| Runs | 250 epochs, `last.ckpt`. The five piecewise `[SGD]` rows are 3 seeds (12345 / 1 / 2); every other row is seed 12345 only. [../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md), sections "muon" and "cosine" |
| Protocol | One post-hoc knob per row, fit on **val** NLL (λ for SNGP arms, T for the baseline); metrics on **test**. `±` is the std across 3 seeds. Bold is the best value per column across all rows |
| Data, commands | [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md), sections "muon" and "cosine" |

## Results

| Arm | Optimizer | Schedule | ℓ | Seeds | Acc | NLL | smECE | knob\* (val) |
|---|---|---|---|---|---:|---:|---:|---:|
| Baseline | [SGD] | piecewise | — | 3 | **0.8061 ± 0.0027** | 0.7574 ± 0.0072 | 0.0364 ± 0.0007 | T 1.31 |
| SNGP (`c = 6.0`) | [SGD] | piecewise | 20 | 3 | 0.8025 ± 0.0045 | 0.7600 ± 0.0065 | 0.0348 ± 0.0022 | λ 32.5 |
| SNGP + SpecReg | [SGD] | piecewise | 20 | 3 | 0.8048 ± 0.0036 | 0.7472 ± 0.0100 | 0.0299 ± 0.0041 | λ 35.3 |
| SNGP + SpecReg + evidence ℓ | [SGD] | piecewise | 7 | 3 | 0.7978 ± 0.0017 | 0.7513 ± 0.0062 | 0.0253 ± 0.0023 | λ 43.1 |
| SNGP + SpecReg + online evidence ℓ | [SGD] | piecewise | online | 3 | 0.7999 ± 0.0003 | 0.7566 ± 0.0039 | 0.0274 ± 0.0011 | λ 49.1 |
| Baseline | [SGD] | cosine | — | 1 | 0.8047 | 0.7691 | 0.0392 | T 1.23 |
| SNGP (`c = 6.0`) | [SGD] | cosine | 7 | 1 | 0.7993 | 0.7697 | 0.0343 | λ 34.3 |
| SNGP + SpecReg | [SGD] | cosine | 7 | 1 | 0.8052 | **0.7427** | 0.0276 | λ 35.9 |
| GP head, no SN, wd 0 | [Muon + AdamW] | piecewise | 7 | 1 | 0.7626 | 0.8855 | 0.0173 | λ 191.1 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | piecewise | 7 | 1 | 0.7729 | 0.8282 | 0.0157 | λ 113.5 |
| GP head, no SN, wd 0 | [Muon + AdamW] | WSD | 7 | 1 | 0.7537 | 0.9129 | 0.0168 | λ 240.4 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | WSD | 7 | 1 | 0.7462 | 0.9003 | **0.0127** | λ 63.5 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | cosine | 7 | 1 | 0.7635 | 0.8501 | 0.0144 | λ 95.0 |
| Baseline, wd 0.1 | [Muon + SGD] | cosine | — | 1 | 0.7963 | 0.7658 | 0.0318 | T 1.51 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 7 | 1 | 0.7906 | 0.7572 | 0.0242 | λ 68.7 |
| GP head, no SN, wd 0 | [Muon + SGD] | cosine | 7 | 1 | 0.7903 | 0.8355 | 0.0354 | λ 29.7 |
| GP head, no SN, wd 0.1, BN spectral norm `c = 3` | [Muon + SGD] | cosine | 7 | 1 | 0.7793 | 0.9184 | 0.1220 | λ 0.0 |
| GP head, no SN, wd 0.1, GP-input LayerNorm | [Muon + SGD] | cosine | 20 | 1 | 0.7888 | 0.7789 | 0.0276 | λ 52.2 |

| Arm | Optimizer | Schedule | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN | Var C-10 | Var SVHN | FPR95 MSP SVHN ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | [SGD] | piecewise | **0.8101 ± 0.0023** | 0.7309 ± 0.0373 | 0.8121 ± 0.0020 | 0.7496 ± 0.0383 | — | — | 0.855 ± 0.025 |
| SNGP (`c = 6.0`) | [SGD] | piecewise | 0.8059 ± 0.0031 | 0.7620 ± 0.0053 | 0.8082 ± 0.0029 | 0.8047 ± 0.0032 | 0.437 ± 0.012 | 0.580 ± 0.018 | 0.828 ± 0.011 |
| SNGP + SpecReg | [SGD] | piecewise | 0.8092 ± 0.0016 | 0.7948 ± 0.0060 | 0.8110 ± 0.0020 | 0.8330 ± 0.0068 | 0.314 ± 0.001 | 0.410 ± 0.046 | 0.794 ± 0.023 |
| SNGP + SpecReg + evidence ℓ | [SGD] | piecewise | 0.7948 ± 0.0014 | 0.8016 ± 0.0269 | 0.7846 ± 0.0036 | 0.8571 ± 0.0230 | 0.336 ± 0.007 | 0.449 ± 0.030 | 0.778 ± 0.036 |
| SNGP + SpecReg + online evidence ℓ | [SGD] | piecewise | 0.7866 ± 0.0066 | 0.7667 ± 0.0113 | 0.7618 ± 0.0127 | 0.8012 ± 0.0143 | 0.294 ± 0.010 | 0.297 ± 0.044 | 0.824 ± 0.017 |
| Baseline | [SGD] | cosine | 0.8096 | 0.7500 | 0.8157 | 0.7593 | — | — | 0.859 |
| SNGP (`c = 6.0`) | [SGD] | cosine | 0.7987 | 0.7578 | 0.7995 | 0.8198 | 0.393 | 0.595 | 0.841 |
| SNGP + SpecReg | [SGD] | cosine | 0.8032 | 0.7984 | 0.7967 | 0.8557 | 0.315 | 0.353 | 0.800 |
| GP head, no SN, wd 0 | [Muon + AdamW] | piecewise | 0.7615 | 0.8120 | 0.7135 | 0.8777 | 0.370 | 0.427 | 0.757 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | piecewise | 0.7685 | 0.7507 | 0.7321 | 0.8441 | 0.410 | 0.416 | 0.796 |
| GP head, no SN, wd 0 | [Muon + AdamW] | WSD | 0.7624 | 0.8249 | 0.7229 | 0.8853 | 0.388 | 0.567 | 0.729 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | WSD | 0.7601 | **0.8301** | 0.7205 | **0.8919** | 0.431 | 0.470 | **0.699** |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | cosine | 0.7724 | 0.7793 | 0.7353 | 0.8674 | 0.416 | 0.450 | 0.750 |
| Baseline, wd 0.1 | [Muon + SGD] | cosine | 0.8095 | 0.7458 | 0.8118 | 0.7927 | — | — | 0.822 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 0.7862 | 0.7465 | 0.7556 | 0.8328 | 0.351 | 0.376 | 0.808 |
| GP head, no SN, wd 0 | [Muon + SGD] | cosine | 0.8044 | 0.8281 | 0.8114 | 0.8523 | 0.297 | 0.430 | 0.736 |
| GP head, no SN, wd 0.1, BN spectral norm `c = 3` | [Muon + SGD] | cosine | 0.8075 | 0.8257 | **0.8139** | 0.8591 | 0.212 | 0.498 | 0.792 |
| GP head, no SN, wd 0.1, GP-input LayerNorm | [Muon + SGD] | cosine | 0.7942 | 0.7887 | 0.7758 | 0.8535 | **0.661** | **0.607** | 0.745 |

Paired at seed 12345 (row − row), each row at its own λ\*:

| | Acc | NLL | smECE | MSP C-10 | MSP SVHN | DS SVHN | Var C-10 | Var SVHN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Muon vs SGD at ℓ = 7* | | | | | | | | |
| [Muon + AdamW] wd 0, piecewise − [SGD] SpecReg + evidence ℓ | −0.0372 | +0.1383 | −0.0095 | −0.0349 | +0.0382 | +0.0467 | +0.0420 | −0.0546 |
| [Muon + AdamW] wd 0.1, piecewise − [SGD] SpecReg + evidence ℓ | −0.0269 | +0.0811 | −0.0110 | −0.0279 | −0.0231 | +0.0131 | +0.0819 | −0.0658 |
| [Muon + AdamW] wd 0, WSD − [SGD] SpecReg + evidence ℓ | −0.0461 | +0.1657 | −0.0099 | −0.0340 | +0.0512 | +0.0544 | +0.0595 | +0.0852 |
| [Muon + AdamW] wd 0.1, WSD − [SGD] SpecReg + evidence ℓ | −0.0536 | +0.1531 | −0.0141 | −0.0363 | +0.0563 | +0.0610 | +0.1027 | −0.0115 |
| [Muon + AdamW] wd 0.1, cosine − [SGD] SpecReg, cosine | −0.0417 | +0.1074 | −0.0133 | −0.0308 | −0.0191 | +0.0117 | +0.1009 | +0.0971 |
| [Muon + SGD] wd 0.1, cosine − [SGD] SpecReg, cosine | −0.0146 | +0.0145 | −0.0035 | −0.0171 | −0.0518 | −0.0229 | +0.0361 | +0.0232 |
| [Muon + SGD] wd 0, cosine − [SGD] SpecReg, cosine | −0.0149 | +0.0928 | +0.0077 | +0.0012 | +0.0298 | −0.0034 | −0.0185 | +0.0774 |
| [Muon + SGD] wd 0.1 + BN SN, cosine − [SGD] SpecReg, cosine | −0.0259 | +0.1757 | +0.0943 | +0.0042 | +0.0273 | +0.0034 | −0.1033 | +0.1453 |
| [Muon + SGD] wd 0.1 + GP-input LN, cosine − [SGD] SpecReg, cosine | −0.0164 | +0.0362 | −0.0001 | −0.0091 | −0.0096 | −0.0022 | +0.3460 | +0.2543 |
| *Schedule* | | | | | | | | |
| [Muon + AdamW] wd 0: WSD − piecewise | −0.0089 | +0.0274 | −0.0004 | +0.0009 | +0.0130 | +0.0076 | +0.0175 | +0.1398 |
| [Muon + AdamW] wd 0.1: WSD − piecewise | −0.0267 | +0.0721 | −0.0030 | −0.0084 | +0.0793 | +0.0479 | +0.0208 | +0.0543 |
| [Muon + AdamW] wd 0.1: cosine − piecewise | −0.0094 | +0.0219 | −0.0013 | +0.0040 | +0.0286 | +0.0233 | +0.0056 | +0.0338 |
| [SGD] SpecReg ℓ = 7: cosine − piecewise | +0.0054 | −0.0045 | +0.0009 | +0.0068 | +0.0246 | +0.0247 | −0.0133 | −0.1291 |
| *Aux optimizer (stem / BN / GP head)* | | | | | | | | |
| Muon wd 0.1, cosine: [Muon + SGD] − [Muon + AdamW] | +0.0271 | −0.0929 | +0.0098 | +0.0137 | −0.0328 | −0.0346 | −0.0648 | −0.0739 |
| *BN spectral norm* | | | | | | | | |
| [Muon + SGD] wd 0.1, cosine: with − without | −0.0113 | +0.1612 | +0.0978 | +0.0213 | +0.0791 | +0.0263 | −0.1394 | +0.1222 |
| *GP-input LayerNorm (ℓ 7 → 20)* | | | | | | | | |
| [Muon + SGD] wd 0.1, cosine: with − without | −0.0018 | +0.0217 | +0.0034 | +0.0080 | +0.0422 | +0.0207 | +0.3099 | +0.2311 |
| *Optimizer alone (linear-head Baseline, cosine)* | | | | | | | | |
| [Muon + SGD] wd 0.1 − [SGD] | −0.0084 | −0.0033 | −0.0074 | −0.0000 | −0.0042 | +0.0334 | — | — |
| *GP head − linear head, same optimizer (cosine, ℓ = 7)* | | | | | | | | |
| [Muon + SGD] wd 0.1: GP head, no SN − Baseline | −0.0057 | −0.0086 | −0.0076 | −0.0233 | +0.0008 | +0.0401 | — | — |
| [SGD]: SNGP (`c = 6.0`) − Baseline | −0.0054 | +0.0006 | −0.0049 | −0.0109 | +0.0078 | +0.0605 | — | — |
| [SGD]: SNGP + SpecReg − Baseline | +0.0005 | −0.0264 | −0.0115 | −0.0063 | +0.0484 | +0.0964 | — | — |

- **Muon + AdamW: worse in-distribution than every SGD row.** Accuracy is down 2.7–5.4 points and NLL
  up 0.08–0.17 against evidence ℓ = 7. The val-fitted λ\* is 63–240, against 33–49 for the SGD rows, so
  the trained logits are far more overconfident. smECE is the lowest of any row only after that large
  correction. Near-OOD (CIFAR-10) is worse on every row. On far-OOD (SVHN), three of the four
  piecewise/WSD rows beat every SGD row, with DS SVHN 0.878–0.892 and the best FPR95 0.70. The GP
  variance is not the source.
- **Schedule: not the lever.** WSD lowered accuracy and raised NLL against piecewise at both weight
  decays. Cosine moved accuracy by +0.5 points (SpecReg) and −0.9 points (Muon), within the SGD rows'
  seed spread.
- **Aux optimizer: the lever for Muon's in-distribution gap.** At Muon wd 0.1, SGD instead of AdamW on
  the stem / BN / GP head gives +2.7 points of accuracy and −0.093 NLL. The final BN |γ| fell from 1.34
  to 0.59, and the GP output layer ‖β‖_F from 44 to 19. The AdamW rows' far-OOD SVHN gain went with
  it. That gain came from a narrowed kernel: ‖h‖/ℓ was 2.0 against 1.23 for SGD.
- **Muon + SGD wd 0** has the best far-OOD of the cosine rows (MSP SVHN 0.828, FPR95 0.736) and the
  SGD rows' BN scale (final BN |γ| 0.31, ‖β‖_F 9.7), but its NLL is 0.836.
- **Optimizer alone (linear-head Baseline, cosine): Muon costs 0.8 points of accuracy.** 0.7963 against
  0.8047 for `[SGD]`. NLL ties at the val-fit T (0.766 vs 0.769), but T\* is 1.51 against 1.23: Muon's raw
  logits are more overconfident (NLL 0.865 vs 0.800 at T = 1), as with the Muon GP rows' larger λ\*. MSP C-10
  ties (0.810); DS SVHN is +0.033 and FPR95 −0.037.
  - **Against the GP rows:** `[Muon + SGD]` wd 0.1 is −0.87 points behind `[SGD]` SNGP and −1.46 behind SpecReg
    (both ℓ = 7). So about 0.8 points of that gap is the optimizer alone. The rest of the SpecReg gap is the
    spectral penalty: SpecReg matches its own Baseline (+0.05 points, NLL −0.026), SNGP does not (−0.54 points).
  - **The GP head (no SN) on Muon** costs −0.57 points against Muon's own Baseline, about what SN-SNGP costs on
    SGD. It buys DS SVHN (+0.040) and loses MSP C-10 (−0.023).
  - The SGD cosine Baseline (0.8047) is inside the piecewise Baseline's seed spread (0.8061 ± 0.0027).
- **BN spectral norm: best near-OOD of the single-seed rows, worst calibration.** DS CIFAR-10 (0.814) is
  the highest in the table. MSP CIFAR-10 (0.808) trails only the 3-seed Baseline (0.810) and SpecReg
  (0.809). MSP SVHN is up 0.08 on the same recipe without the cap. But the
  model is underconfident. The val fit hits λ = 0, the bottom of the range (λ only damps logits), and
  smECE is 0.122.
  - **Mechanism:** the cap binds on 13 of 25 BN layers and shrinks the final BN's gain about 16× (raw
    47 → 3). It divides γ but not β, so the backbone features collapse toward a constant.
  - **Effect:** distances between val features are 0.06–0.11 ℓ, the kernel is about 0.99 between any
    two images, and the logits stay small.
  - **Why ℓ matters:** DUE learns ℓ, which would absorb this; the fixed ℓ = 7 does not.
- **GP-input LayerNorm (ℓ = 20): the only row whose GP variance ranks OOD above ID on both sets.**
  Var AUROC is 0.661 on CIFAR-10 and 0.607 on SVHN, against 0.351 / 0.376 on the same recipe without
  it. Every other row is below 0.44 on CIFAR-10. MSP SVHN is up 0.042 and FPR95 drops from 0.808 to
  0.745. In-distribution it is slightly worse: −0.2 points of accuracy, +0.022 NLL.
  - **It did not stop the λ drift it was run to test.** The val-fit λ still climbs from epoch 150
    (7.7 → 52.2 at epoch 249, against 7.3 → 68.7 without it). ‖β‖_F is 19.9 against 18.8 without it
    and 12.1 for SpecReg, so the drift tracks the GP output layer, not the feature scale.
  - **Feature scale at `last.ckpt`** (2000 val images): L2 shrinks the LayerNorm gain to 0.50, so
    ‖h‖/ℓ is 0.95, not the 1.26 at init. Pairwise distances stay at the no-LayerNorm level: same-class
    d/ℓ 1.00 against 1.07, and 0.75 for SpecReg.

Caveats:
- **Seeds:** every non-piecewise-SGD row is one seed. The SGD rows' seed spread is about ±0.004 for
  accuracy, ±0.01 for NLL and ±0.02–0.04 for DS SVHN.
- **Muon recipe:** Muon's lr and momentum are the reference defaults, untuned on this backbone.

## Fixed mean-field factor λ = π/8

The same rows, plus the four ℓ-sweep rows (ℓ = 2 / 3.5 / 14 / 20), with **nothing fit post hoc**: λ = π/8
(the probit-approximation value) on every GP row and T = 1 on the Baseline. `±` is the std across 3 seeds.
Bold is the best value per column in this table. Accuracy and Var AUROC do not depend on λ, so they match the tables above.
Per-seed CSV: [figures/cifar100_cosine/cifar100_cosine_per_seed_pi8.csv](../../figures/cifar100_cosine/cifar100_cosine_per_seed_pi8.csv).

| Arm | Optimizer | Schedule | ℓ | Seeds | Acc | NLL | smECE |
|---|---|---|---|---:|---:|---:|---:|
| Baseline | [SGD] | piecewise | — | 3 | **0.8061 ± 0.0027** | 0.8074 ± 0.0096 | 0.0779 ± 0.0038 |
| SNGP (`c = 6.0`) | [SGD] | piecewise | 20 | 3 | 0.8025 ± 0.0045 | 0.8184 ± 0.0078 | 0.0800 ± 0.0043 |
| SNGP + SpecReg | [SGD] | piecewise | 20 | 3 | 0.8048 ± 0.0036 | 0.7947 ± 0.0113 | 0.0702 ± 0.0020 |
| SNGP + SpecReg + evidence ℓ | [SGD] | piecewise | 7 | 3 | 0.7978 ± 0.0017 | 0.8224 ± 0.0087 | 0.0816 ± 0.0021 |
| SNGP + SpecReg + online evidence ℓ | [SGD] | piecewise | online | 3 | 0.7999 ± 0.0003 | 0.8466 ± 0.0037 | 0.0895 ± 0.0010 |
| Baseline | [SGD] | cosine | — | 1 | 0.8047 | 0.8001 | 0.0697 |
| SNGP (`c = 6.0`) | [SGD] | cosine | 7 | 1 | 0.7993 | 0.8325 | 0.0778 |
| SNGP + SpecReg | [SGD] | cosine | 7 | 1 | 0.8052 | **0.7937** | 0.0700 |
| GP head, no SN, wd 0 | [Muon + AdamW] | piecewise | 7 | 1 | 0.7626 | 1.4122 | 0.1468 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | piecewise | 7 | 1 | 0.7729 | 1.1062 | 0.1231 |
| GP head, no SN, wd 0 | [Muon + AdamW] | WSD | 7 | 1 | 0.7537 | 1.5775 | 0.1594 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | WSD | 7 | 1 | 0.7462 | 1.0708 | 0.1151 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | cosine | 7 | 1 | 0.7635 | 1.0978 | 0.1247 |
| Baseline, wd 0.1 | [Muon + SGD] | cosine | — | 1 | 0.7963 | 0.8654 | 0.0884 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 7 | 1 | 0.7906 | 0.8972 | 0.1008 |
| GP head, no SN, wd 0 | [Muon + SGD] | cosine | 7 | 1 | 0.7903 | 0.8748 | **0.0674** |
| GP head, no SN, wd 0.1, BN spectral norm `c = 3` | [Muon + SGD] | cosine | 7 | 1 | 0.7793 | 0.9191 | 0.1225 |
| GP head, no SN, wd 0.1, GP-input LayerNorm | [Muon + SGD] | cosine | 20 | 1 | 0.7888 | 0.8793 | 0.0939 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 2 | 1 | 0.7860 | 0.8686 | 0.0921 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 3.5 | 1 | 0.7894 | 0.8756 | 0.0930 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 14 | 1 | 0.7930 | 0.9152 | 0.1019 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 20 | 1 | 0.7964 | 0.8778 | 0.0936 |

| Arm | Optimizer | Schedule | ℓ | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN | Var C-10 | Var SVHN | FPR95 MSP SVHN ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | [SGD] | piecewise | — | 0.8056 ± 0.0024 | 0.7235 ± 0.0378 | 0.8129 ± 0.0020 | 0.7487 ± 0.0382 | — | — | 0.855 ± 0.028 |
| SNGP (`c = 6.0`) | [SGD] | piecewise | 20 | 0.8068 ± 0.0034 | 0.7424 ± 0.0057 | 0.8173 ± 0.0041 | 0.7859 ± 0.0053 | 0.437 ± 0.012 | 0.580 ± 0.018 | 0.849 ± 0.011 |
| SNGP + SpecReg | [SGD] | piecewise | 20 | **0.8113 ± 0.0014** | 0.7823 ± 0.0090 | **0.8212 ± 0.0020** | 0.8228 ± 0.0136 | 0.314 ± 0.001 | 0.410 ± 0.046 | 0.809 ± 0.020 |
| SNGP + SpecReg + evidence ℓ | [SGD] | piecewise | 7 | 0.8001 ± 0.0013 | 0.7844 ± 0.0302 | 0.8122 ± 0.0027 | 0.8491 ± 0.0278 | 0.336 ± 0.007 | 0.449 ± 0.030 | 0.804 ± 0.031 |
| SNGP + SpecReg + online evidence ℓ | [SGD] | piecewise | online | 0.7981 ± 0.0044 | 0.7728 ± 0.0123 | 0.8067 ± 0.0072 | 0.8300 ± 0.0197 | 0.294 ± 0.010 | 0.297 ± 0.044 | 0.830 ± 0.022 |
| Baseline | [SGD] | cosine | — | 0.8057 | 0.7461 | 0.8160 | 0.7593 | — | — | 0.858 |
| SNGP (`c = 6.0`) | [SGD] | cosine | 7 | 0.8008 | 0.7271 | 0.8153 | 0.7903 | 0.393 | 0.595 | 0.866 |
| SNGP + SpecReg | [SGD] | cosine | 7 | 0.8081 | 0.7886 | 0.8166 | 0.8527 | 0.315 | 0.353 | 0.805 |
| GP head, no SN, wd 0 | [Muon + AdamW] | piecewise | 7 | 0.7583 | 0.7690 | 0.7530 | 0.8799 | 0.370 | 0.427 | 0.834 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | piecewise | 7 | 0.7621 | 0.7152 | 0.7534 | 0.8436 | 0.410 | 0.416 | 0.845 |
| GP head, no SN, wd 0 | [Muon + AdamW] | WSD | 7 | 0.7569 | 0.7714 | 0.7561 | 0.8598 | 0.388 | 0.567 | 0.833 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | WSD | 7 | 0.7525 | 0.7973 | 0.7378 | **0.8921** | 0.431 | 0.470 | 0.767 |
| GP head, no SN, wd 0.1 | [Muon + AdamW] | cosine | 7 | 0.7653 | 0.7426 | 0.7560 | 0.8645 | 0.416 | 0.450 | 0.814 |
| Baseline, wd 0.1 | [Muon + SGD] | cosine | — | 0.8011 | 0.7237 | 0.8135 | 0.7900 | — | — | 0.841 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 7 | 0.7886 | 0.7250 | 0.7942 | 0.8422 | 0.351 | 0.376 | 0.840 |
| GP head, no SN, wd 0 | [Muon + SGD] | cosine | 7 | 0.8064 | 0.8189 | 0.8204 | 0.8423 | 0.297 | 0.430 | **0.747** |
| GP head, no SN, wd 0.1, BN spectral norm `c = 3` | [Muon + SGD] | cosine | 7 | 0.8074 | **0.8258** | 0.8139 | 0.8592 | 0.212 | 0.498 | 0.792 |
| GP head, no SN, wd 0.1, GP-input LayerNorm | [Muon + SGD] | cosine | 20 | 0.7905 | 0.7669 | 0.7874 | 0.8625 | **0.661** | **0.607** | 0.779 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 2 | 0.7844 | 0.7573 | 0.7763 | 0.8337 | 0.350 | 0.312 | 0.833 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 3.5 | 0.7827 | 0.7195 | 0.7780 | 0.8210 | 0.354 | 0.405 | 0.832 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 14 | 0.7950 | 0.7444 | 0.8007 | 0.8281 | 0.327 | 0.357 | 0.821 |
| GP head, no SN, wd 0.1 | [Muon + SGD] | cosine | 20 | 0.7985 | 0.7523 | 0.8087 | 0.8311 | 0.313 | 0.339 | 0.826 |

- **NLL rises on every row, roughly in proportion to λ\*.** Against each row's val-fit λ\*: +0.05 to +0.09 for the
  SGD rows (λ\* 32–49) and +0.050 for the Baseline at T = 1, +0.09 to +0.16 for the `[Muon + SGD]` wd 0.1 rows
  (λ\* 45–80), +0.17 to +0.66 for `[Muon + AdamW]` (λ\* 63–240). The BN spectral norm row (λ\* 0) barely moves (+0.001).
  The two SpecReg rows have the best NLL (0.794 / 0.795), ahead of the T = 1 Baseline (0.807). `[Muon + SGD]` wd 0
  (λ\* 29.7) is the best Muon row, with NLL 0.875 and the lowest smECE (0.067).
- **Near-OOD:** DS C-10 is higher at π/8 than at λ\* on every GP row but the BN spectral norm row (unchanged; e.g. evidence ℓ 0.785 → 0.812,
  `[Muon + AdamW]` 0.71–0.74 → 0.74–0.76). Piecewise SpecReg is still the best on MSP and DS C-10.
- **Far-OOD:** without its large λ\*, `[Muon + AdamW]` loses its MSP SVHN lead (0.75–0.83 → 0.72–0.80) and its FPR95
  (0.70–0.80 → 0.77–0.85); its DS SVHN stays highest (0.892, WSD wd 0.1). The best MSP SVHN and FPR95 move to the two
  small-λ\* `[Muon + SGD]` rows: BN spectral norm (0.826) and wd 0 (FPR95 0.747).

## Length-scale sweep (no LayerNorm)

`[Muon + SGD]` wd 0.1, cosine, GP input **not** normalized, seed 12345. Only `model.net.length_scale`
changes; ℓ = 7 is the "GP head, no SN, wd 0.1 [Muon + SGD] cosine" row above. The hollow marker is
the GP-input LayerNorm row (ℓ = 20), the dashed line `[SGD]` SpecReg at ℓ = 7 (cosine).

![CIFAR-100 length-scale sweep](../../figures/cifar100_cosine/cifar100_length_scale_sweep.png)

In-distribution, and the feature scale the kernel sees (2000 val images, `last.ckpt`,
`scripts/metrics/cifar100_gp_head_diag.py`; k is the exact RBF kernel averaged over same- / different-class pairs):

| ℓ | Acc | NLL | smECE | λ\* | ‖h‖ | ‖h‖/ℓ | k same | k diff | ‖β‖_F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.7860 | 0.7798 | 0.0214 | 45.0 | 5.2 | 2.59 | 0.30 | 0.15 | 19.8 |
| 3.5 | 0.7894 | 0.7727 | **0.0194** | 53.9 | 7.9 | 2.26 | 0.36 | 0.19 | 19.4 |
| 7 | 0.7906 | 0.7572 | 0.0242 | 68.7 | 11.2 | 1.60 | 0.57 | 0.39 | 18.8 |
| 14 | 0.7930 | 0.7599 | 0.0262 | 80.1 | 14.1 | 1.01 | 0.82 | 0.68 | 18.3 |
| 20 | **0.7964** | **0.7514** | 0.0267 | 75.8 | 14.7 | 0.74 | 0.90 | 0.80 | 18.5 |
| 20, GP-input LayerNorm | 0.7888 | 0.7789 | 0.0276 | 52.2 | 0.64 (post-LN 19.0) | 0.95 | 0.60 | 0.44 | 19.9 |

OOD AUROC:

| ℓ | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN | Var C-10 | Var SVHN | FPR95 MSP SVHN ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.7838 | 0.7648 | 0.7518 | 0.8165 | 0.350 | 0.312 | 0.818 |
| 3.5 | 0.7817 | 0.7384 | 0.7474 | 0.8120 | 0.354 | 0.405 | 0.795 |
| 7 | 0.7862 | 0.7465 | 0.7556 | 0.8328 | 0.351 | 0.376 | 0.808 |
| 14 | 0.7917 | 0.7589 | 0.7672 | 0.8242 | 0.327 | 0.357 | 0.797 |
| 20 | **0.7964** | 0.7692 | **0.7854** | 0.8322 | 0.313 | 0.339 | 0.803 |
| 20, GP-input LayerNorm | 0.7942 | **0.7887** | 0.7758 | **0.8535** | **0.661** | **0.607** | **0.745** |

- **ℓ is a real knob here, only partly absorbed.** Over ℓ 2 → 20 (×10) the backbone grows ‖h‖ ×2.8,
  so ‖h‖/ℓ still falls 2.59 → 0.74 and the same-class kernel widens 0.30 → 0.90. (In the
  [learnable-ℓ study](CIFAR100_LEARNABLE_LS_RESULTS.md) ‖h‖/ℓ locked at ~2.2 instead.)
- **In-distribution and near-OOD: larger ℓ is better.** ℓ 2 → 20: accuracy +1.0 point (monotone over
  all five ℓ), NLL −0.028, MSP C-10 +0.013, DS C-10 +0.034 (each with one out-of-order step). The ℓ = 20 row is the best `[Muon + SGD]` wd 0.1 row,
  still −0.9 points of accuracy and +0.009 NLL behind `[SGD]` SpecReg.
- **Far-OOD (SVHN): no trend.** MSP 0.738–0.769, DS 0.812–0.833 and FPR95 0.79–0.82 move without
  order, within the ±0.02–0.04 seed spread. A small ℓ does not buy far-OOD.
- **GP variance: inverted at every ℓ** (AUROC 0.31–0.41). Changing ℓ does not un-invert it; the
  LayerNorm does. Ad hoc check (2000 test images per set against 4000 CIFAR-100 val features): without the
  LayerNorm, CIFAR-10 / SVHN features are 3–13 % shorter than CIFAR-100 test features and sit closer to the
  centre of the CIFAR-100 cloud. Their nearest-neighbour distance is only +5 % / +1 % above ID at ℓ = 7,
  and ℓ rescales every distance alike. With the LayerNorm, SVHN's raw ‖h‖ is half of ID's, but the LayerNorm
  normalizes it away, and the nearest-neighbour gap becomes +17 % / +11 %.
- **LayerNorm at matched ℓ = 20:** −0.8 points of accuracy, +0.028 NLL, MSP C-10 −0.002; MSP SVHN +0.020,
  DS SVHN +0.021, FPR95 −0.058, Var AUROC +0.35 / +0.27. So its in-distribution cost is larger than
  the −0.2 points measured against ℓ = 7.
- **λ\* rises with ℓ** (45 → 80) while ‖β‖_F stays at 18–20. So the post-hoc λ is not set by ‖β‖
  alone; it also grows as the kernel widens.

### Length-scale sweep at λ = π/8

The same six checkpoints at λ = π/8 with nothing fit post hoc (rows from the
[fixed-λ section](#fixed-mean-field-factor-λ--π8)). Accuracy and Var AUROC match the tables above. Bold is the best per column.

| ℓ | Acc | NLL | Brier | smECE |
|---|---:|---:|---:|---:|
| 2 | 0.7860 | **0.8686** | 0.3125 | **0.0921** |
| 3.5 | 0.7894 | 0.8756 | 0.3113 | 0.0930 |
| 7 | 0.7906 | 0.8972 | 0.3146 | 0.1008 |
| 14 | 0.7930 | 0.9152 | 0.3140 | 0.1019 |
| 20 | **0.7964** | 0.8778 | **0.3062** | 0.0936 |
| 20, GP-input LayerNorm | 0.7888 | 0.8793 | 0.3125 | 0.0939 |

| ℓ | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN | Var C-10 | Var SVHN | FPR95 MSP SVHN ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.7844 | 0.7573 | 0.7763 | 0.8337 | 0.350 | 0.312 | 0.833 |
| 3.5 | 0.7827 | 0.7195 | 0.7780 | 0.8210 | 0.354 | 0.405 | 0.832 |
| 7 | 0.7886 | 0.7250 | 0.7942 | 0.8422 | 0.351 | 0.376 | 0.840 |
| 14 | 0.7950 | 0.7444 | 0.8007 | 0.8281 | 0.327 | 0.357 | 0.821 |
| 20 | **0.7985** | 0.7523 | **0.8087** | 0.8311 | 0.313 | 0.339 | 0.826 |
| 20, GP-input LayerNorm | 0.7905 | **0.7669** | 0.7874 | **0.8625** | **0.661** | **0.607** | **0.779** |

- **NLL: the larger-ℓ advantage goes away.** At λ\* NLL falls with ℓ (0.780 → 0.751). At π/8 it is
  0.869–0.915 with no order (ℓ = 2 lowest, ℓ = 14 highest). The π/8 penalty grows with λ\*: +0.089 at ℓ = 2
  (λ\* 45), +0.13 to +0.16 at ℓ 7–20 (λ\* 69–80).
- **Near-OOD: the trend holds.** DS C-10 rises monotonically with ℓ (0.776 → 0.809), and MSP C-10 rises with one
  out-of-order step (0.784 → 0.799).
- **Far-OOD: still no trend.** MSP SVHN 0.72–0.76, DS SVHN 0.82–0.84, FPR95 0.82–0.84. FPR95 is worse than at λ\* at every ℓ.
  The LayerNorm row is still the best on far-OOD, but by less: MSP SVHN 0.767, against 0.789 at λ\*.

Caveat: one seed per ℓ.
