# CIFAR-100 / WideResNet-28-10: SpecReg with the length scale tuned online by GP evidence

The two-stage route (score the evidence on a frozen ℓ = 20 backbone, then retrain at the pick,
[CIFAR100_EVIDENCE_LS_RESULTS.md](CIFAR100_EVIDENCE_LS_RESULTS.md)) is replaced here by
tuning during training. The method follows Immer et al. 2021 (arXiv:2104.04975), restricted to
the GP head: the backbone and head weights train by CE, and ℓ moves only up the head's type-II
evidence on training data. No validation data is used.

| | |
|---|---|
| Arm | **SNGP + SpecReg + online evidence ℓ**: the SpecReg recipe, starting at ℓ = 20, plus `src/callbacks/online_length_scale.py`. Nothing else changed: σ² 1, ridge 1.0, `gaussian` |
| Schedule | At epochs 10, 15, …, 155: Gaussian random-feature evidence at type-II (α, s), on 10k train images through the current backbone, over ℓ · 4^[−1, 1] (13 points). Then a half geometric step toward the argmax. ℓ is fixed from epoch 160 (the last LR decay) |
| Runs | seeds 12345 / 1 / 2, 250 epochs, `last.ckpt`; [../checkpoints/CIFAR_CHECKPOINTS.md](../checkpoints/CIFAR_CHECKPOINTS.md), section "online_ls". The other 4 rows are the evidence_ls page's runs |
| Protocol | One post-hoc knob per row, fit on **val** NLL (λ for SNGP arms, T for the baseline); metrics on **test**. `±` is the std across 3 seeds |
| Data, commands | [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md), section "online_ls" |

## Length-scale trajectory

ℓ after the update at each epoch (LR decays at 60 / 120 / 160):

| seed | 10 | 30 | 50 | 70 | 90 | 100 | 120 | 140 | 155 (final) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12345 | 12.6 | 8.9 | 10.0 | 11.2 | 17.8 | 10.0 | 6.3 | 5.6 | **5.61** |
| 1 | 12.6 | 8.9 | 11.2 | 11.2 | 12.6 | 10.0 | 6.3 | 6.3 | **5.61** |
| 2 | 12.6 | 8.9 | 8.9 | 10.0 | 14.1 | 10.0 | 7.9 | 5.6 | **5.00** |

Final backbones: median ρ = ‖x‖/ℓ on train is 1.22 / 1.21 / 1.31, and median ‖x‖ is 6.8. For
comparison, ‖x‖ was 9.1 at fixed ℓ = 7 and 11.9 at ℓ = 20.

## Results

| Arm | Acc | NLL | smECE | knob\* (val) |
|---|---:|---:|---:|---:|
| Baseline | **0.8061 ± 0.0027** | 0.7574 ± 0.0072 | 0.0364 ± 0.0007 | T 1.31 |
| SNGP (`c = 6.0`) | 0.8025 ± 0.0045 | 0.7600 ± 0.0065 | 0.0348 ± 0.0022 | λ 32.5 |
| SNGP + SpecReg | 0.8048 ± 0.0036 | **0.7472 ± 0.0100** | 0.0299 ± 0.0041 | λ 35.3 |
| SNGP + SpecReg + evidence ℓ = 7 | 0.7978 ± 0.0017 | 0.7513 ± 0.0062 | **0.0253 ± 0.0023** | λ 43.1 |
| SNGP + SpecReg + online evidence ℓ | 0.7999 ± 0.0003 | 0.7566 ± 0.0039 | 0.0274 ± 0.0011 | λ 49.1 |

| Arm | MSP C-10 | MSP SVHN | DS C-10 | DS SVHN | Var C-10 | Var SVHN | FPR95 MSP SVHN ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | **0.8101 ± 0.0023** | 0.7309 ± 0.0373 | **0.8121 ± 0.0020** | 0.7496 ± 0.0383 | — | — | 0.855 ± 0.025 |
| SNGP (`c = 6.0`) | 0.8059 ± 0.0031 | 0.7620 ± 0.0053 | 0.8082 ± 0.0029 | 0.8047 ± 0.0032 | **0.437 ± 0.012** | **0.580 ± 0.018** | 0.828 ± 0.011 |
| SNGP + SpecReg | 0.8092 ± 0.0016 | 0.7948 ± 0.0060 | 0.8110 ± 0.0020 | 0.8330 ± 0.0068 | 0.314 ± 0.001 | 0.410 ± 0.046 | 0.794 ± 0.023 |
| SNGP + SpecReg + evidence ℓ = 7 | 0.7948 ± 0.0014 | **0.8016 ± 0.0269** | 0.7846 ± 0.0036 | **0.8571 ± 0.0230** | 0.336 ± 0.007 | 0.449 ± 0.030 | **0.778 ± 0.036** |
| SNGP + SpecReg + online evidence ℓ | 0.7866 ± 0.0066 | 0.7667 ± 0.0113 | 0.7618 ± 0.0127 | 0.8012 ± 0.0143 | 0.294 ± 0.010 | 0.297 ± 0.044 | 0.824 ± 0.017 |

Paired per seed, each at its own λ\*:

| | Acc | NLL | smECE | MSP C-10 | MSP SVHN | DS SVHN | Var C-10 | Var SVHN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| online − SpecReg, mean | −0.0049 | +0.0094 | −0.0026 | −0.0226 | −0.0281 | −0.0318 | −0.0198 | −0.1133 |
| sign-consistent | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| online − evidence ℓ = 7, mean | +0.0020 | +0.0053 | +0.0021 | −0.0083 | −0.0349 | −0.0560 | −0.0419 | −0.1521 |
| sign-consistent | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 |

- **Negative result.** Against SpecReg, online ℓ is worse on accuracy, NLL and every OOD
  AUROC (3/3 seeds each). Against the two-stage ℓ = 7 it gives back all of the far-OOD gain
  (DS SVHN −0.056). Only smECE improves over SpecReg.
- **The GP variance is more inverted.** SVHN variance AUROC is 0.30, the lowest of any arm. On
  the final backbones SVHN features sit at a *smaller* ρ than the CIFAR-100 train features (median 1.04–1.11 vs
  1.21–1.31).
- **ℓ never settles.** It tracks the LR phases: about 9–11 before the epoch-60 decay, 13–18 by
  epoch 90, then down to 5–6 after the epoch-120 decay. At the last update, ℓ\* was still 4.5–5.

## Evidence re-check on the online backbones

Type-II evidence at rff_dim 1024 on the final backbones' features (mean over 3 seeds):

| | ℓ = 20 SpecReg | ℓ = 7 backbones | online backbones |
|---|---:|---:|---:|
| type-II argmax, train / val | 7 / 7 | 7 / 7 | **5 / 5** (seed 2 train: 4) |
| final training ℓ | 20 | 7 | 5.6 / 5.6 / 5.0 |
| median ‖x‖, train | 11.9 | 9.1 | 6.8 |
| log-evidence/N, argmax − ℓ = 20 (train) | +15.8 | +3.2 | +7.8 |

- The run ends near its own fixed point (the evidence picks ℓ ≈ 5 for a backbone trained to
  ℓ ≈ 5). But the backbone shrinks ‖x‖ with each lower ℓ, so ρ stays around 1.2–1.3 and the
  kernel never becomes more distance-aware. This is the same co-adaptation as the CE-trained ℓ
  ([CIFAR100_LEARNABLE_LS_RESULTS.md](CIFAR100_LEARNABLE_LS_RESULTS.md)), only slower.
- The evidence curve is flat over ℓ 4–7 (within 2.5 nats/N on train, 0.6 on val), so on these
  features the evidence barely separates the candidates it chooses between.
