# Wong (ADRC) — SNGP variants vs a linear-head baseline, 5 seeds

The final runs of the ADRC study on full Wong (4 classes, all institutions). Each arm uses the
cell kept from its `adrc_wong_*` grid sweep ([../MASTER_SWEEPS.md](../MASTER_SWEEPS.md)) and is
trained over seeds {12345, 1, 2, 3, 4}. All runs share one recipe: resnet18, SGD + Nesterov
(lr 0.04, wd 6e-4, cosine), 150 epochs, early stopping off, and `best.ckpt` = min raw `val/nll`.
The GP arms use the CIFAR reference head (ℓ = 20, σ² = 1, ridge 1, `binary_logistic` Laplace
weight) with `mean_field_factor` pinned at π/8 for training, selection and test. There is no
post-hoc calibration. Trained 2026-10-05 with `scripts/tmux/adrc_wong_final.sh`, W&B group
`adrc_final_2026-10-05_10-05-27`. Checkpoints:
[../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md) (`adrc_*_wong`). Inference dirs and
reproduce commands: [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md)
(`adrc_wong_final`). Per-run numbers: `figures/wong_adrc/wong_adrc_runs.csv`.

The Muon → SGD two-stage row (recipe in "UC Davis-trained, Muon → SGD two-stage" below, cosine
tail, aux 0.01) was trained later, on 2026-10-06, with `ARMS=muon2stage scripts/tmux/adrc_wong_final.sh`
(W&B group `adrc_final_2026-10-06_21-21-42`). Checkpoints `adrc_muon2stage_wong`, inference
`adrc_wong_final_muon2stage`. The epoch-135 row is the same recipe with the switch + cosine tail at
epoch 135, seed 12345 only (2026-10-07, `ARMS=muon2stage_sw135`, W&B group
`adrc_final_2026-10-07_11-13-57`); checkpoint `adrc_muon2stage_sw135_wong`, inference
`adrc_wong_final_muon2stage_sw135`, per-run numbers
`figures/wong_adrc/wong_adrc_muon2stage_sw135_s12345_runs.csv`.
The `(aux 0.01)` Muon row is the plain Muon arm with only the aux SGD lr (stem / BN / GP output
layer) at the two-stage arms' 0.01 and no switch, seed 12345 only (2026-10-07, `ARMS=muon_aux01`,
W&B group `adrc_final_2026-10-07_13-05-23`); checkpoint `adrc_muon_aux01_wong`, inference
`adrc_wong_final_muon_aux01`, per-run numbers `figures/wong_adrc/wong_adrc_muon_aux01_s12345{,_last}_runs.csv`
(best.ckpt, last.ckpt). Its artifact axes were not run.

| Arm | Experiment | Cell | best.ckpt epoch (s12345, s1–s4) | W&B (s12345, s1–s4) |
|---|---|---|---|---|
| Baseline (linear head) | `baseline_wong_sgd` | dropout 0 | 138, 148, 141, 147, 147 | `glb9x1ee` `y5otcrt8` `ubudgj2e` `xc3s26hd` `07przvzo` |
| SNGP | `sngp_wong_sgd` | c = 1 | 146, 141, 143, 146, 141 | `g02st7tz` `r3c4xpcg` `v7dp31o4` `yta98agx` `4upbt83i` |
| SNGP + BN-SN | `sngp_bnsn_wong_sgd` | c = 8, BN cap tied | 144, 146, 145, 145, 143 | `f3a6in72` `5qnxpq0e` `1prdt22h` `nxgyrx0j` `49kekadv` |
| SNGP + SpecReg | `sngp_specreg_wong_sgd` | λ = 0.003, no SN | 146, 146, 143, 143, 148 | `k4ptgv0v` `5m9ilprx` `zk7eifgn` `hph5tfeg` `2qx6mivs` |
| GP head + Muon | `sngp_muon_wong_sgd` | Muon wd = 0.01, no SN | 146, 141, 143, 146, 148 | `hdmg964v` `qxkd3rwn` `5kixgy2n` `ma6f545j` `9x9a7xp7` |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | `sngp_muon_2stage_wong_sgd` | Muon wd = 0.01 → SGD at epoch 120, no SN | 149, 142, 145, 146, 148 | `fcrtjgun` `7mnbytky` `25j3jpfs` `e22h8269` `njhdpvv7` |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | `sngp_muon_2stage_wong_sgd` | as above, switch at epoch 135 | 149 (s12345 only) | `5hw5q9me` |
| GP head + Muon (aux 0.01) | `sngp_muon_wong_sgd` | Muon wd = 0.01, aux SGD lr 0.01, no SN | 126 (s12345 only) | `m3edt5m7` |

Every cell is mean ± sample std over the 5 training seeds. ECE is 10 equal-width bins (from
inference); aECE is 10 equal-mass bins and smECE is the smooth ECE of Błasiok & Nakkiran with the
reflected kernel, both recomputed from `predictions.csv` (`src/metrics/calibration_variants.py`).
At ~99% accuracy smECE sits at its bandwidth floor, so ID differences in it are small.

### In-distribution — Wong test split (n = 14,010)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.9921 ± 0.0006 | 0.9921 ± 0.0006 | 0.15 ± 0.01 | 0.14 ± 0.08 | 0.37 ± 0.02 | 2.66 ± 0.25 | 1.27 ± 0.11 |
| SNGP, c = 1 | 0.9922 ± 0.0004 | 0.9921 ± 0.0004 | 0.18 ± 0.04 | 0.13 ± 0.10 | 0.38 ± 0.04 | 2.73 ± 0.21 | 1.27 ± 0.11 |
| SNGP + BN-SN, c = 8 | 0.9923 ± 0.0007 | 0.9923 ± 0.0007 | 0.19 ± 0.03 | **0.12 ± 0.04** | 0.41 ± 0.06 | 2.70 ± 0.17 | 1.22 ± 0.07 |
| SNGP + SpecReg, λ = 0.003 | 0.9923 ± 0.0006 | 0.9923 ± 0.0006 | **0.14 ± 0.02** | 0.14 ± 0.04 | 0.36 ± 0.03 | 2.68 ± 0.15 | 1.22 ± 0.08 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **0.9949 ± 0.0002** | **0.9949 ± 0.0002** | 0.22 ± 0.03 | 0.21 ± 0.03 | **0.34 ± 0.01** | **1.88 ± 0.08** | **0.83 ± 0.03** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.9909 ± 0.0008 | 0.9909 ± 0.0008 | 0.30 ± 0.10 | 0.21 ± 0.07 | 0.52 ± 0.08 | 3.00 ± 0.11 | 1.43 ± 0.06 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01), seed 12345 only | 0.9910 | 0.9910 | 0.23 | 0.23 | 0.40 | 3.33 | 1.55 |
| GP head + Muon, wd = 0.01 (aux 0.01), seed 12345 only | 0.9949 | 0.9949 | 0.17 | 0.10 | 0.31 | 1.93 | 0.86 |
| GP head + Muon, wd = 0.01 (aux 0.01), seed 12345 only, last.ckpt | 0.9946 | 0.9946 | 0.16 | 0.15 | 0.32 | 1.79 | 0.79 |

### Entropy AUROC ↑ — Wong test vs each OOD test set

Each seed's AUROC is the mean over the 10 frozen 1,000-row subsamples of
`src/metrics/calculate_ood_metrics.py`. The ± is across training seeds.

| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.930 ± 0.039 | 0.775 ± 0.097 | 0.809 ± 0.042 | 0.720 ± 0.036 | 0.718 ± 0.067 | 0.431 ± 0.028 | 0.731 ± 0.015 |
| SNGP, c = 1 | 0.945 ± 0.022 | 0.860 ± 0.035 | 0.891 ± 0.025 | 0.784 ± 0.019 | 0.822 ± 0.126 | 0.397 ± 0.036 | 0.783 ± 0.030 |
| SNGP + BN-SN, c = 8 | 0.850 ± 0.104 | 0.846 ± 0.097 | 0.835 ± 0.040 | 0.715 ± 0.052 | 0.806 ± 0.104 | 0.452 ± 0.029 | 0.750 ± 0.051 |
| SNGP + SpecReg, λ = 0.003 | 0.911 ± 0.059 | 0.781 ± 0.094 | 0.884 ± 0.014 | 0.731 ± 0.036 | 0.748 ± 0.065 | 0.390 ± 0.015 | 0.741 ± 0.035 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **0.975 ± 0.021** | **0.876 ± 0.083** | **0.956 ± 0.013** | **0.849 ± 0.031** | **0.960 ± 0.024** | **0.453 ± 0.060** | **0.845 ± 0.024** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.891 ± 0.111 | 0.762 ± 0.096 | 0.855 ± 0.049 | 0.692 ± 0.060 | 0.868 ± 0.083 | 0.396 ± 0.026 | 0.744 ± 0.047 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01), seed 12345 only | 0.818 | 0.772 | 0.906 | 0.758 | 0.914 | 0.414 | 0.764 |
| GP head + Muon, wd = 0.01 (aux 0.01), seed 12345 only | 0.979 | 0.870 | 0.966 | 0.894 | 0.977 | 0.445 | 0.855 |
| GP head + Muon, wd = 0.01 (aux 0.01), seed 12345 only, last.ckpt | 0.992 | 0.934 | 0.976 | 0.904 | 0.980 | 0.426 | 0.869 |

### GP-variance AUROC ↑ — Wong test vs each OOD test set

Same protocol, scored on the GP predictive variance (the `uncertainty` column), which does not
depend on the mean-field factor. Below 0.5 means the variance is inverted (lower on OOD).

| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SNGP, c = 1 | 0.747 ± 0.048 | 0.700 ± 0.133 | **0.924 ± 0.021** | 0.835 ± 0.022 | 0.845 ± 0.097 | **0.448 ± 0.040** | 0.750 ± 0.048 |
| SNGP + BN-SN, c = 8 | **0.937 ± 0.031** | 0.822 ± 0.064 | 0.912 ± 0.016 | **0.870 ± 0.041** | 0.887 ± 0.102 | 0.356 ± 0.030 | **0.798 ± 0.030** |
| SNGP + SpecReg, λ = 0.003 | 0.767 ± 0.075 | 0.751 ± 0.112 | 0.906 ± 0.020 | 0.823 ± 0.034 | 0.860 ± 0.054 | 0.420 ± 0.039 | 0.754 ± 0.036 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.719 ± 0.109 | 0.404 ± 0.153 | 0.710 ± 0.037 | 0.575 ± 0.032 | 0.493 ± 0.119 | 0.405 ± 0.110 | 0.551 ± 0.034 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.894 ± 0.098 | **0.828 ± 0.120** | 0.854 ± 0.024 | 0.772 ± 0.041 | **0.904 ± 0.030** | **0.448 ± 0.070** | 0.783 ± 0.045 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01), seed 12345 only | 0.855 | 0.558 | 0.823 | 0.742 | 0.892 | 0.410 | 0.713 |
| GP head + Muon, wd = 0.01 (aux 0.01), seed 12345 only | 0.900 | 0.750 | 0.827 | 0.817 | 0.937 | 0.388 | 0.770 |
| GP head + Muon, wd = 0.01 (aux 0.01), seed 12345 only, last.ckpt | 0.941 | 0.765 | 0.770 | 0.825 | 0.922 | 0.309 | 0.755 |

## Artifact robustness — seed 12345 only

Both artifact-simulation axes on the Wong test split (n = 14,010), each axis with the other
switched off, for the seed-12345 checkpoint of every arm. These are single-seed values, so no ±.

### Config artifact axis — Accuracy ↑ vs count

Wong test, number of real artifact cutouts pasted per image (`artifact_balanced`, procedural off); 0 = real images.

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.9916 | **0.8343** | **0.7133** | **0.6039** | 0.5247 | **0.4655** |
| SNGP, c = 1 | 0.9920 | 0.8264 | 0.6976 | 0.5894 | 0.5137 | 0.4419 |
| SNGP + BN-SN, c = 8 | 0.9914 | 0.8268 | 0.7103 | 0.5977 | **0.5271** | 0.4582 |
| SNGP + SpecReg, λ = 0.003 | 0.9917 | 0.8252 | 0.7023 | 0.5987 | 0.5217 | 0.4587 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **0.9946** | 0.8335 | 0.7090 | 0.5986 | 0.5205 | 0.4583 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.9901 | 0.8073 | 0.6878 | 0.5732 | 0.4972 | 0.4366 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.9910 | 0.8241 | 0.7001 | 0.5844 | 0.5154 | 0.4530 |

### Config artifact axis — NLL (×10⁻²) ↓ vs count

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 3.02 | 87.41 | 152.88 | 219.61 | 266.81 | 301.74 |
| SNGP, c = 1 | 2.94 | 83.96 | 148.49 | 210.91 | 254.78 | 296.73 |
| SNGP + BN-SN, c = 8 | 2.90 | **80.03** | **139.31** | **200.92** | **242.16** | 284.72 |
| SNGP + SpecReg, λ = 0.003 | 2.84 | 87.79 | 152.50 | 212.73 | 255.74 | 294.31 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **1.83** | 82.29 | 146.24 | 203.66 | 243.06 | **276.02** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 3.11 | 86.27 | 149.09 | 210.38 | 249.50 | 285.38 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 3.33 | 82.33 | 144.48 | 204.41 | 245.33 | 279.44 |

### Config artifact axis — ECE (×10⁻²) ↓ vs count

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.15 | 12.42 | 22.02 | 31.27 | 38.52 | 43.03 |
| SNGP, c = 1 | 0.20 | 12.85 | 22.98 | 32.16 | 38.32 | 45.01 |
| SNGP + BN-SN, c = 8 | 0.20 | 12.73 | **21.60** | 31.39 | **37.23** | 43.76 |
| SNGP + SpecReg, λ = 0.003 | **0.13** | 12.68 | 22.63 | **31.01** | 37.82 | 43.11 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.22 | 12.52 | 22.50 | 31.74 | 37.78 | **42.67** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.25 | 13.85 | 23.21 | 33.03 | 39.01 | 44.75 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.23 | **12.33** | 22.16 | 31.93 | 37.44 | 42.93 |

### Procedural artifact axis — Accuracy ↑ vs severity

Wong test, graded acquisition degradations (`procedural_ood`, `histo_c` ladder, cutouts off); 0 = real images.

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.9916 | 0.9880 | 0.9845 | 0.9734 | 0.9345 | 0.8745 |
| SNGP, c = 1 | 0.9920 | 0.9870 | 0.9829 | 0.9647 | 0.9118 | 0.8450 |
| SNGP + BN-SN, c = 8 | 0.9914 | 0.9887 | 0.9848 | 0.9692 | 0.9176 | 0.8545 |
| SNGP + SpecReg, λ = 0.003 | 0.9917 | 0.9874 | 0.9808 | 0.9634 | 0.9178 | 0.8487 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **0.9946** | **0.9928** | **0.9916** | **0.9792** | **0.9494** | 0.8789 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.9901 | 0.9877 | 0.9837 | 0.9696 | 0.9268 | 0.8548 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.9910 | 0.9886 | 0.9845 | 0.9742 | 0.9430 | **0.8847** |

### Procedural artifact axis — NLL (×10⁻²) ↓ vs severity

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 3.02 | 4.26 | 5.20 | 9.52 | 24.37 | 52.94 |
| SNGP, c = 1 | 2.94 | 4.33 | 6.13 | 12.08 | 34.56 | 71.83 |
| SNGP + BN-SN, c = 8 | 2.90 | 3.90 | 5.17 | 9.99 | 28.93 | 64.03 |
| SNGP + SpecReg, λ = 0.003 | 2.84 | 3.97 | 6.47 | 12.27 | 31.92 | 66.17 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **1.83** | **2.63** | **3.20** | **6.99** | 20.43 | 50.33 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 3.11 | 3.95 | 5.27 | 9.28 | 24.79 | 56.06 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 3.33 | 4.12 | 5.04 | 8.03 | **18.91** | **41.05** |

### Procedural artifact axis — ECE (×10⁻²) ↓ vs severity

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.15 | 0.22 | 0.40 | 0.92 | 3.51 | 8.30 |
| SNGP, c = 1 | 0.20 | 0.36 | 0.60 | 1.50 | 5.59 | 11.29 |
| SNGP + BN-SN, c = 8 | 0.20 | **0.13** | 0.44 | 0.94 | 4.50 | 9.81 |
| SNGP + SpecReg, λ = 0.003 | **0.13** | 0.28 | 0.65 | 1.28 | 4.51 | 10.26 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.22 | 0.29 | 0.34 | 1.06 | 3.14 | 8.24 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.25 | 0.23 | **0.12** | **0.56** | 3.38 | 8.95 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.23 | 0.30 | 0.21 | 0.59 | **2.22** | **6.56** |

## UC Davis-trained runs — 5 seeds

The same 5 arms, cells, seeds and recipe as above, trained on Wong `institution=ucdavis` only
(cells not re-tuned on UC Davis). Every model is scored on the Wong test split of each institution:
UC Davis is in-distribution, UPitt and UTSouthwestern are the same 4 classes under institution
shift. All GP checkpoints carry mean_field_factor π/8; no calibration. Trained 2026-10-05 with
`INSTITUTION=ucdavis scripts/tmux/adrc_wong_final.sh`, W&B group
`adrc_final_ucdavis_2026-10-05_21-33-51`. Checkpoints:
[../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md) (`adrc_*_wong_ucdavis`). Inference
dirs and reproduce commands: [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md)
(`adrc_wong_ucdavis_final`). Per-run numbers: `figures/wong_adrc/wong_adrc_ucdavis_runs.csv`.

The Muon → SGD two-stage row (cosine tail, aux 0.01) was trained on 2026-10-06 with
`INSTITUTION=ucdavis ARMS=muon2stage scripts/tmux/adrc_wong_final.sh`. Its seed 12345 is a retrain;
the seed-12345-only section below keeps the earlier run. Checkpoints
`adrc_muon2stage_wong_ucdavis_5seed`, inference `adrc_wong_ucdavis_muon2stage_5seed`.

| Arm | best.ckpt epoch (s12345, s1–s4) | W&B (s12345, s1–s4) |
|---|---|---|
| Baseline (linear head) | 137, 113, 112, 137, 126 | `mc65ywjq` `fftmscz4` `pvzvomh0` `x5hdcjg1` `qeqt80m9` |
| SNGP, c = 1 | 129, 134, 134, 131, 141 | `auzo15ep` `2s55i71f` `0oby1ehy` `o3cmjuqi` `3bmkkdht` |
| SNGP + BN-SN, c = 8 | 132, 139, 138, 137, 136 | `clc3b6yc` `v7w4dfrq` `svb5gsw5` `spey0eb9` `2ub13nea` |
| SNGP + SpecReg, λ = 0.003 | 121, 132, 136, 121, 139 | `n5t5f4jd` `cwwhlvsv` `nf32u6wr` `30cb8ihh` `rm55t8zc` |
| GP head + Muon, wd = 0.01 (aux 0.04) | 115, 132, 89, 145, 128 | `rgc7xeft` `axdomdjc` `xuqjdn6m` `7cgmje1n` `pu2p54w5` |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 129, 132, 149, 126, 137 | `nlpejivj` `74sp39jp` `5bn7raa7` `gf7onnx4` `xvln8nhw` |

### In-distribution — UC Davis test (n = 5,135)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.9925 ± 0.0022 | 0.9923 ± 0.0023 | 0.22 ± 0.06 | **0.19 ± 0.07** | 0.55 ± 0.14 | 2.56 ± 0.54 | 1.19 ± 0.27 |
| SNGP, c = 1 | 0.9919 ± 0.0014 | 0.9917 ± 0.0015 | 0.33 ± 0.07 | 0.21 ± 0.07 | 0.58 ± 0.11 | 2.66 ± 0.29 | 1.22 ± 0.17 |
| SNGP + BN-SN, c = 8 | 0.9922 ± 0.0001 | 0.9920 ± 0.0001 | 0.22 ± 0.04 | 0.20 ± 0.08 | 0.48 ± 0.04 | 2.81 ± 0.10 | 1.26 ± 0.04 |
| SNGP + SpecReg, λ = 0.003 | 0.9917 ± 0.0008 | 0.9914 ± 0.0010 | 0.36 ± 0.09 | 0.29 ± 0.12 | 0.57 ± 0.08 | 2.73 ± 0.23 | 1.25 ± 0.11 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **0.9938 ± 0.0021** | **0.9936 ± 0.0021** | **0.20 ± 0.10** | 0.25 ± 0.08 | **0.46 ± 0.10** | **2.18 ± 0.76** | **1.01 ± 0.38** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.9908 ± 0.0014 | 0.9906 ± 0.0013 | 0.92 ± 0.20 | 0.77 ± 0.24 | 1.02 ± 0.16 | 3.51 ± 0.15 | 1.49 ± 0.10 |

### Institution shift — UPitt test (n = 5,576)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.6592 ± 0.0162 | 0.6469 ± 0.0166 | 25.11 ± 0.83 | 24.85 ± 0.98 | 26.17 ± 1.09 | 151.57 ± 6.93 | 57.81 ± 2.64 |
| SNGP, c = 1 | 0.6696 ± 0.0090 | 0.6528 ± 0.0092 | 22.43 ± 1.47 | 22.36 ± 1.40 | 22.60 ± 1.50 | 125.05 ± 5.16 | 53.67 ± 1.51 |
| SNGP + BN-SN, c = 8 | 0.6716 ± 0.0209 | 0.6581 ± 0.0198 | 22.23 ± 2.29 | 21.98 ± 2.35 | 22.26 ± 2.47 | 121.12 ± 9.07 | 53.52 ± 3.29 |
| SNGP + SpecReg, λ = 0.003 | 0.6810 ± 0.0043 | **0.6651 ± 0.0029** | 20.73 ± 0.90 | 20.71 ± 0.87 | 21.00 ± 0.98 | 115.13 ± 3.02 | 51.11 ± 1.26 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.6569 ± 0.0208 | 0.6333 ± 0.0174 | 25.69 ± 1.88 | 25.56 ± 2.03 | 25.85 ± 1.94 | 148.92 ± 12.83 | 57.81 ± 3.73 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | **0.6860 ± 0.0109** | 0.6631 ± 0.0124 | **18.06 ± 0.75** | **17.88 ± 0.91** | **17.96 ± 0.92** | **104.46 ± 3.76** | **49.00 ± 1.49** |

### Institution shift — UTSouthwestern test (n = 3,299)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.7252 ± 0.0185 | 0.6827 ± 0.0239 | 21.68 ± 1.81 | 21.60 ± 1.74 | 23.87 ± 1.86 | 122.50 ± 11.91 | 47.20 ± 3.54 |
| SNGP, c = 1 | 0.7318 ± 0.0208 | 0.6800 ± 0.0210 | 19.82 ± 2.21 | 19.45 ± 2.04 | 20.01 ± 2.16 | 103.59 ± 12.84 | 44.32 ± 4.41 |
| SNGP + BN-SN, c = 8 | 0.7171 ± 0.0217 | 0.6762 ± 0.0326 | 20.94 ± 2.29 | 20.75 ± 2.48 | 21.23 ± 2.59 | 108.42 ± 7.51 | 47.21 ± 3.25 |
| SNGP + SpecReg, λ = 0.003 | 0.7336 ± 0.0165 | 0.6925 ± 0.0242 | 19.02 ± 1.59 | 18.82 ± 1.75 | 19.23 ± 1.64 | 102.40 ± 5.61 | 44.04 ± 2.62 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.7368 ± 0.0182 | 0.6835 ± 0.0218 | 19.88 ± 2.81 | 19.74 ± 2.69 | 19.99 ± 2.75 | 117.19 ± 16.05 | 45.60 ± 4.80 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | **0.7707 ± 0.0180** | **0.7295 ± 0.0235** | **13.92 ± 2.25** | **13.22 ± 1.88** | **13.29 ± 1.86** | **76.33 ± 6.09** | **36.70 ± 2.59** |

### Entropy AUROC ↑ — UC Davis test vs each other institution's test split

Same frozen 10-subsample protocol as above; the ± is across training seeds.

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| Baseline (linear head) | 0.780 ± 0.006 | 0.684 ± 0.026 | 0.732 ± 0.014 |
| SNGP, c = 1 | **0.832 ± 0.021** | 0.746 ± 0.026 | 0.789 ± 0.022 |
| SNGP + BN-SN, c = 8 | 0.829 ± 0.010 | **0.767 ± 0.010** | **0.798 ± 0.009** |
| SNGP + SpecReg, λ = 0.003 | 0.827 ± 0.019 | 0.755 ± 0.023 | 0.791 ± 0.019 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.824 ± 0.019 | 0.747 ± 0.037 | 0.785 ± 0.025 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.816 ± 0.024 | 0.740 ± 0.027 | 0.778 ± 0.023 |

### Entropy AUPR ↑ — UC Davis test vs each other institution's test split

The shifted institution is the positive class (as in `src/metrics/artifact_quantification.py`):
AUPR is its average precision (0.5 = chance on the balanced 1,000 + 1,000 subsamples), FPR95 the
share of UC Davis images flagged at the threshold that catches 95% of the shifted ones.

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| Baseline (linear head) | 0.821 ± 0.013 | 0.736 ± 0.026 | 0.778 ± 0.018 |
| SNGP, c = 1 | 0.850 ± 0.019 | 0.774 ± 0.021 | 0.812 ± 0.019 |
| SNGP + BN-SN, c = 8 | 0.853 ± 0.007 | 0.790 ± 0.009 | 0.822 ± 0.007 |
| SNGP + SpecReg, λ = 0.003 | 0.848 ± 0.016 | 0.780 ± 0.018 | 0.814 ± 0.016 |
| GP head + Muon, wd = 0.01 (aux 0.04) | **0.854 ± 0.013** | **0.791 ± 0.021** | **0.823 ± 0.013** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.842 ± 0.018 | 0.776 ± 0.019 | 0.809 ± 0.016 |

### Entropy FPR95 ↓ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| Baseline (linear head) | 0.896 ± 0.020 | 0.921 ± 0.017 | 0.908 ± 0.016 |
| SNGP, c = 1 | **0.718 ± 0.061** | 0.839 ± 0.054 | 0.779 ± 0.056 |
| SNGP + BN-SN, c = 8 | 0.743 ± 0.037 | **0.787 ± 0.019** | **0.765 ± 0.024** |
| SNGP + SpecReg, λ = 0.003 | 0.724 ± 0.037 | 0.813 ± 0.035 | 0.768 ± 0.029 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.794 ± 0.076 | 0.880 ± 0.091 | 0.837 ± 0.079 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.767 ± 0.052 | 0.864 ± 0.048 | 0.816 ± 0.047 |

### GP-variance AUROC ↑ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| SNGP, c = 1 | 0.513 ± 0.025 | 0.482 ± 0.033 | 0.497 ± 0.029 |
| SNGP + BN-SN, c = 8 | 0.497 ± 0.044 | 0.465 ± 0.058 | 0.481 ± 0.050 |
| SNGP + SpecReg, λ = 0.003 | 0.512 ± 0.033 | 0.525 ± 0.026 | 0.518 ± 0.029 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.508 ± 0.033 | 0.564 ± 0.027 | 0.536 ± 0.030 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | **0.606 ± 0.017** | **0.611 ± 0.032** | **0.609 ± 0.015** |

### GP-variance AUPR ↑ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| SNGP, c = 1 | 0.522 ± 0.026 | 0.483 ± 0.027 | 0.502 ± 0.027 |
| SNGP + BN-SN, c = 8 | 0.519 ± 0.038 | 0.477 ± 0.040 | 0.498 ± 0.038 |
| SNGP + SpecReg, λ = 0.003 | 0.518 ± 0.019 | 0.514 ± 0.020 | 0.516 ± 0.018 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.512 ± 0.025 | 0.545 ± 0.025 | 0.529 ± 0.025 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | **0.621 ± 0.025** | **0.607 ± 0.039** | **0.614 ± 0.020** |

### GP-variance FPR95 ↓ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| SNGP, c = 1 | 0.946 ± 0.013 | 0.946 ± 0.015 | 0.946 ± 0.014 |
| SNGP + BN-SN, c = 8 | 0.965 ± 0.016 | 0.966 ± 0.015 | 0.965 ± 0.015 |
| SNGP + SpecReg, λ = 0.003 | 0.941 ± 0.031 | 0.925 ± 0.030 | 0.933 ± 0.028 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.933 ± 0.027 | 0.948 ± 0.031 | 0.940 ± 0.028 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | **0.868 ± 0.017** | **0.864 ± 0.017** | **0.866 ± 0.013** |


## UC Davis-trained, Muon → SGD two-stage — seed 12345 only

Two-stage schedules on the UC Davis Muon arm, after Shen et al. 2026 (arXiv:2606.21514), against
the seed-12345 runs of the section above (no spread: one seed each). All rows share epochs 1–5
linear warmup from 0.1×, epochs 6–120 Muon 0.02 (wd 0.01) on the hidden convs + SGD + Nesterov
(L2 6e-4) at the aux LR (in brackets) on the stem / BN / GP output layer, and from epoch 121
every param on SGD + Nesterov (L2 6e-4, momentum reset). The rows differ in the tail and the
aux LR; the plain Muon arm's aux LR is 0.04. The epoch-135 row keeps Muon until epoch 135 and
switches from epoch 136 (15-epoch SGD tail).

| Row | Tail (epochs 121–150) | W&B |
|---|---|---|
| … cosine (aux 0.01) | cosine 0.01 → 1e-4, all groups | `2qjm7osr` |
| … epoch 135, cosine (aux 0.01) | as above, epochs 136–150 | `v72tm0sb` |
| … WSD (aux 0.01) | Wen et al. 2025 (arXiv:2410.05192) Eq. 8, each group's peak → 0.1× | `vuqxbvw3` |
| … WSD (aux 0.02) | as above | `pp4lo9i7` |
| … WSD (aux 0.04) | as above | `dv8p47r6` |
| Muon (aux 0.01), no switch | none: Muon + aux SGD 0.01 on the plain Muon arm's cosine; best.ckpt epoch 82 | `78kdh5jb` |

Launch: `INSTITUTION=ucdavis ARMS=<arm> SEEDS=12345 scripts/tmux/adrc_wong_final.sh` with arm
`muon2stage`, `muon2stage_sw135`, `muon2stage_wsd`, `muon2stage_wsd_aux02`, `muon2stage_wsd_aux04`, `muon_aux01`. Checkpoints / inference:
`adrc_muon2stage*_wong_ucdavis` / `adrc_wong_ucdavis_muon2stage*`. Per-run numbers:
`figures/wong_adrc/wong_adrc_ucdavis_muon2stage_wsd_aux02_s12345_runs.csv` and
`figures/wong_adrc/wong_adrc_ucdavis_muon2stage_sw135_s12345_runs.csv` and
`figures/wong_adrc/wong_adrc_ucdavis_muon_aux01_s12345{,_last}_runs.csv` (best.ckpt, last.ckpt).

### In-distribution — UC Davis test (n = 5,135)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | **0.9949** | **0.9949** | 0.22 | **0.14** | **0.43** | 2.00 | 0.90 |
| SNGP, c = 1 | 0.9895 | 0.9892 | 0.37 | 0.30 | 0.73 | 3.01 | 1.47 |
| SNGP + BN-SN, c = 8 | 0.9922 | 0.9921 | 0.23 | 0.24 | 0.50 | 2.81 | 1.26 |
| SNGP + SpecReg, λ = 0.003 | 0.9916 | 0.9914 | 0.49 | 0.42 | 0.69 | 2.91 | 1.32 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.9945 | 0.9944 | **0.21** | 0.15 | 0.47 | **1.78** | **0.82** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.9914 | 0.9912 | 0.89 | 0.86 | 1.04 | 3.43 | 1.44 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.9908 | 0.9906 | 1.01 | 1.01 | 1.09 | 3.83 | 1.71 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | 0.9914 | 0.9913 | 0.90 | 0.83 | 1.00 | 3.63 | 1.50 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.9936 | 0.9934 | 0.55 | 0.47 | 0.63 | 2.72 | 1.07 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.9924 | 0.9924 | 0.38 | 0.32 | 0.59 | 2.71 | 1.23 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.9914 | 0.9911 | 1.92 | 1.90 | 1.91 | 4.36 | 1.57 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.9944 | 0.9942 | 5.34 | 5.34 | 5.31 | 7.10 | 1.58 |

### Institution shift — UPitt test (n = 5,576)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.6336 | 0.6197 | 26.49 | 26.46 | 27.88 | 160.21 | 61.79 |
| SNGP, c = 1 | 0.6731 | 0.6556 | 22.70 | 22.70 | 22.99 | 124.60 | 52.69 |
| SNGP + BN-SN, c = 8 | 0.6917 | **0.6768** | 20.09 | 19.65 | 19.88 | 109.63 | 50.23 |
| SNGP + SpecReg, λ = 0.003 | 0.6749 | 0.6644 | 21.71 | 21.61 | 21.94 | 119.13 | 52.66 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.6876 | 0.6594 | 23.14 | 22.74 | 23.06 | 133.11 | 52.36 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.6840 | 0.6619 | 17.86 | 17.64 | 17.70 | 103.96 | 49.37 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.6698 | 0.6446 | 20.75 | 20.60 | 20.76 | 123.06 | 52.73 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | 0.6831 | 0.6619 | 19.39 | 18.79 | 18.86 | 106.06 | 50.01 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | **0.6942** | 0.6697 | 20.02 | 20.01 | 20.38 | 112.32 | 49.77 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.6539 | 0.6360 | 26.50 | 26.11 | 27.08 | 143.95 | 57.30 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.6903 | 0.6584 | 15.00 | 14.73 | 14.75 | 94.54 | **46.26** |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.6775 | 0.6535 | **12.73** | **12.38** | **11.14** | **91.68** | 47.94 |

### Institution shift — UTSouthwestern test (n = 3,299)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.7105 | 0.6584 | 23.35 | 23.05 | 25.50 | 135.28 | 50.16 |
| SNGP, c = 1 | 0.7396 | 0.6839 | 19.14 | 18.34 | 18.83 | 96.28 | 42.83 |
| SNGP + BN-SN, c = 8 | 0.7317 | 0.6936 | 19.51 | 19.33 | 19.83 | 104.46 | 45.54 |
| SNGP + SpecReg, λ = 0.003 | 0.7472 | 0.7129 | 17.50 | 17.33 | 17.88 | 94.49 | 41.08 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.7533 | 0.6982 | 17.36 | 17.36 | 17.40 | 108.77 | 42.12 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.7575 | 0.7158 | 14.96 | 14.42 | 14.61 | 83.01 | 38.85 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.7317 | 0.6845 | 17.28 | 16.86 | 17.38 | 97.98 | 43.47 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | **0.7630** | **0.7290** | 15.17 | 14.62 | 14.68 | 82.70 | 39.48 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.7593 | 0.7112 | 15.87 | 15.70 | 16.10 | 83.94 | **37.87** |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.7490 | 0.7075 | 18.68 | 18.59 | 19.78 | 104.74 | 42.48 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.7587 | 0.7120 | 12.68 | 12.70 | 11.53 | 77.49 | 38.94 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.7469 | 0.6875 | **11.60** | **11.31** | **10.81** | **75.25** | 39.83 |

### Entropy AUROC ↑ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| Baseline (linear head) | 0.787 | 0.680 | 0.733 |
| SNGP, c = 1 | 0.805 | 0.738 | 0.771 |
| SNGP + BN-SN, c = 8 | **0.839** | **0.767** | **0.803** |
| SNGP + SpecReg, λ = 0.003 | 0.810 | 0.719 | 0.765 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.797 | 0.737 | 0.767 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.813 | 0.724 | 0.768 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.784 | 0.673 | 0.728 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | 0.813 | 0.747 | 0.780 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.796 | 0.732 | 0.764 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.766 | 0.684 | 0.725 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.819 | 0.756 | 0.787 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.810 | 0.704 | 0.757 |

### Entropy AUPR ↑ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| Baseline (linear head) | 0.837 | 0.745 | 0.791 |
| SNGP, c = 1 | 0.822 | 0.764 | 0.793 |
| SNGP + BN-SN, c = 8 | **0.859** | 0.791 | **0.825** |
| SNGP + SpecReg, λ = 0.003 | 0.829 | 0.752 | 0.790 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.835 | **0.792** | 0.814 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.845 | 0.762 | 0.803 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.812 | 0.731 | 0.772 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | 0.837 | 0.769 | 0.803 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.824 | 0.765 | 0.794 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.793 | 0.727 | 0.760 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.840 | 0.786 | 0.813 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.841 | 0.743 | 0.792 |

### Entropy FPR95 ↓ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| Baseline (linear head) | 0.913 | 0.939 | 0.926 |
| SNGP, c = 1 | 0.772 | 0.867 | 0.820 |
| SNGP + BN-SN, c = 8 | **0.717** | **0.802** | **0.760** |
| SNGP + SpecReg, λ = 0.003 | 0.734 | 0.853 | 0.793 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.886 | 0.948 | 0.917 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.794 | 0.856 | 0.825 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.836 | 0.944 | 0.890 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | 0.775 | 0.822 | 0.798 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.808 | 0.857 | 0.832 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.828 | 0.918 | 0.873 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.801 | 0.883 | 0.842 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.860 | 0.914 | 0.887 |

### GP-variance AUROC ↑ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| SNGP, c = 1 | 0.519 | 0.484 | 0.502 |
| SNGP + BN-SN, c = 8 | 0.447 | 0.382 | 0.414 |
| SNGP + SpecReg, λ = 0.003 | 0.489 | 0.502 | 0.495 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.498 | 0.552 | 0.525 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.602 | 0.616 | 0.609 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.572 | 0.505 | 0.538 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | **0.620** | **0.644** | **0.632** |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.527 | 0.562 | 0.545 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.443 | 0.486 | 0.464 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.578 | 0.625 | 0.602 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.509 | 0.520 | 0.514 |

### GP-variance AUPR ↑ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| SNGP, c = 1 | 0.535 | 0.491 | 0.513 |
| SNGP + BN-SN, c = 8 | 0.482 | 0.423 | 0.452 |
| SNGP + SpecReg, λ = 0.003 | 0.508 | 0.491 | 0.499 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.504 | 0.524 | 0.514 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.633 | 0.628 | 0.631 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.605 | 0.496 | 0.550 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | **0.635** | **0.652** | **0.644** |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.549 | 0.548 | 0.549 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.464 | 0.488 | 0.476 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.624 | 0.629 | 0.626 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.538 | 0.514 | 0.526 |

### GP-variance FPR95 ↓ — UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| SNGP, c = 1 | 0.956 | 0.952 | 0.954 |
| SNGP + BN-SN, c = 8 | 0.987 | 0.988 | 0.987 |
| SNGP + SpecReg, λ = 0.003 | 0.956 | 0.905 | 0.931 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.919 | 0.936 | 0.928 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | **0.861** | 0.858 | 0.859 |
| GP head + Muon → SGD at epoch 135, cosine (aux 0.01) | 0.927 | 0.929 | 0.928 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.01) | 0.872 | **0.835** | **0.853** |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.02) | 0.919 | 0.931 | 0.925 |
| GP head + Muon → SGD at epoch 120, WSD (aux 0.04) | 0.954 | 0.955 | 0.954 |
| GP head + Muon, wd = 0.01 (aux 0.01) | 0.942 | 0.884 | 0.913 |
| GP head + Muon, wd = 0.01 (aux 0.01), last.ckpt | 0.962 | 0.957 | 0.959 |
