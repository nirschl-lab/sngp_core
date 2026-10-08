# Wong (ADRC) — SNGP variants vs a linear-head baseline, 5 seeds

The final runs of the ADRC study on Wong (4 classes). Every arm is trained over seeds
{12345, 1, 2, 3, 4}, and every cell is mean ± sample std over those 5 seeds. Two training sets:
full Wong (all institutions; OOD = the 6 other datasets) and Wong `institution=ucdavis` only
(institution shift = UPitt and UTSouthwestern).

Shared recipe: resnet18, 150 epochs, early stopping off, `best.ckpt` = min raw `val/nll`. SGD +
Nesterov (lr 0.04, wd 6e-4, cosine) unless the arm says otherwise. The GP arms use the CIFAR
reference head (ℓ = 20, σ² = 1, ridge 1, `binary_logistic` Laplace weight) with
`mean_field_factor` pinned at π/8 for training, selection and test. There is no post-hoc
calibration. Each arm uses the cell kept from its `adrc_wong_*` grid sweep
([../MASTER_SWEEPS.md](../MASTER_SWEEPS.md)); cells were not re-tuned on UC Davis.

Single-seed runs (two-stage switch at epoch 135, two-stage WSD tails, UC Davis Muon (aux 0.01)
best/last.ckpt), the seed-12345 artifact-robustness axes and the AUPR / FPR95 tables are in
[archive/WONG_ADRC_RESULTS.md](archive/WONG_ADRC_RESULTS.md).

| Arm | Experiment | Cell / recipe |
|---|---|---|
| Baseline (linear head) | `baseline_wong_sgd` | dropout 0 |
| SNGP | `sngp_wong_sgd` | c = 1 |
| SNGP + BN-SN | `sngp_bnsn_wong_sgd` | c = 8, BN cap tied |
| SNGP + SpecReg | `sngp_specreg_wong_sgd` | λ = 0.003, no SN |
| GP head + Muon (aux 0.04) | `sngp_muon_wong_sgd` | no SN; Muon (lr 0.02, wd 0.01) on the hidden convs, SGD + Nesterov lr 0.04 (L2 6e-4) on the stem, BN and GP output layer |
| GP head + Muon (aux 0.01) | `sngp_muon_wong_sgd` | as above with the aux SGD lr at 0.01, one cosine schedule |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | `sngp_muon_2stage_wong_sgd` | no SN; epochs 1–5 linear warmup from 0.1×, epochs 6–120 Muon (lr 0.02, wd 0.01) + aux SGD lr 0.01, from epoch 121 every param on SGD + Nesterov (L2 6e-4, momentum reset), cosine 0.01 → 1e-4 |

ECE is 10 equal-width bins (from inference); aECE is 10 equal-mass bins and smECE is the smooth
ECE of Błasiok & Nakkiran with the reflected kernel, both recomputed from `predictions.csv`
(`src/metrics/calibration_variants.py`). At ~99% accuracy smECE sits at its bandwidth floor, so
ID differences in it are small. Each seed's AUROC is the mean over the 10 frozen 1,000-row
subsamples of `src/metrics/calculate_ood_metrics.py`. GP-variance AUROC is scored on the GP
predictive variance (the `uncertainty` column), which does not depend on the mean-field factor;
below 0.5 means the variance is inverted (lower on OOD / shifted data).

## Full Wong (all institutions)

Launcher `scripts/tmux/adrc_wong_final.sh` (`ARMS=<arm>` for the later arms). W&B groups:
`adrc_final_2026-10-05_10-05-27` (first 5 arms, 2026-10-05), `adrc_final_2026-10-06_21-21-42`
(two-stage, `ARMS=muon2stage`), `adrc_final_2026-10-07_15-26-25` (Muon aux 0.01,
`ARMS=muon_aux01`). Checkpoints: [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md)
(`adrc_*_wong`, `adrc_muon2stage_wong`, `adrc_muon_aux01_wong`). Inference dirs and reproduce
commands: [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md) (`adrc_wong_final`,
`adrc_wong_final_muon2stage`, `adrc_wong_final_muon_aux01`). Per-run numbers:
`figures/wong_adrc/wong_adrc_runs.csv`.

| Arm | best.ckpt epoch (s12345, s1–s4) | W&B (s12345, s1–s4) |
|---|---|---|
| Baseline (linear head) | 138, 148, 141, 147, 147 | `glb9x1ee` `y5otcrt8` `ubudgj2e` `xc3s26hd` `07przvzo` |
| SNGP, c = 1 | 146, 141, 143, 146, 141 | `g02st7tz` `r3c4xpcg` `v7dp31o4` `yta98agx` `4upbt83i` |
| SNGP + BN-SN, c = 8 | 144, 146, 145, 145, 143 | `f3a6in72` `5qnxpq0e` `1prdt22h` `nxgyrx0j` `49kekadv` |
| SNGP + SpecReg, λ = 0.003 | 146, 146, 143, 143, 148 | `k4ptgv0v` `5m9ilprx` `zk7eifgn` `hph5tfeg` `2qx6mivs` |
| GP head + Muon, wd = 0.01 (aux 0.04) | 146, 141, 143, 146, 148 | `hdmg964v` `qxkd3rwn` `5kixgy2n` `ma6f545j` `9x9a7xp7` |
| GP head + Muon, wd = 0.01 (aux 0.01) | 149, 146, 143, 143, 148 | `r6kexhww` `agker8b9` `to1p5l16` `s9mtcegc` `h113eze9` |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 149, 142, 145, 146, 148 | `fcrtjgun` `7mnbytky` `25j3jpfs` `e22h8269` `njhdpvv7` |

### In-distribution — Wong test split (n = 14,010)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | aECE (×10⁻²) ↓ | smECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.9921 ± 0.0006 | 0.9921 ± 0.0006 | 0.15 ± 0.01 | 0.14 ± 0.08 | 0.37 ± 0.02 | 2.66 ± 0.25 | 1.27 ± 0.11 |
| SNGP, c = 1 | 0.9922 ± 0.0004 | 0.9921 ± 0.0004 | 0.18 ± 0.04 | 0.13 ± 0.10 | 0.38 ± 0.04 | 2.73 ± 0.21 | 1.27 ± 0.11 |
| SNGP + BN-SN, c = 8 | 0.9923 ± 0.0007 | 0.9923 ± 0.0007 | 0.19 ± 0.03 | **0.12 ± 0.04** | 0.41 ± 0.06 | 2.70 ± 0.17 | 1.22 ± 0.07 |
| SNGP + SpecReg, λ = 0.003 | 0.9923 ± 0.0006 | 0.9923 ± 0.0006 | **0.14 ± 0.02** | 0.14 ± 0.04 | 0.36 ± 0.03 | 2.68 ± 0.15 | 1.22 ± 0.08 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.9949 ± 0.0002 | 0.9949 ± 0.0002 | 0.22 ± 0.03 | 0.21 ± 0.03 | 0.34 ± 0.01 | 1.88 ± 0.08 | 0.83 ± 0.03 |
| GP head + Muon, wd = 0.01 (aux 0.01) | **0.9950 ± 0.0003** | **0.9950 ± 0.0003** | 0.15 ± 0.07 | 0.18 ± 0.01 | **0.33 ± 0.04** | **1.72 ± 0.15** | **0.77 ± 0.05** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.9909 ± 0.0008 | 0.9909 ± 0.0008 | 0.30 ± 0.10 | 0.21 ± 0.07 | 0.52 ± 0.08 | 3.00 ± 0.11 | 1.43 ± 0.06 |

### OOD — entropy AUROC ↑, Wong test vs each OOD test set

| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.930 ± 0.039 | 0.775 ± 0.097 | 0.809 ± 0.042 | 0.720 ± 0.036 | 0.718 ± 0.067 | 0.431 ± 0.028 | 0.731 ± 0.015 |
| SNGP, c = 1 | 0.945 ± 0.022 | 0.860 ± 0.035 | 0.891 ± 0.025 | 0.784 ± 0.019 | 0.822 ± 0.126 | 0.397 ± 0.036 | 0.783 ± 0.030 |
| SNGP + BN-SN, c = 8 | 0.850 ± 0.104 | 0.846 ± 0.097 | 0.835 ± 0.040 | 0.715 ± 0.052 | 0.806 ± 0.104 | 0.452 ± 0.029 | 0.750 ± 0.051 |
| SNGP + SpecReg, λ = 0.003 | 0.911 ± 0.059 | 0.781 ± 0.094 | 0.884 ± 0.014 | 0.731 ± 0.036 | 0.748 ± 0.065 | 0.390 ± 0.015 | 0.741 ± 0.035 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.975 ± 0.021 | 0.876 ± 0.083 | 0.956 ± 0.013 | 0.849 ± 0.031 | 0.960 ± 0.024 | **0.453 ± 0.060** | 0.845 ± 0.024 |
| GP head + Muon, wd = 0.01 (aux 0.01) | **0.980 ± 0.018** | **0.946 ± 0.016** | **0.978 ± 0.004** | **0.910 ± 0.021** | **0.982 ± 0.005** | 0.411 ± 0.021 | **0.868 ± 0.007** |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.891 ± 0.111 | 0.762 ± 0.096 | 0.855 ± 0.049 | 0.692 ± 0.060 | 0.868 ± 0.083 | 0.396 ± 0.026 | 0.744 ± 0.047 |

### OOD — GP-variance AUROC ↑, Wong test vs each OOD test set

| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SNGP, c = 1 | 0.747 ± 0.048 | 0.700 ± 0.133 | **0.924 ± 0.021** | 0.835 ± 0.022 | 0.845 ± 0.097 | **0.448 ± 0.040** | 0.750 ± 0.048 |
| SNGP + BN-SN, c = 8 | **0.937 ± 0.031** | 0.822 ± 0.064 | 0.912 ± 0.016 | **0.870 ± 0.041** | 0.887 ± 0.102 | 0.356 ± 0.030 | **0.798 ± 0.030** |
| SNGP + SpecReg, λ = 0.003 | 0.767 ± 0.075 | 0.751 ± 0.112 | 0.906 ± 0.020 | 0.823 ± 0.034 | 0.860 ± 0.054 | 0.420 ± 0.039 | 0.754 ± 0.036 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.719 ± 0.109 | 0.404 ± 0.153 | 0.710 ± 0.037 | 0.575 ± 0.032 | 0.493 ± 0.119 | 0.405 ± 0.110 | 0.551 ± 0.034 |
| GP head + Muon, wd = 0.01 (aux 0.01) | **0.937 ± 0.044** | 0.793 ± 0.071 | 0.842 ± 0.025 | 0.827 ± 0.030 | **0.956 ± 0.012** | 0.370 ± 0.052 | 0.788 ± 0.023 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.894 ± 0.098 | **0.828 ± 0.120** | 0.854 ± 0.024 | 0.772 ± 0.041 | 0.904 ± 0.030 | **0.448 ± 0.070** | 0.783 ± 0.045 |

## UC Davis-trained — institution shift

Trained on Wong `institution=ucdavis` only and scored on the Wong test split of each institution:
UC Davis is in-distribution, UPitt and UTSouthwestern are the same 4 classes under institution
shift. Launcher `INSTITUTION=ucdavis scripts/tmux/adrc_wong_final.sh`; W&B groups
`adrc_final_ucdavis_2026-10-05_21-33-51` (first 5 arms, 2026-10-05) and
`adrc_final_ucdavis_2026-10-06_21-21-44` (two-stage, `ARMS=muon2stage`). Checkpoints:
[../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md) (`adrc_*_wong_ucdavis`,
`adrc_muon2stage_wong_ucdavis_5seed`). Inference dirs and reproduce commands:
[../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md) (`adrc_wong_ucdavis_final`,
`adrc_wong_ucdavis_muon2stage_5seed`). Per-run numbers: `figures/wong_adrc/wong_adrc_ucdavis_runs.csv`.
GP head + Muon (aux 0.01) has only seed 12345 on UC Davis (see the archive).

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

### Shift detection — entropy AUROC ↑, UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| Baseline (linear head) | 0.780 ± 0.006 | 0.684 ± 0.026 | 0.732 ± 0.014 |
| SNGP, c = 1 | **0.832 ± 0.021** | 0.746 ± 0.026 | 0.789 ± 0.022 |
| SNGP + BN-SN, c = 8 | 0.829 ± 0.010 | **0.767 ± 0.010** | **0.798 ± 0.009** |
| SNGP + SpecReg, λ = 0.003 | 0.827 ± 0.019 | 0.755 ± 0.023 | 0.791 ± 0.019 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.824 ± 0.019 | 0.747 ± 0.037 | 0.785 ± 0.025 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | 0.816 ± 0.024 | 0.740 ± 0.027 | 0.778 ± 0.023 |

### Shift detection — GP-variance AUROC ↑, UC Davis test vs each other institution's test split

| Model | UPitt | UTSouthwestern | Mean |
|---|---:|---:|---:|
| SNGP, c = 1 | 0.513 ± 0.025 | 0.482 ± 0.033 | 0.497 ± 0.029 |
| SNGP + BN-SN, c = 8 | 0.497 ± 0.044 | 0.465 ± 0.058 | 0.481 ± 0.050 |
| SNGP + SpecReg, λ = 0.003 | 0.512 ± 0.033 | 0.525 ± 0.026 | 0.518 ± 0.029 |
| GP head + Muon, wd = 0.01 (aux 0.04) | 0.508 ± 0.033 | 0.564 ± 0.027 | 0.536 ± 0.030 |
| GP head + Muon → SGD at epoch 120, cosine (aux 0.01) | **0.606 ± 0.017** | **0.611 ± 0.032** | **0.609 ± 0.015** |
