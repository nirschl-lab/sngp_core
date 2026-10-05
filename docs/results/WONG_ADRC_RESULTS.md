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

| Arm | Experiment | Cell | best.ckpt epoch (s12345, s1–s4) | W&B (s12345, s1–s4) |
|---|---|---|---|---|
| Baseline (linear head) | `baseline_wong_sgd` | dropout 0 | 138, 148, 141, 147, 147 | `glb9x1ee` `y5otcrt8` `ubudgj2e` `xc3s26hd` `07przvzo` |
| SNGP | `sngp_wong_sgd` | c = 1 | 146, 141, 143, 146, 141 | `g02st7tz` `r3c4xpcg` `v7dp31o4` `yta98agx` `4upbt83i` |
| SNGP + BN-SN | `sngp_bnsn_wong_sgd` | c = 8, BN cap tied | 144, 146, 145, 145, 143 | `f3a6in72` `5qnxpq0e` `1prdt22h` `nxgyrx0j` `49kekadv` |
| SNGP + SpecReg | `sngp_specreg_wong_sgd` | λ = 0.003, no SN | 146, 146, 143, 143, 148 | `k4ptgv0v` `5m9ilprx` `zk7eifgn` `hph5tfeg` `2qx6mivs` |
| GP head + Muon | `sngp_muon_wong_sgd` | Muon wd = 0.01, no SN | 146, 141, 143, 146, 148 | `hdmg964v` `qxkd3rwn` `5kixgy2n` `ma6f545j` `9x9a7xp7` |

Every cell is mean ± sample std over the 5 training seeds.

### In-distribution — Wong test split (n = 14,010)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.9921 ± 0.0006 | 0.9921 ± 0.0006 | 0.15 ± 0.01 | 2.66 ± 0.25 | 1.27 ± 0.11 |
| SNGP, c = 1 | 0.9922 ± 0.0004 | 0.9921 ± 0.0004 | 0.18 ± 0.04 | 2.73 ± 0.21 | 1.27 ± 0.11 |
| SNGP + BN-SN, c = 8 | 0.9923 ± 0.0007 | 0.9923 ± 0.0007 | 0.19 ± 0.03 | 2.70 ± 0.17 | 1.22 ± 0.07 |
| SNGP + SpecReg, λ = 0.003 | 0.9923 ± 0.0006 | 0.9923 ± 0.0006 | **0.14 ± 0.02** | 2.68 ± 0.15 | 1.22 ± 0.08 |
| GP head + Muon, wd = 0.01 | **0.9949 ± 0.0002** | **0.9949 ± 0.0002** | 0.22 ± 0.03 | **1.88 ± 0.08** | **0.83 ± 0.03** |

### Entropy AUROC ↑ — Wong test vs each OOD test set

Each seed's AUROC is the mean over the 10 frozen 1,000-row subsamples of
`src/metrics/calculate_ood_metrics.py`. The ± is across training seeds.

| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (linear head) | 0.930 ± 0.039 | 0.775 ± 0.097 | 0.809 ± 0.042 | 0.720 ± 0.036 | 0.718 ± 0.067 | 0.431 ± 0.028 | 0.731 ± 0.015 |
| SNGP, c = 1 | 0.945 ± 0.022 | 0.860 ± 0.035 | 0.891 ± 0.025 | 0.784 ± 0.019 | 0.822 ± 0.126 | 0.397 ± 0.036 | 0.783 ± 0.030 |
| SNGP + BN-SN, c = 8 | 0.850 ± 0.104 | 0.846 ± 0.097 | 0.835 ± 0.040 | 0.715 ± 0.052 | 0.806 ± 0.104 | 0.452 ± 0.029 | 0.750 ± 0.051 |
| SNGP + SpecReg, λ = 0.003 | 0.911 ± 0.059 | 0.781 ± 0.094 | 0.884 ± 0.014 | 0.731 ± 0.036 | 0.748 ± 0.065 | 0.390 ± 0.015 | 0.741 ± 0.035 |
| GP head + Muon, wd = 0.01 | **0.975 ± 0.021** | **0.876 ± 0.083** | **0.956 ± 0.013** | **0.849 ± 0.031** | **0.960 ± 0.024** | **0.453 ± 0.060** | **0.845 ± 0.024** |

### GP-variance AUROC ↑ — Wong test vs each OOD test set

Same protocol, scored on the GP predictive variance (the `uncertainty` column), which does not
depend on the mean-field factor. Below 0.5 means the variance is inverted (lower on OOD).

| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SNGP, c = 1 | 0.747 ± 0.048 | 0.700 ± 0.133 | **0.924 ± 0.021** | 0.835 ± 0.022 | 0.845 ± 0.097 | **0.448 ± 0.040** | 0.750 ± 0.048 |
| SNGP + BN-SN, c = 8 | **0.937 ± 0.031** | **0.822 ± 0.064** | 0.912 ± 0.016 | **0.870 ± 0.041** | **0.887 ± 0.102** | 0.356 ± 0.030 | **0.798 ± 0.030** |
| SNGP + SpecReg, λ = 0.003 | 0.767 ± 0.075 | 0.751 ± 0.112 | 0.906 ± 0.020 | 0.823 ± 0.034 | 0.860 ± 0.054 | 0.420 ± 0.039 | 0.754 ± 0.036 |
| GP head + Muon, wd = 0.01 | 0.719 ± 0.109 | 0.404 ± 0.153 | 0.710 ± 0.037 | 0.575 ± 0.032 | 0.493 ± 0.119 | 0.405 ± 0.110 | 0.551 ± 0.034 |
