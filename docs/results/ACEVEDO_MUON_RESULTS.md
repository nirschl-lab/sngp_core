# Acevedo — GP head on an unconstrained backbone, trained with Muon

The SNGP random-feature GP head on a resnet18 with **no spectral normalization and no
spectral penalty**, trained with Muon (Jordan et al. 2024) on the hidden convs and AdamW on
the rest (`src/models/components/optimizers.py`). Every arm follows the `sngp_acevedo`
protocol: up to 150 epochs, early stopping on `val/nll_cal`, cosine LR, AdamW lr/wd at
Acevedo's swept values. Only the optimizer and the SN switch differ. Trained 2026-09-30 on
branch `acevedo-muon` with `scripts/tmux/acevedo_muon.sh`, W&B group
`Acevedo_muon_2026-09-30_12-47-49`.

| Arm | Experiment | Optimizer | W&B | Stopped at | best.ckpt epoch |
|---|---|---|---|---:|---:|
| Muon wd=0 | `sngp_muon_acevedo`, `model.optimizer.weight_decay=0.0` | Muon lr 0.02, wd 0 | `s7ldm15p` | 83 | 62 |
| Muon wd=0.1 | `sngp_muon_acevedo` | Muon lr 0.02, wd 0.1 | `ri2spl3f` | 150 | 139 |
| AdamW, no SN | `sngp_acevedo`, `use_spectral_norm=false` | AdamW (protocol) | `fkk75th9` | 57 | 36 |

All numbers below are single-seed and post-hoc calibrated (`mean_field_factor` fit on val
NLL). The SNGP `c* = 6.0` and SNGP + Spectral Reg rows are the calibrated rows of
[ACEVEDO_SPECREG_RESULTS.md](ACEVEDO_SPECREG_RESULTS.md#calibrated-comparison--sngp-c--60-vs-sngp--spectral-reg).
Run-to-run accuracy spread is about 0.006 ([ACEVEDO_RESULTS.md](ACEVEDO_RESULTS.md)).
Checkpoints: [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md). Inference
dirs: [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md).

### Fitted `mean_field_factor`

| Model | Fitted | val NLL | val smECE |
|---|---:|---|---|
| Muon wd=0 | 7.3992 | 0.04952 → 0.04525 | 0.09201 → 0.01691 |
| Muon wd=0.1 | 1.4269 | 0.03037 → 0.03035 | 0.02763 → 0.03074 |
| AdamW, no SN | 0 | 0.08709 → 0.08321 | 0.01835 → 0.01588 |

### In-distribution — Acevedo test split (n = 3,419)

| Model | Accuracy ↑ | F1 ↑ | ECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ |
|---|---:|---:|---:|---:|---:|
| SNGP `c* = 6.0` | 0.9874 ± 0.0019 | 0.9860 | **0.39** | 4.50 ± 0.63 | 2.01 ± 0.27 |
| SNGP + Spectral Reg | 0.9751 ± 0.0027 | 0.9691 | 0.64 | 6.81 ± 0.70 | 3.56 ± 0.35 |
| AdamW, no SN | 0.9757 ± 0.0026 | 0.9744 | 0.67 | 8.44 ± 0.80 | 3.86 ± 0.36 |
| Muon wd=0 | 0.9842 ± 0.0021 | 0.9822 | 0.54 | 5.60 ± 0.80 | 2.47 ± 0.32 |
| Muon wd=0.1 | **0.9915 ± 0.0016** | **0.9907** | 0.47 | **3.41 ± 0.65** | **1.50 ± 0.25** |

### Entropy AUROC ↑ — Acevedo test vs each OOD test set

| Model | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Wong | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SNGP `c* = 6.0` | 0.905 | 0.823 | 0.981 | 0.998 | 0.972 | 0.921 | 0.933 |
| SNGP + Spectral Reg | 0.969 | 0.949 | 0.975 | 0.992 | 0.904 | 0.935 | 0.954 |
| AdamW, no SN | 0.971 | 0.778 | 0.947 | 0.990 | 0.984 | 0.905 | 0.929 |
| Muon wd=0 | **0.989** | **0.984** | 0.983 | 0.998 | **1.000** | **0.976** | **0.988** |
| Muon wd=0.1 | 0.988 | 0.900 | **0.987** | **0.999** | 0.977 | 0.958 | 0.968 |

### Procedural artifact axis — NLL (×10⁻²) ↓ vs severity (0 = real test images)

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| SNGP `c* = 6.0` | 4.50 | 5.30 | 7.30 | 16.94 | 50.41 | 100.64 |
| SNGP + Spectral Reg | 6.81 | 7.56 | 9.83 | 18.70 | 41.06 | 72.79 |
| AdamW, no SN | 8.44 | 8.77 | 11.33 | 23.70 | 49.27 | 88.36 |
| Muon wd=0 | 5.60 | 6.01 | 7.58 | 13.02 | 26.75 | **52.48** |
| Muon wd=0.1 | **3.41** | **3.59** | **4.49** | **9.52** | **23.94** | 54.09 |

### Procedural artifact axis — Accuracy ↑ vs severity

| Model | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| SNGP `c* = 6.0` | 0.9874 | 0.9833 | 0.9772 | 0.9523 | 0.8833 | 0.7754 |
| SNGP + Spectral Reg | 0.9751 | 0.9757 | 0.9678 | 0.9377 | 0.8690 | 0.7833 |
| AdamW, no SN | 0.9757 | 0.9746 | 0.9667 | 0.9301 | 0.8725 | 0.7903 |
| Muon wd=0 | 0.9842 | 0.9833 | 0.9769 | 0.9634 | 0.9228 | 0.8502 |
| Muon wd=0.1 | **0.9915** | **0.9909** | **0.9874** | **0.9702** | **0.9286** | **0.8558** |

### Backbone spectra — hidden convs of each `best.ckpt`

Excludes the RGB stem conv, which is on AdamW in the Muon arms. σ is the top singular value
of the kernel reshaped to (out, in·k·k), a lower bound on the conv operator norm. "BN-folded"
multiplies each output channel by γ/√(running_var + ε) of the BatchNorm that follows the
conv, which is what the conv+BN pair applies at inference. Stable rank is
‖W‖²_F / σ², divided by min(out, in·k·k), BN-folded.

| Model | raw σ median / max | BN-folded σ median / max | BN-folded stable rank, median |
|---|---:|---:|---:|
| SNGP `c* = 6.0` | 6.00 / 6.00 | 1.69 / 5.82 | 0.046 |
| SNGP + Spectral Reg | 4.97 / 12.28 | 1.53 / 2.91 | 0.164 |
| AdamW, no SN | 58.71 / 289.08 | 1.50 / 6.35 | 0.030 |
| Muon wd=0 | 18.03 / 32.24 | 1.73 / 2.34 | 0.230 |
| Muon wd=0.1 | 1.04 / 1.59 | 2.35 / 4.90 | 0.129 |
