# In-Distribution 

### Acevedo
| Model | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ | AUROC ↑ | AUPRC ↑ | ECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ | Precision (Micro) ↑ | Recall (Micro) ↑ | F1 (Micro) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.9813 | 0.9798 | 0.9814 | 0.9804 | 0.9996 | 0.9978 | 4.5967 | 9.4385 | — | 0.9813 | 0.9813 | 0.9813 |
| Deep Ensemble | **0.9889** | **0.9880** | **0.9888** | **0.9884** | **0.9999** | **0.9990** | **4.0026** | **6.8317** | — | **0.9889** | **0.9889** | **0.9889** |
| Monte Carlo Dropout | 0.9816 | 0.9802 | 0.9819 | 0.9808 | 0.9996 | 0.9978 | 4.7886 | 9.5676 | — | 0.9816 | 0.9816 | 0.9816 |
| SNGP | 0.9819 | 0.9819 | 0.9792 | 0.9803 | 0.9994 | 0.9963 | 7.6791 | 12.9793 | — | 0.9819 | 0.9819 | 0.9819 |
| SNGP Ensemble | 0.9880 | 0.9870 | 0.9878 | 0.9874 | 0.9997 | 0.9981 | 7.3047 | 10.8912 | — | 0.9880 | 0.9880 | 0.9880 |

### Wong (All institutions)
| Model | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ | AUROC ↑ | AUPRC ↑ | ECE (×10⁻²) ↓ | NLL (×10⁻²) ↓ | Brier (×10⁻²) ↓ | Precision (Micro) ↑ | Recall (Micro) ↑ | F1 (Micro) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.9818 | 0.9820 | 0.9818 | 0.9817 | 0.9981 | 0.9953 | 3.4511 | 8.0095 | 3.6862 | 0.9818 | 0.9818 | 0.9818 |
| Deep Ensemble | **0.9924** | **0.9925** | **0.9924** | **0.9923** | **0.9996** | **0.9989** | **2.3860** | **4.3133** | **1.5861** | **0.9924** | **0.9924** | **0.9924** |
| Monte Carlo Dropout | 0.9818 | 0.9821 | 0.9818 | 0.9817 | 0.9981 | 0.9953 | 3.5365 | 8.0981 | 3.7131 | 0.9818 | 0.9818 | 0.9818 |
| SNGP | 0.9878 | 0.9879 | 0.9878 | 0.9878 | 0.9991 | 0.9977 | 8.8451 | 12.1092 | 4.3607 | 0.9878 | 0.9878 | 0.9878 |
| SNGP Ensemble | 0.9904 | 0.9905 | 0.9904 | 0.9903 | 0.9996 | 0.9989 | 8.0570 | 10.5483 | 3.3336 | 0.9904 | 0.9904 | 0.9904 |
---

# Out-of-Distribution (OOD) settings

Reported OOD-detection numbers use entropy AUROC: the normalized Shannon entropy of
the full predicted class-probability vector, `H(p) / log(K)` for `K` classes, used
directly as the uncertainty score (higher entropy = more OOD-like, no inversion
needed). This differs from max-softmax-probability (MSP), which looks only at the
top predicted class's probability (`1 - max(p)` as the uncertainty score) and
ignores how probability mass is spread over the remaining classes. MSP-based tables
for the same runs are kept in [subdocs/OOD_MSP_AUROC.md](subdocs/OOD_MSP_AUROC.md) for
reference.

### Model Trained on Acevedo and Tested on other datasets

**Entropy AUROC ↑**
| Model | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Wong |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.9700 ± 0.0018 | 0.4382 ± 0.0053 | 0.5107 ± 0.0111 | 0.0573 ± 0.0018 | 0.6533 ± 0.0111 | 0.7280 ± 0.0079 |
| Deep Ensemble | 0.9633 ± 0.0026 | 0.5321 ± 0.0038 | 0.4255 ± 0.0122 | 0.0494 ± 0.0016 | 0.8487 ± 0.0088 | 0.6285 ± 0.0135 |
| Monte Carlo Dropout | 0.9705 ± 0.0018 | 0.4586 ± 0.0053 | 0.5203 ± 0.0109 | 0.0664 ± 0.0019 | 0.6620 ± 0.0109 | 0.7412 ± 0.0074 |
| SNGP | 0.9649 ± 0.0018 | 0.8894 ± 0.0019 | 0.9801 ± 0.0026 | 0.9973 ± 0.0009 | 0.9709 ± 0.0029 | 0.9230 ± 0.0056 |
| SNGP Ensemble | **0.9865 ± 0.0015** | **0.9494 ± 0.0009** | **0.9952 ± 0.0006** | **0.9999 ± 0.0001** | **0.9904 ± 0.0013** | **0.9486 ± 0.0029** |

### Model Trained on Wong (All institutions) and Tested on other datasets

**Entropy AUROC ↑**
| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.7229 ± 0.0104 | 0.6529 ± 0.0069 | 0.6548 ± 0.0045 | 0.6573 ± 0.0154 | 0.8252 ± 0.0054 | 0.3716 ± 0.0052 |
| Deep Ensemble | 0.8087 ± 0.0036 | 0.5983 ± 0.0060 | 0.6833 ± 0.0030 | 0.5875 ± 0.0113 | 0.5140 ± 0.0052 | 0.3152 ± 0.0083 |
| Monte Carlo Dropout | 0.7281 ± 0.0103 | 0.6588 ± 0.0070 | 0.6614 ± 0.0045 | 0.6638 ± 0.0155 | 0.8316 ± 0.0054 | 0.3734 ± 0.0052 |
| SNGP | 0.9974 ± 0.0004 | 0.6851 ± 0.0064 | 0.9587 ± 0.0012 | 0.9000 ± 0.0068 | 0.9648 ± 0.0018 | **0.4606 ± 0.0101** |
| SNGP Ensemble | **0.9989 ± 0.0003** | **0.7938 ± 0.0051** | **0.9808 ± 0.0006** | **0.9360 ± 0.0054** | **0.9974 ± 0.0003** | 0.4095 ± 0.0059 |

**Notes — how these numbers are computed** (`src/metrics/calculate_ood_metrics.py`,
`src/metrics/auc.py`):
- Each run directory has one `predictions.csv` per dataset (full test-set
  predictions from `src/inference/infer.py`, no fold filtering applied here). AUROC
  treats the in-distribution set as label 0 and each OOD set as label 1.
- To get a mean ± std rather than a single point estimate, AUROC is recomputed 10
  times over 10 fixed seeds (`42, 1337, 12345, 8675309, 314159, 271828, 20240427,
  987654321, 3735928559, 777`); each time, `min(1000, len(ID), len(OOD))` rows are
  sampled without replacement from the ID and OOD predictions (`pandas.sample`,
  seeded), scored, and passed to `sklearn.roc_auc_score`. The table reports the
  mean and standard deviation of those 10 AUROC values.
- Sampling is per-comparison, not shared: an ID dataset resampled for the Jung
  comparison is independent of the same ID dataset resampled for the Kather2016
  comparison.

---

# Different Scanner: Distribution shift due to different Institution
### Model Trained on Wong (ucdavis) and Tested Wong - upitt, utsouthwestern, Tang-ucdavis

---

# Artifact Simulation: Testing under the influence of artifacts

### Model Trained on Acevedo and tested with different artifact secnarios

### Model Trained on Wong and tested with different artifact secnarios

---

# Data visualization

### entropy plots

Predictive-entropy KDE per model, trained on Acevedo and evaluated across all
datasets (in-distribution vs. out-of-distribution).

<table>
<tr>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/baseline_acevedo_entropy.png" width="100%"><br>
Baseline
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/deep_ensemble_baseline_acevedo_entropy.png" width="100%"><br>
Deep Ensemble
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/mc_baseline_acevedo_entropy.png" width="100%"><br>
MC Dropout
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/sngp_acevedo_entropy.png" width="100%"><br>
SNGP
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/deep_ensemble_sngp_acevedo_entropy.png" width="100%"><br>
SNGP Ensemble
</td>
</tr>
</table>

Predictive-entropy KDE per model, trained on Wong (all institutions) and
evaluated across all datasets (in-distribution vs. out-of-distribution).

<table>
<tr>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/baseline_wong_entropy.png" width="100%"><br>
Baseline
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/deep_ensemble_baseline_wong_entropy.png" width="100%"><br>
Deep Ensemble
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/mc_baseline_wong_entropy.png" width="100%"><br>
MC Dropout
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/sngp_wong_entropy.png" width="100%"><br>
SNGP
</td>
<td align="center" width="20%">
<img src="../figures/predictive_entropy/deep_ensemble_sngp_wong_entropy.png" width="100%"><br>
SNGP Ensemble
</td>
</tr>
</table>

**Notes — how these plots are made** (`src/visualization/predictive_entropy.py`):
- Per dataset, 500 rows are sampled from `predictions.csv`, class-balanced by the
  `target` column (roughly `500 / num_classes` per class, seed 42; any shortfall
  from small classes is topped up from the remaining rows). This is a fixed
  sample size independent of each dataset's actual size, so datasets are shown at
  comparable density regardless of how many test rows they have.
- Per-row entropy is the raw (unnormalized) Shannon entropy in nats,
  `-Σ p·log(p)`, computed from the `class_probs` column — this is on a different
  scale than the `[0, 1]`-normalized entropy used for AUROC above, since the plot
  is about comparing shape/spread across datasets, not producing a bounded score.
- Each dataset's 500 entropy values are turned into a KDE (`scipy.stats.gaussian_kde`,
  Scott's rule bandwidth) evaluated on a shared x-range across all datasets in the
  plot (padded 5% beyond the global min/max), so curves are directly comparable.
  The in-distribution dataset is drawn as a solid line, OOD datasets as dashed
  lines, colored by `DATASET_COLORS` in `src/visualization/style.py`.
