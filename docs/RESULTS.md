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

### Model Trained on Acevedo and Tested on other datasets

**MSP AUROC ↑**
| Model | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Wong |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.9575 ± 0.0026 | 0.4390 ± 0.0050 | 0.5046 ± 0.0110 | 0.0592 ± 0.0018 | 0.6343 ± 0.0112 | 0.7230 ± 0.0085 |
| Deep Ensemble | 0.9555 ± 0.0035 | 0.5300 ± 0.0041 | 0.4254 ± 0.0121 | 0.0504 ± 0.0016 | 0.8344 ± 0.0093 | 0.6202 ± 0.0135 |
| Monte Carlo Dropout | 0.9578 ± 0.0026 | 0.4602 ± 0.0051 | 0.5142 ± 0.0108 | 0.0689 ± 0.0019 | 0.6423 ± 0.0110 | 0.7362 ± 0.0082 |
| SNGP | 0.9639 ± 0.0020 | 0.8847 ± 0.0020 | 0.9685 ± 0.0031 | 0.9952 ± 0.0010 | 0.9620 ± 0.0035 | 0.9102 ± 0.0063 |
| SNGP Ensemble | **0.9856 ± 0.0012** | **0.9410 ± 0.0009** | **0.9915 ± 0.0009** | **0.9994 ± 0.0004** | **0.9846 ± 0.0016** | **0.9352 ± 0.0035** |

**Entropy AUROC ↑**
| Model | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Wong |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.9700 ± 0.0018 | 0.4382 ± 0.0053 | 0.5107 ± 0.0111 | 0.0573 ± 0.0018 | 0.6533 ± 0.0111 | 0.7280 ± 0.0079 |
| Deep Ensemble | 0.9633 ± 0.0026 | 0.5321 ± 0.0038 | 0.4255 ± 0.0122 | 0.0494 ± 0.0016 | 0.8487 ± 0.0088 | 0.6285 ± 0.0135 |
| Monte Carlo Dropout | 0.9705 ± 0.0018 | 0.4586 ± 0.0053 | 0.5203 ± 0.0109 | 0.0664 ± 0.0019 | 0.6620 ± 0.0109 | 0.7412 ± 0.0074 |
| SNGP | 0.9649 ± 0.0018 | 0.8894 ± 0.0019 | 0.9801 ± 0.0026 | 0.9973 ± 0.0009 | 0.9709 ± 0.0029 | 0.9230 ± 0.0056 |
| SNGP Ensemble | **0.9865 ± 0.0015** | **0.9494 ± 0.0009** | **0.9952 ± 0.0006** | **0.9999 ± 0.0001** | **0.9904 ± 0.0013** | **0.9486 ± 0.0029** |

### Model Trained on Wong (All institutions) and Tested on other datasets

**MSP AUROC ↑**
| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.7160 ± 0.0107 | 0.6457 ± 0.0072 | 0.6455 ± 0.0049 | 0.6506 ± 0.0154 | 0.8180 ± 0.0059 | 0.3698 ± 0.0051 |
| Deep Ensemble | 0.7975 ± 0.0036 | 0.5939 ± 0.0061 | 0.6782 ± 0.0031 | 0.5849 ± 0.0112 | 0.5144 ± 0.0052 | 0.3139 ± 0.0083 |
| Monte Carlo Dropout | 0.7207 ± 0.0107 | 0.6509 ± 0.0072 | 0.6517 ± 0.0050 | 0.6566 ± 0.0154 | 0.8238 ± 0.0059 | 0.3715 ± 0.0051 |
| SNGP | 0.9866 ± 0.0010 | 0.6115 ± 0.0077 | 0.9299 ± 0.0025 | 0.8697 ± 0.0073 | 0.9210 ± 0.0036 | **0.4433 ± 0.0090** |
| SNGP Ensemble | **0.9901 ± 0.0009** | **0.7420 ± 0.0060** | **0.9672 ± 0.0012** | **0.9103 ± 0.0068** | **0.9904 ± 0.0007** | 0.3974 ± 0.0052 |

**Entropy AUROC ↑**
| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.7229 ± 0.0104 | 0.6529 ± 0.0069 | 0.6548 ± 0.0045 | 0.6573 ± 0.0154 | 0.8252 ± 0.0054 | 0.3716 ± 0.0052 |
| Deep Ensemble | 0.8087 ± 0.0036 | 0.5983 ± 0.0060 | 0.6833 ± 0.0030 | 0.5875 ± 0.0113 | 0.5140 ± 0.0052 | 0.3152 ± 0.0083 |
| Monte Carlo Dropout | 0.7281 ± 0.0103 | 0.6588 ± 0.0070 | 0.6614 ± 0.0045 | 0.6638 ± 0.0155 | 0.8316 ± 0.0054 | 0.3734 ± 0.0052 |
| SNGP | 0.9974 ± 0.0004 | 0.6851 ± 0.0064 | 0.9587 ± 0.0012 | 0.9000 ± 0.0068 | 0.9648 ± 0.0018 | **0.4606 ± 0.0101** |
| SNGP Ensemble | **0.9989 ± 0.0003** | **0.7938 ± 0.0051** | **0.9808 ± 0.0006** | **0.9360 ± 0.0054** | **0.9974 ± 0.0003** | 0.4095 ± 0.0059 |

### Model Trained on Tang (full-dataset) and Tested on other datasets 

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
