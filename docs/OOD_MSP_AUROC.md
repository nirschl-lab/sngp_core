# OOD AUROC (MSP-based)

Secondary OOD-detection tables using max-softmax-probability (MSP) as the
uncertainty score, moved out of [RESULTS.md](RESULTS.md) so that document keeps
entropy AUROC as the primary reported metric. See RESULTS.md for what MSP vs.
entropy AUROC measure and why they differ.

### Model Trained on Acevedo and Tested on other datasets

**MSP AUROC ↑**
| Model | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang | Wong |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.9575 ± 0.0026 | 0.4390 ± 0.0050 | 0.5046 ± 0.0110 | 0.0592 ± 0.0018 | 0.6343 ± 0.0112 | 0.7230 ± 0.0085 |
| Deep Ensemble | 0.9555 ± 0.0035 | 0.5300 ± 0.0041 | 0.4254 ± 0.0121 | 0.0504 ± 0.0016 | 0.8344 ± 0.0093 | 0.6202 ± 0.0135 |
| Monte Carlo Dropout | 0.9578 ± 0.0026 | 0.4602 ± 0.0051 | 0.5142 ± 0.0108 | 0.0689 ± 0.0019 | 0.6423 ± 0.0110 | 0.7362 ± 0.0082 |
| SNGP | 0.9639 ± 0.0020 | 0.8847 ± 0.0020 | 0.9685 ± 0.0031 | 0.9952 ± 0.0010 | 0.9620 ± 0.0035 | 0.9102 ± 0.0063 |
| SNGP Ensemble | **0.9856 ± 0.0012** | **0.9410 ± 0.0009** | **0.9915 ± 0.0009** | **0.9994 ± 0.0004** | **0.9846 ± 0.0016** | **0.9352 ± 0.0035** |

### Model Trained on Wong (All institutions) and Tested on other datasets

**MSP AUROC ↑**
| Model | Acevedo | Jung | Kather2016 | Kather2018 | Nirschl2018 | Tang |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Classifier | 0.7160 ± 0.0107 | 0.6457 ± 0.0072 | 0.6455 ± 0.0049 | 0.6506 ± 0.0154 | 0.8180 ± 0.0059 | 0.3698 ± 0.0051 |
| Deep Ensemble | 0.7975 ± 0.0036 | 0.5939 ± 0.0061 | 0.6782 ± 0.0031 | 0.5849 ± 0.0112 | 0.5144 ± 0.0052 | 0.3139 ± 0.0083 |
| Monte Carlo Dropout | 0.7207 ± 0.0107 | 0.6509 ± 0.0072 | 0.6517 ± 0.0050 | 0.6566 ± 0.0154 | 0.8238 ± 0.0059 | 0.3715 ± 0.0051 |
| SNGP | 0.9866 ± 0.0010 | 0.6115 ± 0.0077 | 0.9299 ± 0.0025 | 0.8697 ± 0.0073 | 0.9210 ± 0.0036 | **0.4433 ± 0.0090** |
| SNGP Ensemble | **0.9901 ± 0.0009** | **0.7420 ± 0.0060** | **0.9672 ± 0.0012** | **0.9103 ± 0.0068** | **0.9904 ± 0.0007** | 0.3974 ± 0.0052 |
