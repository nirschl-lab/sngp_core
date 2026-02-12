<div align="center">
  <h1>🧬 SNGP: Spectral-normalized Neural Gaussian Processes</h1>
  <p><em>Uncertainty quantification in medical imaging with robust out-of-distribution detection</em></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
  [![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-purple.svg)](https://lightning.ai)
  [![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2602.02370)
  [![Project](https://img.shields.io/badge/Project-Website-green.svg)](https://sngp.github.io)
  [![Models](https://img.shields.io/badge/Models-HF%20Hub-yellow.svg)](https://huggingface.co/nirschl-lab/sngp-models)
</div>

---

## 📰 News

🎉 **Our paper has been accepted to [ISBI 2026](https://biomedicalimaging.org/2026/)!**

- **Preprint**: [arXiv:2602.02370](https://arxiv.org/abs/2602.02370)
- **Project Website**: [sngp.github.io](https://sngp.github.io)
- **Pretrained Models**: [Hugging Face Hub](https://huggingface.co/nirschl-lab/sngp-models)

---

## 🚀 Quick Inference

Load pretrained SNGP models from Hugging Face Hub and run inference:

### Installation

```bash
# Clone repository
git clone <repository-url>
cd sngp_core

# Install uv
curl -Ls https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Python API

#### SNGP Inference (with uncertainty quantification)

```python
import torch
from scripts.example_inference import quick_sngp_inference

# Create dummy batch [batch_size, channels, height, width]
batch = torch.randn(4, 3, 224, 224)

# Load SNGP model from HF Hub and infer
results = quick_sngp_inference(
    "wong_sngp_resnet18",
    batch,
    device="cuda"  # or "cpu"
)

# Returns:
# - results["logits"]: Model predictions
# - results["predictions"]: Class predictions
# - results["confidence"]: Prediction confidence
# - results["variance"]: Uncertainty estimates
# - results["probabilities"]: Class probabilities

print(f"Predictions: {results['predictions'].tolist()}")
print(f"Confidence: {results['confidence'].tolist()}")
print(f"Uncertainty (variance): {results['variance'].tolist()}")
```

#### Baseline Inference (standard classifier)

```python
import torch
from scripts.example_inference import quick_baseline_inference

# Create dummy batch [batch_size, channels, height, width]
batch = torch.randn(4, 3, 224, 224)

# Load Baseline model from HF Hub and infer
results = quick_baseline_inference(
    "wong_baseline_resnet18",
    batch,
    device="cuda"  # or "cpu"
)

# Returns:
# - results["logits"]: Model predictions
# - results["predictions"]: Class predictions
# - results["confidence"]: Prediction confidence
# - results["probabilities"]: Class probabilities

print(f"Predictions: {results['predictions'].tolist()}")
print(f"Confidence: {results['confidence'].tolist()}")
```

### Command Line

```bash
# Run inference example
python scripts/example_inference.py
```

---

## 📦 Available Models

All models are available on [Hugging Face Hub](https://huggingface.co/nirschl-lab/sngp-models):

**SNGP Models (with uncertainty quantification):**
- `acevedo_sngp_resnet18` - Trained on [Acevedo et al. 2020](https://huggingface.co/datasets/nirschl-lab/acevedo_et_al_2020) (White Blood Cells)
- `wong_sngp_resnet18` - Trained on [Wong et al. 2022](https://huggingface.co/datasets/nirschl-lab/wong_et_al_2022) (Amyloid Plaques)
<!-- - `kather2018_sngp_resnet18` - Trained on [Kather et al. 2016/2018](https://huggingface.co/datasets/nirschl-lab/kather_et_al_2016) (Colorectal Histology) -->
<!-- - `tang_sngp_resnet18` - Trained on [Tang et al. 2019](https://huggingface.co/datasets/nirschl-lab/tang_et_al_2019) (Amyloid Plaques) -->

**Baseline Models (standard classifiers):**
- `acevedo_baseline_resnet18` - Trained on [Acevedo et al. 2020](https://huggingface.co/datasets/nirschl-lab/acevedo_et_al_2020) (White Blood Cells)
- `wong_baseline_resnet18` - Trained on [Wong et al. 2022](https://huggingface.co/datasets/nirschl-lab/wong_et_al_2022) (Amyloid Plaques)
<!-- - `kather2018_baseline_resnet18` - Trained on [Kather et al. 2016/2018](https://huggingface.co/datasets/nirschl-lab/kather_et_al_2016) (Colorectal Histology) -->
<!-- - `tang_baseline_resnet18` - Trained on [Tang et al. 2019](https://huggingface.co/datasets/nirschl-lab/tang_et_al_2019) (Amyloid Plaques) -->

---

## 📊 Datasets

Models are trained and evaluated on medical imaging datasets:

- **[Acevedo et al. 2020](https://huggingface.co/datasets/nirschl-lab/acevedo_et_al_2020)**: White Blood Cells
- **[Wong et al. 2022](https://huggingface.co/datasets/nirschl-lab/wong_et_al_2022)**: Amyloid Plaques
- **[Tang et al. 2019](https://huggingface.co/datasets/nirschl-lab/tang_et_al_2019)**: Amyloid Plaques
- **[Kather et al. 2016/2018](https://huggingface.co/datasets/nirschl-lab/kather_et_al_2016)**: Colorectal Histology

---

## 🏋️ Training & Development

For training, evaluation, and development setup, see [DEVELOPMENT.md](docs/DEVELOPMENT.md).

---

## 📚 Key Features

- ✅ **Uncertainty Quantification**: Obtain uncertainty estimates alongside predictions
- ✅ **Out-of-Distribution Detection**: Robust OOD detection using variance estimates
- ✅ **Multi-Dataset Evaluation**: Cross-dataset generalization testing
- ✅ **Pretrained Models**: Ready-to-use models on Hugging Face Hub
- ✅ **Easy Integration**: Simple Python API for inference

---

## 📖 Citation

If you use SNGP in your research, please cite:

```bibtex
@article{sngp2025,
  title={SNGP: Spectral-normalized Neural Gaussian Processes for Uncertainty Quantification in Medical Imaging},
  author={...},
  journal={ISBI 2025},
  year={2025},
  url={https://arxiv.org/abs/2602.02370}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [PyTorch Lightning](https://lightning.ai) for training
- [Hydra](https://hydra.cc) for configuration
- [Hugging Face](https://huggingface.co) for model hub

For more details, visit [sngp.github.io](https://sngp.github.io)