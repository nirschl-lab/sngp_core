<div align="center">
	<h1>SNGP Lightning + Hydra Experimentation Framework</h1>
</div>

---

## 🚀 Overview

This repository provides a flexible experimentation framework for SNGP models using PyTorch Lightning and Hydra. It supports modular configuration, easy experiment tracking, and reproducible research.

---


## 🛠️ Setup & Reproducibility with uv

### 1. Install uv (if not already installed)
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Install all dependencies exactly as locked
```bash
uv sync
```

This will install all packages as specified in `uv.lock` and `pyproject.toml` for full reproducibility.

---

## ▶️ Training

Run a default training experiment:

```bash
uv run src/train.py
```

By default, this uses the Acevedo dataset, the Acevedo baseline model, and parameters from the default config.

### Customizing Training

Override any config from the command line (names must match config keys):

```bash
uv run src/train.py model=custom_sngp data=acevedo trainer.max_epochs=15 model.optimizer.lr=1e-4 callbacks=default
```

Add or modify callbacks (see `configs/callbacks/` for options):

---

## 🧪 Evaluation

Run evaluation on a dataset:

```bash
uv run src/eval.py data.dataset=tang_et_al_2019
```

---

## ⚙️ Configuration

All experiment settings are managed via Hydra configs in the `configs/` directory. See subfolders for models, data, callbacks, trainers, and more.

---

## 📊 Experiment Tracking

Track your runs and results with [Weights & Biases](https://wandb.ai/):

- [Project Dashboard](https://wandb.ai/maheswararao-university-of-wisconsin-madison/sngp_core?nw=nwusermaheswararao)

---

## 📁 Useful Paths

- Training configs: `configs/train.yaml`
- Callback configs: `configs/callbacks/`
- Model configs: `configs/model/`
- Data configs: `configs/data/`

---