#!/usr/bin/env python3
"""sngp_make_moons.py in scripts.

Two-moons SNGP demo, a PyTorch port of the TensorFlow tutorial "Uncertainty-aware Deep
Learning with SNGP" (https://www.tensorflow.org/tutorials/understanding/sngp). It trains
the tutorial's deterministic `DeepResNet` and its SNGP variant on the same data and plots
each model's class-probability and predictive-uncertainty surfaces, with an OOD cloud
overlaid. The SNGP head is this project's `RandomFeatureGaussianProcess`, configured to
match the tutorial's `nlp_layers.RandomFeatureGaussianProcess` defaults.

Two arms go beyond the tutorial: `sngp_muon`, the project's Muon arm in miniature -- the
same GP head with no spectral norm, Muon on the hidden dense layers and SGD on everything
else (`MuonWithAuxSGD`, the `muon_aux01` recipe at a constant LR) -- and `sngp_muon_adam`,
the same with the tutorial's Adam on everything else (`MuonWithAuxAdamW`).

    uv run scripts/sngp_make_moons.py                  # writes figures/sngp_moons/*.png
    uv run scripts/sngp_make_moons.py --seeds 0 1 42 --show
    uv run scripts/sngp_make_moons.py --models sngp sngp_muon
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.datasets import make_moons
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.components.optimizers import MuonWithAuxAdamW, MuonWithAuxSGD  # noqa: E402
from src.models.components.spectral_norm import bounded_spectral_norm  # noqa: E402
from src.models.outputs import ModelOutput  # noqa: E402
from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess  # noqa: E402
from src.visualization.style import set_default_style  # noqa: E402

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1)
DEFAULT_N_GRID = 100

# Tutorial hyperparameters.
RESNET_CONFIG = dict(num_classes=2, num_layers=6, num_hidden=128, dropout_rate=0.1)
SPEC_NORM_BOUND = 0.9
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 100
# `nlp_layers.RandomFeatureGaussianProcess` defaults: num_inducing=1024, gp_kernel_scale=1
# (l = 1), gp_cov_ridge_penalty=1, normalize_input=False, scale_random_features=True,
# Gaussian random features, a fixed zero output bias, l2_regularization=1e-6.
GP_CONFIG = dict(
    rff_dim=1024,
    length_scale=1.0,
    ridge_penalty=1.0,
    cov_momentum=-1.0,
    normalize_input=False,
    scale_random_features=True,
    random_feature_type="rff",
    output_bias=False,
    likelihood="gaussian",
    mean_field=True,
    mean_field_factor=math.pi / 8,  # the tutorial's `lambda_param`
)
GP_L2_REGULARIZATION = 1e-6
# `muon_aux01` (scripts/tmux/adrc_wong_final.sh) without the cosine schedule, which the
# tutorial does not use: Muon on the hidden layers, SGD + Nesterov on biases and the GP
# output layer. The aux L2 is left to GP_L2_REGULARIZATION, as in the other SNGP arm.
MUON_CONFIG = dict(
    lr=0.02,
    momentum=0.95,
    weight_decay=0.01,
    nesterov=True,
    sgd_lr=0.01,
    sgd_momentum=0.9,
    sgd_nesterov=True,
    sgd_weight_decay=0.0,
)
# Adam aux (`MuonWithAuxAdamW`) at the tutorial's Adam lr, so this arm differs from `sngp`
# only in the hidden layers' optimizer and the dropped spectral norm. The Muon lr is 10x
# below the SGD arm's: at 0.02 every step moves the hidden layers by a spectral norm of 0.02
# while Adam at 1e-4 barely moves the GP output layer, and training stalls at chance (seed 0)
# or ~89% (seed 1). At 0.002 the hidden norms stay ~1.9 and both seeds reach 100%.
MUON_ADAM_CONFIG = dict(
    lr=0.002,
    momentum=0.95,
    weight_decay=0.01,
    nesterov=True,
    adamw_lr=LEARNING_RATE,
    adamw_weight_decay=0.0,
)


# ------------- Data ------------- #
def make_training_data(sample_size: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Two moons, each pushed slightly apart."""
    train_examples, train_labels = make_moons(n_samples=2 * sample_size, noise=0.1)
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]
    return train_examples.astype(np.float32), train_labels


def make_testing_data(
    x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid: int = DEFAULT_N_GRID
) -> np.ndarray:
    """Mesh grid over the data space."""
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1).astype(np.float32)


def make_ood_data(sample_size: int = 500, means=(2.5, -1.75), vars=(0.01, 0.01)) -> np.ndarray:
    return np.random.multivariate_normal(means, cov=np.diag(vars), size=sample_size).astype(np.float32)


# ------------- Models ------------- #
def _keras_dense(in_dim: int, out_dim: int) -> nn.Linear:
    """`nn.Linear` with Keras `Dense` initialization (glorot-uniform kernel, zero bias)."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DeepResNet(nn.Module):
    """Frozen input projection, residual ReLU dense layers with dropout, dense output."""

    def __init__(
        self, num_classes: int, num_layers: int = 3, num_hidden: int = 128, dropout_rate: float = 0.1, in_dim: int = 2
    ):
        super().__init__()
        self.num_hidden = num_hidden
        # The input layer is not trainable: it only lifts the 2-D input to num_hidden.
        self.input_layer = _keras_dense(in_dim, num_hidden).requires_grad_(False)
        self.dense_layers = nn.ModuleList([self.make_dense_layer() for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = self.make_output_layer(num_classes)

    def make_dense_layer(self) -> nn.Module:
        return _keras_dense(self.num_hidden, self.num_hidden)

    def make_output_layer(self, num_classes: int) -> nn.Module:
        return _keras_dense(self.num_hidden, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(x)
        for layer in self.dense_layers:
            hidden = hidden + self.dropout(F.relu(layer(hidden)))
        return hidden

    def forward(self, x: torch.Tensor) -> ModelOutput:
        return ModelOutput(logits=self.classifier(self.features(x)))


class DeepResNetSNGP(DeepResNet):
    """`DeepResNet` with spectral-normalized hidden layers and an RFF-GP output layer.

    `spec_norm_bound=None` drops the spectral norm (the Muon arm).
    """

    def __init__(self, spec_norm_bound: Optional[float] = SPEC_NORM_BOUND, **kwargs):
        self.spec_norm_bound = spec_norm_bound
        super().__init__(**kwargs)

    def make_dense_layer(self) -> nn.Module:
        layer = super().make_dense_layer()
        if self.spec_norm_bound is None:
            return layer
        return bounded_spectral_norm(layer, bound=self.spec_norm_bound)

    def make_output_layer(self, num_classes: int) -> nn.Module:
        return RandomFeatureGaussianProcess(in_dim=self.num_hidden, num_classes=num_classes, **GP_CONFIG)

    def reset_precision(self) -> None:
        self.classifier.reset_precision()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # Train mode: raw logits, precision accumulated. Eval mode: mean-field logits.
        logits, raw_logits, variance = self.classifier(self.features(x))
        return ModelOutput(logits=logits, raw_logits=raw_logits, variance=variance)


# ------------- Train / predict ------------- #
def make_optimizer(model: DeepResNet, optimizer: str) -> torch.optim.Optimizer:
    hidden = {id(layer.weight) for layer in model.dense_layers}
    if optimizer == "muon_sgd":
        return MuonWithAuxSGD(model.parameters(), is_muon=lambda p: id(p) in hidden, **MUON_CONFIG)
    if optimizer == "muon_adam":
        return MuonWithAuxAdamW(model.parameters(), is_muon=lambda p: id(p) in hidden, **MUON_ADAM_CONFIG)
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)


@torch.no_grad()
def hidden_spectral_norms(model: DeepResNet) -> list[float]:
    """Largest singular value of each hidden layer's effective weight (post-SN, if any)."""
    model.eval()
    model(torch.zeros(1, 2))  # refresh SN's cached `.weight` without a power iteration
    return [torch.linalg.matrix_norm(layer.weight, ord=2).item() for layer in model.dense_layers]


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> None:
    x_t, y_t = torch.from_numpy(x), torch.from_numpy(y).long()
    is_sngp = isinstance(model, DeepResNetSNGP)

    model.train()
    for epoch in range(epochs):
        # The tutorial's `ResetCovarianceCallback`: the precision matrix is accumulated
        # over one epoch, so reset it at the start of each one. The last epoch's counts.
        if is_sngp:
            model.reset_precision()
        perm = torch.randperm(len(x_t))
        total_loss, correct = 0.0, 0
        for i in range(0, len(x_t), batch_size):
            idx = perm[i : i + batch_size]
            logits = model(x_t[idx]).logits
            loss = F.cross_entropy(logits, y_t[idx])
            if is_sngp:
                loss = loss + GP_L2_REGULARIZATION * model.classifier.classifier.weight.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(-1) == y_t[idx]).sum().item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"epoch {epoch + 1:3d}/{epochs}  loss {total_loss / len(x_t):.4f}  acc {correct / len(x_t):.4f}")


@torch.no_grad()
def predict_class0_probs(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """p(class 0) on `x`; SNGP's logits are already mean-field adjusted in eval mode."""
    model.eval()
    logits = model(torch.from_numpy(x)).logits
    return F.softmax(logits, dim=-1)[:, 0].numpy()


# ------------- Plotting ------------- #
def plot_uncertainty_surface(test_uncertainty, ax, train_examples, train_labels, ood_examples, cmap="viridis"):
    # Normalize uncertainty for better visualization.
    test_uncertainty = test_uncertainty / np.max(test_uncertainty)

    ax.set_ylim(DEFAULT_Y_RANGE)
    ax.set_xlim(DEFAULT_X_RANGE)
    ax.grid(False)

    pcm = ax.imshow(
        np.reshape(test_uncertainty, [DEFAULT_N_GRID, DEFAULT_N_GRID]),
        cmap=cmap,
        origin="lower",
        extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
        vmin=DEFAULT_NORM.vmin,
        vmax=DEFAULT_NORM.vmax,
        interpolation="bicubic",
        aspect="auto",
    )
    ax.scatter(train_examples[:, 0], train_examples[:, 1], c=train_labels, cmap=DEFAULT_CMAP, alpha=0.5)
    ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)
    return pcm


def plot_predictions(pred_probs, model_name, train_examples, train_labels, ood_examples):
    """Class probability and normalized predictive uncertainty p(1 - p) side by side."""
    uncertainty = pred_probs * (1.0 - pred_probs)
    data = (train_examples, train_labels, ood_examples)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    pcm_0 = plot_uncertainty_surface(pred_probs, axs[0], *data)
    pcm_1 = plot_uncertainty_surface(uncertainty, axs[1], *data)
    fig.colorbar(pcm_0, ax=axs[0])
    fig.colorbar(pcm_1, ax=axs[1])
    axs[0].set_title(f"Class Probability, {model_name}")
    axs[1].set_title(f"(Normalized) Predictive Uncertainty, {model_name}")
    fig.tight_layout()
    return fig


# ------------- Main ------------- #
# tag -> (plot title, model factory, optimizer)
MODELS = {
    "resnet": ("Deterministic Model", lambda: DeepResNet(**RESNET_CONFIG), "adam"),
    "sngp": ("SNGP", lambda: DeepResNetSNGP(spec_norm_bound=SPEC_NORM_BOUND, **RESNET_CONFIG), "adam"),
    "sngp_muon": (
        "SNGP + Muon/SGD (no SN)", lambda: DeepResNetSNGP(spec_norm_bound=None, **RESNET_CONFIG), "muon_sgd"
    ),
    "sngp_muon_adam": (
        "SNGP + Muon/Adam (no SN)", lambda: DeepResNetSNGP(spec_norm_bound=None, **RESNET_CONFIG), "muon_adam"
    ),
}


def run(seed: int, tags: list[str], out_dir: Optional[Path], show: bool, epochs: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_examples, train_labels = make_training_data(sample_size=500)
    test_examples = make_testing_data()
    ood_examples = make_ood_data(sample_size=500)

    for tag in tags:
        model_name, build, optimizer = MODELS[tag]
        model = build()
        logger.info(f"seed {seed}: training {model_name}")
        fit(model, make_optimizer(model, optimizer), train_examples, train_labels, epochs=epochs)
        sigmas = ", ".join(f"{s:.2f}" for s in hidden_spectral_norms(model))
        logger.info(f"{model_name}: hidden-layer spectral norms [{sigmas}]")
        probs = predict_class0_probs(model, test_examples)
        fig = plot_predictions(probs, model_name, train_examples, train_labels, ood_examples)
        if out_dir is not None:
            path = out_dir / f"{tag}_seed{seed}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            logger.info(f"saved {path}")
        if show:
            plt.show()
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures" / "sngp_moons")
    parser.add_argument("--no-save", action="store_true", help="don't write PNGs")
    parser.add_argument("--show", action="store_true", help="open each figure interactively")
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    set_default_style()

    out_dir = None if args.no_save else args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run(seed, args.models, out_dir, args.show, args.epochs)


if __name__ == "__main__":
    main()
