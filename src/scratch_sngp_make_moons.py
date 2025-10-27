#!/usr/bin/env python3
"""scratch_sngp_make_moons.py in src/argusdp."""

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import os

print('torch version - ',torch.__version__)

from src.models.sngp.gaussian_process import mean_field_logits
from src.models.sngp.sngp_classification_layer import SNGP


# set random seed
np.random.seed(0)
torch.manual_seed(0)

plt.rcParams["figure.dpi"] = 90

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(
    vmin=0,
    vmax=1,
)
DEFAULT_N_GRID = 100

moons_plot_save_dir = 'plots/test/'

def make_training_data(sample_size=500):
    """Create two moon training dataset."""
    train_examples, train_labels = sklearn.datasets.make_moons(
        n_samples=2 * sample_size, noise=0.1
    )

    # Adjust data position slightly.
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]

    return train_examples, train_labels


def make_testing_data(
    x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID
):
    """Create a mesh grid in 2D space."""
    # testing data (mesh grid over data space)
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)


def make_ood_data(sample_size=500, means=(2.5, -1.75), vars=(0.01, 0.01)):
    return np.random.multivariate_normal(means, cov=np.diag(vars), size=sample_size)


def plot_uncertainty_surface(test_uncertainty, ax, cmap=None, show_data=True):
    """Visualizes the 2D uncertainty surface.

    For simplicity, assume these objects already exist in the memory:

    test_examples: Array of test examples, shape (num_test, 2).
    train_labels: Array of train labels, shape (num_train, ).
    train_examples: Array of train examples, shape (num_train, 2).

    Arguments:
    test_uncertainty: Array of uncertainty scores, shape (num_test,).
    ax: A matplotlib Axes object that specifies a matplotlib figure.
    cmap: A matplotlib colormap object specifying the palette of the
      predictive surface.

    Returns:
    pcm: A matplotlib PathCollection object that contains the palette
      information of the uncertainty plot.
    """
    # Normalize uncertainty for better visualization.
    test_uncertainty = test_uncertainty / np.max(test_uncertainty)

    # Set view limits.
    ax.set_ylim(DEFAULT_Y_RANGE)
    ax.set_xlim(DEFAULT_X_RANGE)

    # Plot normalized uncertainty surface.
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

    # Plot training data.
    if show_data:
        ax.scatter(
            train_examples[:, 0],
            train_examples[:, 1],
            c=train_labels,
            cmap=DEFAULT_CMAP,
            alpha=0.5,
        )
        ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    return pcm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = X.astype(np.float32), y.astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class BaselineModel(nn.Module):
    def __init__(self, D=2, C=2, H=128, p=0.1, n_hidden=4):
        super().__init__()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.register_buffer("input_W", torch.randn(D, H))
        self.register_buffer("input_b", torch.randn(H))
        self.fcs = nn.ModuleList([nn.Linear(H, H) for _ in range(n_hidden)])
        self.output_layer = nn.Linear(H, C)

    def forward(self, x, masks=None):
        x = x @ self.input_W + self.input_b
        for i in range(len(self.fcs)):
            if masks is not None:
                x = x + self.act(self.fcs[i](x)) * masks[i]
            x = x + self.dropout(self.act(self.fcs[i](x)))
        return self.output_layer(x)


# Load the train, test and OOD datasets.
sample_size = 500
train_examples, train_labels = make_training_data(sample_size=sample_size)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500)

# Visualize
pos_examples = train_examples[train_labels == 0]
neg_examples = train_examples[train_labels == 1]

plt.figure(figsize=(7, 5.5))

plt.scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
plt.scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
plt.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

plt.legend(["Postive", "Negative", "Out-of-Domain"])

plt.ylim(DEFAULT_Y_RANGE)
plt.xlim(DEFAULT_X_RANGE)
plt.title("Make moons dataset")

plt.show()


# model = BaselineModel()

# optim = torch.optim.Adam(model.parameters())
# loader = torch.utils.data.DataLoader(
#     Dataset(train_examples, train_labels), batch_size=64
# )
# loss_fn = nn.CrossEntropyLoss()

# model.train()
# for epoch in range(1, 100 + 1):
#     c, running_loss = 1, 0
#     for x, y in loader:
#         optim.zero_grad()
#         yhat = model(x)
#         loss = loss_fn(yhat, y)
#         loss.backward()
#         optim.step()
#         running_loss += float(loss)
#         c += 1
#     if not epoch % 10:
#         print(f"Epoch {epoch}. Loss: {running_loss / c}")
# model.eval()


# eval_loader = torch.utils.data.DataLoader(
#     Dataset(test_examples, test_examples), batch_size=256
# )
# logits = []
# for x, _ in eval_loader:
#     with torch.no_grad():
#         logits.append(model(x).detach().numpy())
# logits = np.concatenate(logits, axis=0)
# with torch.no_grad():
#     probs = (
#         torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 0]
#         .detach()
#         .numpy()
#     )


# _, ax = plt.subplots(figsize=(7, 5.5))

# pcm = plot_uncertainty_surface(probs, ax)

# plt.colorbar(pcm, ax=ax)
# plt.title("Class Probability, Deterministic Model")

# plt.show()


# ## Monte Carlo Dropout
# def make_masks(n_layers, hidden_dim):
#     return [torch.bernoulli(torch.empty(128).uniform_()) for _ in range(n_layers)]


# n_layers = len(model.fcs)
# hidden_dim = 128

# logit_samples = []

# # These are the results of averaging 24 mc dropout forward passes.
# for i in tqdm(range(24)):
#     masks = make_masks(n_layers, hidden_dim)
#     eval_loader = torch.utils.data.DataLoader(
#         Dataset(test_examples, test_examples), batch_size=256
#     )
#     logits = []
#     for x, _ in eval_loader:
#         with torch.no_grad():
#             logits.append(model(x, masks=masks).detach().numpy())
#     logits = np.concatenate(logits, axis=0)
#     logit_samples.append(
#         logits[None, :]
#     )  # add the sample dimension (size 12) at zero to concatenate along.

# logit_samples = np.concatenate(logit_samples)
# mean = np.mean(logit_samples, axis=0)
# variance = np.var(logit_samples, axis=0) ** 2
# logits = mean / np.sqrt(1 + variance)

# with torch.no_grad():
#     probs = (
#         torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 0]
#         .detach()
#         .numpy()
#     )

# #
# with torch.no_grad():
#     probs = (
#         torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 0]
#         .detach()
#         .numpy()
#     )

# with torch.no_grad():
#     mean_probs = (
#         torch.softmax(torch.tensor(mean, dtype=torch.float32), dim=-1)[:, 0]
#         .detach()
#         .numpy()
#     )

# _, ax = plt.subplots(figsize=(7, 5.5))

# pcm = plot_uncertainty_surface(mean_probs, ax)

# plt.colorbar(pcm, ax=ax)
# plt.title("Class Probability, Deterministic Model")
# plt.savefig("plots/predictive_uncertainty_Deterministic_.png")
# plt.show()

# # MC dropout logits
# resnet_uncertainty = probs * (1 - probs)

# _, ax = plt.subplots(figsize=(7, 5.5))

# pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)

# plt.colorbar(pcm, ax=ax)
# plt.title("Predictive Uncertainty, MC Dropout Model")
# plt.savefig("plots/predictive_uncertainty_MC_Dropout.png")
# plt.show()


# SNGP
class SNGPModel(nn.Module):
    def __init__(self, in_features, num_classes, up_projection_dim):
        super().__init__()
        self.register_buffer("input_W", torch.randn(in_features, up_projection_dim))
        self.register_buffer("input_b", torch.randn(up_projection_dim))
        self.classifier = SNGP(
            in_features=up_projection_dim,
            num_classes=num_classes,
            kernel_scale_trainable=True,
            scale_random_features=True,
            normalize_input=False,
            covariance_momentum=0.999,
            return_dict=False,
        )

    def forward(self, x):
        x = x @ self.input_W + self.input_b
        return self.classifier(x)


input_dim = 2
num_classes = 2
up_projection_dim = 512 #128

model = SNGPModel(
    in_features=input_dim,
    num_classes=num_classes,
    up_projection_dim=up_projection_dim,
)

optim = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
loader = torch.utils.data.DataLoader(
    Dataset(train_examples, train_labels), batch_size=64
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss(reduction="mean")

model.train()
model = model.to(device)
epochs = 150
for epoch in range(1, epochs + 1):
    running_loss, c = 0, 0
    for x, y in loader:
        optim.zero_grad()
        logits, covariance = model(x.to(device))
        loss = loss_fn(logits, y.to(device))
        loss.backward()
        optim.step()
        running_loss += float(loss)
        c += 1
    if not epoch % 10:
        print(f"Epoch {epoch}. Loss: {running_loss / c}")
#     model.classifier.reset_precision()
model.eval()


# SNGP - eval
eval_loader = torch.utils.data.DataLoader(
    Dataset(test_examples, test_examples), batch_size=256
)
logits = []
covs = []
for x, _ in eval_loader:
    with torch.no_grad():
        l, c = model(x.to(device))
        l = mean_field_logits(l, c)
        logits.append(l.detach().cpu().numpy())
        covs.append(np.diag(c.detach().cpu().numpy())[:, None])

logits = np.concatenate(logits, axis=0)
covs = np.concatenate(covs, axis=0)

with torch.no_grad():
    probs2 = torch.softmax(torch.tensor(logits), dim=1)[:, 0].detach().numpy()


#
_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(probs2, ax=ax, show_data=True)

plt.colorbar(pcm, ax=ax)
plt.title("Class Probability, Probabilistic Model")
save_path = os.path.join(moons_plot_save_dir, f"class_prob-Samples-{sample_size}-epochs-{epochs}-up_projection_dim-{up_projection_dim}")
plt.savefig(save_path) 
plt.show()

uncertainty = probs2 * (1 - probs2)

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(uncertainty, ax=ax, show_data=False)

plt.colorbar(pcm, ax=ax)
plt.title("Predictive Uncertainty, Probabilistic Model")
save_path = os.path.join(moons_plot_save_dir, f"uncertainity_surface-Samples-{sample_size}-epochs-{epochs}-up_projection_dim-{up_projection_dim}")
plt.savefig(save_path)
plt.show()
# print("Done")