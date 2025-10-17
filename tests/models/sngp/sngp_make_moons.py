#!/usr/bin/env python3
# sngp_make_moons.py in tests/models/sngp

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import lightning as L
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.sngp.sngp_classification_layer import SNGP


# Two moons dataclass
@dataclass
class MoonsConfig:
    seed: int = 0
    train_size_per_class: int = 500
    batch_size: int = 64
    max_epochs: int = 150

    # backbone dims
    in_dim: int = 2
    up_projection_dim: int = 128

    # SNGP head
    num_classes: int = 2
    normalize_input: bool = True
    scale_random_features: bool = True
    covariance_momentum: float = 0.999
    covariance_ridge: float = 1e-6
    kernel_type: str = "gaussian"
    kernel_scale: Optional[float] = None
    random_features: int = 1024
    trainable_kernel_scale: bool = True

    # lightning
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # plotting
    x_range: Tuple[float, float] = (-3.5, 3.5)
    y_range: Tuple[float, float] = (-2.5, 2.5)
    n_grid: int = 250
    use_gp_uncertainty: bool = False  # True => use GP variance instead of p(1-p)


# define dataset
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MoonsDataModule(L.LightningDataModule):
    def __init__(self, cfg: MoonsConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        from sklearn.datasets import make_moons

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        X, y = make_moons(n_samples=2 * self.cfg.train_size_per_class, noise=0.1)
        X[y == 0] += [-0.1, 0.2]
        X[y == 1] += [0.1, -0.2]
        self.train_ds = NumpyDataset(X, y)

        x = np.linspace(*self.cfg.x_range, self.cfg.n_grid)
        yv = np.linspace(*self.cfg.y_range, self.cfg.n_grid)
        xv, yv = np.meshgrid(x, yv)
        self.test_grid = np.stack([xv.flatten(), yv.flatten()], axis=-1).astype(
            np.float32
        )

        # simple ood cloud for overlay (optional, not used by loaders)
        self.ood = np.random.multivariate_normal(
            mean=(2.5, -1.75), cov=np.diag((0.01, 0.01)), size=500
        ).astype(np.float32)

        self.train_points = X.astype(np.float32)
        self.train_labels = self.train_ds.y

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )


# ------------- Model ------------- #
class TinyBackbone(nn.Module):
    """A super-light “feature lift”: x -> R^H via a fixed linear stem + two residual MLP layers.

    This keeps the example minimal while giving the SNGP head a richer representation.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.input_W = nn.Parameter(torch.randn(in_dim, hidden), requires_grad=False)
        self.input_b = nn.Parameter(torch.randn(hidden), requires_grad=False)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x @ self.input_W + self.input_b
        z = self.act(self.fc1(x))
        x = x + z
        z = self.act(self.fc2(x))
        x = x + z
        return x


class LitSNGP(L.LightningModule):
    def __init__(self, cfg: MoonsConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.backbone = TinyBackbone(cfg.in_dim, cfg.up_projection_dim)

        # - normalize_input=True => RFGP will default ℓ=1.0 (good for normalized features)
        # - scale_random_features=True => apply √(2/m) scaling once (inside GP layer)
        self.sngp = SNGP(
            in_features=cfg.up_projection_dim,
            num_classes=cfg.num_classes,
            reduction_dim=cfg.up_projection_dim,  # SNGP internally will reduce again; keeping dims aligned
            classif_dropout=0.1,
            normalize_input=cfg.normalize_input,
            scale_random_features=cfg.scale_random_features,
            covariance_momentum=cfg.covariance_momentum,
            covariance_ridge_penalty=cfg.covariance_ridge,
            # forward to underlying RFGP
            kernel_type=cfg.kernel_type,
            kernel_scale=cfg.kernel_scale,
            random_features=cfg.random_features,
            trainable_kernel_scale=cfg.trainable_kernel_scale,
        )

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        x = self.backbone(x)
        # SNGP returns a dict with 'logits', 'cov', and mean-field applied in eval mode
        return self.sngp(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    def on_train_epoch_start(self):
        # Recommended when using moving-average precision accumulation:
        # start each epoch fresh (mirrors TF usage commonly seen in SNGP examples)
        try:
            self.sngp.gp_classifier.reset_precision()
        except Exception:
            # in case your SNGP wrapper names differ:
            pass

    def training_step(self, batch, _):
        x, y = batch
        out = self(x)
        logits = out["logits"]
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    # ---- helpers for evaluation/plotting ----
    @torch.no_grad()
    def predict_grid(self, XY: np.ndarray, batch_size: int = 256):
        self.eval()
        device = self.device
        probs, unc, cov_diags = [], [], []
        N = XY.shape[0]

        for i in range(0, N, batch_size):
            xb = torch.from_numpy(XY[i : i + batch_size]).to(device)
            out = self(xb)
            logits = out["logits"]
            cov = out["cov"]  # predictive covariance [B, B]

            # Class probability (softmax)
            p = F.softmax(logits, dim=-1)[..., 0].detach().cpu().numpy()
            probs.append(p)

            # --- Uncertainty choice ---
            if self.cfg.use_gp_uncertainty:
                # GP epistemic uncertainty from predictive covariance
                cov_diag = torch.diagonal(cov).detach().cpu().numpy()
                u = cov_diag
            else:
                # Heuristic: aleatoric-like uncertainty from p(1-p)
                u = p * (1.0 - p)

            unc.append(u)

            # store for sanity checks
            cov_diag = torch.diagonal(cov).detach().cpu().numpy()
            cov_diags.append(cov_diag)

        probs = np.concatenate(probs, axis=0)
        unc = np.concatenate(unc, axis=0)
        cov_diags = np.concatenate(cov_diags, axis=0)

        mean_diag = cov_diags.mean()
        print(f"[Sanity Check] Mean diag(cov) across test grid: {mean_diag:.4f}")

        if self.cfg.use_gp_uncertainty:
            print("[Info] Using GP predictive variance as uncertainty surface.")
            # Normalize GP variance for better visualization scale
            unc = unc / (unc.max() + 1e-12)
        else:
            print("[Info] Using heuristic p(1-p) uncertainty surface.")

        return probs, unc


# train + eval
def main():
    cfg = MoonsConfig()
    L.seed_everything(cfg.seed, workers=True)

    dm = MoonsDataModule(cfg)
    dm.setup()

    model = LitSNGP(cfg)
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=None,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader())

    # ---- Evaluate on grid and plot surfaces ----
    probs, unc = model.predict_grid(dm.test_grid, batch_size=512)

    # Quick single-batch sanity check
    with torch.no_grad():
        xb = torch.from_numpy(dm.test_grid[:256]).to(model.device)
        out = model(xb)
        mean_diag = torch.diagonal(out["cov"]).mean().item()
        print(f"[Sanity Check] Mean diag(cov) for 1 eval batch: {mean_diag:.4f}")

    # title text
    title_text = f"(D={cfg.random_features}, ℓ={'auto' if cfg.kernel_scale is None else cfg.kernel_scale})"
    plot_surfaces(cfg, dm, probs=probs, unc=unc, title_suffix=title_text)


# ------------- Plotting ------------- #
def plot_surfaces(
    cfg: MoonsConfig,
    dm: MoonsDataModule,
    probs: np.ndarray,
    unc: np.ndarray,
    title_suffix="",
):
    DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
    DEFAULT_NORM = colors.Normalize(vmin=0.5, vmax=1)

    def _show_field(ax, field, title, show_data=True):
        field = field / (field.max() + 1e-12)
        ax.set_xlim(cfg.x_range)
        ax.set_ylim(cfg.y_range)
        ax.set_title(title)

        pcm = ax.imshow(
            field.reshape(cfg.n_grid, cfg.n_grid),
            cmap="viridis",
            origin="lower",
            extent=cfg.x_range + cfg.y_range,
            vmin=DEFAULT_NORM.vmin,
            vmax=DEFAULT_NORM.vmax,
            interpolation="bicubic",
            aspect="auto",
        )
        if show_data:
            ax.scatter(
                dm.train_points[:, 0],
                dm.train_points[:, 1],
                c=dm.train_labels,
                cmap=DEFAULT_CMAP,
                alpha=0.5,
                s=10,
                linewidths=0,
            )
        return pcm

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    pcm0 = _show_field(
        axs[0], probs, f"Class Probability {title_suffix}", show_data=True
    )
    plt.colorbar(pcm0, ax=axs[0])

    pcm1 = _show_field(
        axs[1], unc, f"Predictive Uncertainty {title_suffix}", show_data=False
    )
    plt.colorbar(pcm1, ax=axs[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    # to save the figure, uncomment:
    plt.savefig("sngp_moons.png", dpi=300)
