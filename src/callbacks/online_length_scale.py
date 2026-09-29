"""Online type-II choice of the SNGP length scale, after Immer et al. 2021 ("Scalable Marginal
Likelihood Estimation for Model Selection in Deep Learning", arXiv:2104.04975), head only.

The paper trains the network by its usual loss and, every F epochs after a burn-in of B epochs,
moves the hyperparameters up the Laplace marginal likelihood of the training data -- no
validation set. Here the one hyperparameter is the GP head's length scale l, and the model whose
evidence is scored is the head alone: Bayesian linear regression on the random features phi(x; l)
under the Gaussian likelihood, the same model whose posterior gives the SNGP variance. Its
evidence is closed form (`src/metrics/gp_evidence.py`), so the paper's K gradient steps on a
deterministic 1-D objective are replaced by a direct search on a log grid around the current l.

Each update:
  1. a no-grad, eval-mode backbone pass over a fixed random subset of the train set (its
     augmented train transform, as in `scripts/metrics/cifar100_length_scale_evidence.py`);
  2. for every l on `l * search_factor ** linspace(-1, 1, n_grid)`, the log evidence per datum
     at type-II (alpha, s) by MacKay's fixed point, on the head's own random draw;
  3. a damped step `log l += damping * (log l* - log l)`, applied with `net.set_length_scale`.

The backbone and the head's weights keep training by CE; only l moves by evidence, and (alpha, s)
are fitted inside the evidence only to score l -- the head's ridge stays as configured. Updates
stop at `stop_epoch` so the final epochs train, and accumulate the precision, at one l.
"""
import math
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger as log
from torch.utils.data import DataLoader, Subset

from src.metrics.gp_evidence import evidence_basis, optimize_hyperparameters


def _unwrap(net: torch.nn.Module) -> torch.nn.Module:
    """The eager module behind a `torch.compile` wrapper."""
    return getattr(net, "_orig_mod", net)


class OnlineLengthScaleEvidence(Callback):
    """Move the SNGP head's length scale to its type-II evidence optimum during training.

    `epoch` below is Lightning's 0-based `current_epoch`. An update runs at the start of epoch e
    when `burnin_epochs <= e < stop_epoch` and `(e - burnin_epochs) % every_n_epochs == 0`, so it
    scores the backbone as trained through epoch e - 1, and epoch e is the first to train (and
    accumulate the precision) at the new l.
    """

    def __init__(
        self,
        burnin_epochs: int = 10,
        every_n_epochs: int = 5,
        stop_epoch: int = 160,
        n_samples: int = 10_000,
        search_factor: float = 4.0,
        n_grid: int = 13,
        damping: float = 0.5,
        batch_size: int = 512,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if burnin_epochs < 0:
            raise ValueError(f"burnin_epochs must be >= 0, got {burnin_epochs}")
        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")
        if stop_epoch <= burnin_epochs:
            raise ValueError(f"stop_epoch ({stop_epoch}) must be > burnin_epochs ({burnin_epochs})")
        if search_factor <= 1.0:
            raise ValueError(f"search_factor must be > 1, got {search_factor}")
        if n_grid < 3:
            raise ValueError(f"n_grid must be >= 3, got {n_grid}")
        if not 0.0 < damping <= 1.0:
            raise ValueError(f"damping must be in (0, 1], got {damping}")
        self.burnin_epochs = int(burnin_epochs)
        self.every_n_epochs = int(every_n_epochs)
        self.stop_epoch = int(stop_epoch)
        self.n_samples = int(n_samples)
        self.search_factor = float(search_factor)
        self.n_grid = int(n_grid)
        self.damping = float(damping)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._indices: Optional[torch.Tensor] = None
        # The current l, kept in the callback's checkpoint state: a resumed run rebuilds the net
        # from its Hydra config (the initial l) and then loads a `W` that carries the moved one.
        self._length_scale: Optional[float] = None

    # -- schedule ------------------------------------------------------------

    def should_update(self, epoch: int) -> bool:
        return self.burnin_epochs <= epoch < self.stop_epoch and (epoch - self.burnin_epochs) % self.every_n_epochs == 0

    def candidate_length_scales(self, length_scale: float) -> torch.Tensor:
        return length_scale * self.search_factor ** torch.linspace(-1.0, 1.0, self.n_grid, dtype=torch.float64)

    def damped(self, length_scale: float, target: float) -> float:
        return math.exp(math.log(length_scale) + self.damping * (math.log(target) - math.log(length_scale)))

    # -- evidence ------------------------------------------------------------

    @torch.no_grad()
    def score(self, head: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Type-II log evidence per datum over the candidate grid for pooled features `x` [N, d],
        labels `y` [N]. Returns the grid, the per-datum evidences, and the argmax's (l, alpha, s)."""
        grid = self.candidate_length_scales(head.length_scale)
        evidences, fits = [], []
        for ls in grid.tolist():
            basis = evidence_basis(head.features_at(x, ls), y, head.num_classes)
            alpha, noise, ev = optimize_hyperparameters(basis)
            evidences.append(ev / basis.n)
            fits.append((alpha, noise))
        best = max(range(len(evidences)), key=evidences.__getitem__)
        return {
            "grid": grid.tolist(),
            "evidence": evidences,
            "length_scale_star": float(grid[best]),
            "alpha": fits[best][0],
            "noise": fits[best][1],
            "log_evidence_per_datum": evidences[best],
            "at_grid_edge": best in (0, len(grid) - 1),
        }

    @torch.no_grad()
    def collect(self, trainer: "pl.Trainer", net: torch.nn.Module, device: torch.device):
        """Pooled backbone features and labels of the fixed train subset, in eval mode."""
        data_train = trainer.datamodule.data_train
        if self._indices is None:
            g = torch.Generator().manual_seed(self.seed)
            self._indices = torch.randperm(len(data_train), generator=g)[: self.n_samples]
        loader = DataLoader(
            Subset(data_train, self._indices.tolist()),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=trainer.datamodule.hparams.num_workers,
            pin_memory=trainer.datamodule.hparams.pin_memory,
        )
        was_training = net.training
        net.eval()
        feats, labels = [], []
        try:
            for _, images, targets, _ in loader:
                feats.append(net.pooled_features(images.to(device)).float())
                labels.append(targets.to(device))
        finally:
            net.train(was_training)
        return torch.cat(feats), torch.cat(labels)

    # -- Lightning hooks -----------------------------------------------------

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if stage != "fit":
            return
        if trainer.world_size > 1:
            raise NotImplementedError(
                "OnlineLengthScaleEvidence runs single-process only: under DDP each rank would score "
                "its own subset and move l differently. All-reduce the Gram matrix before using it there."
            )
        if not hasattr(_unwrap(pl_module.net), "set_length_scale"):
            raise TypeError(f"{type(_unwrap(pl_module.net)).__name__} has no set_length_scale; this callback is SNGP-only")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        net = _unwrap(pl_module.net)
        if self._length_scale is not None and self._length_scale != net.length_scale:
            # Resumed: the loaded `W` already carries the saved l; bring the attributes in step.
            log.info(f"OnlineLengthScaleEvidence: resuming at l = {self._length_scale:.4g}")
            net.gp_head.length_scale = self._length_scale
            net.length_scale = self._length_scale
        self._length_scale = float(net.length_scale)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        net = _unwrap(pl_module.net)
        epoch = trainer.current_epoch
        if self.should_update(epoch):
            x, y = self.collect(trainer, net, pl_module.device)
            result = self.score(net.gp_head, x, y)
            old = float(net.length_scale)
            new = self.damped(old, result["length_scale_star"])
            net.set_length_scale(new)
            self._length_scale = float(net.length_scale)
            rho = float(x.norm(dim=-1).mean()) / new
            log.info(
                f"OnlineLengthScaleEvidence epoch {epoch}: l {old:.4g} -> {new:.4g} (l* {result['length_scale_star']:.4g}"
                f"{', grid edge' if result['at_grid_edge'] else ''}), rho {rho:.3g}, alpha {result['alpha']:.3g}, "
                f"s {result['noise']:.3g}, log ev/N {result['log_evidence_per_datum']:.5g}"
            )
            pl_module.log_dict(
                {
                    "ls/length_scale_star": result["length_scale_star"],
                    "ls/rho": rho,
                    "ls/alpha": result["alpha"],
                    "ls/noise": result["noise"],
                    "ls/log_evidence_per_datum": result["log_evidence_per_datum"],
                },
                on_step=False,
                on_epoch=True,
            )
        pl_module.log("ls/length_scale", float(net.length_scale), on_step=False, on_epoch=True)

    def state_dict(self) -> Dict[str, Any]:
        return {"length_scale": self._length_scale}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._length_scale = state_dict.get("length_scale")
