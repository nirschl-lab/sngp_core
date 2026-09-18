from typing import Any, Optional, Tuple

import torch

from src.models.components.spectral_reg import (
    DEFAULT_SPEC_REG_WARMUP_ITERATIONS,
    SpectralRegularizer,
)
from src.models.outputs import ModelOutput
from src.models.sngp_lit_module import SNGPLitModule
from src.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class SNGPSpectralRegLitModule(SNGPLitModule):
    """SNGP trained with spectral *regularization* instead of spectral normalization.

    Yang, Zavatone-Veth & Pehlevan (arXiv 2405.17181, "rep-spectral"), eq. (5): the
    backbone's Lipschitz constant is controlled by a loss term

        loss = CE + spec_reg_coef * sum_l sigma_max^2(W_l)

    over every Conv2d/Linear of `net.backbone` -- the GP head (readout) is excluded, as
    the paper drops the last-layer term. The net must be built with
    `use_spectral_norm=False`: SNGP's weight-rescaling hooks and this penalty are two
    mechanisms for the same quantity and are never combined (construction raises).

    Everything else is `SNGPLitModule`: precision-matrix accumulation gated on train mode,
    per-epoch reset, plain CE. Only `model_step` differs, and only while `self.training`
    (validation/test stay pure CE, so `val/nll` is comparable across families).

    Schedule, following the reference ResNet18 recipe:

      * `spec_reg_every_n_steps` -- the penalty is added every N optimizer steps
        (`--reg-freq-update`, 24 in the paper). Its power iteration advances on the same
        cadence, so the sigma estimates track the weights.
      * `spec_reg_burnin_epochs` -- epochs trained *without* the penalty first (paper:
        80% of the budget). During burn-in the penalty is still evaluated under
        `no_grad` on the same cadence, purely so `train/sigma_*` show how far the
        unregularized network drifts before regularization starts.

    Logged (epoch means over the steps where the penalty was evaluated): `train/ce`,
    `train/spec_reg` (raw sum sigma^2), `train/spec_reg_weighted` (0 during burn-in),
    `train/spec_reg_active` (0/1), `train/sigma_max`, `train/sigma_mean`. `train/loss`
    (from `LitModuleBase`) is the total.

    Selection caveat: with a burn-in the unregularized phase can post the lowest
    `val/nll`; pair this module with `ModelCheckpointFromEpoch(start_epoch=burn-in)` so
    `best.ckpt` is a regularized model (configs/experiment/sngp_specreg_acevedo.yaml).
    """

    def __init__(
        self,
        net: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        spec_reg_coef: float = 0.01,
        spec_reg_every_n_steps: int = 24,
        spec_reg_burnin_epochs: int = 0,
        spec_reg_n_power_iterations: int = 1,
        spec_reg_conv_mode: str = "operator",
        spec_reg_warmup_iterations: int = DEFAULT_SPEC_REG_WARMUP_ITERATIONS,
        **kwargs,
    ) -> None:
        super().__init__(
            net=net, optimizer=optimizer, scheduler=scheduler, compile=compile, **kwargs
        )
        if spec_reg_coef < 0:
            raise ValueError(f"spec_reg_coef must be >= 0, got {spec_reg_coef}")
        if spec_reg_every_n_steps < 1:
            raise ValueError(f"spec_reg_every_n_steps must be >= 1, got {spec_reg_every_n_steps}")
        if spec_reg_burnin_epochs < 0:
            raise ValueError(f"spec_reg_burnin_epochs must be >= 0, got {spec_reg_burnin_epochs}")
        self.spec_reg_coef = float(spec_reg_coef)
        self.spec_reg_every_n_steps = int(spec_reg_every_n_steps)
        self.spec_reg_burnin_epochs = int(spec_reg_burnin_epochs)
        self.spec_reg_n_power_iterations = int(spec_reg_n_power_iterations)
        self.spec_reg_conv_mode = spec_reg_conv_mode
        self.spec_reg_warmup_iterations = int(spec_reg_warmup_iterations)

        backbone = getattr(self.net, "backbone", None)
        if backbone is None:
            raise TypeError(
                f"{type(self).__name__} needs a net with a `.backbone` (the representation "
                f"layers to regularize); got {type(self.net).__name__}"
            )
        if any(hasattr(m, "weight_u") for m in backbone.modules()):
            raise ValueError(
                f"{type(self).__name__}: the backbone is spectral-normalized. Spectral "
                "normalization and spectral regularization must not be combined -- set "
                "`model.net.use_spectral_norm: false` "
                "(configs/model/sngp_specreg_classifier.yaml does)."
            )
        # A submodule so it follows `.to(device)`; it owns no persistent state, so the
        # checkpoint state_dict is identical to a plain SNGPLitModule's.
        self.spec_reg = SpectralRegularizer(
            backbone,
            n_power_iterations=self.spec_reg_n_power_iterations,
            conv_mode=self.spec_reg_conv_mode,
            warmup_iterations=self.spec_reg_warmup_iterations,
        )
        logger.info(
            f"Spectral regularization over {len(self.spec_reg)} backbone layers "
            f"(conv_mode={self.spec_reg_conv_mode}): coef={self.spec_reg_coef}, "
            f"every {self.spec_reg_every_n_steps} step(s), "
            f"burn-in {self.spec_reg_burnin_epochs} epoch(s)"
        )

    # ------------------------------------------------------------------ schedule
    def spec_reg_is_active(self) -> bool:
        """True once the burn-in is over (epoch index >= `spec_reg_burnin_epochs`)."""
        return self.current_epoch >= self.spec_reg_burnin_epochs

    def _spec_reg_due(self) -> bool:
        return self.global_step % self.spec_reg_every_n_steps == 0

    # ------------------------------------------------------------------ training loss
    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, ModelOutput
    ]:
        """`LitModuleBase.model_step` (plain CE) plus, in train mode on the regularization
        cadence, `spec_reg_coef * sum sigma^2` -- or the same penalty under `no_grad`, for
        logging only, while still in burn-in. Validation calls land in the first branch
        and return the untouched CE tuple."""
        img_ids, loss, logits, probs, preds, targets, fold, output = super().model_step(batch)
        if not self.training or not self._spec_reg_due():
            return img_ids, loss, logits, probs, preds, targets, fold, output

        active = self.spec_reg_is_active()
        if active:
            reg = self.spec_reg()
            weighted = self.spec_reg_coef * reg.penalty
            total = loss + weighted
        else:
            with torch.no_grad():
                reg = self.spec_reg()
            weighted = torch.zeros_like(reg.penalty)
            total = loss

        self.log("train/ce", loss.detach(), on_step=False, on_epoch=True)
        self.log("train/spec_reg", reg.penalty.detach(), on_step=False, on_epoch=True)
        self.log("train/spec_reg_weighted", weighted.detach(), on_step=False, on_epoch=True)
        self.log("train/spec_reg_active", float(active), on_step=False, on_epoch=True)
        self.log("train/sigma_max", reg.sigmas.max(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/sigma_mean", reg.sigmas.mean(), on_step=False, on_epoch=True)
        return img_ids, total, logits, probs, preds, targets, fold, output
