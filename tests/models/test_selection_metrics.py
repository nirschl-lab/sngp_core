"""Regression test for the probability-based validation metrics (val/auprc,
val/auprc_best, val/nll, val/ece) added to LitModuleBase for hparams_search selection
-- see docs/HPO_GUIDE.md. No network access: uses an in-memory fake dataset, matching
the pattern in tests/checkpointing/test_roundtrip.py.
"""
import hydra
import lightning as L
import torch
from hydra import compose, initialize
from lightning import Callback


class _FakeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8, num_classes: int = 4):
        self.n = n
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return f"img{idx}", torch.randn(3, 224, 224), idx % self.num_classes, "train"


class _RecordAuprcBest(Callback):
    """Captures val/auprc_best once per epoch. trainer.callback_metrics only ever
    holds the LAST epoch's snapshot once fit() returns -- exactly the "objective reads
    the last epoch, not the best" trap the hparams_search objective itself had to be
    fixed to avoid (see src/train.py) -- so a plain post-fit assertion can't tell a
    monotone running max from a random walk.

    Deliberately hooks on_train_epoch_end, not on_validation_epoch_end: Lightning
    calls every registered Callback's on_validation_epoch_end BEFORE the
    LightningModule's own on_validation_epoch_end (see
    lightning.pytorch.loops.evaluation_loop._on_evaluation_epoch_end), so reading
    trainer.callback_metrics from a Callback's on_validation_epoch_end would see last
    epoch's stale value, not the one LitModuleBase just computed. on_train_epoch_end
    fires strictly after that epoch's validation loop (module hook included) has
    already run.
    """

    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "val/auprc_best" in trainer.callback_metrics:
            self.history.append(trainer.callback_metrics["val/auprc_best"].item())


def test_val_auprc_best_is_monotone_and_all_selection_metrics_present(tmp_path):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=tang", "model=baseline_classifier", "model.net.pretrained=false"],
        )
        model = hydra.utils.instantiate(cfg.model)

    dataset = _FakeClassificationDataset(n=8, num_classes=model.num_classes)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    # >=2 val batches: validation_step skips batch_idx 0 in epoch 0 by design (see
    # LitModuleBase), so a single-batch val loader would never update val_auprc/etc.
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    recorder = _RecordAuprcBest()
    trainer = L.Trainer(
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=2,
        num_sanity_val_steps=0,  # sanity-check would otherwise contribute a spurious
        # extra on_validation_epoch_end firing (on a barely-initialized model) before
        # on_train_start's val_auprc_best.reset() ever runs.
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
        callbacks=[recorder],
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    for name in ("val/auprc", "val/auprc_best", "val/nll", "val/ece"):
        assert name in trainer.callback_metrics, f"{name} was not logged"
        value = trainer.callback_metrics[name].item()
        assert torch.isfinite(torch.tensor(value)), f"{name}={value} is not finite"

    assert len(recorder.history) == 3
    assert all(0.0 <= v <= 1.0 for v in recorder.history), recorder.history
    # val_auprc_best is a running max across epochs (MaxMetric) -- it must never
    # decrease, even though the underlying random data gives no reason for the raw
    # per-epoch val/auprc itself to trend in any direction.
    assert recorder.history == sorted(recorder.history), recorder.history
