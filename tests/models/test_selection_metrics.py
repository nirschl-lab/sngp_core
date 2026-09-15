"""Regression test for the probability-based validation metrics LitModuleBase logs for
hparams_search selection -- `val/nll_cal` / `val/nll_cal_best` (the objective), their raw
twins, the fitted calibration knob, and the `val/auprc*` / `val/ece` diagnostics -- see
docs/HPO_GUIDE.md. No network access: uses an in-memory fake dataset, matching the
pattern in tests/checkpointing/test_roundtrip.py.
"""
import hydra
import lightning as L
import pytest
import torch
from hydra import compose, initialize
from lightning import Callback

RUNNING_BEST = ("val/auprc_best", "val/nll_best", "val/nll_cal_best")
PER_EPOCH = ("val/auprc", "val/nll", "val/nll_cal", "val/ece")


class _FakeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8, num_classes: int = 4):
        self.n = n
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return f"img{idx}", torch.randn(3, 224, 224), idx % self.num_classes, "train"


class _RecordSelectionMetrics(Callback):
    """Captures the selection metrics once per epoch. trainer.callback_metrics only ever
    holds the LAST epoch's snapshot once fit() returns -- exactly the "objective reads
    the last epoch, not the best" trap the `*_best` running extrema exist to avoid (see
    src/train.py) -- so a plain post-fit assertion can't tell a monotone running best
    from a random walk.

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
        self.history = {name: [] for name in RUNNING_BEST + PER_EPOCH}

    def on_train_epoch_end(self, trainer, pl_module):
        for name in self.history:
            if name in trainer.callback_metrics:
                self.history[name].append(trainer.callback_metrics[name].item())


def _fit(tmp_path, overrides, epochs: int = 3):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", overrides=["data=tang", *overrides])
        model = hydra.utils.instantiate(cfg.model)

    dataset = _FakeClassificationDataset(n=8, num_classes=model.num_classes)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    # >=2 val batches: validation_step skips batch_idx 0 in epoch 0 by design (see
    # LitModuleBase), so a single-batch val loader would never update val_auprc/etc.
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    recorder = _RecordSelectionMetrics()
    trainer = L.Trainer(
        max_epochs=epochs,
        limit_train_batches=1,
        limit_val_batches=2,
        num_sanity_val_steps=0,  # sanity-check would otherwise contribute a spurious
        # extra on_validation_epoch_end firing (on a barely-initialized model) before
        # on_train_start's *_best.reset() ever runs.
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
        callbacks=[recorder],
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer, recorder


@pytest.mark.parametrize(
    "overrides,knob_metric",
    [
        (["model=baseline_classifier", "model.net.pretrained=false"], "val/temperature_fit"),
        (["model=sngp_classifier", "model.net.pretrained=false", "model.net.rff_dim=64"], "val/mean_field_factor_fit"),
    ],
    ids=["baseline", "sngp"],
)
def test_selection_metrics_present_and_running_bests_are_monotone(tmp_path, overrides, knob_metric):
    trainer, recorder = _fit(tmp_path, overrides)

    for name in RUNNING_BEST + PER_EPOCH + (knob_metric,):
        assert name in trainer.callback_metrics, f"{name} was not logged"
        value = trainer.callback_metrics[name].item()
        assert torch.isfinite(torch.tensor(value)), f"{name}={value} is not finite"
    assert trainer.callback_metrics[knob_metric].item() >= 0.0

    for name in RUNNING_BEST + PER_EPOCH:
        assert len(recorder.history[name]) == 3, name

    # Running extrema across epochs (MaxMetric / MinMetric) -- must never move the wrong
    # way, even though the underlying random data gives the raw per-epoch values no
    # reason to trend in any direction.
    auprc_best = recorder.history["val/auprc_best"]
    assert all(0.0 <= v <= 1.0 for v in auprc_best), auprc_best
    assert auprc_best == sorted(auprc_best), auprc_best
    for name in ("val/nll_best", "val/nll_cal_best"):
        history = recorder.history[name]
        assert history == sorted(history, reverse=True), (name, history)

    # The post-hoc fit always has the identity knob as a candidate, so the calibrated
    # NLL can never be worse than the raw one in the same epoch.
    for nll, nll_cal in zip(recorder.history["val/nll"], recorder.history["val/nll_cal"]):
        assert nll_cal <= nll + 1e-4, (nll, nll_cal)


def test_calibration_knob_can_be_forced_to_temperature_on_sngp(tmp_path):
    """`calibration_knob=temperature` is the like-for-like ablation: SNGP selected with a
    temperature fit instead of its mean-field factor."""
    trainer, _ = _fit(
        tmp_path,
        ["model=sngp_classifier", "model.net.pretrained=false", "model.net.rff_dim=64", "+model.calibration_knob=temperature"],
        epochs=1,
    )
    assert "val/temperature_fit" in trainer.callback_metrics
    assert "val/mean_field_factor_fit" not in trainer.callback_metrics
