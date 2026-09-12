"""End-to-end test of the callback-based test-time analysis path: `trainer.test()`
with `TestArtifactsCallback` attached should compute final metrics and (optionally)
write a prediction CSV -- entirely from the dict `LitModuleBase.test_step` returns, no
state living on the LightningModule. Parametrized over `baseline_classifier` and
`deep_ensemble_classifier`: the ensemble case specifically locks in the fix for the
old bug where `DeepEnsembleLitModule.test_step` populated buffers this callback never
read, so ensemble test artifacts were silently empty.

No network access: uses the same in-memory fake dataset as
`tests/checkpointing/test_roundtrip.py`.
"""
import hydra
import lightning as L
import pandas as pd
import pytest
import torch
from hydra import compose, initialize

from src.callbacks.test_artifacts_callback import TestArtifactsCallback


class _FakeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8, num_classes: int = 8):
        self.n = n
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return f"img{idx}", torch.randn(3, 224, 224), idx % self.num_classes, "test"


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name,overrides",
    [
        ("baseline_classifier", ["model.net.pretrained=false"]),
        ("deep_ensemble_classifier", ["model.net.base_model_spec.pretrained=false"]),
    ],
)
def test_test_artifacts_callback_computes_metrics_and_csv(tmp_path, model_name, overrides):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", overrides=["data=acevedo", f"model={model_name}", *overrides])
        model = hydra.utils.instantiate(cfg.model)

    if hasattr(model.net, "set_active_member"):
        model.net.set_active_member(0)

    csv_dir = tmp_path / "csv"
    callback = TestArtifactsCallback(log_csv=True, csv_save_path=str(csv_dir), log_metrics_per_class=False)

    dataloader = torch.utils.data.DataLoader(_FakeClassificationDataset(), batch_size=4)
    trainer = L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        callbacks=[callback],
        default_root_dir=str(tmp_path),
    )
    trainer.test(model, dataloaders=dataloader)

    for key in ("test/acc_final", "test/precision_final", "test/recall_final", "test/f1_final", "test/ece_final"):
        assert key in trainer.callback_metrics, f"{key} missing from callback_metrics for {model_name}"

    csv_path = csv_dir / "test_predictions.csv"
    assert csv_path.exists(), f"CSV not written for {model_name}"
    df = pd.read_csv(csv_path)
    assert len(df) == 8, f"expected 8 predictions in CSV for {model_name}, got {len(df)}"
    for col in ("image_id", "target", "prediction", "prediction_prob_score", "class_logits", "class_probs"):
        assert col in df.columns

    # The comparable uncertainty columns are written for every family, matching the
    # inference schema -- this writer used to drop uncertainty entirely.
    for col in ("predictive_entropy", "confidence_margin", "dempster_shafer"):
        assert col in df.columns, f"{col} missing from callback CSV for {model_name}"
        assert df[col].notna().all()

    if model_name == "deep_ensemble_classifier":
        # `member_logits` used to be accumulated by this callback and never read.
        for col in ("total_entropy", "aleatoric_entropy", "mutual_information", "uncertainty"):
            assert col in df.columns, f"{col} missing for {model_name}"
        assert (df["mutual_information"] >= 0).all()
        assert df["total_entropy"].values == pytest.approx(
            (df["aleatoric_entropy"] + df["mutual_information"]).values, abs=1e-9
        )
        assert set(df["uncertainty_kind"]) == {"member_logit_variance"}
    else:
        # A plain Baseline has no model-side variance at all.
        assert "uncertainty" not in df.columns
