"""End-to-end checkpoint round-trip: train one step, save, reload through every
supported path, and confirm they agree -- plus that Lightning's own
`trainer.fit(model, ckpt_path=...)` resume path (train.py/eval.py's pattern) still
works. No network access: uses an in-memory fake dataset, matching the pattern in
tests/test_datamodules.py.
"""
import torch
import lightning as L
import hydra
import pytest
from hydra import compose, initialize

from src.checkpointing.io import load_lit_module, load_net, read_meta


class _FakeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8, num_classes: int = 8):
        self.n = n
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return f"img{idx}", torch.randn(3, 224, 224), idx % self.num_classes, "train"


def _build_and_train_one_step(tmp_path, model_name: str, overrides):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", overrides=[f"model={model_name}", *overrides])
        model = hydra.utils.instantiate(cfg.model)

    if hasattr(model.net, "set_active_member"):
        model.net.set_active_member(0)

    dataloader = torch.utils.data.DataLoader(_FakeClassificationDataset(), batch_size=4)
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, train_dataloaders=dataloader)

    ckpt_path = tmp_path / "roundtrip.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    return cfg, ckpt_path


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name,overrides",
    [
        ("baseline_classifier", ["model.net.pretrained=false"]),
        ("sngp_classifier", ["model.net.pretrained=false"]),
    ],
)
def test_checkpoint_roundtrip(tmp_path, model_name, overrides):
    _, ckpt_path = _build_and_train_one_step(tmp_path, model_name, overrides)

    meta = read_meta(ckpt_path)
    assert meta.format_version == 2

    x = torch.randn(2, 3, 224, 224)

    net = load_net(ckpt_path, device="cpu")
    net.eval()
    with torch.no_grad():
        net_logits = net(x).logits

    lit = load_lit_module(ckpt_path, device="cpu")
    with torch.no_grad():
        lit_logits = lit(x).logits

    assert torch.allclose(net_logits, lit_logits, atol=1e-5)
    assert net_logits.shape == (2, meta.num_classes)


@pytest.mark.slow
def test_trainer_fit_ckpt_path_resume_still_works(tmp_path):
    """The train.py/eval.py pattern (fresh Hydra-instantiated model +
    trainer.fit/test(ckpt_path=...)) never depended on unpickling `net`, and must keep
    working unchanged by this refactor."""
    cfg, ckpt_path = _build_and_train_one_step(tmp_path, "baseline_classifier", ["model.net.pretrained=false"])

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg2 = compose(config_name="train.yaml", overrides=["model=baseline_classifier", "model.net.pretrained=false"])
        model2 = hydra.utils.instantiate(cfg2.model)

    dataloader = torch.utils.data.DataLoader(_FakeClassificationDataset(), batch_size=4)
    trainer = L.Trainer(
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model2, train_dataloaders=dataloader, ckpt_path=str(ckpt_path))
    assert trainer.current_epoch == 2
