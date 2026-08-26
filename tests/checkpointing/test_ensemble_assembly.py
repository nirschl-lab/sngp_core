"""End-to-end test for scripts/ensemble/assemble_ensemble_checkpoint.py: train two
tiny baseline "members" independently (as scripts/ensemble/train_members_parallel.sh
would, one per parallel run), assemble them into one DeepEnsemble checkpoint, and
confirm it round-trips through every path a real DeepEnsembleLitModule checkpoint
would: src/checkpointing/io.py's read_meta/load_net/load_lit_module, plus Lightning's
own trainer.test(ckpt_path=...) restore path (src/eval.py's pattern). No network
access: uses an in-memory fake dataset, matching tests/checkpointing/test_roundtrip.py.
"""
from pathlib import Path

import torch
import lightning as L
import hydra
import pytest
from hydra import compose, initialize

from scripts.ensemble.assemble_ensemble_checkpoint import assemble, derive_default_out_path
from src.checkpointing.io import load_lit_module, load_net, read_meta


def test_derive_default_out_path_nests_under_members_dir():
    members_dir = Path(
        "/data1/maheswararao/experiments/uncertaity-aware-ml/train/"
        "baseline_classifier_acevedo/ensemble_members/2026-08-25_15-31-22"
    )
    assert derive_default_out_path(members_dir) == members_dir / "checkpoints" / "ensemble.ckpt"


class _FakeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8, num_classes: int = 8):
        self.n = n
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return f"img{idx}", torch.randn(3, 224, 224), idx % self.num_classes, "train"


def _train_one_baseline_member(tmp_path, seed: int):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=acevedo", "model=baseline_classifier", "model.net.pretrained=false"],
        )

    L.seed_everything(seed, workers=True)
    model = hydra.utils.instantiate(cfg.model)

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

    ckpt_path = tmp_path / f"member_{seed}.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    return ckpt_path


@pytest.mark.slow
def test_assemble_ensemble_checkpoint_roundtrip(tmp_path):
    member_ckpts = [_train_one_baseline_member(tmp_path / f"m{i}", seed=i) for i in range(2)]

    out_path = tmp_path / "ensemble.ckpt"
    assemble(member_ckpts, out_path)

    meta = read_meta(out_path)
    assert meta.format_version == 2
    assert meta.net_spec["name"] == "deep_ensemble"
    assert meta.net_spec["num_estimators"] == 2

    x = torch.randn(2, 3, 224, 224)

    net = load_net(out_path, device="cpu")
    net.eval()
    with torch.no_grad():
        net_out = net(x)
    assert net_out.logits.shape == (2, meta.num_classes)
    assert net_out.member_logits.shape == (2, 2, meta.num_classes)

    lit = load_lit_module(out_path, device="cpu")
    with torch.no_grad():
        lit_out = lit(x)
    assert torch.allclose(net_out.logits, lit_out.logits, atol=1e-5)

    # The two members were trained from different seeds/data -- they must not have
    # collapsed to identical weights (which would make member_logits identical too).
    assert not torch.allclose(net_out.member_logits[0], net_out.member_logits[1])


@pytest.mark.slow
def test_assembled_checkpoint_loads_via_trainer_test(tmp_path):
    """src/eval.py's pattern: hydra.utils.instantiate a fresh model from the
    deep_ensemble_classifier config group, then trainer.test(ckpt_path=...) to load
    the assembled weights -- must work with no changes to eval.py."""
    member_ckpts = [_train_one_baseline_member(tmp_path / f"m{i}", seed=i) for i in range(2)]

    out_path = tmp_path / "ensemble.ckpt"
    assemble(member_ckpts, out_path)

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "data=acevedo",
                "model=deep_ensemble_classifier",
                "model.net.base_model_spec.pretrained=false",
                "model.num_estimators=2",
                "model.net.num_estimators=2",
            ],
        )
    model = hydra.utils.instantiate(cfg.model)

    dataloader = torch.utils.data.DataLoader(_FakeClassificationDataset(), batch_size=4)
    trainer = L.Trainer(
        limit_test_batches=1,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
    )
    trainer.test(model=model, dataloaders=dataloader, ckpt_path=str(out_path))
