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
        cfg = compose(config_name="train.yaml", overrides=["data=acevedo", f"model={model_name}", *overrides])
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
    return cfg, ckpt_path, model


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name,overrides",
    [
        ("baseline_classifier", ["model.net.pretrained=false"]),
        ("sngp_classifier", ["model.net.pretrained=false"]),
        # Bounded spectral norm (Liu et al. eq. 15): the bound must persist in the spec
        # and rebuild the *bounded* hook, not the stock one.
        ("sngp_classifier", ["model.net.pretrained=false", "++model.net.spectral_norm_bound=2.0"]),
        ("deep_ensemble_classifier", ["model.net.base_model_spec.pretrained=false"]),
    ],
)
def test_checkpoint_roundtrip(tmp_path, model_name, overrides):
    _, ckpt_path, trained = _build_and_train_one_step(tmp_path, model_name, overrides)

    meta = read_meta(ckpt_path)
    assert meta.format_version == 2
    # Every spec key -- including additive ones like `temperature` / `spectral_norm_bound`
    # -- must survive the write; `load_net` rebuilds from exactly this dict.
    assert meta.net_spec == trained.net.spec

    x = torch.randn(2, 3, 224, 224)

    net = load_net(ckpt_path, device="cpu")
    assert net.spec == trained.net.spec
    net.eval()
    with torch.no_grad():
        net_logits = net(x).logits

    lit = load_lit_module(ckpt_path, device="cpu")
    with torch.no_grad():
        lit_logits = lit(x).logits

    assert torch.allclose(net_logits, lit_logits, atol=1e-5)
    assert net_logits.shape == (2, meta.num_classes)

    # Compare against the *in-memory trained* model, not just two reloads of the same
    # file. Without this the test cannot detect state that silently fails to persist:
    # two reloads of a checkpoint missing a buffer agree with each other perfectly.
    trained.eval()
    with torch.no_grad():
        trained_logits = trained(x).logits
    assert torch.allclose(trained_logits, net_logits, atol=1e-5)


@pytest.mark.slow
def test_sngp_precision_matrix_survives_roundtrip(tmp_path):
    """The GP precision accumulator is the state SNGP's whole uncertainty estimate rests
    on, and it lives only in a buffer. Nothing asserted it round-tripped before."""
    _, ckpt_path, trained = _build_and_train_one_step(
        tmp_path, "sngp_classifier", ["model.net.pretrained=false"]
    )

    trained_accum = trained.net.gp_head.precision_accum
    assert torch.count_nonzero(trained_accum) > 0, "training should have populated the precision matrix"

    net = load_net(ckpt_path, device="cpu")
    assert torch.equal(net.gp_head.precision_accum, trained_accum)

    # `_cov_stale` is deliberately non-persistent, so the reloaded net must recompute the
    # covariance from the precision rather than trust whatever was cached at save time.
    x = torch.randn(2, 3, 224, 224)
    trained.eval()
    with torch.no_grad():
        expected = trained(x).variance
        actual = net(x).variance
    assert expected is not None
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_trainer_fit_ckpt_path_resume_still_works(tmp_path):
    """The train.py/eval.py pattern (fresh Hydra-instantiated model +
    trainer.fit/test(ckpt_path=...)) never depended on unpickling `net`, and must keep
    working unchanged by this refactor."""
    cfg, ckpt_path, _ = _build_and_train_one_step(tmp_path, "baseline_classifier", ["model.net.pretrained=false"])

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg2 = compose(
            config_name="train.yaml",
            overrides=["data=acevedo", "model=baseline_classifier", "model.net.pretrained=false"],
        )
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
