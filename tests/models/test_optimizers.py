"""Guards for `src/models/components/optimizers.py`.

The Muon / AdamW split is by parameter shape, not name, so a wrong rule would silently
send the stem, BatchNorm or the GP output layer to Muon (or every conv to AdamW) and
still train.
"""
import pytest
import torch

from src.models.components.optimizers import MuonWithAuxAdamW, is_muon_param, zeropower_via_newtonschulz5
from src.models.sngp.sngp_classifier import SNGPClassifier


@pytest.fixture(scope="module")
def net() -> SNGPClassifier:
    torch.manual_seed(0)
    return SNGPClassifier(num_classes=8, arch="resnet18", use_spectral_norm=False)


def _group_ids(optimizer: MuonWithAuxAdamW) -> tuple[set[int], set[int]]:
    muon, adamw = optimizer.param_groups
    assert muon["use_muon"] and not adamw["use_muon"]
    return {id(p) for p in muon["params"]}, {id(p) for p in adamw["params"]}


def test_split_on_sngp_resnet18(net: SNGPClassifier):
    optimizer = MuonWithAuxAdamW(net.parameters())
    muon, adamw = _group_ids(optimizer)
    named = dict(net.named_parameters())
    assert id(named["backbone.conv1.weight"]) in adamw  # RGB stem
    assert id(named["gp_head.classifier.weight"]) in adamw  # GP output layer
    assert all(id(p) in adamw for n, p in named.items() if p.ndim == 1)  # BN / LN / biases
    hidden_convs = [n for n, p in named.items() if p.ndim == 4 and n != "backbone.conv1.weight"]
    assert len(hidden_convs) == 19
    assert {id(named[n]) for n in hidden_convs} == muon
    assert len(muon) + len(adamw) == len(named)


def test_rejects_models_without_conv_kernels():
    with pytest.raises(ValueError, match="no hidden conv kernels"):
        MuonWithAuxAdamW(torch.nn.Linear(4, 4).parameters())


@pytest.mark.parametrize("shape", [(64, 576), (512, 128), (32, 32)])
def test_newton_schulz_is_near_orthogonal(shape):
    torch.manual_seed(0)
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(torch.randn(shape)).float())
    assert s.min() > 0.5 and s.max() < 1.3  # quintic NS: ~[0.7, 1.2], not exactly 1


def test_step_updates_both_groups_and_muon_update_is_spectrally_bounded():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3), torch.nn.BatchNorm2d(32))
    lr = 0.02
    optimizer = MuonWithAuxAdamW(model.parameters(), lr=lr, adamw_lr=1e-2)
    before = [p.detach().clone() for p in model.parameters()]
    (model(torch.randn(4, 16, 8, 8)) * 1e3).square().mean().backward()  # large grad on purpose
    optimizer.step()
    for b, p in zip(before, model.parameters()):
        assert not torch.equal(b, p)
    conv = model[0].weight
    delta = (conv.detach() - before[0]).reshape(len(conv), -1)
    # Orthogonalized: the step's spectral norm is ~lr whatever the gradient scale.
    assert torch.linalg.matrix_norm(delta, ord=2) < 1.3 * lr
    # Bias of the conv is 1D -> AdamW.
    assert is_muon_param(conv) and not is_muon_param(model[0].bias)


def test_weight_decay_shrinks_muon_weights_only_by_lr_times_wd():
    model = torch.nn.Conv2d(8, 8, 3, bias=False)
    optimizer = MuonWithAuxAdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    before = model.weight.detach().clone()
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.step()  # zero grad: orthogonalizing 0 gives 0, only decay acts
    torch.testing.assert_close(model.weight.detach(), before * (1 - 0.1 * 0.5))


def test_state_dict_roundtrip_and_cosine_schedule(net: SNGPClassifier):
    optimizer = MuonWithAuxAdamW(net.parameters(), lr=0.02, adamw_lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    net.train()
    out = net(torch.randn(2, 3, 64, 64), update_precision=False)
    out.logits.sum().backward()
    optimizer.step()
    scheduler.step()
    # Both groups decay from their own initial LR.
    factor = optimizer.param_groups[0]["lr"] / 0.02
    assert 0 < factor < 1
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-3 * factor)

    fresh = MuonWithAuxAdamW(net.parameters(), lr=0.02, adamw_lr=1e-3)
    fresh.load_state_dict(optimizer.state_dict())
    for a, b in zip(fresh.param_groups, optimizer.param_groups):
        assert a["lr"] == b["lr"] and a["use_muon"] == b["use_muon"]
    assert len(fresh.state) == len(optimizer.state)
