"""Guards for `src/models/components/optimizers.py`.

The Muon / aux split is by parameter shape, not name, so a wrong rule would silently
send the stem, BatchNorm or the GP output layer to Muon (or every conv to the aux
optimizer) and still train. `MuonWithAuxSGD`'s aux group must be exactly the SGD arms'
update rule, or the Muon-vs-SGD comparison is no longer confined to the hidden convs.
"""
import pytest
import torch

from src.models.components.optimizers import (
    MuonWithAuxAdamW,
    MuonWithAuxSGD,
    is_muon_param,
    zeropower_via_newtonschulz5,
)
from src.models.sngp.sngp_classifier import SNGPClassifier


@pytest.fixture(scope="module")
def net() -> SNGPClassifier:
    torch.manual_seed(0)
    return SNGPClassifier(num_classes=8, arch="resnet18", use_spectral_norm=False)


def _group_ids(optimizer: torch.optim.Optimizer) -> tuple[set[int], set[int]]:
    muon, adamw = optimizer.param_groups
    assert muon["use_muon"] and not adamw["use_muon"]
    return {id(p) for p in muon["params"]}, {id(p) for p in adamw["params"]}


@pytest.mark.parametrize("cls", [MuonWithAuxAdamW, MuonWithAuxSGD])
def test_split_on_sngp_resnet18(net: SNGPClassifier, cls):
    optimizer = cls(net.parameters())
    muon, adamw = _group_ids(optimizer)
    named = dict(net.named_parameters())
    assert id(named["backbone.conv1.weight"]) in adamw  # RGB stem
    assert id(named["gp_head.classifier.weight"]) in adamw  # GP output layer
    assert all(id(p) in adamw for n, p in named.items() if p.ndim == 1)  # BN / LN / biases
    hidden_convs = [n for n, p in named.items() if p.ndim == 4 and n != "backbone.conv1.weight"]
    assert len(hidden_convs) == 19
    assert {id(named[n]) for n in hidden_convs} == muon
    assert len(muon) + len(adamw) == len(named)


@pytest.mark.parametrize("cls", [MuonWithAuxAdamW, MuonWithAuxSGD])
def test_rejects_models_without_conv_kernels(cls):
    with pytest.raises(ValueError, match="no hidden conv kernels"):
        cls(torch.nn.Linear(4, 4).parameters())


@pytest.mark.parametrize("cls", [MuonWithAuxAdamW, MuonWithAuxSGD])
def test_custom_is_muon_selects_linear_weights(cls):
    """An MLP passes its own predicate: its hidden 2D weights go to Muon, and Muon's step
    keeps them spectrally bounded just as it does a flattened conv kernel."""
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    hidden = mlp[0].weight
    optimizer = cls(mlp.parameters(), lr=0.02, is_muon=lambda p: p is hidden)
    muon, aux = _group_ids(optimizer)
    assert muon == {id(hidden)}
    assert aux == {id(mlp[0].bias), id(mlp[2].weight), id(mlp[2].bias)}

    before = hidden.detach().clone()
    (mlp(torch.randn(8, 16)) * 1e3).square().mean().backward()
    optimizer.step()
    assert torch.linalg.matrix_norm(hidden.detach() - before, ord=2) < 1.3 * 0.02


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


@pytest.mark.parametrize("cls, aux_lr", [(MuonWithAuxAdamW, "adamw_lr"), (MuonWithAuxSGD, "sgd_lr")])
def test_state_dict_roundtrip_and_cosine_schedule(net: SNGPClassifier, cls, aux_lr):
    optimizer = cls(net.parameters(), lr=0.02, **{aux_lr: 1e-3})
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

    fresh = cls(net.parameters(), lr=0.02, **{aux_lr: 1e-3})
    fresh.load_state_dict(optimizer.state_dict())
    for a, b in zip(fresh.param_groups, optimizer.param_groups):
        assert a["lr"] == b["lr"] and a["use_muon"] == b["use_muon"]
    assert len(fresh.state) == len(optimizer.state)


@pytest.mark.parametrize("nesterov", [True, False])
def test_sgd_aux_group_matches_torch_sgd(nesterov):
    """The aux step is torch.optim.SGD (coupled L2, dampening 0), step for step."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(8, 8, 3), torch.nn.BatchNorm2d(8),
                                torch.nn.Flatten(), torch.nn.Linear(8 * 6 * 6, 5))
    hp = dict(lr=0.04, momentum=0.9, nesterov=nesterov, weight_decay=6e-4)
    optimizer = MuonWithAuxSGD(model.parameters(), sgd_lr=hp["lr"], sgd_momentum=hp["momentum"],
                               sgd_nesterov=nesterov, sgd_weight_decay=hp["weight_decay"])
    aux = optimizer.param_groups[1]["params"]
    assert len(aux) == 5  # conv bias, BN weight/bias, linear weight/bias
    ref_params = [p.detach().clone().requires_grad_() for p in aux]
    reference = torch.optim.SGD(ref_params, **hp)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.randn(4, 8, 8, 8)).square().mean().backward()
        for r, p in zip(ref_params, aux):
            r.grad = p.grad.clone()
        optimizer.step()
        reference.step()
        for r, p in zip(ref_params, aux):
            torch.testing.assert_close(p.detach(), r.detach())


def test_switch_muon_group_to_sgd_matches_torch_sgd_and_persists():
    """After the two-stage switch the hidden convs step exactly as torch.optim.SGD from a fresh
    buffer (Muon's EMA buffer dropped), and the switch survives a state_dict round-trip."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(8, 8, 3), torch.nn.BatchNorm2d(8),
                                torch.nn.Flatten(), torch.nn.Linear(8 * 6 * 6, 5))
    optimizer = MuonWithAuxSGD(model.parameters(), lr=0.02, weight_decay=0.01, sgd_lr=0.01)

    def backward() -> None:
        optimizer.zero_grad()
        model(torch.randn(4, 8, 8, 8)).square().mean().backward()

    backward()
    optimizer.step()  # one Muon step, so group 0 carries a Muon momentum buffer
    hidden = optimizer.param_groups[0]["params"]
    assert all("momentum_buffer" in optimizer.state[p] for p in hidden)

    hp = dict(lr=0.01, momentum=0.9, nesterov=True, weight_decay=6e-4)
    optimizer.param_groups[0]["lr"] = hp["lr"]
    optimizer.switch_muon_group_to_sgd(momentum=hp["momentum"], nesterov=True,
                                       weight_decay=hp["weight_decay"])
    group = optimizer.param_groups[0]
    assert not group["use_muon"] and group["momentum"] == 0.9 and group["weight_decay"] == 6e-4
    assert all("momentum_buffer" not in optimizer.state[p] for p in hidden)

    ref_params = [p.detach().clone().requires_grad_() for p in hidden]
    reference = torch.optim.SGD(ref_params, **hp)
    for _ in range(3):
        backward()
        for r, p in zip(ref_params, hidden):
            r.grad = p.grad.clone()
        optimizer.step()
        reference.step()
        for r, p in zip(ref_params, hidden):
            torch.testing.assert_close(p.detach(), r.detach())

    fresh = MuonWithAuxSGD(model.parameters(), lr=0.02, sgd_lr=0.01)
    assert fresh.param_groups[0]["use_muon"]
    fresh.load_state_dict(optimizer.state_dict())
    assert not fresh.param_groups[0]["use_muon"]
    assert fresh.param_groups[0]["weight_decay"] == 6e-4
