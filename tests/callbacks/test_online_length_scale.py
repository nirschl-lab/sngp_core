import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from src.callbacks.online_length_scale import OnlineLengthScaleEvidence
from src.metrics.gp_evidence import evidence_basis, optimize_hyperparameters
from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess, SNGPClassifier


def _head(length_scale: float = 4.0, in_dim: int = 8, rff_dim: int = 256, num_classes: int = 3):
    torch.manual_seed(0)
    return RandomFeatureGaussianProcess(
        in_dim=in_dim, num_classes=num_classes, rff_dim=rff_dim, length_scale=length_scale, ridge_penalty=1.0,
        normalize_input=False, scale_random_features=False,
    )


def _clusters(n: int = 600, in_dim: int = 8, num_classes: int = 3, spread: float = 1.0):
    g = torch.Generator().manual_seed(1)
    centres = 3.0 * torch.randn(num_classes, in_dim, generator=g)
    y = torch.arange(n) % num_classes
    return centres[y] + spread * torch.randn(n, in_dim, generator=g), y


class TestSchedule:
    def test_update_epochs(self):
        cb = OnlineLengthScaleEvidence(burnin_epochs=10, every_n_epochs=5, stop_epoch=30)
        assert [e for e in range(40) if cb.should_update(e)] == [10, 15, 20, 25]

    def test_grid_is_log_symmetric_around_current(self):
        cb = OnlineLengthScaleEvidence(search_factor=4.0, n_grid=5)
        torch.testing.assert_close(cb.candidate_length_scales(8.0), torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0], dtype=torch.float64))

    def test_damping_is_geometric(self):
        assert OnlineLengthScaleEvidence(damping=0.5).damped(16.0, 4.0) == pytest.approx(8.0)
        assert OnlineLengthScaleEvidence(damping=1.0).damped(16.0, 4.0) == pytest.approx(4.0)

    @pytest.mark.parametrize(
        "kwargs",
        [dict(burnin_epochs=-1), dict(every_n_epochs=0), dict(burnin_epochs=5, stop_epoch=5), dict(search_factor=1.0),
         dict(n_grid=2), dict(damping=0.0), dict(damping=1.5)],
    )
    def test_rejects_bad_args(self, kwargs):
        with pytest.raises(ValueError):
            OnlineLengthScaleEvidence(**kwargs)


class TestScore:
    def test_argmax_matches_brute_force(self):
        head = _head(length_scale=4.0)
        x, y = _clusters()
        cb = OnlineLengthScaleEvidence(search_factor=4.0, n_grid=9)
        result = cb.score(head, x, y)
        brute = []
        for ls in cb.candidate_length_scales(4.0).tolist():
            basis = evidence_basis(head.features_at(x, ls), y, 3)
            brute.append(optimize_hyperparameters(basis)[2] / basis.n)
        assert result["evidence"] == pytest.approx(brute)
        assert result["length_scale_star"] == pytest.approx(cb.candidate_length_scales(4.0)[max(range(9), key=brute.__getitem__)].item())

    def test_interior_optimum_found_from_either_side(self):
        """The evidence has an interior optimum; starting far above or below, the search moves toward it."""
        x, y = _clusters()
        cb = OnlineLengthScaleEvidence(search_factor=4.0, n_grid=13)
        low, high = cb.score(_head(0.5), x, y), cb.score(_head(64.0), x, y)
        assert low["length_scale_star"] > 0.5
        assert high["length_scale_star"] < 64.0

    def test_score_does_not_mutate_head(self):
        head = _head(4.0)
        W = head.W.clone()
        OnlineLengthScaleEvidence(n_grid=5).score(head, *_clusters(n=90))
        assert head.length_scale == 4.0
        assert torch.equal(head.W, W)


class _ImageSet(Dataset):
    def __init__(self, n: int = 12):
        g = torch.Generator().manual_seed(2)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.arange(n) % 4

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return f"img{i}", self.x[i], self.y[i], 0


class _Module:
    """Just what the callback touches on the LightningModule."""

    def __init__(self, net):
        self.net = net
        self.device = torch.device("cpu")
        self.logged = {}

    def log(self, name, value, **_):
        self.logged[name] = value

    def log_dict(self, d, **_):
        self.logged.update(d)


def _trainer(epoch: int = 0, world_size: int = 1):
    datamodule = SimpleNamespace(data_train=_ImageSet(), hparams=SimpleNamespace(num_workers=0, pin_memory=False))
    return SimpleNamespace(current_epoch=epoch, world_size=world_size, datamodule=datamodule)


class TestHooks:
    @pytest.fixture
    def net(self):
        torch.manual_seed(0)
        return SNGPClassifier(num_classes=4, arch="resnet18", rff_dim=64, normalize_input=False, length_scale=20.0)

    def test_update_moves_length_scale_and_logs(self, net):
        cb = OnlineLengthScaleEvidence(burnin_epochs=0, every_n_epochs=1, stop_epoch=5, n_samples=8, batch_size=4, damping=0.5)
        module = _Module(net)
        net.train()
        cb.on_train_start(_trainer(), module)
        cb.on_train_epoch_start(_trainer(epoch=0), module)
        star = module.logged["ls/length_scale_star"]
        assert net.length_scale == pytest.approx(math.sqrt(20.0 * star))
        assert net.spec["length_scale"] == net.length_scale == net.gp_head.length_scale
        assert module.logged["ls/length_scale"] == net.length_scale
        assert net.training, "the eval-mode feature pass must restore train mode"
        assert cb.state_dict() == {"length_scale": net.length_scale}

    def test_no_update_outside_schedule(self, net):
        cb = OnlineLengthScaleEvidence(burnin_epochs=3, every_n_epochs=1, stop_epoch=5, n_samples=8)
        module = _Module(net)
        cb.on_train_start(_trainer(), module)
        cb.on_train_epoch_start(_trainer(epoch=1), module)
        assert net.length_scale == 20.0
        assert "ls/length_scale_star" not in module.logged

    def test_resume_restores_length_scale_attributes(self, net):
        """A resumed run rebuilds the net at the config's l, then loads a `W` that carries the moved one."""
        moved = SNGPClassifier(num_classes=4, arch="resnet18", rff_dim=64, normalize_input=False, length_scale=20.0)
        moved.set_length_scale(7.0)
        net.load_state_dict(moved.state_dict())
        cb = OnlineLengthScaleEvidence()
        cb.load_state_dict({"length_scale": 7.0})
        cb.on_train_start(_trainer(), _Module(net))
        assert net.length_scale == net.gp_head.length_scale == 7.0
        x = torch.randn(5, 512)
        torch.testing.assert_close(net.gp_head._features(x), moved.gp_head._features(x))
        torch.testing.assert_close(net.gp_head.features_at(x, 7.0), moved.gp_head._features(x))

    def test_ddp_refused(self, net):
        with pytest.raises(NotImplementedError):
            OnlineLengthScaleEvidence().setup(_trainer(world_size=2), _Module(net), "fit")

    def test_non_sngp_net_refused(self):
        with pytest.raises(TypeError):
            OnlineLengthScaleEvidence().setup(_trainer(), _Module(torch.nn.Linear(2, 2)), "fit")
