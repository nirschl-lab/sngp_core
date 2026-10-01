"""`SNGPSpectralRegLitModule`: the penalty is added only in train mode, only on the
regularization cadence, only after burn-in; validation stays plain CE; construction
refuses a spectral-normalized net; hyperparameters stay primitive. The trainer-less
`model_step` calls below see `current_epoch == 0` and `global_step == 0` (Lightning's
defaults without a trainer); other phases are reached by patching those properties."""
import json

import pytest
import torch

from src.models.lit_module_base import LitModuleBase
from src.models.sngp.sngp_classifier import SNGPClassifier
from src.models.sngp_lit_module import SNGPLitModule
from src.models.sngp_specreg_lit_module import SNGPSpectralRegLitModule


def make_net():
    return SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64, use_spectral_norm=False)


def make_module(**kwargs):
    defaults = dict(spec_reg_every_n_steps=1, spec_reg_burnin_epochs=0, spec_reg_warmup_iterations=50)
    defaults.update(kwargs)
    return SNGPSpectralRegLitModule(net=make_net(), optimizer=None, scheduler=None, num_classes=4, compile=False, **defaults)


def make_batch(n: int = 2):
    return [f"img{i}" for i in range(n)], torch.randn(n, 3, 224, 224), torch.arange(n) % 4, ["train"] * n


def base_ce(module, batch) -> torch.Tensor:
    """The plain-CE loss `LitModuleBase.model_step` would return for this module/batch."""
    return LitModuleBase.model_step(module, batch)[1]


class TestConstruction:
    def test_refuses_spectral_normalized_net(self):
        net = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64)  # use_spectral_norm=True
        with pytest.raises(ValueError, match="must not be combined"):
            SNGPSpectralRegLitModule(net=net, optimizer=None, scheduler=None, num_classes=4, compile=False)

    def test_regularizes_the_backbone_only(self):
        module = make_module()
        assert len(module.spec_reg) == 20
        assert not any("gp_head" in name for name in module.spec_reg.layer_names)

    def test_hparams_are_primitive_and_include_the_knobs(self):
        module = make_module(spec_reg_coef=0.25, spec_reg_every_n_steps=3, spec_reg_burnin_epochs=7)
        hparams = dict(module.hparams)
        json.dumps(hparams)
        assert "net" not in hparams and "optimizer" not in hparams and "scheduler" not in hparams
        assert hparams["net_spec"]["use_spectral_norm"] is False
        assert hparams["spec_reg_coef"] == 0.25
        assert hparams["spec_reg_every_n_steps"] == 3
        assert hparams["spec_reg_burnin_epochs"] == 7
        assert hparams["spec_reg_conv_mode"] == "operator"

    def test_state_dict_is_identical_to_a_plain_sngp_module(self):
        """The regularizer adds no persistent state, so checkpoints of this module load
        into the same keys a plain (SN-free) SNGPLitModule has -- nothing special to migrate."""
        module = make_module()
        module.train()
        module.model_step(make_batch())  # creates the (non-persistent) power-iteration vectors
        plain = SNGPLitModule(net=make_net(), optimizer=None, scheduler=None, num_classes=4, compile=False)
        assert set(module.state_dict()) == set(plain.state_dict())

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(spec_reg_coef=-1.0), "spec_reg_coef"),
            (dict(spec_reg_every_n_steps=0), "spec_reg_every_n_steps"),
            (dict(spec_reg_burnin_epochs=-1), "spec_reg_burnin_epochs"),
            (dict(spec_reg_conv_mode="fft"), "conv_mode"),
        ],
    )
    def test_rejects_bad_knobs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_module(**kwargs)


class TestTrainingLoss:
    def test_active_phase_adds_the_weighted_penalty(self):
        torch.manual_seed(0)
        coef = 0.5
        module = make_module(spec_reg_coef=coef, spec_reg_warmup_iterations=200)
        module.train()
        batch = make_batch()

        ce = base_ce(module, batch)
        _, total, *_ = module.model_step(batch)
        # One more (converged) power-iteration round moves sigma negligibly, so the
        # penalty read back now is the one that was added.
        with torch.no_grad():
            penalty = module.spec_reg().penalty
        assert total.item() > ce.item()
        assert (total - ce).item() == pytest.approx(coef * penalty.item(), rel=1e-3)

        total.backward()
        conv = module.net.backbone.conv1
        assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()

    def test_burnin_phase_returns_plain_ce_but_still_measures_sigma(self):
        torch.manual_seed(0)
        module = make_module(spec_reg_burnin_epochs=5)  # current_epoch is 0 without a trainer
        module.train()
        batch = make_batch()
        assert not module.spec_reg_is_active()

        ce = base_ce(module, batch)
        _, loss, *_ = module.model_step(batch)
        torch.testing.assert_close(loss, ce)
        assert dict(module.spec_reg.named_buffers()), "sigma is still estimated during burn-in, for logging"

    def test_becomes_active_once_burnin_is_over(self, monkeypatch):
        module = make_module(spec_reg_burnin_epochs=5)
        monkeypatch.setattr(SNGPSpectralRegLitModule, "current_epoch", property(lambda self: 5))
        assert module.spec_reg_is_active()
        module.train()
        batch = make_batch()
        ce = base_ce(module, batch)
        _, total, *_ = module.model_step(batch)
        assert total.item() > ce.item()

    def test_off_cadence_steps_are_plain_ce(self, monkeypatch):
        module = make_module(spec_reg_every_n_steps=24)
        monkeypatch.setattr(SNGPSpectralRegLitModule, "global_step", property(lambda self: 5))
        module.train()
        batch = make_batch()
        ce = base_ce(module, batch)
        _, loss, *_ = module.model_step(batch)
        torch.testing.assert_close(loss, ce)
        assert not dict(module.spec_reg.named_buffers()), "the penalty is not even evaluated off-cadence"

    def test_on_cadence_step_is_regularized(self, monkeypatch):
        module = make_module(spec_reg_every_n_steps=24)
        monkeypatch.setattr(SNGPSpectralRegLitModule, "global_step", property(lambda self: 48))
        module.train()
        batch = make_batch()
        ce = base_ce(module, batch)
        _, total, *_ = module.model_step(batch)
        assert total.item() > ce.item()

    def test_eval_mode_is_plain_ce_and_never_touches_the_regularizer(self):
        """Validation (and the sanity check) call `model_step` in eval mode: `val/nll`
        must stay comparable to every other family's."""
        torch.manual_seed(0)
        module = make_module()
        module.train()
        module(torch.randn(4, 3, 224, 224))  # populate the precision matrix
        module.eval()
        batch = make_batch()
        ce = base_ce(module, batch)
        _, loss, _, _, _, _, _, output = module.model_step(batch)
        torch.testing.assert_close(loss, ce)
        assert output.variance is not None  # eval-mode SNGP output, untouched
        assert not dict(module.spec_reg.named_buffers())


class TestInheritedSNGPBehaviour:
    def test_precision_is_accumulated_in_train_and_reset_per_epoch(self):
        module = make_module()
        module.train()
        module(torch.randn(2, 3, 224, 224))
        assert torch.count_nonzero(module.net.gp_head.precision_accum) > 0
        module.on_train_epoch_start()
        assert torch.count_nonzero(module.net.gp_head.precision_accum) == 0


class TestBatchNormTarget:
    def test_batchnorm_target_regularizes_the_backbone_bns(self):
        from src.models.components.spectral_reg import BatchNormSpectralRegularizer

        module = make_module(spec_reg_target="batchnorm")
        assert isinstance(module.spec_reg, BatchNormSpectralRegularizer)
        assert len(module.spec_reg) == 20
        assert not any("gp_head" in name for name in module.spec_reg.layer_names)
        assert module.hparams["spec_reg_target"] == "batchnorm"
        json.dumps(dict(module.hparams))

    def test_batchnorm_target_adds_the_weighted_penalty(self):
        module = make_module(spec_reg_target="batchnorm", spec_reg_coef=0.5)
        # Module in train mode (so the penalty branch runs) but the net in eval mode, so the
        # forward leaves the BN running stats -- and hence CE and penalty -- unchanged.
        module.train()
        module.net.eval()
        batch = make_batch()
        ce = base_ce(module, batch)
        penalty = module.spec_reg().penalty
        total = module.model_step(batch)[1]
        assert penalty.item() > 0
        torch.testing.assert_close(total, ce + 0.5 * penalty)

    def test_rejects_an_unknown_target(self):
        with pytest.raises(ValueError, match="spec_reg_target"):
            make_module(spec_reg_target="layernorm")
