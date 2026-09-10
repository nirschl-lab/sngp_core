"""Coverage for `SNGPLitModule`'s two responsibilities: gating the precision update on
train mode, and resetting the accumulator each epoch. Neither had any test before."""

import torch

from src.models.sngp.sngp_classifier import SNGPClassifier
from src.models.sngp_lit_module import SNGPLitModule


def make_module():
    net = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64)
    return SNGPLitModule(net=net, optimizer=None, scheduler=None, num_classes=4, compile=False)


class TestForwardGating:
    def test_train_mode_accumulates_precision(self):
        module = make_module()
        module.train()
        module(torch.randn(2, 3, 224, 224))
        assert torch.count_nonzero(module.net.gp_head.precision_accum) > 0

    def test_eval_mode_does_not_accumulate(self):
        module = make_module()
        module.eval()
        before = module.net.gp_head.precision_accum.clone()
        module(torch.randn(2, 3, 224, 224))
        assert torch.equal(module.net.gp_head.precision_accum, before)

    def test_train_mode_output_is_uncorrected(self):
        """`LitModuleBase.model_step` takes `.logits` for the CE loss, so this is what
        keeps the mean-field correction out of training."""
        module = make_module()
        module.train()
        output = module(torch.randn(2, 3, 224, 224))
        assert output.variance is None
        assert torch.equal(output.logits, output.raw_logits)

    def test_eval_mode_output_is_corrected(self):
        module = make_module()
        module.train()
        module(torch.randn(4, 3, 224, 224))
        module.eval()
        output = module(torch.randn(2, 3, 224, 224))
        assert output.variance is not None
        assert not torch.allclose(output.logits, output.raw_logits)


class TestEpochReset:
    def test_on_train_epoch_start_resets_precision(self):
        module = make_module()
        module.train()
        module(torch.randn(2, 3, 224, 224))
        assert torch.count_nonzero(module.net.gp_head.precision_accum) > 0

        module.on_train_epoch_start()
        assert torch.count_nonzero(module.net.gp_head.precision_accum) == 0

    def test_accumulator_holds_only_the_current_epoch(self):
        """The per-epoch reset is what makes the accumulator equal exactly one pass over
        the data -- the paper's 'final epoch' precision update.

        The backbone is pinned to eval mode so BatchNorm running stats and the spectral
        norm power iteration (both of which mutate on every train-mode forward) cannot
        drift the features between the two epochs; only the reset is under test.
        """
        torch.manual_seed(0)
        module = make_module()
        batch = torch.randn(2, 3, 224, 224)

        module.train()
        module.net.backbone.eval()

        module.on_train_epoch_start()
        module(batch)
        epoch_one = module.net.gp_head.precision_accum.clone()

        module.on_train_epoch_start()
        module(batch)
        # Same data, same weights (no optimizer step): a reset epoch reproduces the
        # previous one exactly.
        torch.testing.assert_close(module.net.gp_head.precision_accum, epoch_one)

        # Without the reset it would keep growing -- a second batch doubles it.
        module(batch)
        torch.testing.assert_close(module.net.gp_head.precision_accum, epoch_one * 2)
