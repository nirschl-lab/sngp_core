"""`SpectralRegularizer` (src/models/components/spectral_reg.py): the rep-spectral penalty
`sum_l sigma_max^2(W_l)`. The estimator is checked against exact singular values -- for
convs against the *materialized* conv operator, since "true operator norm" is the whole
point of `conv_mode="operator"` -- and the analytic gradient `2 sigma u v^T` is checked
explicitly. No network access, tiny tensors, CPU only.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.spectral_reg import CONV_MODES, SpectralRegOutput, SpectralRegularizer
from src.models.sngp.sngp_classifier import SNGPClassifier


def _dense_conv_operator(conv: nn.Conv2d, in_hw) -> torch.Tensor:
    """Materialize K~ (the conv as one matrix) by pushing every input basis vector through
    the conv. Small shapes only."""
    c_in = conv.in_channels
    h, w = in_hw
    n_in = c_in * h * w
    basis = torch.eye(n_in).reshape(n_in, c_in, h, w)
    with torch.no_grad():
        out = F.conv2d(basis, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
    return out.reshape(n_in, -1).t()  # [n_out, n_in]


def _exact_operator_sigma(conv: nn.Conv2d, in_hw) -> float:
    return torch.linalg.matrix_norm(_dense_conv_operator(conv, in_hw), ord=2).item()


def _exact_matrix_sigma(weight: torch.Tensor) -> float:
    return torch.linalg.matrix_norm(weight.reshape(weight.shape[0], -1), ord=2).item()


class TestEstimator:
    @pytest.mark.parametrize(
        "kernel,stride,padding,in_hw",
        [
            (3, 1, 1, (8, 8)),  # resnet basic-block conv
            (3, 2, 1, (8, 8)),  # strided block entry
            (1, 2, 0, (8, 8)),  # downsample shortcut
            (7, 2, 3, (16, 16)),  # stem
            (3, 1, 1, (7, 7)),  # odd spatial size
            (3, 2, 1, (7, 7)),  # odd + stride: exercises output_padding
        ],
    )
    def test_operator_sigma_matches_materialized_conv(self, kernel, stride, padding, in_hw):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 5, kernel, stride=stride, padding=padding, bias=False)
        net = nn.Sequential(conv)
        reg = SpectralRegularizer(net, conv_mode="operator", warmup_iterations=500)
        net(torch.randn(1, 3, *in_hw))  # records the input shape
        out = reg()
        assert isinstance(out, SpectralRegOutput)
        assert out.sigmas.shape == (1,)
        assert out.sigmas[0].item() == pytest.approx(_exact_operator_sigma(conv, in_hw), rel=1e-2)

    def test_reshape_sigma_matches_reshaped_kernel_matrix_norm(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 5, 3, padding=1, bias=False)
        reg = SpectralRegularizer(nn.Sequential(conv), conv_mode="reshape", warmup_iterations=500)
        out = reg()  # no forward needed: reshape mode does not depend on the input shape
        assert out.sigmas[0].item() == pytest.approx(_exact_matrix_sigma(conv.weight), rel=1e-3)

    def test_linear_sigma_matches_matrix_norm(self):
        torch.manual_seed(0)
        lin = nn.Linear(12, 7)
        for mode in CONV_MODES:  # conv_mode is irrelevant for Linear
            reg = SpectralRegularizer(nn.Sequential(lin), conv_mode=mode, warmup_iterations=500)
            assert reg().sigmas[0].item() == pytest.approx(_exact_matrix_sigma(lin.weight), rel=1e-3)

    def test_one_by_one_strided_conv_operator_equals_matrix_norm(self):
        """A 1x1 conv is pure channel mixing (stride only subsamples), so both estimators
        must agree -- the one case where the two quantities coincide exactly."""
        torch.manual_seed(0)
        conv = nn.Conv2d(4, 6, 1, stride=2, bias=False)
        net = nn.Sequential(conv)
        op_reg = SpectralRegularizer(net, conv_mode="operator", warmup_iterations=500)
        rs_reg = SpectralRegularizer(net, conv_mode="reshape", warmup_iterations=500)
        net(torch.randn(1, 4, 8, 8))  # after construction: the operator-mode hook must see it
        op = op_reg().sigmas[0].item()
        rs = rs_reg().sigmas[0].item()
        assert op == pytest.approx(rs, rel=1e-3) == pytest.approx(_exact_matrix_sigma(conv.weight), rel=1e-3)

    def test_operator_norm_exceeds_reshape_estimate_for_3x3(self):
        """The reshaped-kernel estimate under-shoots the true operator norm -- the
        documented reason `spectral_norm_bound` values are not comparable to these sigmas."""
        torch.manual_seed(0)
        conv = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        net = nn.Sequential(conv)
        reg = SpectralRegularizer(net, conv_mode="operator", warmup_iterations=500)
        net(torch.randn(1, 8, 16, 16))
        assert reg().sigmas[0].item() > _exact_matrix_sigma(conv.weight)

    def test_power_iteration_tracks_weight_changes(self):
        """sigma is linear in W for fixed (u, v): scaling the kernel scales the estimate."""
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        net = nn.Sequential(conv)
        reg = SpectralRegularizer(net, warmup_iterations=200)
        net(torch.randn(1, 3, 8, 8))
        before = reg().sigmas[0].item()
        with torch.no_grad():
            conv.weight.mul_(3.0)
        assert reg().sigmas[0].item() == pytest.approx(3.0 * before, rel=1e-3)

    def test_input_shape_change_reinitializes_vectors(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
        net = nn.Sequential(conv)
        reg = SpectralRegularizer(net, warmup_iterations=500)
        net(torch.randn(1, 3, 8, 8))
        reg()
        assert reg.v_0.shape[-2:] == (8, 8)

        net(torch.randn(1, 3, 12, 12))
        out = reg()
        assert reg.v_0.shape[-2:] == (12, 12)
        assert out.sigmas[0].item() == pytest.approx(_exact_operator_sigma(conv, (12, 12)), rel=1e-2)


class TestPenaltyAndGradient:
    def _net_and_reg(self, **kwargs):
        """conv -> BN -> ReLU -> flatten -> linear, with the regularizer attached *before*
        the shape-recording forward pass."""
        torch.manual_seed(0)
        net = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 6),
        )
        reg = SpectralRegularizer(net, **kwargs)
        net(torch.randn(2, 3, 8, 8))
        return net, reg

    def test_penalty_is_sum_of_squared_sigmas(self):
        _, reg = self._net_and_reg(warmup_iterations=50)
        out = reg()
        torch.testing.assert_close(out.penalty, (out.sigmas**2).sum())
        assert out.penalty.requires_grad
        assert not out.sigmas.requires_grad

    def test_gradient_is_the_analytic_two_sigma_u_v_transpose(self):
        """d sigma^2 / dW = 2 sigma u v^T (the paper's closed form). Checked on the Linear
        layer, where u v^T is a plain outer product of the stored vectors."""
        net, reg = self._net_and_reg(warmup_iterations=500)
        out = reg()
        out.penalty.backward()

        lin = net[4]
        lin_idx = reg.layer_names.index("4")
        u, v = getattr(reg, f"u_{lin_idx}"), getattr(reg, f"v_{lin_idx}")
        sigma = out.sigmas[lin_idx]
        torch.testing.assert_close(lin.weight.grad, 2 * sigma * torch.outer(u, v), rtol=1e-4, atol=1e-6)

    def test_gradient_reaches_conv_and_linear_weights_only(self):
        net, reg = self._net_and_reg(warmup_iterations=20)
        reg().penalty.backward()
        conv, bn, lin = net[0], net[1], net[4]
        assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
        assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
        assert conv.bias.grad is None and lin.bias.grad is None
        assert bn.weight.grad is None and bn.bias.grad is None

    def test_only_conv_and_linear_layers_are_enumerated(self):
        _, reg = self._net_and_reg(conv_mode="reshape")
        assert reg.layer_names == ["0", "4"]
        assert len(reg) == 2


class TestIntegrationWithSNGP:
    def _backbone_reg(self, **kwargs):
        torch.manual_seed(0)
        net = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64, use_spectral_norm=False)
        return net, SpectralRegularizer(net.backbone, **kwargs)

    def test_regularizes_every_resnet18_conv_and_nothing_in_the_gp_head(self):
        _, reg = self._backbone_reg()
        names = reg.layer_names
        assert len(names) == 20  # stem + 16 block convs + 3 downsample shortcuts; fc is Identity
        assert "conv1" in names and "layer4.1.conv2" in names
        assert "layer2.0.downsample.0" in names  # paper regularizes the shortcut convs too
        assert not any("gp_head" in n or "classifier" in n for n in names)

    def test_penalty_on_untrained_resnet18_is_finite_and_of_the_measured_scale(self):
        """sum sigma^2 of an untrained torchvision resnet18 at 224px is ~130 under the
        operator norm (measured while designing this); guard against a silently broken
        estimator returning ~0 or ~1e6, not against init noise."""
        net, reg = self._backbone_reg(warmup_iterations=20)
        net.train()
        net(torch.randn(2, 3, 224, 224))
        out = reg()
        assert torch.isfinite(out.penalty)
        assert 30.0 < out.penalty.item() < 500.0
        assert (out.sigmas > 0).all()

    def test_refuses_a_spectral_normalized_backbone(self):
        net = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64)  # SN on
        with pytest.raises(ValueError, match="must not be combined"):
            SpectralRegularizer(net.backbone)

    def test_no_persistent_state(self):
        """u/v never enter a checkpoint: a regularized net's state_dict is a plain net's."""
        net, reg = self._backbone_reg()
        net(torch.randn(1, 3, 224, 224))
        reg()
        assert dict(reg.named_buffers())  # vectors exist ...
        assert reg.state_dict() == {}  # ... but are not persisted


class TestValidation:
    def test_operator_mode_requires_a_forward_pass_first(self):
        reg = SpectralRegularizer(nn.Sequential(nn.Conv2d(3, 4, 3)), conv_mode="operator")
        with pytest.raises(RuntimeError, match="forward pass"):
            reg()

    def test_forward_before_construction_is_not_enough(self):
        """The shape hook is attached at construction; an earlier forward is invisible."""
        net = nn.Sequential(nn.Conv2d(3, 4, 3))
        net(torch.randn(1, 3, 8, 8))
        reg = SpectralRegularizer(net, conv_mode="operator")
        with pytest.raises(RuntimeError, match=r"\*after\* constructing"):
            reg()

    def test_rejects_bad_arguments(self):
        net = nn.Sequential(nn.Linear(3, 2))
        with pytest.raises(ValueError, match="conv_mode"):
            SpectralRegularizer(net, conv_mode="fft")
        with pytest.raises(ValueError, match="n_power_iterations"):
            SpectralRegularizer(net, n_power_iterations=0)
        with pytest.raises(ValueError, match="no Conv2d/Linear"):
            SpectralRegularizer(nn.Sequential(nn.ReLU()))

    def test_rejects_non_zero_padding_modes_in_operator_mode(self):
        conv = nn.Conv2d(3, 4, 3, padding=1, padding_mode="circular")
        with pytest.raises(NotImplementedError, match="padding_mode"):
            SpectralRegularizer(nn.Sequential(conv), conv_mode="operator")
        SpectralRegularizer(nn.Sequential(conv), conv_mode="reshape")  # fine: kernel-only


class TestBatchNormSpectralRegularizer:
    """`BatchNormSpectralRegularizer`: sum_l (max_i |gamma_i| / sqrt(running_var_i + eps))^2."""

    def _model(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 4, 3), nn.BatchNorm2d(4))
        with torch.no_grad():
            model[1].weight.uniform_(0.5, 2.0)
            model[4].weight.uniform_(-3.0, 3.0)
            model[1].running_var.uniform_(0.1, 2.0)
            model[4].running_var.uniform_(0.1, 2.0)
        return model

    def test_penalty_is_sum_of_squared_bn_gains(self):
        from src.models.components.spectral_reg import BatchNormSpectralRegularizer

        model = self._model()
        reg = BatchNormSpectralRegularizer(model)
        assert reg.layer_names == ["1", "4"]
        out = reg()
        gains = torch.stack([(bn.weight.abs() / (bn.running_var + bn.eps).sqrt()).max() for bn in (model[1], model[4])])
        torch.testing.assert_close(out.sigmas, gains.detach())
        torch.testing.assert_close(out.penalty, (gains**2).sum())

    def test_gradient_reaches_bn_gamma_only(self):
        from src.models.components.spectral_reg import BatchNormSpectralRegularizer

        model = self._model()
        BatchNormSpectralRegularizer(model)().penalty.backward()
        for bn in (model[1], model[4]):
            assert (bn.weight.grad != 0).sum() == 1  # the arg-max channel only
            assert bn.bias.grad is None
        assert model[0].weight.grad is None and model[3].weight.grad is None

    def test_enumerates_every_resnet18_batchnorm_and_adds_no_state(self):
        from src.models.components.spectral_reg import BatchNormSpectralRegularizer

        net = SNGPClassifier(num_classes=4, arch="resnet18", pretrained=False, rff_dim=64, use_spectral_norm=False)
        reg = BatchNormSpectralRegularizer(net.backbone)
        assert len(reg) == 20
        assert torch.isfinite(reg().penalty)
        assert len(reg.state_dict()) == 0

    def test_refuses_a_module_without_batchnorm(self):
        from src.models.components.spectral_reg import BatchNormSpectralRegularizer

        with pytest.raises(ValueError, match="no BatchNorm2d"):
            BatchNormSpectralRegularizer(nn.Sequential(nn.Linear(4, 4)))
