import pytest
import torch

from src.models.baseline.baseline_models import BaselineClassifier
from src.models.ensemble.deep_ensemble_model import DeepEnsemble
from src.models.outputs import ModelOutput
from src.models.sngp.sngp_classifier import SNGPClassifier

NUM_CLASSES = 4
BATCH_SIZE = 2


def _assert_valid_output(output, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES):
    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (batch_size, num_classes)
    assert torch.isfinite(output.logits).all()


class TestModelOutputContract:
    def test_baseline_classifier(self):
        net = BaselineClassifier(arch="resnet18", num_classes=NUM_CLASSES, pretrained=False)
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        _assert_valid_output(net(x))

    def test_sngp_classifier(self):
        net = SNGPClassifier(arch="resnet18", num_classes=NUM_CLASSES, pretrained=False, rff_dim=64)
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        output = net(x)
        _assert_valid_output(output)
        assert output.raw_logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert output.variance.shape == (BATCH_SIZE, 1)

    def test_deep_ensemble_train_mode(self):
        net = DeepEnsemble(
            base_model_spec={"name": "baseline_classifier", "arch": "resnet18", "num_classes": NUM_CLASSES, "pretrained": False},
            num_estimators=3,
        )
        net.train()
        net.set_active_member(0)
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        _assert_valid_output(net(x))

    def test_deep_ensemble_eval_mode(self):
        net = DeepEnsemble(
            base_model_spec={"name": "baseline_classifier", "arch": "resnet18", "num_classes": NUM_CLASSES, "pretrained": False},
            num_estimators=3,
        )
        net.eval()
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        output = net(x)
        _assert_valid_output(output)
        assert output.member_logits.shape == (3, BATCH_SIZE, NUM_CLASSES)
        assert output.variance.shape == (BATCH_SIZE, 1)
