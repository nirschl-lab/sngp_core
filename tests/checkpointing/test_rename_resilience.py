"""Simulates the exact failure this refactor fixes: a model class moving/being
renamed after a checkpoint was saved. Pre-refactor, `hyper_parameters["net"]` pickled
the live object, so unpickling required the *original* class's import path to still
resolve (this is why `src/models/torch_vision_base.py` had to exist as a compat shim).
Post-refactor, checkpoints only store a registry key + plain-data spec, so loading
only cares what is *currently* registered under that key -- never where the class
used to live.
"""
import pytest
import torch

from src.checkpointing.io import load_net
from src.models.baseline.baseline_models import BaselineClassifier
from src.models.registry import NET_REGISTRY
from tests.checkpointing.test_roundtrip import _build_and_train_one_step


class _RelocatedBaselineClassifier(BaselineClassifier):
    """Stand-in for `BaselineClassifier` after a hypothetical file move/rename: a
    distinct class object at a different (fake) location, structurally compatible."""


@pytest.mark.slow
def test_load_net_does_not_depend_on_the_original_class_still_resolving(tmp_path, monkeypatch):
    _, ckpt_path, _ = _build_and_train_one_step(tmp_path, "baseline_classifier", ["model.net.pretrained=false"])

    # Simulate "the code moved": whatever is registered under this key at load time is
    # no longer the class the checkpoint was originally saved with.
    monkeypatch.setitem(NET_REGISTRY, "baseline_classifier", _RelocatedBaselineClassifier)

    net = load_net(ckpt_path, device="cpu")
    assert isinstance(net, _RelocatedBaselineClassifier)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = net(x).logits
    assert out.shape[0] == 2
