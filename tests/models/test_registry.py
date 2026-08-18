import pytest

# Import the net modules so their @register_net decorators run.
import src.models.baseline.baseline_models  # noqa: F401
import src.models.ensemble.deep_ensemble_model  # noqa: F401
import src.models.sngp.sngp_classifier  # noqa: F401
from src.models.registry import NET_REGISTRY, build_net


@pytest.mark.parametrize("name", ["baseline_classifier", "sngp_classifier", "deep_ensemble"])
def test_expected_nets_are_registered(name):
    assert name in NET_REGISTRY


def test_build_net_unknown_key_raises():
    with pytest.raises(KeyError):
        build_net({"name": "not_a_registered_net"})


def test_build_net_baseline_classifier():
    net = build_net({"name": "baseline_classifier", "arch": "resnet18", "num_classes": 3, "pretrained": False})
    assert net.num_classes == 3
    assert net.spec["name"] == "baseline_classifier"


def test_build_net_deep_ensemble():
    net = build_net({
        "name": "deep_ensemble",
        "base_model_spec": {"name": "baseline_classifier", "arch": "resnet18", "num_classes": 3, "pretrained": False},
        "num_estimators": 2,
    })
    assert net.num_estimators == 2
    assert len(net.ensemble_members) == 2
