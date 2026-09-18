"""Regression guard for the original bug: `LitModuleBase.save_hyperparameters()` used
to pickle the live `net` object (and optimizer/scheduler partials) into
`checkpoint["hyper_parameters"]`, which is exactly why checkpoints ended up depending
on the code structure they were saved with. This test fails loudly if that ever
regresses, for every model family the project ships.
"""
import json

import hydra
import pytest
from hydra import compose, initialize


@pytest.mark.parametrize(
    "model_name,overrides",
    [
        ("baseline_classifier", ["model.net.pretrained=false"]),
        ("sngp_classifier", ["model.net.pretrained=false"]),
        ("sngp_specreg_classifier", ["model.net.pretrained=false"]),
        ("deep_ensemble_classifier", ["model.net.base_model_spec.pretrained=false"]),
    ],
)
def test_hparams_are_json_serializable(model_name, overrides):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", overrides=["data=acevedo", f"model={model_name}", *overrides])
        model = hydra.utils.instantiate(cfg.model)

    hparams = dict(model.hparams)
    json.dumps(hparams)  # raises TypeError if anything non-primitive slipped in

    assert "net" not in hparams
    assert "optimizer" not in hparams
    assert "scheduler" not in hparams
    assert "net_spec" in hparams
    assert isinstance(hparams["net_spec"], dict)
    assert "name" in hparams["net_spec"]
