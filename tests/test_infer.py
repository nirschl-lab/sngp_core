import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.inference import infer


class DummyDataModule:
    def __init__(self, artifact_csv_path: str, artifact_config_path: str, **kwargs) -> None:
        self.artifact_csv_path = artifact_csv_path
        self.artifact_config_path = artifact_config_path


def test_infer_datamodule_resolves_paths_root_dir(monkeypatch):
    monkeypatch.setattr(infer.hydra.utils, "instantiate", lambda cfg, **kwargs: DummyDataModule(**cfg))

    cfg = OmegaConf.create(
        {
            "data": {
                "datamodule": {
                    "_target_": "tests.test_infer.DummyDataModule",
                    "artifact_csv_path": "${paths.root_dir}/data/artifact/artifacts.csv",
                    "artifact_config_path": "${paths.root_dir}/configs/artifact/balanced.yaml",
                }
            },
            "infer": {"runtime": {"batch_size_override": None}},
            "paths": {"root_dir": "/tmp/fake-root"},
        }
    )

    datamodule = infer._instantiate_datamodule(cfg)

    assert datamodule.artifact_csv_path == "/tmp/fake-root/data/artifact/artifacts.csv"
    assert datamodule.artifact_config_path == "/tmp/fake-root/configs/artifact/balanced.yaml"


def test_extract_ckpt_run_id_finds_the_project_timestamp():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/last.ckpt"
    )
    assert infer._extract_ckpt_run_id(ckpt_path) == "2026-08-20_14-03-11"


def test_extract_ckpt_run_id_falls_back_when_no_timestamp_present():
    run_id = infer._extract_ckpt_run_id("/some/renamed/checkpoint.ckpt")
    assert infer._RUN_ID_PATTERN.fullmatch(run_id)


def test_derive_default_run_name_uses_ckpt_run_id_and_dataset():
    ckpt_path = ".../runs/2026-08-20_14-03-11/checkpoints/last.ckpt"
    run_name = infer.derive_default_run_name(ckpt_path, net_name="sngp_classifier", data_name="acevedo")
    assert run_name == "sngp_classifier/2026-08-20_14-03-11/acevedo"


def test_derive_default_run_name_fallback_shape():
    run_name = infer.derive_default_run_name("/no/timestamp/here.ckpt", net_name="deep_ensemble", data_name="tang")
    assert re.fullmatch(rf"deep_ensemble/{infer._RUN_ID_PATTERN.pattern}/tang", run_name)
