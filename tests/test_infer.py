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
            "infer": {"model": {"batch_size_override": None}},
            "paths": {"root_dir": "/tmp/fake-root"},
        }
    )

    datamodule = infer._instantiate_datamodule(cfg)

    assert datamodule.artifact_csv_path == "/tmp/fake-root/data/artifact/artifacts.csv"
    assert datamodule.artifact_config_path == "/tmp/fake-root/configs/artifact/balanced.yaml"
