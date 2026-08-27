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
    ckpt_path = (
        "/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/best.ckpt"
    )
    run_name = infer.derive_default_run_name(ckpt_path, net_name="baseline_classifier", data_name="acevedo")
    assert run_name == "baseline_classifier_acevedo/2026-08-20_14-03-11/acevedo"


def test_derive_default_run_name_fallback_shape():
    run_name = infer.derive_default_run_name(
        "/tmp/wandb-artifacts/model-v3/model.ckpt", net_name="deep_ensemble", data_name="tang"
    )
    assert re.fullmatch(rf"deep_ensemble/{infer._RUN_ID_PATTERN.pattern}/tang", run_name)


def test_derive_default_run_name_tags_mc_dropout():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/best.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path, net_name="sngp_classifier", data_name="acevedo", use_mc_dropout=True
    )
    assert run_name == "mc_sngp_classifier_acevedo/2026-08-20_14-03-11/acevedo"


def test_extract_netname_dataset_parses_runs_segment():
    ckpt_path = (
        "/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_acevedo/"
        "runs/2026-08-25_14-31-36/checkpoints/best.ckpt"
    )
    assert infer._extract_netname_dataset(ckpt_path, fallback="model") == "baseline_classifier_acevedo"


def test_extract_netname_dataset_parses_ensemble_members_segment():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "ensemble_members/2026-08-20_14-03-11/checkpoints/ensemble.ckpt"
    )
    assert infer._extract_netname_dataset(ckpt_path, fallback="model") == "sngp_classifier_acevedo"


def test_extract_netname_dataset_falls_back_when_no_convention_segment():
    netname_dataset = infer._extract_netname_dataset("/tmp/wandb-artifacts/model-v3/model.ckpt", fallback="model")
    assert netname_dataset == "model"


def test_derive_default_run_name_deep_ensemble_assembled_path():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "ensemble_members/2026-08-20_14-03-11/checkpoints/ensemble.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path, net_name="deep_ensemble", data_name="acevedo", is_deep_ensemble=True
    )
    assert run_name == "deep_ensemble_sngp_classifier_acevedo/2026-08-20_14-03-11/acevedo"


def test_derive_default_run_name_deep_ensemble_sequential_path_no_doubled_prefix():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/deep_ensemble_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/best.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path, net_name="deep_ensemble", data_name="acevedo", is_deep_ensemble=True
    )
    assert run_name == "deep_ensemble_classifier_acevedo/2026-08-20_14-03-11/acevedo"
    assert "deep_ensemble_deep_ensemble" not in run_name


def test_derive_default_run_name_deep_ensemble_precedes_mc_dropout():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/baseline_classifier_tang/"
        "ensemble_members/2026-08-20_14-03-11/checkpoints/ensemble.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path,
        net_name="deep_ensemble",
        data_name="tang",
        use_mc_dropout=True,
        is_deep_ensemble=True,
    )
    assert run_name == "deep_ensemble_baseline_classifier_tang/2026-08-20_14-03-11/tang"
    assert "mc_" not in run_name
