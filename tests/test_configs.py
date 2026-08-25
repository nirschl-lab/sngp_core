"""test_configs.py in tests."""

import glob
import os

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

_DATA_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "data")


def _dataset_config_names() -> list[str]:
    """Names of the per-dataset `configs/data/*.yaml` configs (excludes the
    artifact-eval config, which has a different schema)."""
    paths = glob.glob(os.path.join(_DATA_CONFIG_DIR, "*.yaml"))
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return sorted(n for n in names if not n.startswith("artifact"))


class TestTrainConfig:
    """Grouped tests for the training configuration available via the `cfg_train` fixture.

    Each method checks only one top-level section (data/model/trainer) and raises an
    informative AssertionError when instantiation fails.
    """

    @pytest.fixture(autouse=True)
    def _setup_cfg(self, cfg_train: DictConfig):
        # sanity checks
        assert cfg_train is not None, "cfg_train fixture returned None"
        assert isinstance(cfg_train, DictConfig), "cfg_train must be an omegaconf.DictConfig"
        assert getattr(cfg_train, "data", None) is not None, "cfg_train is missing 'data' section"
        assert getattr(cfg_train, "model", None) is not None, "cfg_train is missing 'model' section"
        assert getattr(cfg_train, "trainer", None) is not None, "cfg_train is missing 'trainer' section"

        HydraConfig().set_config(cfg_train)
        self.cfg_train = cfg_train

    def _instantiate(self, node, node_name: str):
        try:
            hydra.utils.instantiate(node)
        except Exception as exc:
            # Raise an AssertionError so pytest reports it as a test failure with a clear message
            raise AssertionError(
                f"Failed to instantiate '{node_name}' from cfg_train: {exc}"
            ) from exc

    def test_data_instantiation(self):
        """Instantiate data."""
        self._instantiate(self.cfg_train.data, "data")

    def test_model_instantiation(self):
        """Instantiate model only."""
        self._instantiate(self.cfg_train.model, "model")

    def test_trainer_instantiation(self):
        """Instantiate trainer."""
        self._instantiate(self.cfg_train.trainer, "trainer")


class TestEvalConfig:
    """Grouped tests for the eval config."""

    @pytest.fixture(autouse=True)
    def _setup_cfg(self, cfg_eval: DictConfig):
        # sanity check
        assert cfg_eval is not None, "cfg_eval fixture returned None"
        assert isinstance(cfg_eval, DictConfig), "cfg_eval must be an omegaconf.DictConfig"
        assert getattr(cfg_eval, "data", None) is not None, "cfg_eval is missing 'data' section"
        assert getattr(cfg_eval, "model", None) is not None, "cfg_eval is missing 'model' section"
        assert getattr(cfg_eval, "trainer", None) is not None, "cfg_eval is missing 'trainer' section"

        HydraConfig().set_config(cfg_eval)
        self.cfg_eval = cfg_eval

    def _instantiate(self, node, node_name: str):
        try:
            hydra.utils.instantiate(node)
        except Exception as exc:
            # Raise an AssertionError so pytest reports it as a test failure with a clear message
            raise AssertionError(
                f"Failed to instantiate '{node_name}' from cfg_eval: {exc}"
            ) from exc

    def test_data_instantiation(self):
        """Instantiate data."""
        self._instantiate(self.cfg_eval.data, "data")

    def test_model_instantiation(self):
        """Instantiate model."""
        self._instantiate(self.cfg_eval.model, "model")

    def test_trainer_instantiation(self):
        """Instantiate trainer."""
        self._instantiate(self.cfg_eval.trainer, "trainer")


class TestModelConfigs:
    """Regression guard: smoke-instantiate every `configs/model/*.yaml` individually,
    not just whichever one happens to be the Hydra default -- mirrors
    `TestDatasetConfigDrift`'s per-file approach so a broken config is attributable to
    one file."""

    @pytest.mark.parametrize(
        "model_config_name",
        ["baseline_classifier", "sngp_classifier", "deep_ensemble_classifier"],
    )
    def test_each_model_config_instantiates(self, model_config_name: str):
        with hydra.initialize(version_base="1.3", config_path="../configs"):
            cfg = hydra.compose(
                config_name="train.yaml", overrides=["data=acevedo", f"model={model_config_name}"]
            )
        hydra.utils.instantiate(cfg.model)


class TestDatasetConfigDrift:
    """Regression guard for `configs/data/*.yaml`: catches a config that was
    copy-pasted from another dataset without every field being updated to match."""

    @pytest.mark.parametrize("data_config_name", _dataset_config_names())
    def test_num_classes_matches_class_to_idx(self, data_config_name: str):
        with hydra.initialize(version_base="1.3", config_path="../configs"):
            cfg = hydra.compose(config_name="train.yaml", overrides=[f"data={data_config_name}"])
        datamodule_cfg = cfg.data.datamodule
        assert datamodule_cfg.num_classes == len(datamodule_cfg.class_to_idx), (
            f"configs/data/{data_config_name}.yaml: num_classes="
            f"{datamodule_cfg.num_classes} but class_to_idx has "
            f"{len(datamodule_cfg.class_to_idx)} entries"
        )

    def test_dataset_names_are_distinct(self):
        dataset_names = {}
        for data_config_name in _dataset_config_names():
            with hydra.initialize(version_base="1.3", config_path="../configs"):
                cfg = hydra.compose(config_name="train.yaml", overrides=[f"data={data_config_name}"])
            dataset_names[data_config_name] = cfg.data.datamodule.dataset_name
        assert len(set(dataset_names.values())) == len(dataset_names), (
            f"duplicate dataset_name across configs/data/*.yaml: {dataset_names}"
        )

    def test_data_short_names_are_distinct(self):
        """`data.name` drives task_name/output-directory naming (see
        docs/OUTPUT_LAYOUT.md) -- a duplicate would make two datasets share one
        output tree."""
        short_names = {}
        for data_config_name in _dataset_config_names():
            with hydra.initialize(version_base="1.3", config_path="../configs"):
                cfg = hydra.compose(config_name="train.yaml", overrides=[f"data={data_config_name}"])
            short_names[data_config_name] = cfg.data.name
        assert len(set(short_names.values())) == len(short_names), (
            f"duplicate data.name across configs/data/*.yaml: {short_names}"
        )


class TestExperimentClassFreqConsistency:
    """Regression guard for fair cross-model-family comparison: every model family's
    experiment config for the same dataset must resolve `model.class_freq` to the
    identical, data-derived value -- imbalance handling can't differ by model family."""

    _EXPERIMENTS_BY_DATASET = {
        "tang": ["baseline_tang", "sngp_tang"],
        "kather2018": ["baseline_kather2018", "sngp_kather2018"],
        "wong": ["baseline_wong", "sngp_wong", "deep_ensemble_wong"],
        "acevedo": ["baseline_acevedo", "sngp_acevedo", "deep_ensemble_acevedo"],
    }

    @pytest.mark.parametrize("dataset", sorted(_EXPERIMENTS_BY_DATASET))
    def test_class_freq_identical_across_model_families(self, dataset: str):
        experiments = self._EXPERIMENTS_BY_DATASET[dataset]
        resolved = {}
        for experiment in experiments:
            with hydra.initialize(version_base="1.3", config_path="../configs"):
                cfg = hydra.compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
            resolved[experiment] = list(cfg.model.class_freq)

        values = list(resolved.values())
        assert all(v == values[0] for v in values), (
            f"model.class_freq differs across experiment configs for dataset {dataset!r}: {resolved}"
        )
