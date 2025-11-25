"""test_configs.py in tests."""

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


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
