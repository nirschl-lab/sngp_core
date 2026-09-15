"""test_configs.py in tests."""

import functools
import glob
import inspect
import logging
import logging.config
import os

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict, read_write

_DATA_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "data")


def _dataset_config_names() -> list[str]:
    """Names of the per-dataset `configs/data/*.yaml` configs.

    Excludes the artifact-eval config: it has a different schema, and it interpolates
    `${paths.artifact_bank_dir}`, which resolves `HISTO_ARTIFACTS_BANK` -- an asset bank
    outside the repo that is not present in a bare checkout or in CI.
    """
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


class TestJobLoggingConfig:
    """Regression guard: `task_name` now contains "/" (e.g.
    "train/baseline_classifier_acevedo"), and `configs/hydra/default.yaml`'s
    `job_logging.handlers.file.filename` doesn't create intermediate directories, so a
    slash there raises FileNotFoundError during job-logging setup -- before the task
    function ever runs, and before `hydra.utils.instantiate` touches anything, so
    TestTrainConfig/TestEvalConfig's per-group instantiation checks don't exercise
    this path at all. Calls Hydra's own `configure_log` (the exact function
    `run_job()` calls) directly against a real, resolved job_logging config."""

    @pytest.mark.parametrize(
        "config_name,overrides",
        [
            ("train.yaml", ["data=acevedo", "model=deep_ensemble_classifier"]),
            ("eval.yaml", ["data=acevedo", "model=baseline_classifier", "ckpt_path=."]),
        ],
    )
    def test_job_logging_configures_without_error(self, config_name, overrides, tmp_path):
        with hydra.initialize(version_base="1.3", config_path="../configs"):
            cfg = hydra.compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        with open_dict(cfg):
            cfg.paths.log_dir = str(tmp_path)

        # Mirrors hydra.core.utils.run_job's own sequence: resolve hydra.run.dir into
        # hydra.runtime.output_dir, register the config, then create the directory --
        # all *before* configure_log runs, exactly as it happens in a real job.
        output_dir = str(OmegaConf.select(cfg, "hydra.run.dir"))
        with read_write(cfg.hydra.runtime):
            with open_dict(cfg.hydra.runtime):
                cfg.hydra.runtime.output_dir = os.path.abspath(output_dir)
        HydraConfig.instance().set_config(cfg)
        os.makedirs(output_dir, exist_ok=True)

        root_logger = logging.getLogger()
        saved_handlers = list(root_logger.handlers)
        saved_level = root_logger.level
        try:
            configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
        finally:
            for handler in root_logger.handlers:
                if handler not in saved_handlers:
                    handler.close()
            root_logger.handlers = saved_handlers
            root_logger.setLevel(saved_level)

        assert os.path.isfile(os.path.join(output_dir, "run.log")), (
            f"expected a flat run.log directly under {output_dir}"
        )


SELECTION_METRIC = "val/nll_cal"  # configs/callbacks/default.yaml; see docs/HPO_GUIDE.md

_EXPERIMENT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment")
_MODEL_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "model")


@functools.lru_cache(maxsize=None)
def _compose_experiment(experiment: str) -> DictConfig:
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        return hydra.compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])


@functools.lru_cache(maxsize=None)
def _compose_bare_model(model: str) -> DictConfig:
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        return hydra.compose(config_name="train.yaml", overrides=["data=acevedo", f"model={model}"])


def _experiment_config_names() -> list[str]:
    paths = glob.glob(os.path.join(_EXPERIMENT_CONFIG_DIR, "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


class TestExperimentProtocolConsistency:
    """Fair cross-family comparison (docs/HPO_GUIDE.md): for one dataset every family
    trains under the same loss, the same fixed family constants, and the same
    checkpoint / early-stopping selection metric, and Deep Ensemble inherits its
    partner's optimizer. Replaces the old class_freq-equality guard, which assumed a
    non-null class_freq (the protocol is now plain CE everywhere)."""

    _EXPERIMENTS_BY_DATASET = {
        "tang": ["baseline_tang", "sngp_tang", "deep_ensemble_tang"],
        "kather2018": ["baseline_kather2018", "sngp_kather2018", "deep_ensemble_kather2018"],
        "wong": ["baseline_wong", "sngp_wong", "deep_ensemble_wong"],
        "acevedo": ["baseline_acevedo", "sngp_acevedo", "deep_ensemble_acevedo"],
        "wong_ucdavis": [
            "baseline_wong_ucdavis",
            "sngp_wong_ucdavis",
            "deep_ensemble_baseline_wong_ucdavis",
            "deep_ensemble_sngp_wong_ucdavis",
        ],
        "wong_upitt": ["baseline_wong_upitt", "sngp_wong_upitt"],
        "wong_utsouthwestern": ["baseline_wong_utsouthwestern", "sngp_wong_utsouthwestern"],
    }
    _ALL_EXPERIMENTS = sorted(e for group in _EXPERIMENTS_BY_DATASET.values() for e in group)

    # (deep-ensemble experiment, the single-model experiment whose optimizer it must track)
    _DE_INHERITS = [
        ("deep_ensemble_tang", "baseline_tang"),
        ("deep_ensemble_kather2018", "baseline_kather2018"),
        ("deep_ensemble_wong", "baseline_wong"),
        ("deep_ensemble_acevedo", "baseline_acevedo"),
        ("deep_ensemble_baseline_wong_ucdavis", "baseline_wong_ucdavis"),
        ("deep_ensemble_sngp_wong_ucdavis", "sngp_wong_ucdavis"),
    ]

    _LOSS_KEYS = ("class_freq", "class_weights", "cb_beta", "focal_gamma", "label_smoothing")

    # Fixed to Liu et al. (2022) Table 9 in configs/model/sngp_classifier.yaml; never
    # overridden per experiment. `spectral_norm_bound` is swept, but its *default* is
    # protocol too, so an experiment may not silently pin a different one.
    _SNGP_FIXED_KEYS = (
        "rff_dim",
        "length_scale",
        "ridge_penalty",
        "cov_momentum",
        "mean_field",
        "mean_field_factor",
        "normalize_input",
        "likelihood",
        "output_bias",
        "random_feature_type",
        "n_power_iterations_sn",
        "spectral_norm_bound",
    )

    @classmethod
    def _loss_signature(cls, model_cfg: DictConfig) -> tuple:
        def norm(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple, ListConfig)):
                return tuple(float(x) for x in value)
            return float(value) if isinstance(value, (int, float)) else value

        return tuple(norm(model_cfg.get(key)) for key in cls._LOSS_KEYS)

    def test_every_experiment_config_is_registered(self):
        """A new configs/experiment/*.yaml must be added to _EXPERIMENTS_BY_DATASET (and,
        if it is a deep ensemble, to _DE_INHERITS) to be covered by these guards."""
        assert set(_experiment_config_names()) == set(self._ALL_EXPERIMENTS)
        assert {de for de, _ in self._DE_INHERITS} == {e for e in self._ALL_EXPERIMENTS if e.startswith("deep_ensemble_")}

    @pytest.mark.parametrize("dataset", sorted(_EXPERIMENTS_BY_DATASET))
    def test_loss_config_identical_across_model_families(self, dataset: str):
        signatures = {e: self._loss_signature(_compose_experiment(e).model) for e in self._EXPERIMENTS_BY_DATASET[dataset]}
        assert len(set(signatures.values())) == 1, (
            f"loss config {self._LOSS_KEYS} differs across families for {dataset!r}: {signatures}"
        )

    @pytest.mark.parametrize("experiment", _ALL_EXPERIMENTS)
    def test_loss_is_plain_cross_entropy(self, experiment: str):
        """Protocol: plain CE everywhere. `class_weights` is the documented fallback;
        using it is a deliberate protocol change, so update this test alongside."""
        model_cfg = _compose_experiment(experiment).model
        assert model_cfg.get("class_freq") is None, f"{experiment}: class_freq must be null (not ${{data.class_freq}})"
        assert model_cfg.get("class_weights") is None, f"{experiment}: class_weights must be null"

    @pytest.mark.parametrize("experiment", [e for e in _ALL_EXPERIMENTS if e.startswith("sngp_")])
    def test_sngp_experiment_does_not_override_fixed_constants(self, experiment: str):
        cfg = _compose_experiment(experiment)
        reference = _compose_bare_model("sngp_classifier").model.net
        for key in self._SNGP_FIXED_KEYS:
            assert cfg.model.net.get(key) == reference.get(key), (
                f"{experiment}: model.net.{key}={cfg.model.net.get(key)!r} overrides the fixed constant {reference.get(key)!r}"
            )

    def test_sngp_ctor_defaults_match_model_config(self):
        """deep_ensemble_sngp_* members are built by build_net() from SNGPClassifier's ctor
        defaults, not from configs/model/sngp_classifier.yaml -- so the two must agree.
        `spectral_norm_bound` is the one deliberate exception (ctor default None keeps
        pre-bound checkpoints reproducible); see the next test for how it is covered."""
        from src.models.sngp.sngp_classifier import SNGPClassifier

        params = inspect.signature(SNGPClassifier.__init__).parameters
        net = OmegaConf.load(os.path.join(_MODEL_CONFIG_DIR, "sngp_classifier.yaml")).net
        for key in self._SNGP_FIXED_KEYS:
            if key == "spectral_norm_bound":
                assert params[key].default is None
                continue
            assert params[key].default == net[key], (
                f"SNGPClassifier.__init__ default {key}={params[key].default!r} != configs/model/sngp_classifier.yaml {net[key]!r}"
            )

    def test_deep_ensemble_sngp_members_get_the_protocol_bound(self):
        cfg = _compose_experiment("deep_ensemble_sngp_wong_ucdavis")
        expected = _compose_bare_model("sngp_classifier").model.net.spectral_norm_bound
        assert cfg.model.net.base_model_spec.spectral_norm_bound == expected

    @pytest.mark.parametrize("experiment", _ALL_EXPERIMENTS)
    def test_checkpoint_and_early_stopping_track_selection_metric(self, experiment: str):
        callbacks = _compose_experiment(experiment).callbacks
        for name in ("model_checkpoint", "early_stopping"):
            assert callbacks[name].monitor == SELECTION_METRIC, f"{experiment}: callbacks.{name}.monitor"
            assert callbacks[name].mode == "min", f"{experiment}: callbacks.{name}.mode"

    @pytest.mark.parametrize("de_experiment,base_experiment", _DE_INHERITS)
    def test_deep_ensemble_inherits_partner_optimizer(self, de_experiment: str, base_experiment: str):
        de, base = _compose_experiment(de_experiment), _compose_experiment(base_experiment)
        assert float(de.model.optimizer.lr) == float(base.model.optimizer.lr)
        assert float(de.model.optimizer.weight_decay) == float(base.model.optimizer.weight_decay)
        assert de.data.datamodule.batch_size == base.data.datamodule.batch_size

    @pytest.mark.parametrize("experiment", [e for e in _ALL_EXPERIMENTS if e.startswith("deep_ensemble_")])
    def test_deep_ensemble_scheduler_restart_matches_epochs_per_member(self, experiment: str):
        cfg = _compose_experiment(experiment)
        assert cfg.model.scheduler.T_0 == cfg.trainer.max_epochs // cfg.model.num_estimators


class TestInferDefaultCfgKeysMatchYaml:
    """`src/inference/infer.py`'s `DEFAULT_INFER_{RUNTIME,METRICS,SAVE}_CFG` dicts are a
    no-Hydra fallback so the module works without config composition -- but that means
    each one duplicates a `configs/infer/<section>/default.yaml`, and nothing enforces
    they stay in sync. Assert matching *keys*, not values: `runtime.batch_size_override`
    already disagrees (1024 in yaml, None in the code default) pre-existing this test --
    see docs/KNOWN_ISSUES.md -- so a value-equality check would fail today for reasons
    unrelated to what this guards against (a new flag added to one side and not the
    other, e.g. `save_run_json`)."""

    @pytest.mark.parametrize(
        "cfg_attr,yaml_path,section",
        [
            ("DEFAULT_INFER_RUNTIME_CFG", "infer/runtime/default.yaml", "runtime"),
            ("DEFAULT_INFER_METRICS_CFG", "infer/metrics/default.yaml", "metrics"),
            ("DEFAULT_INFER_SAVE_CFG", "infer/save/default.yaml", "save"),
        ],
    )
    def test_keys_match(self, cfg_attr: str, yaml_path: str, section: str):
        from src.inference import infer

        code_default = getattr(infer, cfg_attr)
        yaml_default = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "configs", yaml_path))

        assert set(code_default) == set(yaml_default["infer"][section]), (
            f"{cfg_attr} and configs/{yaml_path} have diverged -- a save/runtime/metrics "
            "flag was added to one but not the other."
        )
