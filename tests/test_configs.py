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
        ["baseline_classifier", "sngp_classifier", "sngp_specreg_classifier", "deep_ensemble_classifier"],
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
        "acevedo": ["baseline_acevedo", "sngp_acevedo", "sngp_specreg_acevedo", "deep_ensemble_acevedo"],
        "wong_ucdavis": [
            "baseline_wong_ucdavis",
            "sngp_wong_ucdavis",
            "deep_ensemble_baseline_wong_ucdavis",
            "deep_ensemble_sngp_wong_ucdavis",
        ],
        "wong_upitt": ["baseline_wong_upitt", "sngp_wong_upitt"],
        "wong_utsouthwestern": ["baseline_wong_utsouthwestern", "sngp_wong_utsouthwestern"],
        # The CIFAR-100 / WRN-28-10 reproduction of the SNGP benchmark. Not part of the
        # biomedical protocol -- see docs/models/CIFAR100_BENCHMARK.md.
        "cifar100": [
            "baseline_cifar100",
            "sngp_cifar100",
            "sngp_specreg_cifar100",
            "sngp_specreg_cifar100_literal",
        ],
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

    # Deliberate, documented deviations from the protocol selection metric, per experiment.
    # `monitor`: what ModelCheckpoint (and EarlyStopping, if present) must track instead
    # of SELECTION_METRIC. `early_stopping`: None means the experiment must have NO early
    # stopping callback (fixed budget). Anything not listed here gets the protocol check.
    #
    # sngp_specreg_acevedo -- pilot of spectral *regularization* (rep-spectral) in place of
    # spectral normalization: selects on raw val/nll and trains a fixed 100-epoch budget with
    # a 50-epoch burn-in (the paper's recipe), so patience-based stopping is disabled. See
    # configs/experiment/sngp_specreg_acevedo.yaml for the reasoning behind each choice.
    #
    # The four *_cifar100* experiments reproduce the SNGP CIFAR benchmark, whose recipe
    # trains a fixed 250-epoch budget and reports on plain validation loss with no
    # calibration-aware selection. All four share the rule, so the SNGP-vs-spectral-reg
    # comparison does not turn on it. See docs/models/CIFAR100_BENCHMARK.md.
    _SELECTION_OVERRIDES = {
        "sngp_specreg_acevedo": {"monitor": "val/nll", "early_stopping": None},
        "baseline_cifar100": {"monitor": "val/loss", "early_stopping": None},
        "sngp_cifar100": {"monitor": "val/loss", "early_stopping": None},
        "sngp_specreg_cifar100": {"monitor": "val/loss", "early_stopping": None},
        "sngp_specreg_cifar100_literal": {"monitor": "val/loss", "early_stopping": None},
    }

    # sngp_specreg_* experiments normally must have weight_decay == 0, so the spectral
    # penalty is the only weight regularizer. `sngp_specreg_cifar100` is the deliberate
    # exception: it is a controlled comparison against `sngp_cifar100`, so it holds
    # everything but the regularizer fixed -- including that arm's weight decay. (Its
    # companion, sngp_specreg_cifar100_literal, keeps 0 and is not exempt.)
    _SPECREG_WD_EXEMPT = frozenset({"sngp_specreg_cifar100"})

    # The Liu et al. (2022) Table 9 constants, as pinned in configs/model/sngp_classifier.yaml.
    #
    # These used to be additionally guarded against per-experiment overrides. That guard is
    # gone: the CIFAR-100 reproduction has to set the reference's own CIFAR values for
    # `length_scale`, `ridge_penalty`, `normalize_input` and `mean_field_factor`, which are
    # different constants for a different benchmark, not protocol drift.
    #
    # What remains below is a narrower and separate guarantee -- that the ctor defaults and
    # the model config agree -- and it is still load-bearing: deep_ensemble_sngp_* members
    # are built by `build_net()` from `SNGPClassifier.__init__` defaults, NOT from the model
    # config, so drift between the two silently changes ensemble architecture.
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
        "scale_random_features",
        "feature_map",
        "n_power_iterations_sn",
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

    def test_sngp_ctor_defaults_match_model_config(self):
        """deep_ensemble_sngp_* members are built by build_net() from SNGPClassifier's ctor
        defaults, not from configs/model/sngp_classifier.yaml -- so the two must agree."""
        from src.models.sngp.sngp_classifier import SNGPClassifier

        params = inspect.signature(SNGPClassifier.__init__).parameters
        net = OmegaConf.load(os.path.join(_MODEL_CONFIG_DIR, "sngp_classifier.yaml")).net
        for key in self._SNGP_FIXED_KEYS:
            assert params[key].default == net[key], (
                f"SNGPClassifier.__init__ default {key}={params[key].default!r} != configs/model/sngp_classifier.yaml {net[key]!r}"
            )
        # `spectral_norm_bound` is the one key whose ctor default deliberately does NOT
        # match the config: `None` is stock hard normalization, so a checkpoint written
        # before the bound existed (its spec lacks the key) rebuilds bit-identically.
        assert params["spectral_norm_bound"].default is None
        # `use_spectral_norm` defaults on for the same back-compat reason: every spec written
        # before the key existed is a spectral-normed SNGP. Not in _SNGP_FIXED_KEYS because
        # the spectral-regularization variant (sngp_specreg_*) must switch it off.
        assert params["use_spectral_norm"].default is True
        assert net["use_spectral_norm"] is True

    def test_deep_ensemble_sngp_members_match_partner_bound(self):
        """An SNGP ensemble's members must be built with the same bound as the single SNGP
        model it ensembles -- `spectral_norm_bound` changes the trained function, so a
        mismatch would make the ensemble a different architecture than its partner. The
        reference is the partner *experiment* (which may pin its own swept winner), not
        the family default in configs/model/sngp_classifier.yaml."""
        for de_experiment, base_experiment in self._DE_INHERITS:
            cfg = _compose_experiment(de_experiment)
            spec = cfg.model.net.get("base_model_spec")
            if spec is None or spec.get("name") != "sngp_classifier":
                continue  # baseline-member ensemble; no bound involved
            expected = _compose_experiment(base_experiment).model.net.spectral_norm_bound
            assert spec.spectral_norm_bound == expected, (
                f"{de_experiment}: base_model_spec.spectral_norm_bound={spec.spectral_norm_bound!r} "
                f"!= {base_experiment}'s {expected!r}"
            )

    @pytest.mark.parametrize("experiment", _ALL_EXPERIMENTS)
    def test_checkpoint_and_early_stopping_track_selection_metric(self, experiment: str):
        callbacks = _compose_experiment(experiment).callbacks
        override = self._SELECTION_OVERRIDES.get(experiment, {})
        expected_monitor = override.get("monitor", SELECTION_METRIC)
        assert callbacks["model_checkpoint"].monitor == expected_monitor, f"{experiment}: callbacks.model_checkpoint.monitor"
        assert callbacks["model_checkpoint"].mode == "min", f"{experiment}: callbacks.model_checkpoint.mode"
        if "early_stopping" in override and override["early_stopping"] is None:
            assert callbacks.get("early_stopping") is None, f"{experiment}: early stopping must be disabled"
        else:
            assert callbacks["early_stopping"].monitor == expected_monitor, f"{experiment}: callbacks.early_stopping.monitor"
            assert callbacks["early_stopping"].mode == "min", f"{experiment}: callbacks.early_stopping.mode"

    @pytest.mark.parametrize("experiment", [e for e in _ALL_EXPERIMENTS if e.startswith("sngp_specreg_")])
    def test_specreg_experiment_is_internally_consistent(self, experiment: str):
        """Spectral-regularization runs: no spectral norm anywhere, no weight decay (the
        penalty is the only weight regularizer), and best.ckpt ranked only over
        regularized epochs -- the checkpoint's start_epoch must equal the burn-in."""
        cfg = _compose_experiment(experiment)
        assert cfg.model.net.use_spectral_norm is False
        assert cfg.model.net.spectral_norm_bound is None
        if experiment not in self._SPECREG_WD_EXEMPT:
            assert float(cfg.model.optimizer.weight_decay) == 0.0
        assert cfg.callbacks.model_checkpoint._target_.endswith("ModelCheckpointFromEpoch")
        assert cfg.callbacks.model_checkpoint.start_epoch == cfg.model.spec_reg_burnin_epochs
        assert 0 < cfg.model.spec_reg_burnin_epochs < cfg.trainer.max_epochs

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
