"""test_sweeps.py in tests.

Two kinds of tests: the slow, subprocess-driven Hydra multirun / experiment smoke tests
(unchanged from the template), and fast drift guards for the W&B sweep definitions in
configs/hparams_search/wandb/*.yaml against the Hydra run presets they drive.
"""
from pathlib import Path

import hydra
import pytest
import yaml
from hydra.types import RunMode

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]

_WANDB_SWEEP_DIR = Path(__file__).resolve().parent.parent / "configs" / "hparams_search" / "wandb"


def _wandb_sweep_families() -> list[str]:
    return sorted(p.stem for p in _WANDB_SWEEP_DIR.glob("*.yaml"))


def _load_sweep(family: str) -> dict:
    # Plain YAML on purpose: `${env}` / `${program}` / `${args_no_hyphens}` are W&B command
    # macros that OmegaConf would try to resolve as interpolations.
    with open(_WANDB_SWEEP_DIR / f"{family}.yaml") as fh:
        return yaml.safe_load(fh)


def _compose_preset(family: str):
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        return hydra.compose(
            config_name="train.yaml",
            overrides=[f"experiment={family}_acevedo", f"hparams_search={family}"],
            return_hydra_config=True,
        )


def _select(cfg, dotted: str):
    node = cfg
    for part in dotted.split("."):
        assert part in node, f"{dotted!r} does not resolve in the composed config (missing {part!r})"
        node = node[part]
    return node


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """Test default hydra sweep.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "experiment=baseline_tang",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
    """Test default hydra sweep with ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "experiment=baseline_tang",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hparams_search_preset_single_run(tmp_path: Path) -> None:
    """A W&B sweep trial is an ordinary single run with `hparams_search=<family>` composed
    in. The preset must compose without any sweeper and the objective must be logged and
    retrievable at the end. Not `fast_dev_run`: that skips `on_validation_epoch_end`, so
    the objective would never be logged and `get_metric_value` would raise.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "hparams_search=baseline",
        "experiment=baseline_tang",
        "hydra.run.dir=" + str(tmp_path),
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=2",
        "+trainer.limit_val_batches=3",
    ] + overrides
    run_sh_command(command)


class TestWandbSweepConfigDrift:
    """The search space (configs/hparams_search/wandb/<family>.yaml, native W&B) and the
    run preset (configs/hparams_search/<family>.yaml, Hydra) are two files that must agree
    -- this is the guard that keeps them in sync."""

    @pytest.mark.parametrize("family", _wandb_sweep_families())
    def test_every_swept_parameter_resolves_in_the_composed_config(self, family: str):
        sweep, cfg = _load_sweep(family), _compose_preset(family)
        assert sweep["parameters"], f"{family}: empty search space"
        for key in sweep["parameters"]:
            _select(cfg, key)

    @pytest.mark.parametrize("family", _wandb_sweep_families())
    def test_objective_matches_preset_and_is_a_running_best_validation_metric(self, family: str):
        sweep, cfg = _load_sweep(family), _compose_preset(family)
        metric = sweep["metric"]
        assert metric["name"] == cfg.optimized_metric, f"{family}: metric.name != optimized_metric"
        assert metric["goal"] == "minimize"
        assert metric["name"].endswith("_best"), "W&B reads the run summary (= last logged value); only a running best makes last == best"
        assert not metric["name"].startswith("test/")

    @pytest.mark.parametrize("family", _wandb_sweep_families())
    def test_monitors_track_the_epoch_level_objective(self, family: str):
        sweep, cfg = _load_sweep(family), _compose_preset(family)
        epoch_metric = sweep["metric"]["name"][: -len("_best")]
        for name in ("early_stopping", "model_checkpoint"):
            assert cfg.callbacks[name].monitor == epoch_metric, f"{family}: callbacks.{name}.monitor"
            assert cfg.callbacks[name].mode == "min", f"{family}: callbacks.{name}.mode"

    @pytest.mark.parametrize("family", _wandb_sweep_families())
    def test_trials_land_under_sweeps_and_write_no_checkpoints(self, family: str):
        cfg = _compose_preset(family)
        # Guards the defaults-list ORDER in configs/train.yaml: `- hydra: default` must come
        # before `- hparams_search`, or configs/hydra/default.yaml's runs/<timestamp> template
        # wins (last-in-defaults-list) and every trial pollutes the real-training tree.
        assert "/sweeps/" in cfg.hydra.run.dir, f"{family}: trial dir is not under sweeps/ -- {cfg.hydra.run.dir}"
        # sweeps/<sweep id>/<timestamp>_<run id>: the per-sweep parent keeps a pilot, a re-sweep
        # and a different backbone in separate folders. Resolves to "local/<ts>_local" here,
        # since hydra.compose sees none of the env vars `wandb agent` exports.
        tail = cfg.hydra.run.dir.split("/sweeps/", 1)[1]
        assert tail.count("/") == 1, f"{family}: expected sweeps/<sweep id>/<trial>, got {tail!r}"
        # A trial's weights are never loaded: the protocol retrains the top-3 from scratch at
        # the experiment's full max_epochs (docs/HPO_GUIDE.md), so writing best+last per trial
        # is 278 MB x run_cap of pure waste.
        assert cfg.callbacks.model_checkpoint.save_top_k == 0, f"{family}: sweep trials must not save checkpoints"
        assert cfg.callbacks.model_checkpoint.save_last is False, f"{family}: sweep trials must not save last.ckpt"

    @pytest.mark.parametrize("family", _wandb_sweep_families())
    def test_preset_is_a_plain_single_run(self, family: str):
        cfg = _compose_preset(family)
        assert "sweep_fail_safe" not in cfg, "a fail-safe floor would rank as the best trial under a minimized objective"
        assert cfg.hydra.mode != RunMode.MULTIRUN
        assert cfg.logger.wandb.name is None, "W&B names sweep trials; a fixed name would be shared by every trial"
        assert cfg.logger.wandb.log_model is False

    @pytest.mark.parametrize("family", _wandb_sweep_families())
    def test_command_hands_hydra_the_preset_and_the_overrides(self, family: str):
        sweep = _load_sweep(family)
        command = list(sweep["command"])
        assert sweep["program"] == "src/train.py"
        assert "${program}" in command
        assert f"hparams_search={family}" in command
        assert command[-1] == "${args_no_hyphens}", "overrides must be the last element so sweep.py can insert experiment= before them"
        assert not any(c.startswith("experiment=") for c in command), "experiment is injected per dataset by scripts/hpo/sweep.py"

    @pytest.mark.parametrize("family", _wandb_sweep_families())
    def test_hyperband_cannot_kill_before_lightning_min_epochs(self, family: str):
        sweep, cfg = _load_sweep(family), _compose_preset(family)
        if "early_terminate" not in sweep:
            pytest.skip("no early_terminate block")
        assert sweep["early_terminate"]["type"] == "hyperband"
        assert sweep["early_terminate"]["min_iter"] >= cfg.trainer.min_epochs


class TestSweepLauncher:
    """scripts/hpo/sweep.py joins the W&B search space and the Hydra preset for one
    dataset; these are the pure parts (no W&B API, no sbatch)."""

    def test_build_sweep_config_injects_experiment_and_overrides(self):
        from scripts.hpo.sweep import build_sweep_config

        cfg = build_sweep_config("sngp", "acevedo", extra_overrides=["trainer.max_epochs=2"])
        assert cfg["name"] == "acevedo_sngp_resnet18_hpo"
        assert cfg["command"][-3:] == ["experiment=sngp_acevedo", "trainer.max_epochs=2", "${args_no_hyphens}"]
        assert "hparams_search=sngp" in cfg["command"]
        assert cfg["run_cap"] == 48
        assert cfg["metric"] == {"name": "val/nll_cal_best", "goal": "minimize"}
        assert "sngp x acevedo" in cfg["description"]

    def test_build_sweep_config_trials_override_run_cap(self):
        from scripts.hpo.sweep import build_sweep_config

        assert build_sweep_config("baseline", "tang", trials=2)["run_cap"] == 2

    def test_build_sweep_config_rejects_objective_mismatch(self, monkeypatch):
        import scripts.hpo.sweep as sweep_module

        real = sweep_module.load_sweep_definition

        def mismatched(family):
            cfg = real(family)
            cfg["metric"]["name"] = "val/auprc_best"
            return cfg

        monkeypatch.setattr(sweep_module, "load_sweep_definition", mismatched)
        with pytest.raises(ValueError, match="optimized_metric"):
            sweep_module.build_sweep_config("baseline", "tang")

    def test_register_appends_one_bullet(self, tmp_path):
        from scripts.hpo.sweep import register

        registry = tmp_path / "MASTER_SWEEPS.md"
        cfg = {"name": "tang_baseline_resnet18_hpo", "metric": {"name": "val/nll_cal_best", "goal": "minimize"}, "run_cap": 48}
        register("me/proj/abc123", cfg, registry=registry)
        register("me/proj/def456", cfg, registry=registry)
        text = registry.read_text()
        assert text.startswith("# Master list of W&B sweeps")
        assert text.count("\n- `tang_baseline_resnet18_hpo`") == 2
        assert "`me/proj/abc123`" in text and "`me/proj/def456`" in text


class TestSummarizeSweep:
    class _Run:
        def __init__(self, name, state, summary, config):
            self.name, self.id, self.state, self.summary, self.config = name, name, state, summary, config

    def _runs(self):
        return [
            self._Run("a", "finished", {"val/nll_cal_best": 0.30}, {"model.optimizer.lr": 1e-3}),
            self._Run("b", "finished", {"val/nll_cal_best": 0.20}, {"model": {"optimizer": {"lr": 2e-3}}}),
            self._Run("c", "crashed", {"val/nll_cal_best": 0.05}, {"model.optimizer.lr": 3e-3}),
            self._Run("d", "finished", {}, {"model.optimizer.lr": 4e-3}),
        ]

    def test_rank_runs_ascending_finished_with_metric_only(self):
        from scripts.hpo.summarize_sweep import rank_runs

        ranked = rank_runs(self._runs(), "val/nll_cal_best", "minimize")
        assert [run.name for _, run in ranked] == ["b", "a"]
        assert [value for value, _ in ranked] == [0.20, 0.30]

    def test_rank_runs_can_include_other_states(self):
        from scripts.hpo.summarize_sweep import rank_runs

        ranked = rank_runs(self._runs(), "val/nll_cal_best", "minimize", states=("finished", "crashed"))
        assert [run.name for _, run in ranked] == ["c", "b", "a"]

    def test_lookup_dotted_and_nested(self):
        from scripts.hpo.summarize_sweep import format_overrides, lookup

        assert lookup({"model.optimizer.lr": 1e-3}, "model.optimizer.lr") == 1e-3
        assert lookup({"model": {"optimizer": {"lr": 2e-3}}}, "model.optimizer.lr") == 2e-3
        assert lookup({"model": {}}, "model.optimizer.lr") is None
        assert format_overrides({"model.optimizer.lr": 0.001, "data.datamodule.batch_size": 64},
                                ["model.optimizer.lr", "data.datamodule.batch_size"]) == \
            "data.datamodule.batch_size=64 model.optimizer.lr=0.001"
