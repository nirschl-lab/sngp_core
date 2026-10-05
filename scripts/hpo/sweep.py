"""Create a W&B sweep for one model family x dataset and (optionally) submit SLURM agents.

The search space lives in configs/hparams_search/wandb/<family>.yaml (native W&B); the
run preset lives in configs/hparams_search/<family>.yaml (Hydra). This script joins the
two for one dataset: it loads the W&B YAML, inserts `experiment=<family>_<dataset>` (plus
any --override) into its `command` right before `${args_no_hyphens}`, names the sweep
after the experiment, registers it with W&B, appends it to docs/MASTER_SWEEPS.md, and
submits a SLURM array of `wandb agent`s (scripts/slurm/wandb_agent.sbatch).

Usage:
    uv run scripts/hpo/sweep.py sngp acevedo                       # create + register + submit 48 trials, 8 at a time
    uv run scripts/hpo/sweep.py baseline tang --trials 2 --parallel 2 \\
        --override trainer.max_epochs=2 --override +trainer.limit_train_batches=0.05 \\
        --no-register                                             # pilot
    uv run scripts/hpo/sweep.py sngp acevedo --dry-run             # print the final sweep YAML, no API call
    uv run scripts/hpo/sweep.py sngp acevedo --no-submit           # create only; run agents by hand:
        #   uv run wandb agent --count 1 <entity>/<project>/<sweep_id>
    uv run scripts/hpo/sweep.py sngp_probit wong --experiment sngp_bnsn_wong_sgd \\
        --name adrc_wong_sngp_bnsn --agents-per-gpu 4 --time 08:00:00  # one family, another experiment;
        # named sweep + W&B group; 4 concurrent trials share each GPU

Each trial is an ordinary single-run `src/train.py` invocation; the W&B controller owns
the loop and reads the objective (`val/nll_cal_best`) from the logged metrics. See
docs/HPO_GUIDE.md for the protocol.
"""
import argparse
import datetime as _dt
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import rootutils
import yaml

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)  # also loads .env

from hydra import compose, initialize  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

import src.utils  # noqa: E402,F401  (registers the tags_with / dataset_label resolvers)

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DEFINITION_DIR = REPO_ROOT / "configs" / "hparams_search" / "wandb"
SWEEP_REGISTRY = REPO_ROOT / "docs" / "MASTER_SWEEPS.md"
AGENT_SBATCH = REPO_ROOT / "scripts" / "slurm" / "wandb_agent.sbatch"
ARGS_MACRO = "${args_no_hyphens}"
# One agent's share of an array task -- must match the #SBATCH defaults in AGENT_SBATCH.
CPUS_PER_AGENT = 8
MEM_GB_PER_AGENT = 32


def compose_preset(family: str, experiment: str) -> DictConfig:
    """Hydra-compose the run preset for this family x experiment (for `name`/`optimized_metric`)."""
    with initialize(version_base="1.3", config_path="../../configs"):
        return compose(
            config_name="train.yaml",
            overrides=[f"experiment={experiment}", f"hparams_search={family}"],
        )


def load_sweep_definition(family: str) -> dict:
    """Plain YAML on purpose: `${env}` / `${program}` / `${args_no_hyphens}` are W&B
    command macros that OmegaConf would try to resolve as interpolations."""
    path = SWEEP_DEFINITION_DIR / f"{family}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No W&B sweep definition for family {family!r}: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # not a git checkout, git missing, ...
        return "unknown"


def build_sweep_config(
    family: str,
    dataset: str,
    extra_overrides: Sequence[str] = (),
    trials: Optional[int] = None,
    experiment: Optional[str] = None,
    name: Optional[str] = None,
) -> dict:
    """The final sweep dict handed to `wandb.sweep()`.

    `experiment` defaults to `<family>_<dataset>`; pass it when one family's search space
    drives several experiments (e.g. sngp_probit for sngp_wong_sgd and sngp_bnsn_wong_sgd).
    `name` replaces the default sweep name `<experiment name>_hpo` and is also handed to
    every trial as its W&B group, so a study's trials can be found by one name.
    """
    experiment = experiment or f"{family}_{dataset}"
    preset = compose_preset(family, experiment)
    sweep = load_sweep_definition(family)

    metric = sweep["metric"]["name"]
    if metric != preset.optimized_metric:
        raise ValueError(
            f"configs/hparams_search/wandb/{family}.yaml optimizes {metric!r} but the Hydra preset "
            f"configs/hparams_search/{family}.yaml sets optimized_metric={preset.optimized_metric!r}"
        )

    command = list(sweep["command"])
    if ARGS_MACRO not in command:
        raise ValueError(f"sweep command must contain {ARGS_MACRO!r}: {command}")
    insert_at = command.index(ARGS_MACRO)
    group_override = [f"logger.wandb.group={name}"] if name else []
    command[insert_at:insert_at] = [f"experiment={experiment}", *group_override, *extra_overrides]
    sweep["command"] = command

    sweep["name"] = name or f"{preset.name}_hpo"
    if trials is not None:
        sweep["run_cap"] = int(trials)
    sweep["description"] = (
        f"{family} x {dataset} | experiment {experiment} | objective {metric} ({sweep['metric']['goal']}) | "
        f"git {git_sha()} | {_dt.date.today().isoformat()}"
    )
    return sweep


def create_sweep(sweep_cfg: dict, project: str, entity: Optional[str] = None) -> str:
    """Register the sweep with W&B; returns the fully-qualified `entity/project/sweep_id`."""
    import wandb

    sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)
    entity = entity or wandb.Api().default_entity
    return f"{entity}/{project}/{sweep_id}"


def register(sweep_path: str, sweep_cfg: dict, registry: Path = SWEEP_REGISTRY) -> None:
    """Append one bullet to docs/MASTER_SWEEPS.md (same convention as the other MASTER_* docs)."""
    header = (
        "# Master list of W&B sweeps\n\n"
        "One bullet per sweep, appended automatically by `scripts/hpo/sweep.py` (pilots run\n"
        "with `--no-register` are not listed). Read results with\n"
        "`uv run scripts/hpo/summarize_sweep.py --sweep <path>`; protocol in docs/HPO_GUIDE.md.\n\n"
    )
    if not registry.exists():
        registry.write_text(header)
    line = (
        f"- `{sweep_cfg['name']}`: `{sweep_path}` (created {_dt.date.today().isoformat()}, "
        f"objective `{sweep_cfg['metric']['name']}` {sweep_cfg['metric']['goal']}, "
        f"run_cap {sweep_cfg.get('run_cap', '?')}, git {git_sha()})\n"
    )
    with open(registry, "a") as fh:
        fh.write(line)


def sbatch_command(
    sweep_path: str,
    trials: int,
    parallel: int,
    count: int = 1,
    time_limit: str = "03:00:00",
    agents_per_gpu: int = 1,
) -> List[str]:
    """The `sbatch --array` argv: each array task holds one GPU and runs `agents_per_gpu`
    concurrent agents on it, each running `count` trials then exiting.

    Default one agent, one trial per task: per-trial wall-clock limit and SLURM log, crash
    isolation, and `%parallel` gives the same concurrency the old `array_parallelism: 8` did.
    With agents_per_gpu > 1 the task's CPUs / memory scale with it (CPUS_PER_AGENT /
    MEM_GB_PER_AGENT, the sbatch file's one-agent defaults) -- worth it when a trial is
    data-loader bound and leaves most of the GPU idle. Surplus tasks past `run_cap` exit
    immediately.
    """
    agents_per_gpu = max(1, agents_per_gpu)
    n_tasks = max(1, math.ceil(trials / (max(1, count) * agents_per_gpu)))
    cmd = ["sbatch", f"--array=0-{n_tasks - 1}%{parallel}", f"--time={time_limit}"]
    if agents_per_gpu > 1:
        cmd += [f"--cpus-per-task={CPUS_PER_AGENT * agents_per_gpu}", f"--mem={MEM_GB_PER_AGENT * agents_per_gpu}G"]
    return cmd + [str(AGENT_SBATCH), sweep_path, str(count), str(agents_per_gpu)]


def submit_agents(
    sweep_path: str,
    trials: int,
    parallel: int,
    count: int = 1,
    time_limit: str = "03:00:00",
    agents_per_gpu: int = 1,
) -> str:
    """Submit `sbatch_command(...)`. `uv sync --frozen` runs once here so the concurrent
    `uv run --no-sync` agents never race to sync .venv."""
    subprocess.run(["uv", "sync", "--frozen"], cwd=REPO_ROOT, check=True)
    cmd = sbatch_command(sweep_path, trials, parallel, count, time_limit, agents_per_gpu)
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()
    print(out)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", help="baseline | sngp (a configs/hparams_search/wandb/<family>.yaml must exist)")
    parser.add_argument("dataset", help="tang | acevedo | wong | kather2018 | wong_ucdavis | ... (configs/experiment/<family>_<dataset>.yaml)")
    parser.add_argument("--trials", type=int, default=None, help="total trials (run_cap); default from the sweep YAML")
    parser.add_argument("--parallel", type=int, default=8, help="max concurrent SLURM array tasks (default 8)")
    parser.add_argument("--count", type=int, default=1, help="trials per agent/array task (default 1)")
    parser.add_argument("--time", default="03:00:00", help="SLURM time limit per array task (default 03:00:00)")
    parser.add_argument("--agents-per-gpu", type=int, default=1,
                        help="concurrent agents (trials) sharing each array task's GPU (default 1)")
    parser.add_argument("--experiment", default=None,
                        help="experiment config to sweep (default <family>_<dataset>)")
    parser.add_argument("--name", default=None,
                        help="sweep name and W&B group for every trial (default <experiment name>_hpo)")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="extra Hydra override for every trial (repeatable), e.g. trainer.max_epochs=2")
    parser.add_argument("--project", default=None, help="W&B project (default: $PROJECT_NAME from .env)")
    parser.add_argument("--entity", default=None, help="W&B entity (default: $WANDB_ENTITY, else your default entity)")
    parser.add_argument("--dry-run", action="store_true", help="print the final sweep YAML and exit")
    parser.add_argument("--no-register", action="store_true", help="do not append to docs/MASTER_SWEEPS.md (pilots)")
    parser.add_argument("--no-submit", action="store_true", help="create the sweep but do not sbatch agents")
    args = parser.parse_args(argv)

    experiment = args.experiment or f"{args.family}_{args.dataset}"
    if not (REPO_ROOT / "configs" / "experiment" / f"{experiment}.yaml").is_file():
        parser.error(f"no configs/experiment/{experiment}.yaml")

    sweep_cfg = build_sweep_config(args.family, args.dataset, extra_overrides=args.override, trials=args.trials,
                                   experiment=experiment, name=args.name)
    if args.dry_run:
        print(yaml.safe_dump(sweep_cfg, sort_keys=False))
        return 0

    project = args.project or os.environ.get("PROJECT_NAME")
    if not project:
        parser.error("pass --project or set PROJECT_NAME in .env (see env_example)")
    entity = args.entity or os.environ.get("WANDB_ENTITY") or None

    sweep_path = create_sweep(sweep_cfg, project=project, entity=entity)
    print(f"sweep: {sweep_path}")
    print(f"       name={sweep_cfg['name']} run_cap={sweep_cfg.get('run_cap')} metric={sweep_cfg['metric']['name']}")
    if not args.no_register:
        register(sweep_path, sweep_cfg)
        print(f"registered in {SWEEP_REGISTRY.relative_to(REPO_ROOT)}")

    if args.no_submit:
        print(f"run agents by hand:  uv run wandb agent --count {args.count} {sweep_path}")
        return 0
    submit_agents(sweep_path, trials=int(sweep_cfg.get("run_cap", args.trials or 48)), parallel=args.parallel,
                  count=args.count, time_limit=args.time, agents_per_gpu=args.agents_per_gpu)
    print(f"read results:  uv run scripts/hpo/summarize_sweep.py --sweep {sweep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
