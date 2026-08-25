"""Summarize a persisted Optuna hyperparameter-search study.

Prints the top-K trials with their parameter overrides, ready to paste onto a
`uv run src/train.py experiment=...` command line for the full-budget retrain step
(see docs/HPO_GUIDE.md -- proxy-budget sweeps retrain their top-3, not just the
winner, since proxy-budget rankings are noisy).

Not part of the importable `src` package (per CLAUDE.md's convention for
scripts/ -- cluster/utility scripts live outside it), so it's a plain script, not a
Hydra entrypoint.

Usage:
    uv run scripts/hpo/summarize_study.py --study tang_baseline_resnet18_hpo
    uv run scripts/hpo/summarize_study.py --study tang_baseline_resnet18_hpo --top-k 5
    uv run scripts/hpo/summarize_study.py --study tang_baseline_resnet18_hpo \\
        --storage sqlite:////absolute/path/to/tang_baseline_resnet18_hpo.db
"""
import argparse
import os
from pathlib import Path

import optuna
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def resolve_default_storage(study_name: str) -> str:
    """Match the `hydra.sweeper.storage` path built by configs/hparams_search/*.yaml:
    `sqlite:///${paths.log_dir}/optuna/${name}_hpo.db`, where `paths.log_dir` is
    `${EXPERIMENTS_HOME}/${PROJECT_NAME}`."""
    experiments_home = os.environ.get("EXPERIMENTS_HOME")
    project_name = os.environ.get("PROJECT_NAME")
    if not experiments_home or not project_name:
        raise SystemExit(
            "Pass --storage explicitly, or export EXPERIMENTS_HOME/PROJECT_NAME "
            "(see env_example) so the default sqlite path can be resolved."
        )
    db_path = Path(experiments_home) / project_name / "optuna" / f"{study_name}.db"
    return f"sqlite:///{db_path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--study", required=True, help="Optuna study_name, e.g. tang_baseline_resnet18_hpo"
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URI. Defaults to the sqlite path configs/hparams_search "
        "builds from EXPERIMENTS_HOME/PROJECT_NAME.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top trials to print (default: 3)")
    args = parser.parse_args()

    storage = args.storage or resolve_default_storage(args.study)
    study = optuna.load_study(study_name=args.study, storage=storage)

    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)

    print(f"Study: {args.study}  ({len(study.trials)} trials total, {len(completed)} completed)")
    print(f"Storage: {storage}\n")

    if not completed:
        print("No completed trials yet.")
        return

    for rank, trial in enumerate(completed[: args.top_k], start=1):
        overrides = " ".join(f"{k}={v}" for k, v in sorted(trial.params.items()))
        print(f"#{rank}  trial={trial.number}  val/auprc_best={trial.value:.4f}")
        print(f"    {overrides}\n")


if __name__ == "__main__":
    main()
