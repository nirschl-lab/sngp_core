"""Summarize a W&B sweep: top-K finished trials with paste-ready Hydra overrides.

Prints the top-K trials ranked by the sweep's own objective (`metric.name`, e.g.
`val/nll_cal_best`, minimized), each with its swept parameters as `key=value` overrides
ready to paste onto `uv run src/train.py experiment=...` for the full-budget retrain step
(docs/HPO_GUIDE.md -- proxy-budget sweeps retrain their top-3, not just the winner, since
proxy-budget rankings are noisy).

Usage:
    uv run scripts/hpo/summarize_sweep.py --sweep <entity>/<project>/<sweep_id>
    uv run scripts/hpo/summarize_sweep.py --sweep <entity>/<project>/<sweep_id> --top-k 5
    uv run scripts/hpo/summarize_sweep.py --sweep ... --include-unfinished   # also rank running/killed trials

Sweep paths are recorded in docs/MASTER_SWEEPS.md by scripts/hpo/sweep.py.
"""
import argparse
import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

FINISHED = ("finished",)
DIAGNOSTICS = ("val/nll_best", "val/temperature_fit", "val/mean_field_factor_fit", "val/auprc_best", "epoch")


def lookup(config: dict, dotted: str) -> Any:
    """Read a swept parameter from `run.config`. W&B stores sweep parameters under their
    dotted name verbatim (`model.optimizer.lr`); Lightning's `log_hyperparams` also dumps
    the nested Hydra config, so fall back to walking `config["model"]["optimizer"]["lr"]`."""
    if dotted in config:
        return config[dotted]
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def rank_runs(runs: Iterable[Any], metric: str, goal: str = "minimize", states: Sequence[str] = FINISHED) -> List[Tuple[float, Any]]:
    """`(value, run)` pairs sorted best-first, skipping runs in other states or whose
    summary lacks the metric (crashed before the first validation epoch). Sorted
    client-side: server-side `order` strings with `/` in the key are brittle."""
    ranked = []
    for run in runs:
        if run.state not in states:
            continue
        value = run.summary.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        ranked.append((float(value), run))
    ranked.sort(key=lambda pair: pair[0], reverse=(goal == "maximize"))
    return ranked


def format_overrides(config: dict, param_keys: Sequence[str]) -> str:
    return " ".join(f"{key}={lookup(config, key)}" for key in sorted(param_keys))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="entity/project/sweep_id (see docs/MASTER_SWEEPS.md)")
    parser.add_argument("--top-k", type=int, default=3, help="number of trials to print (default 3)")
    parser.add_argument("--include-unfinished", action="store_true", help="also rank running / killed / crashed trials")
    args = parser.parse_args(argv)

    import wandb

    api = wandb.Api()
    sweep = api.sweep(args.sweep)
    metric = sweep.config["metric"]["name"]
    goal = sweep.config["metric"].get("goal", "minimize")
    param_keys = list(sweep.config.get("parameters", {}))
    runs = list(sweep.runs)

    states: dict = {}
    for run in runs:
        states[run.state] = states.get(run.state, 0) + 1
    ranked = rank_runs(runs, metric, goal, states=tuple(states) if args.include_unfinished else FINISHED)

    print(f"Sweep: {sweep.name}  ({args.sweep})  state={sweep.state}")
    print(f"Objective: {metric} ({goal})   runs by state: {states}")
    print(f"Swept: {', '.join(sorted(param_keys))}\n")
    if not ranked:
        print("No ranked trials yet.")
        return 0

    for rank, (value, run) in enumerate(ranked[: args.top_k], start=1):
        extras = []
        for key in DIAGNOSTICS:
            if key in run.summary:
                extras.append(f"{key}={run.summary[key]:.4g}" if isinstance(run.summary[key], float) else f"{key}={run.summary[key]}")
        print(f"#{rank}  run={run.name}  id={run.id}  {metric}={value:.4f}  ({', '.join(extras)})")
        print(f"    {format_overrides(dict(run.config), param_keys)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
