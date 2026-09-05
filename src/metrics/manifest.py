"""Load, validate, and generate docs from `configs/runs/infer_manifest.yaml`.

Everything a run's identity took to reconstruct by hand -- which checkpoint, which
inference run-folder, which datasets it was evaluated against, whether it's an
ensemble or MC-Dropout -- previously lived only in two hand-typed docs
(`docs/MASTER_INFER_RESULTS_PATH.md`, `docs/MASTER_CHECKPONT_PATHS.md`), referenced by
no code (`grep` for either filename returns nothing outside `docs/`), with three label
typos and an inconsistent path granularity. This module makes that data
machine-readable, and the two docs generated output rather than a second source of
truth to keep in sync by hand -- see `render_infer_master_doc`/`render_ckpt_master_doc`
and this file's `main()` (`--write-master-docs`).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import os
import rootutils
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.inference.infer import _RUN_ID_PATTERN  # noqa: E402

DEFAULT_MANIFEST_PATH = ROOT / "configs" / "runs" / "infer_manifest.yaml"
MASTER_INFER_DOC_PATH = ROOT / "docs" / "MASTER_INFER_RESULTS_PATH.md"
MASTER_CKPT_DOC_PATH = ROOT / "docs" / "MASTER_CHECKPONT_PATHS.md"

_GENERATED_HEADER = (
    "<!-- generated from configs/runs/infer_manifest.yaml -- do not edit by hand.\n"
    "     Regenerate with `uv run src/metrics/manifest.py --write-master-docs`. -->\n\n"
)

_WONG_UCDAVIS_NOTE = (
    "Wong (ucdavis) models. Each run is the in-distribution test split of the UC Davis\n"
    "institution subset (`data=wong data.datamodule.institution=ucdavis fold=test`); the\n"
    "`wong_ucdavis` leaf comes from `data.name=wong_ucdavis`, set so per-institution runs\n"
    "off the same checkpoint don't collide in a shared `wong/` folder.\n\n"
)


@dataclass(frozen=True)
class RunEntry:
    label: str
    method: str
    train_dataset: str
    is_ensemble: bool
    uses_mc_dropout: bool
    run_dir: str
    ckpt: str
    id_dataset: str
    eval_datasets: Tuple[str, ...]
    paired_streams: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    infer_root: Path
    runs: Tuple[RunEntry, ...]

    def get(self, label: str) -> RunEntry:
        for run in self.runs:
            if run.label == label:
                return run
        raise KeyError(f"No manifest entry with label {label!r}. Known labels: {sorted(r.label for r in self.runs)}")


def resolve_default_infer_root() -> str:
    """Match `configs/paths/default.yaml`'s `log_dir`:
    `${EXPERIMENTS_HOME}/${PROJECT_NAME}/infer`."""
    experiments_home = os.environ.get("EXPERIMENTS_HOME")
    project_name = os.environ.get("PROJECT_NAME")
    if not experiments_home or not project_name:
        raise SystemExit(
            "Pass --infer-root explicitly, or export EXPERIMENTS_HOME/PROJECT_NAME "
            "(see env_example) so the default infer root can be resolved."
        )
    return str(Path(experiments_home) / project_name / "infer")


def load_manifest(
    path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    *,
    infer_root: Union[str, Path, None] = None,
) -> Manifest:
    """Load and parse the run manifest.

    If `infer_root` is given, it overrides the YAML's `infer_root` field before
    resolution -- the `${oc.env:EXPERIMENTS_HOME}/${oc.env:PROJECT_NAME}/infer`
    interpolation is then never touched, which is what lets this be called in a
    test/CI environment with neither variable exported. If `infer_root` is `None`,
    resolving the manifest requires both to already be set (see `env_example`).
    """
    cfg = OmegaConf.load(path)
    if infer_root is not None:
        cfg.infer_root = str(infer_root)
    raw = OmegaConf.to_container(cfg, resolve=True)

    runs = tuple(
        RunEntry(
            label=r["label"],
            method=r["method"],
            train_dataset=r["train_dataset"],
            is_ensemble=bool(r["is_ensemble"]),
            uses_mc_dropout=bool(r["uses_mc_dropout"]),
            run_dir=r["run_dir"],
            ckpt=r["ckpt"],
            id_dataset=r["id_dataset"],
            eval_datasets=tuple(r["eval_datasets"]),
            paired_streams=tuple(r.get("paired_streams") or ()),
        )
        for r in raw["runs"]
    )
    return Manifest(schema_version=int(raw["schema_version"]), infer_root=Path(raw["infer_root"]), runs=runs)


def _find_run_id(text: str) -> Optional[str]:
    match = _RUN_ID_PATTERN.search(text)
    return match.group(0) if match else None


def validate_manifest(manifest: Manifest, *, check_disk: bool = True) -> List[str]:
    """Return a list of human-readable problems; empty means the manifest is valid.

    Always checked (no disk access): labels are unique and filename-safe, each
    `id_dataset` is one of its own `eval_datasets`, and the run-id embedded in
    `run_dir` matches the one embedded in `ckpt` (this project's own
    `${now:%Y-%m-%d}_${now:%H-%M-%S}` stamp, see `configs/hydra/default.yaml`) --
    catching an entry that was hand-typed with a mismatched or copy-pasted path.

    Additionally checked when `check_disk=True` (needs `manifest.infer_root` to be a
    real, resolved path): every `run_dir` exists, every `eval_datasets`/
    `paired_streams` leaf has a `predictions.csv`, and every directory actually on
    disk under a `run_dir` is claimed by exactly one of those two lists (an
    `images/` leaf is whitelisted) -- catching drift in both directions between the
    manifest and what inference has actually produced.
    """
    problems: List[str] = []

    seen_labels = set()
    for run in manifest.runs:
        if run.label in seen_labels:
            problems.append(f"duplicate label: {run.label!r}")
        seen_labels.add(run.label)
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", run.label):
            problems.append(f"label {run.label!r} is not filename-safe")

    for run in manifest.runs:
        if run.id_dataset not in run.eval_datasets:
            problems.append(
                f"{run.label}: id_dataset {run.id_dataset!r} is not in its own eval_datasets {run.eval_datasets}"
            )

    for run in manifest.runs:
        run_dir_id = _find_run_id(run.run_dir)
        ckpt_id = _find_run_id(run.ckpt)
        if run_dir_id is None:
            problems.append(f"{run.label}: no run-id timestamp found in run_dir {run.run_dir!r}")
        elif ckpt_id is None:
            problems.append(f"{run.label}: no run-id timestamp found in ckpt {run.ckpt!r}")
        elif run_dir_id != ckpt_id:
            problems.append(
                f"{run.label}: run_dir run-id {run_dir_id!r} does not match ckpt run-id {ckpt_id!r}"
            )

    if not check_disk:
        return problems

    for run in manifest.runs:
        run_dir_path = manifest.infer_root / run.run_dir
        if not run_dir_path.is_dir():
            problems.append(f"{run.label}: run_dir does not exist on disk: {run_dir_path}")
            continue

        claimed = set(run.eval_datasets) | set(run.paired_streams)
        for dataset in sorted(claimed):
            leaf = run_dir_path / dataset
            if not (leaf / "predictions.csv").exists():
                problems.append(f"{run.label}: no predictions.csv at {leaf}")

        for child in sorted(run_dir_path.iterdir()):
            if not child.is_dir() or child.name == "images":
                continue
            if child.name not in claimed:
                problems.append(f"{run.label}: {child} is on disk but not claimed by eval_datasets/paired_streams")

    return problems


def _group_by_id_dataset(runs: Sequence[RunEntry]) -> List[List[RunEntry]]:
    """Consecutive runs sharing an `id_dataset`, in manifest order -- the manifest is
    authored pre-grouped (acevedo, then wong, then wong_ucdavis), so this recovers the
    same grouping the original hand-written docs used without a separate field."""
    groups: List[List[RunEntry]] = []
    for run in runs:
        if groups and groups[-1][-1].id_dataset == run.id_dataset:
            groups[-1].append(run)
        else:
            groups.append([run])
    return groups


def render_infer_master_doc(manifest: Manifest) -> str:
    lines = [_GENERATED_HEADER]
    for group in _group_by_id_dataset(manifest.runs):
        if group[0].id_dataset == "wong_ucdavis":
            lines.append(_WONG_UCDAVIS_NOTE)
        for run in group:
            path = manifest.infer_root / run.run_dir
            lines.append(f"{run.label}:\n```bash\n{path}\n```\n\n")
        lines.append("---\n\n")
    # Drop the trailing separator after the last group.
    if lines[-1] == "---\n\n":
        lines.pop()
    return "".join(lines).rstrip("\n") + "\n"


def render_ckpt_master_doc(manifest: Manifest) -> str:
    log_dir = manifest.infer_root.parent
    lines = [_GENERATED_HEADER]
    for group in _group_by_id_dataset(manifest.runs):
        if group[0].id_dataset == "wong_ucdavis":
            lines.append(_WONG_UCDAVIS_NOTE)
        # MC-Dropout entries reuse their baseline sibling's checkpoint -- one entry
        # per underlying checkpoint, not one per inference run.
        for run in group:
            if run.uses_mc_dropout:
                continue
            path = log_dir / run.ckpt
            lines.append(f"{run.label}:\n```bash\n{path}\n```\n\n")
        lines.append("---\n\n")
    if lines[-1] == "---\n\n":
        lines.pop()
    return "".join(lines).rstrip("\n") + "\n"


def check_master_docs_are_current(manifest: Manifest) -> List[str]:
    """Return a problem for each of the two generated docs that doesn't exist or
    doesn't match what `render_*_master_doc` would write for `manifest` -- the
    "checked-in markdown matches the manifest" half of `--validate`."""
    problems: List[str] = []
    for path, render in (
        (MASTER_INFER_DOC_PATH, render_infer_master_doc),
        (MASTER_CKPT_DOC_PATH, render_ckpt_master_doc),
    ):
        if not path.exists() or path.read_text() != render(manifest):
            problems.append(f"{path} is stale or missing -- run --write-master-docs")
    return problems


def main() -> None:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description="Validate/regenerate docs for configs/runs/infer_manifest.yaml.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), type=Path)
    parser.add_argument("--infer-root", default=None, help="Override infer_root instead of resolving it from env.")
    parser.add_argument("--validate", action="store_true", help="Run validate_manifest and exit non-zero on problems.")
    parser.add_argument(
        "--write-master-docs", action="store_true", help="Regenerate docs/MASTER_{INFER_RESULTS_PATH,CHECKPONT_PATHS}.md."
    )
    args = parser.parse_args()

    infer_root = args.infer_root or resolve_default_infer_root()
    manifest = load_manifest(args.manifest, infer_root=infer_root)

    if args.write_master_docs:
        MASTER_INFER_DOC_PATH.write_text(render_infer_master_doc(manifest))
        MASTER_CKPT_DOC_PATH.write_text(render_ckpt_master_doc(manifest))
        print(f"Wrote {MASTER_INFER_DOC_PATH}")
        print(f"Wrote {MASTER_CKPT_DOC_PATH}")

    if args.validate:
        problems = validate_manifest(manifest, check_disk=True) + check_master_docs_are_current(manifest)

        if problems:
            for problem in problems:
                print(f"PROBLEM: {problem}")
            raise SystemExit(f"{len(problems)} problem(s) found.")
        print(f"OK -- {len(manifest.runs)} manifest entries validated.")


if __name__ == "__main__":
    main()
