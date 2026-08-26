#!/usr/bin/env python3
"""Assemble N independently-trained checkpoints (all the same net type/architecture --
e.g. all baseline_classifier, or all sngp_classifier) into one DeepEnsemble checkpoint
-- the counterpart to `scripts/ensemble/train_members_parallel.sh`, which trains those
N members as separate, fully parallel (multi-GPU) runs instead of
DeepEnsembleLitModule's sequential single-run cycling (the only `train_strategy` it
implements today, see src/models/deep_ensemble_lit_module.py).

The output satisfies the same checkpoint contract as one produced by an actual
DeepEnsembleLitModule training run (src/checkpointing/spec.py's format_version-2
`sngp_core` block + `hyper_parameters` + `state_dict`), so it loads through
src/checkpointing/io.py's read_meta/load_net/load_lit_module -- and therefore through
src/eval.py and src/inference/infer.py -- exactly like any other checkpoint.

Usage:
    # discover member_0/checkpoints/best.ckpt, member_1/..., ... under a directory
    # (the layout scripts/ensemble/train_members_parallel.sh produces). --out defaults
    # to <members-dir>/checkpoints/ensemble.ckpt when omitted (see
    # derive_default_out_path below):
    uv run scripts/ensemble/assemble_ensemble_checkpoint.py --members-dir <dir>

    # or override the output path explicitly:
    uv run scripts/ensemble/assemble_ensemble_checkpoint.py \
        --members-dir <dir> --out <path/to/ensemble.ckpt>

    # or list member checkpoints explicitly, in member order (--out is required here --
    # there's no shared members directory to derive a default from):
    uv run scripts/ensemble/assemble_ensemble_checkpoint.py \
        --ckpts m0/checkpoints/best.ckpt m1/checkpoints/best.ckpt m2/checkpoints/best.ckpt \
        --out ensemble.ckpt
"""
import argparse
import sys
from pathlib import Path
from typing import List

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import lightning  # noqa: E402
import torch  # noqa: E402

from src.checkpointing.io import load_net, read_meta  # noqa: E402
from src.checkpointing.spec import build_meta  # noqa: E402
from src.models.deep_ensemble_lit_module import DeepEnsembleLitModule  # noqa: E402
from src.models.ensemble.deep_ensemble_model import DeepEnsemble  # noqa: E402


def _discover_member_checkpoints(members_dir: Path) -> List[Path]:
    member_dirs = sorted(
        (p for p in members_dir.glob("member_*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    if not member_dirs:
        raise FileNotFoundError(f"No member_* directories found under {members_dir}")

    ckpts = []
    for member_dir in member_dirs:
        ckpt = member_dir / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Expected checkpoint at {ckpt}")
        ckpts.append(ckpt)
    return ckpts


def derive_default_out_path(members_dir: Path) -> Path:
    """Default `--out` when left blank: `<members_dir>/checkpoints/ensemble.ckpt` -- a
    sibling of `member_0/`, `member_1/`, ... under the same `ensemble_members/<run_id>/`
    directory, matching the `checkpoints/` subdir convention `runs/<run_id>/` and
    `member_i/` already use."""
    return members_dir / "checkpoints" / "ensemble.ckpt"


def assemble(ckpt_paths: List[Path], out_path: Path) -> None:
    metas = [read_meta(p) for p in ckpt_paths]

    reference = metas[0]
    reference_arch = {k: v for k, v in reference.net_spec.items() if k != "pretrained"}
    for meta, ckpt_path in zip(metas[1:], ckpt_paths[1:]):
        arch = {k: v for k, v in meta.net_spec.items() if k != "pretrained"}
        if arch != reference_arch:
            raise ValueError(
                f"{ckpt_path} has architecture {arch}, expected {reference_arch} "
                f"(from {ckpt_paths[0]}). All ensemble members must be independently-trained "
                "runs of the same net type/architecture (e.g. all baseline_classifier, or all "
                "sngp_classifier) to be assembled into one DeepEnsemble."
            )
        if meta.dataset_name != reference.dataset_name:
            raise ValueError(
                f"{ckpt_path} was trained on dataset {meta.dataset_name!r}, expected "
                f"{reference.dataset_name!r} (from {ckpt_paths[0]})."
            )

    base_model_spec = dict(reference.net_spec)
    base_model_spec["pretrained"] = False  # weights are overwritten by state dicts below

    ensemble_net = DeepEnsemble(
        base_model_spec=base_model_spec,
        num_estimators=len(ckpt_paths),
        task="classification",
    )
    for i, ckpt_path in enumerate(ckpt_paths):
        member_net = load_net(ckpt_path, device="cpu")
        ensemble_net.ensemble_members[i].load_state_dict(member_net.state_dict())

    lit_module = DeepEnsembleLitModule(
        net=ensemble_net,
        num_estimators=len(ckpt_paths),
        num_classes=reference.num_classes,
        train_strategy="sequential",
    )

    checkpoint = {
        "state_dict": lit_module.state_dict(),
        "hyper_parameters": dict(lit_module.hparams),
        "sngp_core": build_meta(
            lit_module,
            net_spec=ensemble_net.spec,
            num_classes=reference.num_classes,
            idx_to_class=reference.idx_to_class,
            dataset_name=reference.dataset_name,
        ),
        "pytorch-lightning_version": lightning.__version__,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Warning: overwriting existing checkpoint at {out_path}", file=sys.stderr)
    torch.save(checkpoint, out_path)
    print(f"Wrote {len(ckpt_paths)}-member ensemble checkpoint to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--members-dir",
        type=Path,
        help="Directory with member_0/, member_1/, ... subdirs, as produced by train_members_parallel.sh",
    )
    group.add_argument(
        "--ckpts",
        type=Path,
        nargs="+",
        help="Explicit list of member checkpoint paths, in member order",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path for the assembled ensemble checkpoint. Defaults to "
            "<members-dir>/checkpoints/ensemble.ckpt when --members-dir is given; "
            "required when using --ckpts."
        ),
    )
    args = parser.parse_args()

    ckpt_paths = _discover_member_checkpoints(args.members_dir) if args.members_dir else args.ckpts
    if len(ckpt_paths) < 2:
        print("Need at least 2 member checkpoints to build an ensemble.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        if args.members_dir is None:
            print("--out is required when using --ckpts (no --members-dir to derive a default from).", file=sys.stderr)
            sys.exit(1)
        out_path = derive_default_out_path(args.members_dir)
        print(f"No --out given; auto-derived {out_path}")

    assemble(ckpt_paths, out_path)


if __name__ == "__main__":
    main()
