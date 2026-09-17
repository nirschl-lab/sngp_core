"""One-time migration: re-save pre-v2 checkpoints (pickled `net` object in
`hyper_parameters`) as v2 (plain-data `net_spec` + a `sngp_core` metadata block).

After this migration, `src/checkpointing/io.py` (used by everything -- train/eval,
inference, HF export) refuses to load anything below format_version 2, so this script
is the only supported path from an old checkpoint to a usable one.

Usage:
    uv run scripts/checkpoints/migrate_checkpoints.py --in "logs/**/*.ckpt" --out migrated_checkpoints/
    uv run scripts/checkpoints/migrate_checkpoints.py --in old.ckpt --out migrated/ --arch resnet18 --dry-run

`--arch` is required when migrating a legacy SNGP checkpoint (its pickled net object
never recorded which backbone it used -- see src/checkpointing/legacy.py) and ignored
otherwise. Migrating checkpoints of different SNGP archs in one run isn't supported;
run the script once per arch.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.checkpointing.legacy import (  # noqa: E402
    derive_net_spec,
    load_legacy_checkpoint,
    remap_legacy_state_dict,
)
from src.checkpointing.spec import FORMAT_VERSION  # noqa: E402


def migrate_one(ckpt_path: Path, out_dir: Path, arch_hint: str, dry_run: bool) -> Dict[str, Any]:
    row = {"checkpoint": str(ckpt_path), "status": "skipped", "detail": ""}

    checkpoint = load_legacy_checkpoint(str(ckpt_path))
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        row["status"] = "error"
        row["detail"] = "not a Lightning checkpoint dict (missing 'state_dict')"
        return row

    if isinstance(checkpoint.get("sngp_core"), dict):
        row["status"] = "already-v2"
        row["detail"] = f"format_version={checkpoint['sngp_core'].get('format_version')}"
        return row

    hparams = checkpoint.get("hyper_parameters", {})
    net = hparams.get("net")
    if net is None:
        row["status"] = "error"
        row["detail"] = "no hyper_parameters['net'] to derive a net_spec from"
        return row

    try:
        net_spec = derive_net_spec(net, arch_hint=arch_hint)
    except (TypeError, ValueError) as exc:
        row["status"] = "error"
        row["detail"] = str(exc)
        return row

    # Infer which LightningModule class this checkpoint belongs to from the net type,
    # since legacy checkpoints don't record it explicitly the way v2 checkpoints do.
    lit_module_path = {
        "baseline_classifier": "src.models.baseline_lit_module.BaselineLitModule",
        "sngp_classifier": "src.models.sngp_lit_module.SNGPLitModule",
        "deep_ensemble": "src.models.deep_ensemble_lit_module.DeepEnsembleLitModule",
    }.get(net_spec["name"])
    if lit_module_path is None:
        row["status"] = "error"
        row["detail"] = f"no known LightningModule for net_spec name '{net_spec['name']}'"
        return row

    num_classes = hparams.get("num_classes", getattr(net, "num_classes", None))
    new_state_dict = remap_legacy_state_dict(checkpoint["state_dict"], net_spec)

    meta = {
        "format_version": FORMAT_VERSION,
        "lit_module": lit_module_path,
        "net_spec": net_spec,
        "num_classes": num_classes,
        "idx_to_class": None,
        "dataset_name": None,
    }

    new_checkpoint = dict(checkpoint)
    new_checkpoint["state_dict"] = new_state_dict
    # New hparams: only primitives + net_spec, matching what LitModuleBase now saves.
    primitive_hparams = {
        k: v for k, v in hparams.items()
        if k not in {"net", "optimizer", "scheduler", "calibration_cfg"}
        and isinstance(v, (int, float, str, bool, type(None), dict, list))
    }
    primitive_hparams["net_spec"] = net_spec
    new_checkpoint["hyper_parameters"] = primitive_hparams
    new_checkpoint["sngp_core"] = meta

    row["status"] = "migrated"
    row["detail"] = f"lit_module={lit_module_path}, net_spec.name={net_spec['name']}"

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ckpt_path.name
        torch.save(new_checkpoint, out_path)
        row["detail"] += f", written to {out_path}"

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_glob", required=True, help="Path or glob of checkpoint(s) to migrate")
    parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="Directory to write migrated checkpoints into")
    parser.add_argument("--arch", default=None, help="Backbone arch, required for legacy SNGP checkpoints (e.g. resnet18)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing files")
    parser.add_argument("--report", type=Path, default=None, help="Optional CSV path to write the per-checkpoint report to")
    args = parser.parse_args()

    paths: List[Path]
    if any(ch in args.in_glob for ch in "*?["):
        paths = sorted(Path().glob(args.in_glob))
    else:
        paths = [Path(args.in_glob)]

    if not paths:
        print(f"No checkpoints matched: {args.in_glob}", file=sys.stderr)
        sys.exit(1)

    rows = [migrate_one(p, args.out_dir, args.arch, args.dry_run) for p in paths]

    width = max(len(r["checkpoint"]) for r in rows)
    for r in rows:
        print(f"{r['status']:>12}  {r['checkpoint']:<{width}}  {r['detail']}")

    if args.report:
        with open(args.report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", "status", "detail"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nReport written to {args.report}")

    n_errors = sum(1 for r in rows if r["status"] == "error")
    if n_errors:
        print(f"\n{n_errors} checkpoint(s) failed to migrate.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
