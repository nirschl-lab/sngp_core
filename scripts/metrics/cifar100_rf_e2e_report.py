"""Score the `rf_e2e` study: SpecReg CIFAR-100 trained end-to-end per random-feature head.

Reads the prediction CSVs written by `scripts/tmux/cifar100_rf_e2e_infer.sh` (CIFAR-100 test,
CIFAR-10 / SVHN as OOD, all from `last.ckpt`) and the validation-fitted mean-field factor
lambda* from that launcher's `lambda_<label>.log` files. Every metric is on test; lambda is
fitted on val because the two heads were trained at different factors (pi/8 vs 7.5).

Two rows per run: at its trained lambda (the gate: the offline softmax must reproduce the
written `class_probs`) and at lambda*(val), the reported one. GP-variance AUROC is lambda-free.
The l = 20 cos/orf control is the existing SpecReg seed-12345 run (`cifar100last_specreg`).

Metric helpers are imported from the sibling mean-field sweep, not restated, so this and
docs/results/CIFAR100_RESULTS.md cannot drift on how a number is computed.

    uv run python scripts/metrics/cifar100_rf_e2e_report.py \\
        --tag rf_e2e_2026-09-26_16-55-28 \\
        --lambda-logs $EXPERIMENTS_HOME/$PROJECT_NAME/tmux_logs/rf_e2e_infer_2026-09-26_16-55-28 \\
        --csv figures/cifar100_rf_e2e/cifar100_rf_e2e.csv
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(Path(__file__).parent))
from cifar100_mean_field_sweep import _sweep_rows  # noqa: E402
from cifar100_predictive_links import _collect  # noqa: E402

from src.checkpointing.io import read_meta  # noqa: E402
from src.metrics.auc import AUROC  # noqa: E402
from src.metrics.posthoc_calibration import mean_field_scale  # noqa: E402

# label -> (head, feature_map, coupling). Order is the table order.
RUNS: Dict[str, Tuple[str, str, str]] = {
    "control_specreg_s12345": ("l20recipe", "cos", "orf"),
    "l20recipe_positive_orf": ("l20recipe", "positive", "orf"),
    "l20recipe_positive_simrf": ("l20recipe", "positive", "simrf"),
    "l20recipe_hyperbolic_orf": ("l20recipe", "hyperbolic", "orf"),
    "l20recipe_hyperbolic_simrf": ("l20recipe", "hyperbolic", "simrf"),
    "l2paper_cos_orf": ("l2paper", "cos", "orf"),
    "l2paper_cos_simrf": ("l2paper", "cos", "simrf"),
}
CONTROL = "control_specreg_s12345"
CONTROL_INFER_LABEL = "cifar100last_specreg"  # first-run convention: no tag prefix
CONTROL_CKPT = "train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt"

FIELDS = [
    "label", "head", "feature_map", "coupling", "lambda_kind", "lambda",
    "acc", "nll", "brier", "smece",
    "auroc_msp_cifar10", "auroc_msp_svhn", "auroc_ds_cifar10", "auroc_ds_svhn",
    "auroc_var_cifar10", "auroc_var_svhn",
]


def _fitted_lambda(log: Path) -> float:
    m = re.search(r"fitted mean_field_factor=([0-9.eE+-]+)", log.read_text())
    if m is None:
        raise SystemExit(f"no 'fitted mean_field_factor=' line in {log}")
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lambda-logs", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    args = ap.parse_args()

    base = Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"]
    infer_root = base / "infer"
    stamp = args.tag.removeprefix("rf_e2e_")

    rows: List[Dict[str, object]] = []
    for label, (head, fmap, coupling) in RUNS.items():
        if label == CONTROL:
            arm = _collect(infer_root, "", CONTROL_INFER_LABEL, label, head)
            ckpt = base / CONTROL_CKPT
        else:
            arm = _collect(infer_root, args.tag, label, label, head)
            ckpt = base / "train" / "rf_e2e" / stamp / label / "checkpoints" / "last.ckpt"
        if arm is None:
            raise SystemExit(f"missing predictions for {label}")
        trained = float(read_meta(ckpt).net_spec["mean_field_factor"])
        fitted = _fitted_lambda(args.lambda_logs / f"lambda_{label}.log")

        raw, var, _ = arm.id
        probs = torch.softmax(mean_field_scale(raw, var, trained), dim=1).numpy()
        gate = float(np.abs(probs - arm.published_probs).max())
        print(f"gate {label:28s} trained lambda {trained:<9.4g} max |delta prob| {gate:.1e}   lambda*(val) {fitted:.4g}")
        if gate > 1e-4:
            raise SystemExit(f"{label}: offline softmax does not reproduce the written class_probs")

        id_var = var.squeeze(1).numpy()
        var_auroc = {
            f"auroc_var_{ds}": float(AUROC(id_var, ovar.squeeze(1).numpy(), score_is_uncertainty=True))
            for ds, (_, ovar) in arm.ood.items()
        }
        swept = {r["lambda"]: r for r in _sweep_rows(arm, [trained, fitted]) if r["kind"] == "sweep"}
        for kind, lam in (("trained", trained), ("val_fit", fitted)):
            rows.append({
                "label": label, "head": head, "feature_map": fmap, "coupling": coupling,
                "lambda_kind": kind, **swept[lam], "lambda": lam, **var_auroc,
            })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv} ({len(rows)} rows)\n")

    cols = [("acc", "Acc"), ("nll", "NLL"), ("smece", "smECE"), ("lambda", "λ*"),
            ("auroc_msp_cifar10", "MSP C-10"), ("auroc_msp_svhn", "MSP SVHN"),
            ("auroc_ds_svhn", "DS SVHN"), ("auroc_var_svhn", "Var SVHN")]
    print("| Run | Head | " + " | ".join(c for _, c in cols) + " |")
    print("|---|---|" + "---:|" * len(cols))
    for r in rows:
        if r["lambda_kind"] != "val_fit":
            continue
        name = f"{r['feature_map']} / {r['coupling']}" + (" (control)" if r["label"] == CONTROL else "")
        vals = [f"{r[k]:.3g}" if k == "lambda" else f"{r[k]:.4f}" for k, _ in cols]
        print(f"| {name} | {r['head']} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
