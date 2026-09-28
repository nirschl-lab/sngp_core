"""Score the `evidence_ls` benchmark: Baseline / SNGP / SNGP+SpecReg / SNGP+SpecReg+evidence l.

Four rows, three seeds each (12345 / 1 / 2), all from `last.ckpt` (epoch 249), under the
one-post-hoc-knob protocol: each row gets exactly one scalar fitted on VALIDATION NLL by
`scripts/checkpoints/calibrate_checkpoint.py` -- the mean-field factor lambda for the SNGP
arms, a temperature T for the baseline -- and every metric is on TEST.

Inputs:
  * prediction CSVs: the new arm's from `scripts/tmux/cifar100_evidence_ls_infer.sh`
    (`infer/<tag>_els_specreg_s<seed>__<ds>`), the existing arms' from the first run
    (`infer/cifar100last_<arm>__<ds>`, seed 12345) and the overnight run
    (`infer/overnight_2026-09-20_21-38-42_s<seed>_<arm>__<ds>`);
  * the `fit_<arm>_s<seed>.log` files that launcher writes (one or more `--fit-logs` dirs;
    a later dir wins when both carry the same row).

The baseline is scored through the SAME code path as the SNGP arms: its logits are divided by
T and given a constant, negligible variance, so `_sweep_rows(..., [0.0])` returns softmax(logits / T) and
Dempster-Shafer on logits / T. Metric helpers are imported from the sibling mean-field sweep
rather than restated, so this page and docs/results/CIFAR100_RESULTS.md cannot drift on how a
number is computed.

Gates, all fatal:
  * the offline softmax at each row's trained knob reproduces the written `class_probs`;
  * accuracy is identical at the trained and the fitted knob (the knob only rescales logits);
  * with `--expect-committed`, the SNGP / SpecReg rows reproduce the val-fit table of
    docs/results/CIFAR100_RESULTS.md (NLL 0.7600 / 0.7472).

    uv run python scripts/metrics/cifar100_evidence_ls_report.py \\
        --tag evidence_ls_2026-09-28_16-10-10 \\
        --fit-logs $EXPERIMENTS_HOME/$PROJECT_NAME/tmux_logs/cifar100_evidence_ls_2026-09-28_16-10-10_infer \\
        --csv figures/cifar100_evidence_ls/cifar100_evidence_ls_per_seed.csv
"""
import argparse
import csv
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)
sys.path.insert(0, str(Path(__file__).parent))
from cifar100_mean_field_sweep import _sweep_rows  # noqa: E402
from cifar100_overnight_report import LABEL_COLS, _infer_dir  # noqa: E402
from cifar100_predictive_links import ArmData, _collect, _msp  # noqa: E402
from cifar100_rf_head_swap import fpr_at_95_tpr  # noqa: E402

from src.metrics.auc import AUROC  # noqa: E402
from src.metrics.io import load_predictions, logits_array, probs_array  # noqa: E402
from src.metrics.posthoc_calibration import mean_field_scale  # noqa: E402

SEEDS = (12345, 1, 2)
OVERNIGHT_TAG = "overnight_2026-09-20_21-38-42"
OOD_SETS = ("cifar10", "svhn")
# key -> display. Order is the table order.
ARMS: Dict[str, str] = {
    "baseline": "Baseline",
    "sngp": "SNGP (c = 6.0)",
    "specreg": "SNGP + SpecReg",
    "els": "SNGP + SpecReg + evidence ℓ = 7",
}
# Committed val-fit numbers (CIFAR100_RESULTS.md, "λ fitted on validation"), for the gate.
COMMITTED_NLL = {"sngp": 0.7600, "specreg": 0.7472}
_NO_VAR = 1e-12

FIT_RE = re.compile(r"^fitted (\w+)=([0-9.eE+-]+)", re.M)
CURRENT_RE = re.compile(r"knob=(\w+)\s+current=([0-9.eE+-]+)")

FIELDS = [
    "arm", "seed", "knob", "trained", "fitted",
    "acc", "nll", "brier", "smece",
    "auroc_msp_cifar10", "auroc_msp_svhn", "auroc_ds_cifar10", "auroc_ds_svhn",
    "auroc_var_cifar10", "auroc_var_svhn", "fpr95_msp_cifar10", "fpr95_msp_svhn",
]


def _source(arm: str, seed: int, tag: str) -> Tuple[str, str, str]:
    """(infer tag, infer label, fit-log row) for one arm x seed."""
    if arm == "els":
        return tag, f"els_specreg_s{seed}", f"els_specreg_s{seed}"
    if seed == 12345:
        return "", f"cifar100last_{arm}", f"{arm}_s12345"  # first-run convention: no tag prefix
    return OVERNIGHT_TAG, f"s{seed}_{arm}", f"{arm}_s{seed}"


def _fit(log_dirs: Sequence[Path], row: str) -> Tuple[str, float, float]:
    """(knob, trained value, val-fitted value) from the last `fit_<row>.log` found."""
    log = next((d / f"fit_{row}.log" for d in reversed(log_dirs) if (d / f"fit_{row}.log").exists()), None)
    if log is None:
        raise SystemExit(f"no fit_{row}.log in {[str(d) for d in log_dirs]}")
    text = log.read_text()
    fitted, current = FIT_RE.search(text), CURRENT_RE.search(text)
    if fitted is None or current is None:
        raise SystemExit(f"{log}: no 'fitted <knob>=' / 'knob= current=' line")
    return fitted.group(1), float(current.group(2)), float(fitted.group(2))


def _baseline_arm(root: Path, tag: str, label: str, temperature: float) -> Tuple[ArmData, np.ndarray, torch.Tensor]:
    """The baseline as an ArmData at logits / T, plus its written probs and raw logits.

    The variance is a constant `_NO_VAR`, not 0: at lambda = 0 it never enters a metric
    (sqrt(1 + 0 * var) == 1 exactly), but `_sweep_rows` also builds a lambda -> infinity row
    that divides by sigma.
    """
    arm = ArmData(label, ARMS["baseline"], "baseline")
    frame = load_predictions(str(_infer_dir(root, tag, label, "cifar100") / "predictions.csv"), fold="test")
    logits = torch.from_numpy(logits_array(frame)).to(torch.float64)
    col = next(c for c in LABEL_COLS if c in frame.df.columns)
    targets = torch.from_numpy(frame.df[col].to_numpy().astype(int)).long()
    arm.id = (logits / temperature, torch.full((len(logits), 1), _NO_VAR, dtype=torch.float64), targets)
    for ds in OOD_SETS:
        ood = load_predictions(str(_infer_dir(root, tag, label, ds) / "predictions.csv"), fold="test")
        ologits = torch.from_numpy(logits_array(ood)).to(torch.float64)
        arm.ood[ds] = (ologits / temperature, torch.full((len(ologits), 1), _NO_VAR, dtype=torch.float64))
    return arm, probs_array(frame), logits


def _score(arm: ArmData, lam: float, with_variance: bool) -> Dict[str, float]:
    row = next(r for r in _sweep_rows(arm, [lam]) if r["kind"] == "sweep")
    out = {k: v for k, v in row.items() if k in FIELDS}
    raw, var, _ = arm.id
    id_msp = _msp(raw, var, "softmax", lam)
    for ds, (oraw, ovar) in arm.ood.items():
        out[f"fpr95_msp_{ds}"] = fpr_at_95_tpr(id_msp, _msp(oraw, ovar, "softmax", lam))
        if with_variance:  # lambda-free; the baseline has no variance to rank by
            out[f"auroc_var_{ds}"] = float(
                AUROC(var.squeeze(1).numpy(), ovar.squeeze(1).numpy(), score_is_uncertainty=True)
            )
    return out


def _mean_sd(values: List[float], places: int = 4) -> str:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{places}f}"
    return f"{statistics.mean(vals):.{places}f} ± {statistics.stdev(vals):.{places}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="the new arm's tag, e.g. evidence_ls_2026-09-28_16-10-10")
    ap.add_argument("--fit-logs", required=True, type=Path, nargs="+", help="dirs holding fit_<arm>_s<seed>.log")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--expect-committed", action="store_true", help="gate the SNGP / SpecReg rows on CIFAR100_RESULTS.md")
    args = ap.parse_args()

    infer_root = Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "infer"
    rows: List[Dict[str, object]] = []
    for arm_key in args.arms:
        for seed in SEEDS:
            tag, label, fit_row = _source(arm_key, seed, args.tag)
            knob, trained, fitted = _fit(args.fit_logs, fit_row)
            if arm_key == "baseline":
                arm, published, logits = _baseline_arm(infer_root, tag, label, fitted)
                probs_trained = torch.softmax(logits / trained, dim=1).numpy()
                at_trained = _baseline_arm(infer_root, tag, label, trained)[0]
                scored, scored_trained = _score(arm, 0.0, False), _score(at_trained, 0.0, False)
            else:
                arm = _collect(infer_root, tag, label, ARMS[arm_key], arm_key)
                if arm is None or set(arm.ood) != set(OOD_SETS):
                    raise SystemExit(f"missing predictions for {arm_key} seed {seed} ({tag}_{label})")
                raw, var, _ = arm.id
                published = arm.published_probs
                probs_trained = torch.softmax(mean_field_scale(raw, var, trained), dim=1).numpy()
                scored, scored_trained = _score(arm, fitted, True), _score(arm, trained, False)

            gate = float(np.abs(probs_trained - published).max())
            if gate > 1e-4:
                raise SystemExit(f"{arm_key} s{seed}: offline softmax at {knob}={trained:g} does not reproduce class_probs ({gate:.1e})")
            if scored["acc"] != scored_trained["acc"]:
                raise SystemExit(f"{arm_key} s{seed}: accuracy moved with the calibration knob")
            print(f"gate {arm_key:9s} s{seed:<6d} {knob}={trained:g} -> {fitted:.4g} (val)   max |delta prob| {gate:.1e}")
            rows.append({"arm": arm_key, "seed": seed, "knob": knob, "trained": trained, "fitted": fitted, **scored})

    if args.expect_committed:
        for arm_key, want in COMMITTED_NLL.items():
            got = statistics.mean(r["nll"] for r in rows if r["arm"] == arm_key)
            if abs(got - want) > 5e-4:
                raise SystemExit(f"gate: {arm_key} val-fit NLL {got:.4f} != committed {want:.4f}")
            print(f"gate {arm_key}: val-fit NLL {got:.4f} reproduces committed {want:.4f}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv} ({len(rows)} rows)\n")

    by_arm = {a: [r for r in rows if r["arm"] == a] for a in args.arms}
    tables = [
        ("In-distribution (CIFAR-100 test)", [("acc", "Acc"), ("nll", "NLL"), ("brier", "Brier"), ("smece", "smECE"), ("fitted", "knob*")]),
        ("OOD AUROC", [("auroc_msp_cifar10", "MSP C-10"), ("auroc_msp_svhn", "MSP SVHN"), ("auroc_ds_cifar10", "DS C-10"),
                       ("auroc_ds_svhn", "DS SVHN"), ("auroc_var_cifar10", "Var C-10"), ("auroc_var_svhn", "Var SVHN"),
                       ("fpr95_msp_svhn", "FPR95 MSP SVHN ↓")]),
    ]
    for title, cols in tables:
        print(f"### {title}\n")
        print("| Arm | " + " | ".join(c for _, c in cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for a, rs in by_arm.items():
            cells = [_mean_sd([r.get(k) for r in rs], 3 if k == "fitted" else 4) for k, _ in cols]
            print(f"| {ARMS[a]} | " + " | ".join(cells) + " |")
        print()

    pairs = [(b, a) for a, b in (("sngp", "specreg"), ("specreg", "els")) if a in by_arm and b in by_arm]
    delta_cols = ["acc", "nll", "smece", "auroc_msp_cifar10", "auroc_msp_svhn", "auroc_ds_svhn", "auroc_var_cifar10", "auroc_var_svhn"]
    for hi, lo in pairs:
        print(f"### Paired per seed: {ARMS[hi]} − {ARMS[lo]}\n")
        print("| seed | " + " | ".join(delta_cols) + " |")
        print("|---|" + "---:|" * len(delta_cols))
        deltas = {k: [] for k in delta_cols}
        for seed in SEEDS:
            h = next(r for r in by_arm[hi] if r["seed"] == seed)
            l = next(r for r in by_arm[lo] if r["seed"] == seed)
            d = {k: h[k] - l[k] for k in delta_cols}
            for k, v in d.items():
                deltas[k].append(v)
            print(f"| {seed} | " + " | ".join(f"{v:+.4f}" for v in d.values()) + " |")
        print("| **mean** | " + " | ".join(f"{statistics.mean(v):+.4f}" for v in deltas.values()) + " |")
        signs = [max(sum(x > 0 for x in v), sum(x < 0 for x in v)) for v in deltas.values()]
        print("| sign-consistent | " + " | ".join(f"{s}/{len(SEEDS)}" for s in signs) + " |\n")


if __name__ == "__main__":
    main()
