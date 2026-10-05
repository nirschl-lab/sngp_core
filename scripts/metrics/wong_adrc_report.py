"""Score the ADRC study's final Wong runs: 5 arms x 5 seeds (`scripts/tmux/adrc_wong_final.sh`).

Reads the inference tree `scripts/inference/run_eval_suite_parallel.sh` writes per checkpoint
(`<infer>/<model.name>_wong/<run>/<dataset>/{metrics.json,predictions.csv}`, LEVELS="", all
from `best.ckpt`). Every GP arm was trained and is scored at mean_field_factor pi/8 -- the value
each checkpoint carries -- so `class_probs` is used as written; no post-hoc fit.

Per run: Wong test metrics from metrics.json, and Wong-vs-OOD AUROC for entropy (all arms) and
GP predictive variance (GP arms) under the frozen 10-subsample protocol of
src/metrics/calculate_ood_metrics.py. Per arm: mean and sample std (ddof=1) across the 5 seeds.

    uv run python scripts/metrics/wong_adrc_report.py --stamp 2026-10-05_10-05-27 \\
        --out-dir figures/wong_adrc
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.auc import AUROC, sample_rate, seeds  # noqa: E402
from src.metrics.calculate_ood_metrics import load_ood_scores  # noqa: E402

# arm -> (model.name, table label). Order is the table order.
ARMS: Dict[str, Tuple[str, str]] = {
    "baseline": ("baseline_sgd_classifier", "Baseline (linear head)"),
    "sngp": ("sngp_sgd_classifier", "SNGP, c = 1"),
    "bnsn": ("sngp_bnsn_sgd_classifier", "SNGP + BN-SN, c = 8"),
    "specreg": ("sngp_specreg_sgd_classifier", "SNGP + SpecReg, λ = 0.003"),
    "muon": ("sngp_muon_sgd_classifier", "GP head + Muon, wd = 0.01"),
}
SEEDS = [12345, 1, 2, 3, 4]
INDIST = "wong"
OOD = ["acevedo", "jung", "kather2016", "kather2018", "nirschl2018", "tang"]
ID_METRICS = ["acc", "f1", "ece", "nll", "brier"]


def _subsampled_auroc(id_scores: pd.Series, ood_scores: pd.Series) -> float:
    """Mean AUROC over the frozen subsample seeds, same as calculate_ood_metrics."""
    n = min(sample_rate, len(id_scores), len(ood_scores))
    return float(np.mean([
        AUROC(id_scores.sample(n, random_state=s), ood_scores.sample(n, random_state=s), score_is_uncertainty=True)
        for s in seeds
    ]))


def _variance(predictions_csv: Path) -> pd.Series:
    return pd.to_numeric(pd.read_csv(predictions_csv, usecols=["uncertainty"])["uncertainty"]).dropna()


def _run_row(run_dir: Path, arm: str, seed: int) -> Dict[str, float]:
    row: Dict[str, float] = {"arm": arm, "seed": seed}
    metrics = json.loads((run_dir / INDIST / "metrics.json").read_text())
    row.update({m: metrics[m] for m in ID_METRICS})
    _, id_ent = load_ood_scores(run_dir / INDIST / "predictions.csv")
    is_gp = arm != "baseline"
    id_var = _variance(run_dir / INDIST / "predictions.csv") if is_gp else None
    for name in OOD:
        _, ood_ent = load_ood_scores(run_dir / name / "predictions.csv")
        row[f"auroc_ent_{name}"] = _subsampled_auroc(id_ent, ood_ent)
        row[f"auroc_var_{name}"] = (
            _subsampled_auroc(id_var, _variance(run_dir / name / "predictions.csv")) if is_gp else np.nan
        )
    for kind in ("ent", "var"):
        row[f"auroc_{kind}_mean"] = float(np.mean([row[f"auroc_{kind}_{n}"] for n in OOD]))
    return row


def _fmt(values: pd.Series, scale: float, digits: int) -> str:
    v = values.to_numpy() * scale
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def _table(summary_rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    lines += ["| " + " | ".join(r) + " |" for r in summary_rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", required=True, help="Launcher stamp, e.g. 2026-10-05_10-05-27.")
    ap.add_argument("--infer-root", type=Path,
                    default=Path(os.environ.get("EXPERIMENTS_HOME", "")) / os.environ.get("PROJECT_NAME", "") / "infer")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    for arm, (model_name, _) in ARMS.items():
        for seed in SEEDS:
            run_dir = args.infer_root / f"{model_name}_wong" / f"{args.stamp}_{arm}_s{seed}"
            rows.append(_run_row(run_dir, arm, seed))
    runs = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "wong_adrc_runs.csv", index=False)

    groups = {arm: runs[runs.arm == arm] for arm in ARMS}
    id_rows = [
        [ARMS[a][1], _fmt(g.acc, 1, 4), _fmt(g.f1, 1, 4), _fmt(g.ece, 100, 2), _fmt(g.nll, 100, 2),
         _fmt(g.brier, 100, 2)]
        for a, g in groups.items()
    ]
    print(_table(id_rows, ["Model", "Accuracy ↑", "F1 ↑", "ECE (×10⁻²) ↓", "NLL (×10⁻²) ↓", "Brier (×10⁻²) ↓"]))
    cols = OOD + ["mean"]
    header = ["Model"] + [c.capitalize() if c != "mean" else "Mean" for c in cols]
    for kind, title in (("ent", "Entropy AUROC"), ("var", "GP-variance AUROC")):
        print(f"\n{title}")
        print(_table(
            [[ARMS[a][1]] + [_fmt(g[f"auroc_{kind}_{c}"], 1, 3) for c in cols]
             for a, g in groups.items() if not g[f"auroc_{kind}_mean"].isna().all()],
            header,
        ))
    print(f"\nPer-run CSV: {args.out_dir / 'wong_adrc_runs.csv'}")


if __name__ == "__main__":
    main()
