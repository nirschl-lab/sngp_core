"""GP evidence over the length scale on CIFAR-100: an in-distribution-only rule for choosing l.

`docs/results/CIFAR100_DECOUPLED_VARIANCE_RESULTS.md` found the variance AUROC peaks at
l ~ 3 on the frozen SpecReg backbones, but read that l off the *test OOD sets*. This asks
whether the GP's own model-selection criterion -- the log marginal likelihood of the
random-feature Bayesian linear model under the Gaussian likelihood, the same model whose
posterior gives the SNGP variance -- lands there using the training set alone.

Per SpecReg (matched) backbone seed, from the head-swap feature cache
(`scripts/metrics/cifar100_rf_head_swap.py extract`), for l x rff_dim:
  * features of the decoupled study's variance head (`_variance_head`: cos / orf, unscaled,
    no input norm, same seed), on the augmented train pass `train_view0`;
  * log evidence per datum at the recipe's (alpha = ridge = 1, s = 1), and at type-II
    (alpha, s) by MacKay's fixed point (`src/metrics/gp_evidence.py`);
  * the same type-II evidence on the held-out *val* features. The backbone was trained on
    the train images and all but interpolates them (type-II noise s ~ 1e-3), so train
    evidence mostly measures fit; val features never shaped the backbone. Still ID-only;
  * joined to the decoupled study's variance AUROC at the same (l, rff_dim), when present.
One `median` row per seed gives the median-heuristic l of the train features.

    uv run python scripts/metrics/cifar100_length_scale_evidence.py \\
        --csv figures/cifar100_length_scale_evidence/cifar100_length_scale_evidence_per_seed.csv
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
import rootutils
import torch
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)
sys.path.insert(0, str(Path(__file__).parent))
from cifar100_decoupled_variance import _variance_head  # noqa: E402
from cifar100_rf_head_swap import SPECREG_CKPTS, _load, default_cache_dir  # noqa: E402

from src.metrics.gp_evidence import evidence_basis, log_evidence, median_heuristic, optimize_hyperparameters  # noqa: E402

LENGTH_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0)
RFF_DIMS = (1024, 4096)
DECOUPLED_SUMMARY = ROOT / "figures" / "cifar100_decoupled_variance" / "cifar100_decoupled_variance_summary.csv"


@torch.no_grad()
def _head_features(head, h: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    head.eval()
    return torch.cat([head._features(h[i : i + batch]) for i in range(0, len(h), batch)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(default_cache_dir()))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SPECREG_CKPTS))
    ap.add_argument("--length-scales", type=float, nargs="+", default=list(LENGTH_SCALES))
    ap.add_argument("--rff-dims", type=int, nargs="+", default=list(RFF_DIMS))
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument(
        "--no-auroc-join", action="store_true",
        help="skip joining the decoupled study's variance AUROC, which was measured on the l = 20 "
             "SpecReg backbones and so does not describe any other --cache-dir",
    )
    args = ap.parse_args()

    device = "cuda"
    cache = Path(args.cache_dir)
    rows: List[dict] = []
    t0 = time.time()
    for seed in args.seeds:
        h_train, y_train = _load(cache, seed, "train_view0", device)
        h_val, y_val = _load(cache, seed, "val", device)
        n = len(h_train)
        norm = h_train.norm(dim=1).median().item()
        l_med = median_heuristic(h_train, seed=seed)
        rows.append(dict(seed=seed, kind="median", length_scale=l_med, rff_dim=0, median_norm=norm))
        logger.info(f"seed {seed}: N {n}, median ||x|| {norm:.2f}, median-heuristic l {l_med:.2f}")

        for m in args.rff_dims:
            for l in args.length_scales:
                head = _variance_head(l, m, seed, device)
                basis = evidence_basis(_head_features(head, h_train), y_train, num_classes=100)
                recipe = log_evidence(basis, 1.0, 1.0)
                alpha, noise, opt = optimize_hyperparameters(basis)
                vbasis = evidence_basis(_head_features(head, h_val), y_val, num_classes=100)
                v_alpha, v_noise, v_opt = optimize_hyperparameters(vbasis)
                rows.append(dict(
                    seed=seed, kind="evidence", length_scale=l, rff_dim=m, rho=norm / l,
                    log_ev_recipe=recipe / n, log_ev_type2=opt / n, alpha=alpha, noise=noise,
                    log_ev_val_type2=v_opt / len(h_val), val_alpha=v_alpha, val_noise=v_noise,
                    median_norm=norm,
                ))
                logger.info(
                    f"seed {seed} m={m:5d} l={l:4g} (rho {norm / l:5.2f}): log-ev/N recipe {recipe / n:.4f} "
                    f"type-II {opt / n:.4f} (alpha {alpha:.3g}, s {noise:.3g}) | val type-II {v_opt / len(h_val):.4f} "
                    f"(alpha {v_alpha:.3g}, s {v_noise:.3g}) ({time.time() - t0:.0f}s)"
                )

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    ev = df[df.kind == "evidence"]
    metrics = ["log_ev_recipe", "log_ev_type2", "alpha", "noise", "log_ev_val_type2", "val_alpha", "val_noise", "rho"]
    summary = ev.groupby(["rff_dim", "length_scale"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    if DECOUPLED_SUMMARY.exists() and not args.no_auroc_join:
        dec = pd.read_csv(DECOUPLED_SUMMARY)
        dec = dec[dec.variance == "decoupled"][["rff_dim", "length_scale", "auroc_var_cifar10_mean", "auroc_var_svhn_mean"]]
        summary = summary.merge(dec, on=["rff_dim", "length_scale"], how="left")
    summary.to_csv(args.csv.with_name(args.csv.stem.replace("_per_seed", "") + "_summary.csv"), index=False)

    # Argmax per seed, so the reported l carries its own seed spread.
    best = ev.loc[ev.groupby(["seed", "rff_dim"])[["log_ev_recipe"]].idxmax().values.ravel()]
    best2 = ev.loc[ev.groupby(["seed", "rff_dim"])[["log_ev_type2"]].idxmax().values.ravel()]
    best_val = ev.loc[ev.groupby(["seed", "rff_dim"])[["log_ev_val_type2"]].idxmax().values.ravel()]
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string())
    print("\nargmax l, recipe (alpha, s) = (1, 1):\n", best[["seed", "rff_dim", "length_scale"]].to_string(index=False))
    print("\nargmax l, type-II (alpha, s):\n", best2[["seed", "rff_dim", "length_scale"]].to_string(index=False))
    print("\nargmax l, type-II on val:\n", best_val[["seed", "rff_dim", "length_scale"]].to_string(index=False))
    print("\nmedian heuristic:\n", df[df.kind == "median"][["seed", "length_scale", "median_norm"]].to_string(index=False))
    logger.info(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
