"""Decoupled GP variance on CIFAR-100: logits from the trained l = 20 head, variance from a
second random-feature map at a smaller length scale, fit post-hoc on a frozen backbone.

Under the Gaussian likelihood the SNGP predictive variance

    var(x) = phi(x)^T (ridge * I + sum_i phi_i phi_i^T)^-1 phi(x)

needs only the training *features* -- no labels, no classifier weights. So the length scale
that sets it can be chosen independently of the one the logits were trained with. This
keeps the end-to-end head's logits (and so its accuracy, exactly) and asks how
distance-aware a small-l variance is on the same backbone.

Per SpecReg (matched) backbone seed, from the head-swap feature cache
(`scripts/metrics/cifar100_rf_head_swap.py extract`), for l x rff_dim:
  * variance head: cos / orf, unscaled, ridge 1.0 (the CIFAR recipe's head, l varied), precision
    from one augmented train pass (`train_view0`, what the final training epoch accumulates);
  * variance-only OOD score: AUROC and FPR@95%TPR vs CIFAR-10 and SVHN;
  * combined: the original head's raw logits with the NEW variance in the mean-field
    correction, lambda* fitted on val NLL, reported on test (NLL, smECE, MSP / DS AUROC);
  * rank average: the original head's MSP uncertainty (1 - MSP at its own lambda*) and the new
    variance, ranked over the pooled ID + OOD set and averaged -- parameter-free, so nothing is
    tuned on OOD data -- to test whether the variance adds to MSP rather than only rivals it.
One `original` row per seed scores the trained head with its own variance -- the gate: it
must reproduce the head-swap page's original-head numbers (lambda* ~35.3, var SVHN ~0.41).

Accuracy is an invariant, not a result: the logits are the original head's.

    uv run python scripts/metrics/cifar100_decoupled_variance.py \\
        --csv figures/cifar100_decoupled_variance/cifar100_decoupled_variance_per_seed.csv
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.stats import rankdata

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)
sys.path.insert(0, str(Path(__file__).parent))
from cifar100_rf_head_swap import (  # noqa: E402
    OOD_SETS,
    SPECREG_CKPTS,
    _finalize_precision,
    _load,
    _make_head,
    _raw_and_var,
    default_cache_dir,
    fpr_at_95_tpr,
)

from src.metrics.auc import AUROC  # noqa: E402
from src.metrics.posthoc_calibration import fit_mean_field_factor, mean_field_scale  # noqa: E402
from src.metrics.smooth_ece import smECE_fast_compat  # noqa: E402
from src.metrics.uncertainty import dempster_shafer  # noqa: E402
from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess  # noqa: E402

LENGTH_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
RFF_DIMS = (1024, 4096, 16384)


def _variance_head(
    length_scale: float, rff_dim: int, seed: int, device: str,
    feature_map: str = "cos", random_feature_type: str = "orf",
) -> RandomFeatureGaussianProcess:
    """The study's variance head. The defaults (cos / orf) are the committed results; the evidence
    script overrides the feature map and coupling to score positive-feature heads."""
    torch.manual_seed(2_000 + seed)
    return RandomFeatureGaussianProcess(
        in_dim=640, num_classes=100, rff_dim=rff_dim, length_scale=length_scale, ridge_penalty=1.0,
        normalize_input=False, scale_random_features=False, cov_momentum=-1.0,
        random_feature_type=random_feature_type, feature_map=feature_map,
    ).to(device)


@torch.no_grad()
def _invert_precision_fp64(head: RandomFeatureGaussianProcess) -> None:
    """Posterior covariance via a float64 Cholesky, cached on the head.

    At large rff_dim and large l the precision's top eigenvalues reach ~1e8 against the unit
    ridge, so the head's own float32 Cholesky fails and it falls back to a float64 `pinv` --
    an SVD that takes many minutes at rff_dim 16384. The precision is positive definite
    (ridge > 0), so a float64 Cholesky gives the same inverse in seconds.
    """
    eye = torch.eye(head.rff_dim, dtype=torch.float64, device=head.precision_accum.device)
    precision = head.ridge * eye + head.precision_accum.double()
    head.covariance.copy_(torch.cholesky_inverse(torch.linalg.cholesky(precision)).to(head.covariance.dtype))
    head._cov_stale.fill_(False)


def _score(
    raw: Dict[str, torch.Tensor], var: Dict[str, torch.Tensor], y_val: torch.Tensor, y_test: torch.Tensor
) -> dict:
    """Variance-only OOD scores, then the combined mean-field predictive at lambda*(val)."""
    row: dict = {"mean_var_test": var["test"].mean().item()}
    for ood in OOD_SETS:  # higher variance = more OOD
        row[f"auroc_var_{ood}"] = AUROC(var["test"].numpy(), var[ood].numpy(), score_is_uncertainty=True)
        row[f"fpr95_var_{ood}"] = fpr_at_95_tpr(-var["test"].numpy(), -var[ood].numpy())

    lam, _ = fit_mean_field_factor(raw["val"], var["val"].unsqueeze(1), y_val)

    def logits(name: str) -> torch.Tensor:
        return mean_field_scale(raw[name], var[name].unsqueeze(1), lam)

    lt = logits("test")
    probs = lt.softmax(1)
    conf, pred = probs.max(1)
    correct = (pred == y_test).double()
    assert torch.equal(pred, raw["test"].argmax(1)), "mean-field correction changed argmax"
    row.update(
        lam=lam, acc=correct.mean().item(), nll=F.cross_entropy(lt, y_test).item(),
        smece=float(smECE_fast_compat(conf.numpy(), correct.numpy())),
    )
    for ood in OOD_SETS:
        lo = logits(ood)
        row[f"auroc_msp_{ood}"] = AUROC(conf.numpy(), lo.softmax(1).max(1).values.numpy())
        row[f"auroc_ds_{ood}"] = AUROC(dempster_shafer(lt).numpy(), dempster_shafer(lo).numpy(), score_is_uncertainty=True)
    return row


def _msp_uncertainty(raw: Dict[str, torch.Tensor], var: Dict[str, torch.Tensor], lam: float) -> Dict[str, np.ndarray]:
    """1 - MSP of the mean-field predictive, per split. Higher = more OOD."""
    return {n: 1.0 - mean_field_scale(raw[n], var[n].unsqueeze(1), lam).softmax(1).max(1).values.numpy() for n in raw}


def _rank_average(msp_u: Dict[str, np.ndarray], var: Dict[str, torch.Tensor]) -> dict:
    row = {}
    for ood in OOD_SETS:
        n_id = len(msp_u["test"])
        a = rankdata(np.concatenate([msp_u["test"], msp_u[ood]]))
        b = rankdata(np.concatenate([var["test"].numpy(), var[ood].numpy()]))
        comb = (a + b) / 2
        row[f"auroc_rank_{ood}"] = AUROC(comb[:n_id], comb[n_id:], score_is_uncertainty=True)
        row[f"fpr95_rank_{ood}"] = fpr_at_95_tpr(-comb[:n_id], -comb[n_id:])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(default_cache_dir()))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SPECREG_CKPTS))
    ap.add_argument("--length-scales", type=float, nargs="+", default=list(LENGTH_SCALES))
    ap.add_argument("--rff-dims", type=int, nargs="+", default=list(RFF_DIMS))
    ap.add_argument("--csv", required=True, type=Path)
    args = ap.parse_args()

    device = "cuda"
    cache = Path(args.cache_dir)
    rows: List[dict] = []
    t0 = time.time()
    for seed in args.seeds:
        h_train, _ = _load(cache, seed, "train_view0", device)
        data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {n: _load(cache, seed, n, device) for n in ("val", "test", *OOD_SETS)}
        y_val, y_test = data["val"][1].cpu(), data["test"][1].cpu()
        rho = (data["test"][0].norm(dim=1)).median().item()

        original = _make_head("cos", "orf", 20.0, seed, device)
        original.load_state_dict(torch.load(cache / f"seed{seed}" / "original_gp_head.pt"))
        raw, own_var = {}, {}
        for name, (h, _) in data.items():
            raw[name], own_var[name], _ = _raw_and_var(original, h)
        orig_row = _score(raw, own_var, y_val, y_test)
        msp_u = _msp_uncertainty(raw, own_var, orig_row["lam"])
        for ood in OOD_SETS:
            orig_row[f"fpr95_msp_{ood}"] = fpr_at_95_tpr(-msp_u["test"], -msp_u[ood])
        rows.append(dict(seed=seed, variance="original", length_scale=20.0, rff_dim=1024, **orig_row))
        logger.info(f"seed {seed} original: lam {rows[-1]['lam']:.3g} nll {rows[-1]['nll']:.4f} var SVHN {rows[-1]['auroc_var_svhn']:.4f}  (median ||x|| {rho:.2f})")

        for m in args.rff_dims:
            for l in args.length_scales:
                head = _variance_head(l, m, seed, device)
                _finalize_precision(head, h_train)
                _invert_precision_fp64(head)
                var = {name: _raw_and_var(head, h)[1] for name, (h, _) in data.items()}
                row = dict(seed=seed, variance="decoupled", length_scale=l, rff_dim=m, **_score(raw, var, y_val, y_test), **_rank_average(msp_u, var))
                rows.append(row)
                logger.info(
                    f"seed {seed} m={m:5d} l={l:4g} (rho {rho / l:5.2f}): var C10 {row['auroc_var_cifar10']:.4f} "
                    f"SVHN {row['auroc_var_svhn']:.4f} | lam {row['lam']:.3g} nll {row['nll']:.4f} smece {row['smece']:.4f} "
                    f"MSP SVHN {row['auroc_msp_svhn']:.4f} DS SVHN {row['auroc_ds_svhn']:.4f} | rank SVHN {row['auroc_rank_svhn']:.4f} ({time.time() - t0:.0f}s)"
                )

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    metrics = ["auroc_var_cifar10", "auroc_var_svhn", "fpr95_var_cifar10", "fpr95_var_svhn",
               "auroc_rank_cifar10", "auroc_rank_svhn", "fpr95_rank_cifar10", "fpr95_rank_svhn",
               "fpr95_msp_cifar10", "fpr95_msp_svhn", "lam", "acc", "nll", "smece",
               "auroc_msp_cifar10", "auroc_msp_svhn", "auroc_ds_cifar10", "auroc_ds_svhn"]
    summary = df.groupby(["variance", "rff_dim", "length_scale"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(args.csv.with_name(args.csv.stem.replace("_per_seed", "") + "_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summary[[f"{m}_mean" for m in metrics]].round(4).to_string())
    logger.info(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
