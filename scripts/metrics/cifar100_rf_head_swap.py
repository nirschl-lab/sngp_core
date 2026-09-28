"""Head-swap study on CIFAR-100: positive / hyperbolic random features x ORF / SimRF.

The kernel study (`scripts/metrics/random_feature_kernel_mse.py`) found that SimRF cuts
kernel MSE to ~0.1x ORF with *positive* random features at rho = ||x||/l ~ 0.5, while
merely tying ORF for the cos features SNGP uses. The CIFAR-100 GP head (no input norm,
l = 20) sits at rho ~ 0.57, the favourable regime. This asks whether the better kernel
estimate buys calibration or OOD detection.

Head-only, by design. The three SpecReg (matched) backbones are frozen, their 640-d GP
inputs cached once, and only the GP head is retrained per arm:

    feature_map {cos, positive, hyperbolic} x coupling {orf, simrf} x l {10, 20, 40}

Every head is the CIFAR recipe's (`rff_dim` 1024, ridge 1.0, unscaled features, no input
norm) with only those three knobs varied. Every arm at a given backbone seed draws its
random features from the same RNG state, and all arms share one optimizer recipe:
SGD + Nesterov with (LR, weight decay) picked once, on the cos/orf/l=20 control only, by
validation NLL at its own lambda*, then frozen for every arm and seed. The recipe's own
weight decay (6e-4) is in that grid but loses: on a frozen backbone it caps the logit
scale and leaves the head *under*confident (lambda* = 0), where the end-to-end head had
co-scaled its features for 250 epochs. The comparison is always against the *retrained*
control, never against the original head. The original head is scored through the same
pipeline only as a gate: the retrained control has to match its accuracy, and how far it
falls short on calibration/OOD is the measured cost of head-only training.

Per head: lambda* = argmin val NLL of the mean-field predictive (`fit_mean_field_factor`).
On test it reports accuracy, NLL, Brier, top-label smECE, and AUROC / FPR@95%TPR vs
CIFAR-10 and SVHN (full sets) for three scores: MSP at lambda*, Dempster-Shafer at
lambda*, and the raw GP variance.

Caveat, stated in the report too: the backbones co-adapted with a cos head for 250
epochs, so this measures the head's contribution on a cos-shaped representation, not what
end-to-end training with each feature map would reach.

Usage:

    # 1. cache features (GPU, ~minutes per seed)
    uv run python scripts/metrics/cifar100_rf_head_swap.py extract
    # ... or cache another arm's backbones into their own cache dir
    uv run python scripts/metrics/cifar100_rf_head_swap.py extract --cache-dir <dir> \\
        --ckpts 12345=<last.ckpt> 1=<last.ckpt> 2=<last.ckpt>
    # 2. retrain heads and score them (GPU)
    uv run python scripts/metrics/cifar100_rf_head_swap.py sweep \\
        --csv figures/cifar100_rf_head_swap/cifar100_rf_head_swap_per_seed.csv
"""
import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.checkpointing.io import load_net  # noqa: E402
from src.metrics.auc import AUROC  # noqa: E402
from src.metrics.brier import brier_score  # noqa: E402
from src.metrics.posthoc_calibration import fit_mean_field_factor, mean_field_scale  # noqa: E402
from src.metrics.smooth_ece import smECE_fast_compat  # noqa: E402
from src.metrics.uncertainty import dempster_shafer  # noqa: E402
from src.models.sngp.sngp_classifier import RandomFeatureGaussianProcess  # noqa: E402

_RUNS = "/data1/maheswararao/experiments/uncertainty-aware-ml"
_B = f"{_RUNS}/overnight/overnight_2026-09-20_21-38-42"
# SpecReg (matched) last.ckpt per seed -- docs/checkpoints/CIFAR_CHECKPOINTS.md. The seed-12345
# file is the matched arm's (spec_reg_burnin_epochs = 1), despite the run-directory collision.
SPECREG_CKPTS: Dict[int, str] = {
    12345: f"{_RUNS}/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt",
    1: f"{_B}/s1_specreg/checkpoints/last.ckpt",
    2: f"{_B}/s2_specreg/checkpoints/last.ckpt",
}
OOD_SETS = ("cifar10", "svhn")
FEATURE_MAPS = ("cos", "positive", "hyperbolic")
COUPLINGS = ("orf", "simrf")
LENGTH_SCALES = (10.0, 20.0, 40.0)
CONTROL = ("cos", "orf", 20.0)
HEAD_KW = dict(rff_dim=1024, ridge_penalty=1.0, normalize_input=False, scale_random_features=False, cov_momentum=-1.0)


def default_cache_dir() -> Path:
    return Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "feature_cache" / "cifar100_rf_head_swap"


# -- extract ----------------------------------------------------------------------------


def _datamodule(dataset: str, stage: str):
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose("train.yaml", overrides=["experiment=sngp_specreg_cifar100", f"data={dataset}"])
    # Mirror src/train.py: the `cifar32` preset (CIFAR normalization; pad/crop/flip for
    # train) is passed in explicitly. Without it every split silently falls back to the
    # ImageNet-normalization default and the train split is not augmented at all.
    aug = cfg.data.img_augmentations
    dm = hydra.utils.instantiate(
        cfg.data.datamodule,
        train_augmentations=hydra.utils.instantiate(aug.train),
        val_augmentations=hydra.utils.instantiate(aug.val),
        test_augmentations=hydra.utils.instantiate(aug.test),
    )
    dm.setup(stage)
    return dm


@torch.no_grad()
def _features(net, loader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backbone features and labels. Batches are `[ids, image, label]`."""
    feats, labels = [], []
    for batch in loader:
        h = net.backbone(batch[1].to(device, non_blocking=True))
        feats.append(h.flatten(1).half().cpu())
        labels.append(batch[2].cpu())
    return torch.cat(feats), torch.cat(labels)


def _parse_ckpts(items: List[str]) -> Dict[int, str]:
    """`seed=path` pairs -> {seed: path}, for caching backbones other than `SPECREG_CKPTS`."""
    out: Dict[int, str] = {}
    for item in items:
        seed, sep, path = item.partition("=")
        if not sep or not Path(path).is_file():
            raise SystemExit(f"--ckpts expects seed=<existing checkpoint>, got {item!r}")
        out[int(seed)] = path
    return out


def extract(args) -> None:
    device = "cuda"
    cache = Path(args.cache_dir)
    ckpts = _parse_ckpts(args.ckpts) if args.ckpts else SPECREG_CKPTS
    seeds = list(ckpts) if args.ckpts else args.seeds
    train_dm, test_dm = _datamodule("cifar100", "fit"), _datamodule("cifar100", "test")
    ood_dms = {ds: _datamodule(ds, "test") for ds in OOD_SETS}
    for seed in seeds:
        out = cache / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        net = load_net(ckpts[seed], device=device).eval()
        l = net.length_scale
        splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for k in range(args.train_views):
            splits[f"train_view{k}"] = _features(net, train_dm.train_dataloader(), device)
        splits["val"] = _features(net, train_dm.val_dataloader(), device)
        splits["test"] = _features(net, test_dm.test_dataloader(), device)
        for ds, dm in ood_dms.items():
            splits[ds] = _features(net, dm.test_dataloader(), device)
        for name, (h, y) in splits.items():
            torch.save({"h": h, "y": y}, out / f"{name}.pt")
            q = (h.float().norm(dim=1) / l).quantile(torch.tensor([0.05, 0.5, 0.95]))
            logger.info(f"seed {seed} {name:12s} n={len(y):6d} rho p5/med/p95 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}")
        # The original head, for gate 1: same scoring pipeline as the retrained ones.
        torch.save(net.gp_head.state_dict(), out / "original_gp_head.pt")


# -- sweep ------------------------------------------------------------------------------


def _load(cache: Path, seed: int, name: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    d = torch.load(cache / f"seed{seed}" / f"{name}.pt")
    return d["h"].to(device).float(), d["y"].to(device)


def _make_head(feature_map: str, coupling: str, length_scale: float, seed: int, device: str) -> RandomFeatureGaussianProcess:
    # Same RNG state for every arm at a backbone seed: arms differ only in their knobs.
    torch.manual_seed(1_000 + seed)
    return RandomFeatureGaussianProcess(
        in_dim=640, num_classes=100, length_scale=length_scale, random_feature_type=coupling,
        feature_map=feature_map, **HEAD_KW,
    ).to(device)


def _train_head(
    head, views: List[Tuple[torch.Tensor, torch.Tensor]], lr: float, wd: float, epochs: int, batch: int
) -> None:
    """SGD + Nesterov with a cosine LR on the head alone. Epoch e uses augmented view
    e mod K -- one fresh augmentation per image per epoch."""
    opt = torch.optim.SGD(head.classifier.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    steps = epochs * math.ceil(len(views[0][1]) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    head.train()
    for epoch in range(epochs):
        h, y = views[epoch % len(views)]
        perm = torch.randperm(len(y), device=h.device)
        for i in range(0, len(y), batch):
            idx = perm[i : i + batch]
            with torch.no_grad():
                phi = head._features(h[idx])
            loss = F.cross_entropy(head.classifier(phi), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()


@torch.no_grad()
def _finalize_precision(head, h: torch.Tensor, batch: int = 4096) -> None:
    """One exact pass of `sum_i w_i phi_i phi_i^T` over an augmented train view -- what
    the lit module's final epoch accumulates."""
    head.reset_precision()
    for i in range(0, len(h), batch):
        phi = head._features(h[i : i + batch])
        head.update_precision(phi, head.classifier(phi))


@torch.no_grad()
def _raw_and_var(head, h: torch.Tensor, batch: int = 4096) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    head.eval()
    raws, vars_, finite = [], [], True
    for i in range(0, len(h), batch):
        phi = head._features(h[i : i + batch])
        finite &= bool(torch.isfinite(phi).all())
        raws.append(head.classifier(phi))
        vars_.append(head.predictive_variance(phi))
    return torch.cat(raws).cpu(), torch.cat(vars_).squeeze(1).cpu(), finite


def fpr_at_95_tpr(id_conf: np.ndarray, ood_conf: np.ndarray) -> float:
    """Fraction of OOD inputs above the confidence threshold that keeps 95% of ID inputs."""
    return float((ood_conf >= np.quantile(id_conf, 0.05)).mean())


def score_head(head, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> dict:
    out = {name: _raw_and_var(head, h) for name, (h, _) in data.items() if name != "train"}
    raw_val, var_val, _ = out["val"]
    lam, _ = fit_mean_field_factor(raw_val, var_val, data["val"][1].cpu())

    def logits(name):
        raw, var, _ = out[name]
        return mean_field_scale(raw, var, lam)

    y = data["test"][1].cpu()
    lt = logits("test")
    probs = lt.softmax(1)
    conf, pred = probs.max(1)
    correct = (pred == y).double()
    assert torch.equal(pred, out["test"][0].argmax(1)), "mean-field correction changed argmax"  # gate 3
    row = dict(
        lam=lam,
        acc=correct.mean().item(),
        nll=F.cross_entropy(lt, y).item(),
        brier=brier_score(probs, y, probs.shape[1]),
        smece=float(smECE_fast_compat(conf.numpy(), correct.numpy())),
        all_finite=all(v[2] for v in out.values()),
        mean_var_test=out["test"][1].mean().item(),
    )
    id_scores = {"msp": conf.numpy(), "ds": -dempster_shafer(lt).numpy(), "var": -out["test"][1].numpy()}
    for ood in OOD_SETS:
        lo = logits(ood)
        ood_scores = {"msp": lo.softmax(1).max(1).values.numpy(), "ds": -dempster_shafer(lo).numpy(), "var": -out[ood][1].numpy()}
        for s in id_scores:  # all as confidences: higher = more in-distribution
            row[f"auroc_{s}_{ood}"] = AUROC(-id_scores[s], -ood_scores[s], score_is_uncertainty=True)
            row[f"fpr95_{s}_{ood}"] = fpr_at_95_tpr(id_scores[s], ood_scores[s])
    return row


def sweep(args) -> None:
    device = "cuda"
    cache = Path(args.cache_dir)
    rows: List[dict] = []
    t0 = time.time()
    for seed in args.seeds:
        views = [_load(cache, seed, f"train_view{k}", device) for k in range(args.train_views)]
        data = {name: _load(cache, seed, name, device) for name in ("val", "test", *OOD_SETS)}

        original = _make_head("cos", "orf", 20.0, seed, device)
        original.load_state_dict(torch.load(cache / f"seed{seed}" / "original_gp_head.pt"))
        rows.append(dict(seed=seed, feature_map="cos", coupling="orf", length_scale=20.0, arm="original_head", lr=np.nan, wd=np.nan, **score_head(original, data)))
        logger.info(f"seed {seed} original head: acc {rows[-1]['acc']:.4f} MSP-SVHN {rows[-1]['auroc_msp_svhn']:.4f}")

        if args.lr is None or args.wd is None:
            # Pick once, on the control, by val NLL at its own lambda* -- then frozen for
            # every arm and every remaining seed. Test is never looked at.
            best = None
            for cand_lr in args.lr_grid:
                for cand_wd in args.wd_grid:
                    head = _make_head(*CONTROL, seed, device)
                    _train_head(head, views, cand_lr, cand_wd, args.epochs, args.batch_size)
                    _finalize_precision(head, views[0][0])
                    raw_val, var_val, _ = _raw_and_var(head, data["val"][0])
                    lam, val_nll = fit_mean_field_factor(raw_val, var_val, data["val"][1].cpu())
                    logger.info(f"seed {seed} lr {cand_lr} wd {cand_wd:g}: control val NLL {val_nll:.4f} at lambda* {lam:.3g}")
                    if best is None or val_nll < best[2]:
                        best = (cand_lr, cand_wd, val_nll)
            args.lr, args.wd = best[0], best[1]
            logger.info(f"Frozen: lr {args.lr}, wd {args.wd:g} (control val NLL {best[2]:.4f})")
        lr, wd = args.lr, args.wd

        for fm in args.feature_maps:
            for coupling in args.couplings:
                for l in args.length_scales:
                    head = _make_head(fm, coupling, l, seed, device)
                    _train_head(head, views, lr, wd, args.epochs, args.batch_size)
                    _finalize_precision(head, views[0][0])
                    row = dict(seed=seed, feature_map=fm, coupling=coupling, length_scale=l, arm="retrained", lr=lr, wd=wd, **score_head(head, data))
                    rows.append(row)
                    logger.info(
                        f"seed {seed} {fm:>10}/{coupling:<5} l={l:4.0f}: acc {row['acc']:.4f} nll {row['nll']:.4f} "
                        f"smece {row['smece']:.4f} lam {row['lam']:.3g} | MSP C10 {row['auroc_msp_cifar10']:.4f} "
                        f"SVHN {row['auroc_msp_svhn']:.4f} | var SVHN {row['auroc_var_svhn']:.4f} "
                        f"{'' if row['all_finite'] else 'NONFINITE '}({time.time() - t0:.0f}s)"
                    )
        pd.DataFrame(rows).to_csv(args.csv, index=False)  # checkpoint progress per seed

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    summary = summarize(df)
    summary_path = Path(args.csv).with_name(Path(args.csv).stem.replace("_per_seed", "") + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote {args.csv} ({len(df)} rows) and {summary_path}")


METRICS = (
    "acc", "nll", "brier", "smece", "lam",
    "auroc_msp_cifar10", "auroc_msp_svhn", "auroc_ds_cifar10", "auroc_ds_svhn", "auroc_var_cifar10", "auroc_var_svhn",
    "fpr95_msp_cifar10", "fpr95_msp_svhn", "fpr95_ds_cifar10", "fpr95_ds_svhn", "fpr95_var_cifar10", "fpr95_var_svhn",
)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across backbone seeds per arm, plus the paired per-seed delta vs the
    retrained control and how many seeds agree in sign with it."""
    ctrl = df[(df.arm == "retrained") & (df.feature_map == CONTROL[0]) & (df.coupling == CONTROL[1]) & (df.length_scale == CONTROL[2])]
    ctrl = ctrl.set_index("seed")
    out = []
    for (arm, fm, coupling, l), g in df.groupby(["arm", "feature_map", "coupling", "length_scale"]):
        row = dict(arm=arm, feature_map=fm, coupling=coupling, length_scale=l, n_seeds=len(g))
        g = g.set_index("seed")
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1)
            delta = g[m] - ctrl[m].reindex(g.index)
            row[f"{m}_delta"] = delta.mean()
            row[f"{m}_delta_pos"] = int((delta > 0).sum())
            row[f"{m}_delta_neg"] = int((delta < 0).sum())
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--cache-dir", default=str(default_cache_dir()))
        p.add_argument("--seeds", type=int, nargs="+", default=list(SPECREG_CKPTS))
        p.add_argument("--train-views", type=int, default=4, help="augmented views of the train set to cache / cycle")
    sub.choices["extract"].add_argument(
        "--ckpts", nargs="+", default=None,
        help="seed=path pairs to cache instead of SPECREG_CKPTS (use a separate --cache-dir); overrides --seeds",
    )
    sp = sub.choices["sweep"]
    sp.add_argument("--csv", default=str(ROOT / "figures" / "cifar100_rf_head_swap" / "cifar100_rf_head_swap_per_seed.csv"))
    sp.add_argument("--feature-maps", nargs="+", default=list(FEATURE_MAPS))
    sp.add_argument("--couplings", nargs="+", default=list(COUPLINGS))
    sp.add_argument("--length-scales", type=float, nargs="+", default=list(LENGTH_SCALES))
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--batch-size", type=int, default=128)
    sp.add_argument("--lr", type=float, default=None, help="with --wd, skip the pick and use these")
    sp.add_argument("--wd", type=float, default=None)
    sp.add_argument("--lr-grid", type=float, nargs="+", default=[0.01, 0.04, 0.1])
    sp.add_argument("--wd-grid", type=float, nargs="+", default=[6e-4, 1e-4, 0.0], help="6e-4 is the recipe's")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args)
    else:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        sweep(args)


if __name__ == "__main__":
    main()
