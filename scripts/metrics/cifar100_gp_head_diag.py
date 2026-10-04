"""Feature scale the GP kernel sees, per CIFAR-100 SNGP checkpoint: does the backbone absorb l?

The RBF kernel only sees ||h - h'|| / l, so a backbone free to rescale h can undo any fixed l (in the
learnable-l study, ||h|| co-shrank with l and rho = ||h|| / l locked at ~2.2). Measured on the first
N val images (val augmentations, as training preprocessed them) at each checkpoint:

  * ||h||       mean backbone feature norm (the GP head's raw input);
  * ||h_in||/l  the same after the head's LayerNorm, if any, over l: the kernel's input scale;
  * d/l         mean pairwise ||h_in - h_in'|| / l over same-class and different-class pairs, and
                the mean exact RBF kernel exp(-d^2 / 2 l^2) over the same pairs;
  * ||beta||_F  the GP output layer's weight norm (what the val-fit mean_field_factor tracks);
  * final-BN |gamma|, the LayerNorm's mean |gain| (if any), and the mean max |raw logit|.

    uv run python scripts/metrics/cifar100_gp_head_diag.py --tag cosine_2026-10-01_10-17-06 \\
        --labels cos_muonsgd_wd0.1_s12345 cos_muonsgd_ln_l20_wd0.1_s12345 \\
        --csv figures/cifar100_cosine/cifar100_gp_head_diag.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import rootutils
import torch
from torch import nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)
sys.path.insert(0, str(Path(__file__).parents[1] / "checkpoints"))
from calibrate_checkpoint import _build_loader  # noqa: E402

from src.checkpointing.io import load_net  # noqa: E402

FIELDS = [
    "label", "length_scale", "h_norm", "h_in_norm_over_l", "d_over_l_same", "d_over_l_diff",
    "kernel_same", "kernel_diff", "beta_fro", "final_bn_abs_gamma", "ln_abs_gain", "max_abs_raw_logit",
]


@torch.no_grad()
def _val_batch(experiment: str, n: int, device: str):
    """The first `n` val images and labels. Batches are `[ids, image, label]`."""
    loader, _ = _build_loader(experiment, "val", 250)
    xs, ys = [], []
    for batch in loader:
        xs.append(batch[1])
        ys.append(batch[2])
        if sum(len(x) for x in xs) >= n:
            break
    return torch.cat(xs)[:n].to(device), torch.cat(ys)[:n].to(device)


def _final_bn(backbone: nn.Module) -> nn.BatchNorm2d:
    """WRN's post-block BN (`backbone.bn`), else the last BatchNorm2d registered."""
    bn = getattr(backbone, "bn", None)
    if isinstance(bn, nn.BatchNorm2d):
        return bn
    return [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)][-1]


@torch.no_grad()
def diagnose(net: nn.Module, x: torch.Tensor, y: torch.Tensor, n_pairs: int = 1000,
             chunk: int = 250) -> Dict[str, float]:
    """The feature-scale numbers for one net on images `x` with labels `y`."""
    gp = net.gp_head
    ls = float(gp.length_scale)
    h = torch.cat([net.pooled_features(x[i:i + chunk]) for i in range(0, len(x), chunk)])
    h_in = gp.input_norm(h) if gp.input_norm is not None else h
    raw = gp.classifier(gp._features(h_in))

    d = torch.cdist(h_in[:n_pairs], h_in[:n_pairs]) / ls
    yp = y[:n_pairs]
    same = (yp[:, None] == yp[None, :]) & ~torch.eye(len(yp), dtype=torch.bool, device=y.device)
    diff = yp[:, None] != yp[None, :]
    k = torch.exp(-0.5 * d**2)
    ln = gp.input_norm
    return {
        "length_scale": ls,
        "h_norm": h.norm(dim=1).mean().item(),
        "h_in_norm_over_l": (h_in.norm(dim=1).mean() / ls).item(),
        "d_over_l_same": d[same].mean().item(),
        "d_over_l_diff": d[diff].mean().item(),
        "kernel_same": k[same].mean().item(),
        "kernel_diff": k[diff].mean().item(),
        "beta_fro": gp.classifier.weight.norm().item(),
        "final_bn_abs_gamma": _final_bn(net.backbone).weight.abs().mean().item(),
        "ln_abs_gain": ln.weight.abs().mean().item() if ln is not None and ln.weight is not None else float("nan"),
        "max_abs_raw_logit": raw.abs().max(dim=1).values.mean().item(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="run tree under overnight/, e.g. cosine_2026-10-01_10-17-06")
    parser.add_argument("--labels", nargs="+", required=True, help="run labels; reads <label>/checkpoints/last.ckpt")
    parser.add_argument("--experiment", default="sngp_muon_sgd_cifar100_cosine",
                        help="experiment config that builds the val loader (only the data part is used)")
    parser.add_argument("--n", type=int, default=2000, help="val images (default 2000)")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    root = Path(os.environ["EXPERIMENTS_HOME"]) / os.environ["PROJECT_NAME"] / "overnight" / args.tag
    ckpts = {label: root / label / "checkpoints" / "last.ckpt" for label in args.labels}
    missing = [str(p) for p in ckpts.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = _val_batch(args.experiment, args.n, device)
    rows: List[Dict[str, object]] = []
    for label, ckpt in ckpts.items():
        net = load_net(ckpt, device=device).eval()
        rows.append({"label": label, **diagnose(net, x, y)})
        del net

    print("| label | ℓ | ‖h‖ | ‖h_in‖/ℓ | d/ℓ same | d/ℓ diff | k same | k diff | ‖β‖_F | final-BN |γ| | LN |gain| | max|raw logit| |")
    print("|---|" + "---:|" * (len(FIELDS) - 1))
    for r in rows:
        print("| " + " | ".join([str(r["label"])] + [f"{r[f]:.3g}" for f in FIELDS[1:]]) + " |")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
