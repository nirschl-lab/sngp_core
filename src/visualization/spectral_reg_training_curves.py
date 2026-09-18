"""spectral_reg_training_curves.py in src/visualization.

Training trajectory of a spectral-*regularization* run (`SNGPSpectralRegLitModule`): the
per-layer spectral norms the penalty acts on (`train/sigma_max`, `train/sigma_mean`), the
raw penalty `train/spec_reg` (= sum_l sigma_l^2) and the validation NLL / accuracy, all
against epoch, with the burn-in boundary marked. This is the one figure specific to the
method -- it shows how far the unregularized backbone drifts during burn-in and how fast
the penalty pulls it back -- so it lives here rather than in a notebook.

The history is read from W&B (the module logs these keys via `self.log`) and cached to a
tidy CSV next to the figure, so the figure is reproducible offline from that CSV.

Usage:
    uv run src/visualization/spectral_reg_training_curves.py \\
        --run nirschl-lab/uncertainty-aware-ml/5bnnbxdb --burnin-epoch 50 \\
        --figures-dir figures/acevedo_specreg
    # or, offline, from a previously written CSV:
    uv run src/visualization/spectral_reg_training_curves.py \\
        --history-csv figures/acevedo_specreg/spectral_reg_training_curves.csv --burnin-epoch 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualization.style import set_default_style  # noqa: E402

HISTORY_KEYS = [
    "train/sigma_max",
    "train/sigma_mean",
    "train/spec_reg",
    "train/spec_reg_active",
    "train/ce",
    "train/loss",
    "val/nll",
    "val/acc",
    "lr",
]


def fetch_history(run_path: str) -> pd.DataFrame:
    """One row per epoch with the last logged value of every key in `HISTORY_KEYS`.

    Lightning logs the `train/*` keys at train-epoch end and the `val/*` keys at
    validation end, i.e. in different W&B rows of the same `epoch`; `groupby.last()`
    takes the last non-null value per column, which merges them.
    """
    import wandb  # local import: only the online path needs it

    run = wandb.Api(timeout=60).run(run_path)
    rows = pd.DataFrame(list(run.scan_history()))
    if "epoch" not in rows.columns:
        raise RuntimeError(f"{run_path}: W&B history has no `epoch` column")
    rows = rows[rows["epoch"].notna()].copy()
    rows["epoch"] = rows["epoch"].astype(int)
    keys = [k for k in HISTORY_KEYS if k in rows.columns]
    history = rows.groupby("epoch")[keys].last()
    # The test stage / final summary logs an `epoch` row that carries none of these keys.
    history = history.dropna(how="all").reset_index()
    return history.sort_values("epoch").reset_index(drop=True)


def plot_training_curves(history: pd.DataFrame, burnin_epoch: int | None, title: str) -> plt.Figure:
    """Three panels: sigma_max / sigma_mean, sum sigma^2, val NLL (+ val acc on a twin axis)."""
    set_default_style()
    fig, (ax_sigma, ax_pen, ax_val) = plt.subplots(1, 3, figsize=(16, 4.8))
    epochs = history["epoch"]

    ax_sigma.plot(epochs, history["train/sigma_max"], marker=".", label="σ_max (worst layer)")
    ax_sigma.plot(epochs, history["train/sigma_mean"], marker=".", label="σ_mean (over layers)")
    ax_sigma.set_yscale("log")
    ax_sigma.set_ylabel("spectral norm (conv-operator estimate)")
    ax_sigma.set_title("Backbone spectral norms")

    ax_pen.plot(epochs, history["train/spec_reg"], marker=".", color="C3", label="Σ σ_max² (raw penalty)")
    ax_pen.set_yscale("log")
    ax_pen.set_ylabel("Σ σ²")
    ax_pen.set_title("Penalty magnitude")

    ax_val.plot(epochs, history["val/nll"], marker=".", color="C0", label="val/nll")
    ax_val.set_ylabel("val NLL")
    ax_val.set_yscale("log")
    ax_acc = ax_val.twinx()
    ax_acc.plot(epochs, history["val/acc"], marker=".", color="C2", label="val/acc")
    ax_acc.set_ylabel("val accuracy")
    ax_acc.set_ylim(top=1.0)
    ax_val.set_title("Validation")
    h1, l1 = ax_val.get_legend_handles_labels()
    h2, l2 = ax_acc.get_legend_handles_labels()
    ax_val.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")

    for ax in (ax_sigma, ax_pen, ax_val):
        ax.set_xlabel("epoch")
        if burnin_epoch is not None:
            ax.axvline(burnin_epoch, color="0.3", linestyle="--", linewidth=1)
            ax.text(
                burnin_epoch,
                ax.get_ylim()[1],
                " penalty on",
                ha="left",
                va="top",
                fontsize=10,
                color="0.3",
            )
    ax_sigma.legend(frameon=False, loc="lower left")
    ax_pen.legend(frameon=False, loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="W&B run path, e.g. entity/project/run_id.")
    source.add_argument("--history-csv", help="Previously written per-epoch CSV (offline path).")
    parser.add_argument("--burnin-epoch", type=int, default=None, help="First regularized epoch (dashed line).")
    parser.add_argument("--title", default="SNGP + spectral regularization — Acevedo pilot")
    parser.add_argument("--figures-dir", default=str(ROOT / "figures" / "acevedo_specreg"))
    parser.add_argument("--stem", default="spectral_reg_training_curves")
    args = parser.parse_args()

    if args.history_csv:
        history = pd.read_csv(args.history_csv)
    else:
        history = fetch_history(args.run)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(figures_dir / f"{args.stem}.csv", index=False)

    fig = plot_training_curves(history, args.burnin_epoch, args.title)
    fig.savefig(figures_dir / f"{args.stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{args.stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    # A few anchor epochs for the results table, printed so they can be copied verbatim.
    anchors = {0, int(history["epoch"].max())}
    if args.burnin_epoch is not None:
        anchors |= {args.burnin_epoch - 1, args.burnin_epoch, args.burnin_epoch + 3}
        post = history[history["epoch"] >= args.burnin_epoch]
        if len(post):
            anchors.add(int(post.loc[post["val/nll"].idxmin(), "epoch"]))
    cols = [c for c in ["epoch", "train/sigma_max", "train/sigma_mean", "train/spec_reg", "val/nll", "val/acc"] if c in history]
    print(history[history["epoch"].isin(sorted(anchors))][cols].to_string(index=False))
    print(f"Wrote {figures_dir / args.stem}.{{csv,png,pdf}}")


if __name__ == "__main__":
    main()
