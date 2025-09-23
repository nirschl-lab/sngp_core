#!/usr/bin/env python3
"""demo_smooth_ece.py in src/metrics.

Synthetic and real-world datasets for Smooth ECE visualizations with kernel regression diagrams.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.smooth_ece import smoothed_ece_logit, smoothed_ece_logit_search
from src.visualization.reliability import rel_diagram


def make_toy_dataset(n: int = 2000, seed: int = 0):
    """Generate a toy binary classification dataset with logistic ground-truth model."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))

    w_true = np.array([1.2, -1.0, 0.5])
    b_true = -0.3

    z_true = X @ w_true + b_true
    p_true = 1.0 / (1.0 + np.exp(-z_true))
    y = rng.binomial(1, p_true)

    return X, y, p_true, z_true


def temperature_scale_probs(p: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling on probabilities via logits."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p) - np.log1p(-p)
    zT = z / T
    return 1.0 / (1.0 + np.exp(-zT))


def demo_smooth_ece(seed: int = 0) -> None:
    """Generate toy and real datasets, apply miscalibration, and plot reliability diagrams."""
    # === Synthetic toy dataset ===
    _, y, p_calib, _ = make_toy_dataset(n=4000, seed=seed)
    p_over = temperature_scale_probs(p_calib, T=0.5)  # over-confident
    p_under = temperature_scale_probs(p_calib, T=2.0)  # under-confident

    scenarios = {
        "Calibrated": p_calib,
        "Over-confident": p_over,
        "Under-confident": p_under,
    }

    # Side-by-side plots for synthetic scenarios
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (name, probs) in zip(axs, scenarios.items()):
        best_sigma = smoothed_ece_logit_search(probs, y)
        ece_val = smoothed_ece_logit(probs, y, sigma=best_sigma)
        print(f"{name}: smoothed ECE = {ece_val:.4f}")
        rel_diagram(
            probs,
            y,
            fig=fig,
            ax=ax,
            plot_density_ticks=True,
            plot_density=False,
            plot_confidence_band=True,
            simple_main_line=False,
        )
        ax.set_title(f"{name}\n(smECE={ece_val:.3f})")

    fig.suptitle("Synthetic Toy Dataset: Calibration Scenarios")
    fig.tight_layout()
    plt.show()

    # === Real-world dataset (POP3 precipitation forecasts) ===
    url = "https://www.cawcr.gov.au/projects/verification/POP3/POP_3cat_2003.txt"
    df = pd.read_csv(url, delim_whitespace=True, header=0)

    obs = df["obs(mm)"]
    df = df.loc[obs.abs() < 100]
    df = df.loc[(df["p24_cat0"] >= 0) & (df["p24_cat0"] <= 1)]

    y_real = (df["obs(mm)"] > 0.2).to_numpy() * 1.0
    f_real = 1.0 - df["p24_cat0"].to_numpy()

    best_sigma = smoothed_ece_logit_search(f_real, y_real)
    ece_val_real = smoothed_ece_logit(f_real, y_real, sigma=best_sigma)
    fig, ax = plt.subplots(figsize=(6, 6))
    rel_diagram(
        f_real,
        y_real,
        fig=fig,
        ax=ax,
        plot_density_ticks=True,
        plot_density=True,
        plot_confidence_band=True,
        simple_main_line=False,
    )
    ax.set_title(f"POP3 Dataset\n(smECE={ece_val_real:.3f})")
    fig.suptitle("Real-World Precipitation Forecast Calibration")
    fig.tight_layout()
    plt.show()

    print(f"POP3 Dataset: smoothed ECE = {ece_val_real:.4f}")

    # additional demos from
    # https://github.com/apple/ml-calibration/blob/main/notebooks/paper_experiments.ipynb
    def load_data(suffix, fname):
        url = f'https://raw.githubusercontent.com/hollance/reliability-diagrams/master/results/{suffix}'
        df = pd.read_csv(url)
        f = df['confidence'].to_numpy()
        y = (df['true_label'] == df['pred_label']).to_numpy() * 1.0
        return f, y


    #
    dataset = 'ImageNet_pytorch-image-models/resnet34.csv'
    dataset_name = dataset.split("_")[-1]
    model = dataset.split("/")[0].replace(".csv", "")
    f_resnet, y_resnet = load_data(dataset, f"{model}.png")

    best_sigma = smoothed_ece_logit_search(f_resnet, y_resnet)
    ece_val_real = smoothed_ece_logit(f_resnet, y_resnet, sigma=best_sigma)
    fig, ax = plt.subplots(figsize=(6, 6))
    rel_diagram(
        f_resnet,
        y_resnet,
        fig=fig,
        ax=ax,
        plot_density_ticks=True,
        plot_density=True,
        plot_confidence_band=True,
        simple_main_line=False,
    )
    ax.set_title(f"{dataset_name} {model.capitalize()}\n(smECE={ece_val_real:.3f}) Expected=0.078")
    fig.suptitle(f"{dataset_name} {model.capitalize()} Calibration")
    fig.tight_layout()
    plt.show()

    # load_plot_save('ImageNet_pytorch-image-models/resnext50_32x4d.csv', 'resnext50_32x4d.png')
    dataset = "ImageNet_pytorch-image-models/resnext50_32x4d.csv"
    dataset_name = dataset.split("_")[0]
    model = dataset.split("/")[-1].replace(".csv", "")
    f_resnet, y_resnet = load_data(dataset, f"{model}.png")

    best_sigma = smoothed_ece_logit_search(f_resnet, y_resnet)
    ece_val_real = smoothed_ece_logit(f_resnet, y_resnet, sigma=best_sigma)
    fig, ax = plt.subplots(figsize=(6, 6))
    rel_diagram(
        f_resnet,
        y_resnet,
        fig=fig,
        ax=ax,
        plot_density_ticks=True,
        plot_density=True,
        plot_confidence_band=True,
        simple_main_line=False,
    )
    ax.set_title(f"{dataset_name} {model.capitalize()}\n(smECE={ece_val_real:.3f}) Expected=0.058")
    fig.suptitle(f"{dataset_name} {model.capitalize()} Calibration")
    fig.tight_layout()
    plt.show()

    # solar flares
    solar_df = pd.read_csv('https://raw.githubusercontent.com/TimoDimi/replication_DGJ20/master/data/SF.FC.C1.csv')
    f_solar = solar_df['DAFFS'].to_numpy().copy()
    y_solar =solar_df['rlz.C1'].to_numpy().copy()
    print(solar_df[['DAFFS', 'NOAA', 'rlz.C1']].head())

    best_sigma = smoothed_ece_logit_search(f_solar, y_solar)
    ece_val_real = smoothed_ece_logit(f_solar, y_solar, sigma=best_sigma)
    fig, ax = plt.subplots(figsize=(6, 6))
    rel_diagram(
        f_solar,
        y_solar,
        fig=fig,
        ax=ax,
        plot_density_ticks=True,
        plot_density=True,
        plot_confidence_band=True,
        simple_main_line=False,
    )
    ax.set_title(f"Solar Flares\n(smECE={ece_val_real:.3f}) Expected: 0.067")
    fig.suptitle("Solar Flares Calibration")
    fig.tight_layout()
    plt.show()

    #

    def prepare_dataset(n, skew_function):
        res = []
        fa = []
        ya = []
        for _ in range(n):
            f = np.random.uniform()
            y = int(np.random.uniform() > 1 - skew_function(f))
            fa.append(f)
            ya.append(y)
        return np.array(fa), np.array(ya)

    def temp_scale(f, beta):
        return np.power(f, beta) / (np.power(f, beta) + np.power(1 - f, beta))

    def get_temp_scaled_dataset(N, beta):
        S = prepare_dataset(N, lambda x: x)  # perfectly calibrated
        S[0] = temp_scale(S[0], beta)
        return S


if __name__ == "__main__":
    demo_smooth_ece(seed=42)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from typing import Tuple
#
# from src.metrics.smooth_ece import smoothed_ece_logit
#
#
# def make_toy_dataset(
#     n: int = 2000, seed: int = 0
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Generate a toy binary classification dataset with logistic ground-truth model."""
#     rng = np.random.default_rng(seed)
#     X = rng.normal(size=(n, 3))
#
#     w_true = np.array([1.2, -1.0, 0.5])
#     b_true = -0.3
#
#     z_true = X @ w_true + b_true
#     p_true = 1.0 / (1.0 + np.exp(-z_true))
#     y = rng.binomial(1, p_true)
#
#     return X, y, p_true, z_true
#
#
# def temperature_scale_probs(p: np.ndarray, T: float) -> np.ndarray:
#     """Apply temperature scaling on probabilities via logits."""
#     eps = 1e-12
#     p = np.clip(p, eps, 1 - eps)
#     z = np.log(p) - np.log1p(-p)
#     zT = z / T
#     return 1.0 / (1.0 + np.exp(-zT))
#
#
# def plot_reliability_subplot(ax, p: np.ndarray, y: np.ndarray, title: str = "", n_bins: int = 10):
#     """Plot a reliability diagram (ECE-style binning) on a given axis."""
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#     digitized = np.digitize(p, bins) - 1
#
#     accs, confs = [], []
#     for i in range(n_bins):
#         mask = digitized == i
#         if not np.any(mask):
#             continue
#         confs.append(p[mask].mean())
#         accs.append(y[mask].mean())
#
#     ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
#     ax.plot(confs, accs, "o-", label="Empirical")
#     ax.set_xlabel("Predicted confidence")
#     ax.set_ylabel("Empirical accuracy")
#     ax.set_title(title)
#     ax.legend()
#
#
# def demo_smooth_ece(seed: int = 0) -> None:
#     """Generate toy and real datasets, apply miscalibration, and plot reliability diagrams."""
#     # === Synthetic toy dataset ===
#     _, y, p_calib, _ = make_toy_dataset(n=4000, seed=seed)
#     p_over = temperature_scale_probs(p_calib, T=0.5)   # over-confident
#     p_under = temperature_scale_probs(p_calib, T=2.0)  # under-confident
#
#     scenarios = {
#         "Calibrated": p_calib,
#         "Over-confident": p_over,
#         "Under-confident": p_under,
#     }
#
#     # Side-by-side plots for synthetic scenarios
#     fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
#     for ax, (name, probs) in zip(axs, scenarios.items()):
#         ece_val = smoothed_ece_logit(probs, y, sigma=0.2)
#         print(f"{name}: smoothed ECE = {ece_val:.4f}")
#         plot_reliability_subplot(ax, probs, y, title=f"{name}\n(Smoothed ECE={ece_val:.3f})")
#     fig.suptitle("Synthetic Toy Dataset: Calibration Scenarios")
#     fig.tight_layout()
#     plt.show()
#
#     # === Real-world dataset (POP3 precipitation forecasts) ===
#     url = "https://www.cawcr.gov.au/projects/verification/POP3/POP_3cat_2003.txt"
#     df = pd.read_csv(url, delim_whitespace=True, header=0)
#
#     obs = df["obs(mm)"]
#     df = df.loc[obs.abs() < 100]
#     df = df.loc[(df["p24_cat0"] >= 0) & (df["p24_cat0"] <= 1)]
#
#     y_real = (df["obs(mm)"] > 0.2).to_numpy() * 1.0
#     f_real = 1.0 - df["p24_cat0"].to_numpy()
#
#     ece_val_real = smoothed_ece_logit(f_real, y_real, sigma=0.2)
#     fig, ax = plt.subplots(figsize=(5, 5))
#     plot_reliability_subplot(ax, f_real, y_real, title=f"POP3 Dataset\n(Smoothed ECE={ece_val_real:.3f})")
#     fig.suptitle("Real-World Precipitation Forecast Calibration")
#     fig.tight_layout()
#     plt.show()
#
#     print(f"POP3 Dataset: smoothed ECE = {ece_val_real:.4f}")
#
#
# if __name__ == "__main__":
#     demo_smooth_ece(seed=42)
#
#
