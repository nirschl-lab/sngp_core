#!/usr/bin/env python3
"""demo_smooth_ece.py in tests/metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.metrics.smooth_ece import smECE_fast_compat
from src.metrics.utils import _bootstrap_ci_width
from src.visualization.reliability import rel_diagram_binned, rel_diagram_smoothed

# set vars
n_bootstrap = 100
confidence = 0.999
atol = 1e-2
nbins = 10
sigma = 0.1

# read sample_data.json in the same directory
sample_data_filepath = (
    Path(__file__).parents[2].joinpath("tests", "metrics", "sample_data.json")
)
if not sample_data_filepath.is_file():
    raise FileNotFoundError(f"File {sample_data_filepath} not found.")

with open(sample_data_filepath, "r") as f:
    sample_data = json.load(f)


# load data
# data in the format: {
#  'resnext50_32x4d': {'url': 'https://raw.githubusercontent.com/hollance/reliability-diagrams/master/results//ImageNet_pytorch-image-models/resnext50_32x4d.csv',
#   'dataset_name': 'ImageNet',
#   'model': 'resnext50_32x4d',
#   'ece_expected': 0.05804360030577724,
#   'ece_ci_width': 0.004293579205581294}
# }


def load_data(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset not in sample_data:
        raise ValueError(f"Dataset {dataset} not found in sample_data.json")

    # train dataset
    dataset_name = sample_data[dataset]["dataset_name"]
    model = sample_data[dataset]["model"]
    logger.debug(f"Loading dataset {dataset_name} with model {model}")

    url = sample_data[dataset]["url"]
    df = pd.read_csv(url)
    if "pop3" in dataset_name.lower():
        df = pd.read_csv(url, sep="\s+", header=0)

        # POP3 dataset has different column names
        obs = df["obs(mm)"]
        df = df.loc[obs.abs() < 100]
        df = df.loc[(df["p24_cat0"] >= 0) & (df["p24_cat0"] <= 1)]

        f = 1.0 - df["p24_cat0"].to_numpy()
        y = (df["obs(mm)"] > 0.2).to_numpy() * 1.0
    elif "solar" in dataset_name.lower():
        # Solar dataset has different column names
        f = df["DAFFS"].to_numpy().copy()
        y = df["rlz.C1"].to_numpy().copy()
    elif "imagenet" in dataset_name.lower():
        # standard dataset format
        f = df["confidence"].to_numpy()
        y = (df["true_label"] == df["pred_label"]).to_numpy() * 1.0
    else:
        raise ValueError(f"Unknown dataset format for {dataset_name}")

    return f, y


# Example usage:
for dataset in sample_data.keys():
    dataset_name = sample_data[dataset]["dataset_name"]
    model = sample_data[dataset]["model"]
    ece_expected = sample_data[dataset]["ece_expected"]
    ci_width = sample_data[dataset]["ece_ci_width"]

    # load data
    f, y = load_data(dataset)
    print(f"Loaded dataset {dataset} with {len(y)} samples.")

    # compute smECE (global ECE)
    ece_val = smECE_fast_compat(f, y)
    ece_ci_width = _bootstrap_ci_width(f, y, smECE_fast_compat, confidence=confidence)

    # assert
    assert np.isclose(ece_val, ece_expected, atol=atol), f"ECE mismatch for {dataset}"
    assert np.isclose(
        ece_ci_width, ci_width, atol=atol
    ), f"CI width mismatch for {dataset}"
    print(
        f"{dataset}: smECE = {ece_val:.6f}, expected ECE = {sample_data[dataset]['ece_expected']:.6f}"
    )
    print(
        f"{dataset}: CI width = {ece_ci_width:.6f}, expected CI width = {sample_data[dataset]['ece_ci_width']:.6f}"
    )

    #
    fig, ax = rel_diagram_binned(f, y, nbins=nbins)
    plt.title(f"{dataset}: ECE{nbins}")
    plt.show()

    # test plot reliability diagram
    fig, ax = rel_diagram_smoothed(f, y, sigma=sigma, n_bootstrap=n_bootstrap)
    plt.title(f"{dataset}: smECE w/bootstrap {n_bootstrap}")
    plt.show()

#
print("All datasets processed successfully.")
