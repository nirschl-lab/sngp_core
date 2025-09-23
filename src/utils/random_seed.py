#!/usr/bin/env python3
"""random_seed.py in src/sngp_core/utils."""

import random

import numpy as np
import pytorch_lightning as pl
import torch


def set_random_seed(random_seed: int = 8675309) -> None:
    """Set the random seed for reproducibility.

    Args:
        random_seed (int, optional): The random seed to set.

    Returns:
        None

    Examples:
        >>> set_random_seed(8675309)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    pl.seed_everything(random_seed)
