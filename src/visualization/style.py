#!/usr/bin/env python3
"""style.py in src/visualization."""

import matplotlib as mpl
import seaborn as sns
from loguru import logger


def set_default_style(use_tex: bool = False) -> None:
    """Set global matplotlib/seaborn style for calibration plots."""
    mpl.rc_file_defaults()
    sns.set_style("whitegrid")
    sns.set_palette("pastel", color_codes=True)

    mpl.rcParams.update(
        {
            "axes.edgecolor": "0.5",
            "font.size": 22,
            "legend.frameon": False,
            "patch.force_edgecolor": False,
            "figure.figsize": [6.0, 6.0],
            "axes.titlepad": 20,
        }
    )

    if use_tex:
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "text.usetex": True,
                "text.latex.preamble": r"""
                    \usepackage{libertine}
                    \usepackage[libertine]{newtxmath}
                """,
            }
        )

    logger.debug("Matplotlib/seaborn default style set (use_tex=%s).", use_tex)
