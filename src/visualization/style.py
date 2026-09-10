#!/usr/bin/env python3
"""style.py in src/visualization."""

import matplotlib as mpl
import seaborn as sns
from loguru import logger

# Colorblind-safe categorical palette (Wong 2011, Nature Methods), one fixed color per
# project dataset so the same dataset reads as the same color across every figure.
DATASET_COLORS: dict[str, str] = {
    "acevedo": "#0173B2",
    "wong": "#DE8F05",
    "tang": "#029E73",
    "kather2018": "#D55E00",
    "kather2016": "#CC78BC",
    "jung": "#CA9161",
    "nirschl2018": "#949494",
}

# Canonical display order for this project's 5 model variants, shared by every figure
# that compares them (artifact_ablation_curves.py, artifact_severity_curves.py) so
# panel/row order stays identical across figures.
MODEL_ROW_ORDER: list[str] = [
    "Baseline Classifier",
    "Deep Ensemble",
    "Monte Carlo Dropout",
    "SNGP",
    "SNGP Ensemble",
]


def set_default_style(use_tex: bool = False) -> None:
    """Set global matplotlib/seaborn style for calibration plots."""
    mpl.rc_file_defaults()
    sns.set_style("whitegrid")
    sns.set_palette("pastel", color_codes=True)

    mpl.rcParams.update(
        {
            "axes.edgecolor": "0.5",
            "font.size": 14,
            "legend.frameon": False,
            "patch.force_edgecolor": False,
            "figure.figsize": [8.0, 6.0],
            "axes.titlepad": 16,
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
