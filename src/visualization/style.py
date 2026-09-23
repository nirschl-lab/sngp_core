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

# Canonical display order for this project's model variants, shared by every figure
# that compares them (artifact_ablation_curves.py, artifact_severity_curves.py,
# render_artifact_results_tables.py) so panel/row order stays identical across figures.
MODEL_ROW_ORDER: list[str] = [
    "Baseline Classifier",
    "Deep Ensemble",
    "Monte Carlo Dropout",
    "SNGP",
    "SNGP Ensemble",
    "SNGP + Spectral Reg",
]

# One fixed color per model variant, from the same Wong 2011 palette as `DATASET_COLORS`,
# so a method reads as the same color across every figure that compares methods -- the
# counterpart of `DATASET_COLORS` for the model axis.
#
# The baseline is grey on purpose: it is the deterministic reference the uncertainty
# methods are read against, and it is usually drawn as a reference line rather than as a
# series of its own, so it should recede rather than compete for attention.
#
# Figures that predate this dict build their own local maps
# (`artifact_ablation_curves.py::MODEL_COLORS`, the two module-level constants in
# `spectral_norm_bound_curve.py`); those are not wired up to this and stay as they are
# until someone is regenerating those figures anyway.
METHOD_COLORS: dict[str, str] = {
    "Baseline Classifier": "#949494",
    "Deep Ensemble": "#DE8F05",
    "Monte Carlo Dropout": "#CC78BC",
    "SNGP": "#0173B2",
    "SNGP Ensemble": "#56B4E9",
    "SNGP + Spectral Reg": "#D55E00",
}


def order_models(names) -> list[str]:
    """Canonical display order for whatever subset of models a figure or table actually
    has: `MODEL_ROW_ORDER`'s order for the known names, then any unknown names sorted.

    Every consumer used to index the fixed list directly, which raised on any sidecar
    that lacked one of the five original models or added a sixth -- a one-model results
    write-up (e.g. a new training variant on its own) must still render.
    """
    present = list(dict.fromkeys(names))
    known = [m for m in MODEL_ROW_ORDER if m in present]
    extra = sorted(m for m in present if m not in MODEL_ROW_ORDER)
    return known + extra


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
