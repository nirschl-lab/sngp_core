#!/usr/bin/env python3
"""utils.py in src/visualization.

Small helpers for smooth reliability plotting.
"""

import numpy as np
from numpy.typing import NDArray


def scale_density_sizes(
    density: NDArray[np.float64],
    size_scale: float = 240.0,
) -> NDArray[np.float64]:
    """
    Maps density values to point sizes for visualization, using a nonlinear scaling.

    This function scales density values to marker sizes, using a quadratic mapping for low densities
    and a square root mapping for high densities, to enhance visual contrast in plots.

    Args:
        density: Array of density values to be mapped to sizes.
        size_scale: Maximum size scaling factor (default 240.0).

    Returns:
        An array of point sizes corresponding to the input densities.
    """
    #Map density → point sizes: quadratic for low density, sqrt for high density
    d = np.asarray(density, dtype=np.float64).reshape(-1)
    if d.max() <= 0:
        return np.full_like(d, fill_value=size_scale * 0.1)
    dn = d / d.max()
    small = dn < 1.0
    shaped = (dn**2) * small + np.sqrt(dn) * (~small)
    sizes = size_scale * shaped
    return sizes


def clip01(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
