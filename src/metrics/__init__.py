#!/usr/bin/env python3
"""__init__.py in src/metrics."""

# Populates src.metrics.registry.METRIC_REGISTRY as a side effect of importing this
# package -- see src/metrics/registered.py's docstring for why that matters under
# pytest-randomly's randomized test order.
from src.metrics import registered  # noqa: E402,F401
