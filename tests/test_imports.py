#!/usr/bin/env python3
"""test_imports.py in tests."""

import pytest


@pytest.mark.smoke
def test_imports():
    """Test package imports."""
    import src.data # noqa: F401
    import src.models # noqa: F401
    import src.utils # noqa: F401
    from src.models.sngp.sngp_classifier import SNGPClassifier
    from src.models.baseline.baseline_models import BaselineClassifier
    from src.models.lit_module_base import LitModuleBase
    from src.models.baseline_lit_module import BaselineLitModule
    from src.models.sngp_lit_module import SNGPLitModule
    from src.models.deep_ensemble_lit_module import DeepEnsembleLitModule