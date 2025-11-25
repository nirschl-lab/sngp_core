#!/usr/bin/env python3
"""test_imports.py in tests."""

import pytest


@pytest.mark.smoke
def test_imports():
    """Test package imports."""
    import src.data # noqa: F401
    import src.models # noqa: F401
    import src.utils # noqa: F401