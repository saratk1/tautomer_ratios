"""
Unit and regression test for the taut_diff package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import taut_diff


def test_taut_diff_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "taut_diff" in sys.modules
