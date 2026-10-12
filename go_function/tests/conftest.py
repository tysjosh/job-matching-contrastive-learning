"""Shared pytest configuration for go_function tests.

- ``slow``: real-data integration tests, skipped unless ``GO_FUNCTION_SLOW=1``.
- ``needs_dgz``: skipped when the pinned DGZ release is absent
  (dataset/go_function/dgz/go.obo).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DGZ = ROOT / "dataset" / "go_function" / "dgz"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: real-data integration test; run with GO_FUNCTION_SLOW=1")
    config.addinivalue_line(
        "markers", "needs_dgz: needs the pinned DGZ release in dataset/go_function/dgz/")


def pytest_collection_modifyitems(config, items):
    run_slow = os.environ.get("GO_FUNCTION_SLOW") == "1"
    has_dgz = (DGZ / "go.obo").exists()
    skip_slow = pytest.mark.skip(reason="slow test; set GO_FUNCTION_SLOW=1 to run")
    skip_dgz = pytest.mark.skip(reason="DGZ release absent (dataset/go_function/dgz/go.obo)")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "needs_dgz" in item.keywords and not has_dgz:
            item.add_marker(skip_dgz)


@pytest.fixture
def mini_obo() -> Path:
    return FIXTURES / "mini.obo"


@pytest.fixture
def mini_norm() -> Path:
    return FIXTURES / "mini.norm"
