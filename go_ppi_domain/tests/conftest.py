"""Shared pytest configuration for go_ppi_domain tests.

Registers the ``slow`` marker (withholding-study integration tests with real
configs and MPNet). Slow tests are skipped unless ``GO_WITHHOLDING_SLOW=1``.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: real-trainer/MPNet integration test; run with GO_WITHHOLDING_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("GO_WITHHOLDING_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow test; set GO_WITHHOLDING_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
