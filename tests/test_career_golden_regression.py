#!/usr/bin/env python3
"""
Post-seam career byte-identical regression test.

Feature: cve-vulnerability-ranking, Task 2.5 (Requirements 12.2, 12.3)

This test re-runs the two career loader scenarios (``infonce_streaming`` and
``ordinal_grouped``) through the POST-seam ``contrastive_learning.DataLoader``
(default ``domain_adapter="career"``) and asserts the loader reproduces the
committed pre-seam golden artifact, ``career_golden_output.json``, byte-for-byte.

It reuses the exact capture logic from the golden generator
(``generate_career_golden._run_scenario`` and the same serialization used by
``main``), so the only thing under test is whether the seam changed observable
loader behavior. Any diff is a regression in the default career-domain code path.

Run:  .venv/bin/pytest tests/test_career_golden_regression.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate the spec test assets and make the repo root importable so that both
# this test and the imported generator resolve `contrastive_learning`.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_ASSETS = (
    _REPO_ROOT
    / ".kiro"
    / "specs"
    / "cve-vulnerability-ranking"
    / "test_assets"
)
_GOLDEN_PATH = _TEST_ASSETS / "career_golden_output.json"
_GENERATOR_PATH = _TEST_ASSETS / "generate_career_golden.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_generator_module():
    """Import the golden generator module by file path (it lives outside any package)."""
    spec = importlib.util.spec_from_file_location(
        "generate_career_golden", _GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_post_seam_artifact(gen):
    """
    Reconstruct the golden artifact exactly as ``generate_career_golden.main``
    does, but using the CURRENT (post-seam) DataLoader via the generator's
    reused ``_run_scenario`` logic.
    """
    return {
        "_description": (
            "Pre-seam golden regression output of the unmodified "
            "contrastive_learning.DataLoader on career_sample.jsonl. "
            "Captured for cve-vulnerability-ranking task 2.1 / Requirement 12.3. "
            "Task 2.5 asserts the post-seam loader reproduces this byte-for-byte."
        ),
        "seed": gen.SEED,
        "sample_file": gen.SAMPLE_PATH.name,
        "scenarios": {
            "infonce_streaming": gen._run_scenario("infonce_streaming", "infonce"),
            "ordinal_grouped": gen._run_scenario("ordinal_grouped", "ordinal"),
        },
    }


def _serialize(artifact):
    """Serialize identically to the generator (same options + trailing newline)."""
    return json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@pytest.fixture(scope="module")
def generator():
    assert _GENERATOR_PATH.exists(), f"Missing generator: {_GENERATOR_PATH}"
    assert _GOLDEN_PATH.exists(), f"Missing golden artifact: {_GOLDEN_PATH}"
    return _load_generator_module()


def test_post_seam_output_is_byte_identical_to_golden(generator):
    """The full post-seam artifact must equal the committed golden JSON byte-for-byte."""
    produced = _serialize(_build_post_seam_artifact(generator))
    expected = _GOLDEN_PATH.read_text(encoding="utf-8")
    assert produced == expected, (
        "Post-seam DataLoader output diverged from the pre-seam career golden "
        "artifact — the Domain_Adapter_Seam changed observable career behavior."
    )


@pytest.mark.parametrize(
    "scenario_name,loss_type",
    [("infonce_streaming", "infonce"), ("ordinal_grouped", "ordinal")],
)
def test_scenario_matches_golden(generator, scenario_name, loss_type):
    """Per-scenario equality for finer diagnostics (samples, grouping, batches, stats)."""
    expected = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))["scenarios"][
        scenario_name
    ]
    produced = generator._run_scenario(scenario_name, loss_type)
    assert produced == expected, (
        f"Scenario '{scenario_name}' diverged from the pre-seam golden output."
    )
