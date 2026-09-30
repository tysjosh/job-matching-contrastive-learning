"""The disease study varies one scorer or one negative tier at a time."""

import subprocess
import sys
from pathlib import Path

from scripts.run_mesh_disease_decomposition import (
    ROOT, SCOPES, arm_config, command,
)


def test_exact_vs_hierarchy_changes_only_scorer():
    for scope in SCOPES:
        exact = arm_config("exact", scope)
        hierarchy = arm_config("hierarchy", scope)
        assert {key for key in exact if exact[key] != hierarchy[key]} == {
            "mesh_similarity_mode"}


def test_pool_scope_changes_only_tier_scope():
    for mode in ("exact", "hierarchy"):
        both = arm_config(mode, "both")
        for scope in ("ineligible", "not_relevant"):
            arm = arm_config(mode, scope)
            assert {key for key in both if both[key] != arm[key]} == {
                "trials_mesh_tier_scope"}


def test_training_plan_uses_2021_and_never_test():
    cmd = command(Path("config.json"), "disease_exact_both", 100, 13)
    assert "preprocess/trec_ct_lc/frac_100/train.jsonl" in " ".join(cmd)
    assert "preprocess/trec_ct_lc/validation_positive.jsonl" in " ".join(cmd)
    assert "test.jsonl" not in " ".join(cmd)


def test_test_evaluation_requires_declared_arms():
    script = ROOT / "scripts/run_mesh_disease_decomposition.py"
    result = subprocess.run(
        [sys.executable, str(script), "--evaluate-test", "--seeds", "13"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires --selected-arms" in result.stderr
