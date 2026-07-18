#!/usr/bin/env python3
"""Tests for the ORCA experiment matrix and the runner's checkpoint saving.

Covers:
  * every experiment in ``orca/experiments/experiments.json`` resolves (base
    config + overrides) into a valid ``TrainingConfig`` that ORCA accepts
    (``OrcaConfig.validate``), with ``orca_enabled`` on and a recognized variant;
  * the per-variant implied flags are consistent (e.g. full => alignment on,
    denominator/external_weight => adaptive sampling off);
  * ``run_orca_training._save_orca_checkpoint`` writes a ``best_checkpoint.pt``
    carrying ``model_state_dict`` (what the Phase-1 embedding evaluation loads),
    so ORCA runs are evaluable by the same script as the baselines.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_experiments_and_checkpoint.py -v
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contrastive_learning.data_structures import TrainingConfig  # noqa: E402
from orca.config import RECOGNIZED_VARIANTS, OrcaConfig  # noqa: E402
from orca.reliability_model import ReliabilityMLP  # noqa: E402

_MANIFEST = _REPO_ROOT / "orca" / "experiments" / "experiments.json"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Experiment matrix
# --------------------------------------------------------------------------- #
def test_manifest_experiments_resolve_to_valid_orca_configs():
    """Base config + each variant's overrides yields a valid ORCA config."""
    manifest = _load(_MANIFEST)
    base = _load(_REPO_ROOT / manifest["base_config"])

    assert manifest["experiments"], "manifest has no experiments"

    for exp in manifest["experiments"]:
        merged = copy.deepcopy(base)
        merged.update(exp.get("orca_overrides", {}))
        merged["orca_enabled"] = True

        cfg = TrainingConfig.from_dict(merged)
        assert cfg.orca_enabled is True
        assert cfg.orca_variant in RECOGNIZED_VARIANTS, exp["id"]

        # Per-variant consistency guard must accept every matrix entry.
        oc = OrcaConfig.from_training_config(cfg)
        oc.validate()


def test_manifest_variant_flags_are_consistent():
    """Implied flags per variant match the design table."""
    manifest = _load(_MANIFEST)
    by_id = {e["id"]: e.get("orca_overrides", {}) for e in manifest["experiments"]}

    # denominator / external_weight keep adaptive sampling OFF (Req 6.1/6.2).
    assert by_id["ER-DEN"]["orca_variant"] == "denominator"
    assert by_id["ER-DEN"]["orca_adaptive_sampling"] is False
    assert by_id["ER-EXT"]["orca_variant"] == "external_weight"
    assert by_id["ER-EXT"]["orca_adaptive_sampling"] is False

    # full enables alignment; no_align does not.
    assert by_id["ER-FULL"]["orca_variant"] == "full"
    assert by_id["ER-FULL"]["orca_use_alignment"] is True
    assert by_id["ER-NOALIGN"]["orca_use_alignment"] is False

    # no_ontology strips ontology features.
    assert by_id["ER-NOONT"]["orca_use_ontology_features"] is False


def test_manifest_baselines_present():
    """The InfoNCE + OSCAR baselines ORCA is compared against are documented."""
    manifest = _load(_MANIFEST)
    assert "E4-InfoNCE" in manifest["baselines"]
    assert "E4-OSCAR-Skill" in manifest["baselines"]


# --------------------------------------------------------------------------- #
# Checkpoint saving
# --------------------------------------------------------------------------- #
class _StubOrchestrator:
    def __init__(self, reliability_model):
        self.reliability_model = reliability_model

        class _Phase:
            name = "JOINT"

        self.current_phase = _Phase()


class _StubTrainer:
    def __init__(self, dim=8):
        self.model = nn.Linear(dim, dim)  # projection head stand-in


def test_save_orca_checkpoint_writes_model_state_dict(tmp_path):
    """The runner persists best_checkpoint.pt with the loadable model_state_dict."""
    import run_orca_training as r

    cfg = TrainingConfig(orca_enabled=True, orca_variant="denominator",
                         projection_dim=8, training_seed=42)
    trainer = _StubTrainer(dim=8)
    reliability = ReliabilityMLP(embed_dim=8, feature_dim=5)
    orch = _StubOrchestrator(reliability)

    out_dir = tmp_path / "phase1_pretraining"
    ckpt_path = r._save_orca_checkpoint(trainer, str(out_dir), cfg, orch)

    assert Path(ckpt_path).exists()
    payload = torch.load(ckpt_path, weights_only=False)
    # The key the Phase-1 embedding evaluation loads.
    assert "model_state_dict" in payload
    assert payload["orca_variant"] == "denominator"
    assert payload["completed_phase"] == "JOINT"
    assert payload["reliability_model_state_dict"] is not None

    # The saved projection weights round-trip into a fresh model.
    fresh = nn.Linear(8, 8)
    fresh.load_state_dict(payload["model_state_dict"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
