"""Tests for the run_orca_training.py domain-dispatch and ontology-wiring guards.

The guards matter more than usual here because the failure they prevent is
*silent*: an unwired trials run trains on random negatives with no ontology
signal and still produces plausible metrics. The pre-existing
``--require-ontology`` check inspects only the data, which passes for trials
records, so a wiring check is the only thing standing between a broken run and a
believable-looking table.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import run_orca_training as runner


class _BP:
    """Minimal batch-processor stand-in with the effective-matcher resolver."""

    def __init__(self, matcher=None, override=None):
        self.skill_matcher = matcher
        self.ontology_matcher_override = override

    def _effective_skill_matcher(self):
        return self.ontology_matcher_override or self.skill_matcher


def _trainer(batch_processor):
    return types.SimpleNamespace(batch_processor=batch_processor)


# ------------------------------------------------------- wiring verification
def test_wiring_ok_when_esco_matcher_present():
    """Career path: a matcher built from esco_* paths satisfies the check."""
    runner._verify_ontology_wiring(
        _trainer(_BP(matcher=object())),
        types.SimpleNamespace(domain_adapter="career"),
        require=True,
    )  # must not raise


def test_wiring_ok_when_override_injected():
    """Trials path: an injected matcher satisfies the check."""
    runner._verify_ontology_wiring(
        _trainer(_BP(override=object())),
        types.SimpleNamespace(domain_adapter="trials"),
        require=True,
    )


def test_wiring_missing_exits_under_require_ontology():
    """The whole point: no matcher + --require-ontology must be fatal."""
    with pytest.raises(SystemExit) as excinfo:
        runner._verify_ontology_wiring(
            _trainer(_BP()),
            types.SimpleNamespace(domain_adapter="trials"),
            require=True,
        )
    assert excinfo.value.code == 1


def test_wiring_missing_only_warns_without_require(caplog):
    """Without the flag it degrades to a warning, matching the data check."""
    with caplog.at_level("WARNING"):
        runner._verify_ontology_wiring(
            _trainer(_BP()),
            types.SimpleNamespace(domain_adapter="trials"),
            require=False,
        )
    assert any("No ontology matcher is active" in r.message for r in caplog.records)


def test_wiring_check_tolerates_trainer_without_batch_processor():
    runner._verify_ontology_wiring(
        types.SimpleNamespace(), types.SimpleNamespace(domain_adapter="career"),
        require=True,
    )


def test_wiring_check_falls_back_when_resolver_absent():
    """An older BatchProcessor without the resolver still reports correctly."""
    legacy = types.SimpleNamespace(skill_matcher=object())
    runner._verify_ontology_wiring(
        _trainer(legacy), types.SimpleNamespace(domain_adapter="career"), require=True
    )

    legacy_empty = types.SimpleNamespace(skill_matcher=None)
    with pytest.raises(SystemExit):
        runner._verify_ontology_wiring(
            _trainer(legacy_empty),
            types.SimpleNamespace(domain_adapter="career"),
            require=True,
        )


# ----------------------------------------------------------- domain dispatch
def test_attach_domain_is_noop_for_career(monkeypatch):
    """The career path must not touch the trials wiring at all."""
    called = []
    monkeypatch.setitem(
        sys.modules,
        "trials_domain.run_config",
        types.SimpleNamespace(
            attach_trials_domain=lambda *a, **k: called.append(a) or {}
        ),
    )
    runner._attach_domain(
        _trainer(_BP(matcher=object())),
        types.SimpleNamespace(domain_adapter="career"),
    )
    assert called == []


def test_attach_domain_exits_when_pools_are_empty(monkeypatch):
    """Zero indexed pools means graded negatives are absent -> fail loudly."""
    monkeypatch.setitem(
        sys.modules,
        "trials_domain.run_config",
        types.SimpleNamespace(
            attach_trials_domain=lambda *a, **k: {
                "mesh_descriptors": 29917,
                "curriculum": {"start_hard_ratio": 0.2, "end_hard_ratio": 0.6},
                "pools": {"topics": 0, "hard_total": 0, "easy_total": 0},
            }
        ),
    )
    with pytest.raises(SystemExit) as excinfo:
        runner._attach_domain(
            _trainer(_BP()),
            types.SimpleNamespace(
                domain_adapter="trials", trials_split_dir="preprocess/trec_ct_splits"
            ),
        )
    assert excinfo.value.code == 1


def test_attach_domain_exits_when_attach_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("set_ontology_matcher missing")

    monkeypatch.setitem(
        sys.modules,
        "trials_domain.run_config",
        types.SimpleNamespace(attach_trials_domain=boom),
    )
    with pytest.raises(SystemExit):
        runner._attach_domain(
            _trainer(_BP()), types.SimpleNamespace(domain_adapter="trials")
        )


def test_attach_domain_succeeds_with_populated_pools(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "trials_domain.run_config",
        types.SimpleNamespace(
            attach_trials_domain=lambda *a, **k: {
                "mesh_descriptors": 29917,
                "curriculum": {"start_hard_ratio": 0.2, "end_hard_ratio": 0.6},
                "pools": {"topics": 60, "hard_total": 4973, "easy_total": 19052},
            }
        ),
    )
    runner._attach_domain(
        _trainer(_BP(override=object())),
        types.SimpleNamespace(domain_adapter="trials"),
    )  # must not raise


# --------------------------------------------------------------- config fields
def test_trials_config_fields_are_declared_not_dropped():
    """from_json silently drops undeclared keys; these must survive the round trip.

    Without the dataclass declarations, tuning mesh_alpha or the curriculum in the
    config would read back as the hardcoded fallback and have no effect.
    """
    from contrastive_learning.data_structures import TrainingConfig

    config = TrainingConfig.from_json(
        str(ROOT / "config" / "orca_trials_denominator_config.json")
    )
    assert config.domain_adapter == "trials"
    assert config.mesh_alpha == pytest.approx(0.5)
    assert config.mesh_max_hops == 12
    assert config.trials_split_dir == "preprocess/trec_ct_splits"
    assert config.trials_converted_dir == "preprocess/trec_ct"
    assert config.trials_start_hard_ratio == pytest.approx(0.2)
    assert config.trials_end_hard_ratio == pytest.approx(0.6)
    assert config.embedding_cache_path == "embedding_cache/trials_text_embeddings.pt"


def test_career_config_unaffected_by_trials_fields():
    """The career config must still load, with the trials fields inert."""
    from contrastive_learning.data_structures import TrainingConfig

    config = TrainingConfig.from_json(
        str(ROOT / "config" / "orca_denominator_config.json")
    )
    assert config.domain_adapter == "career"
    assert config.esco_graph_path
    assert config.use_pathway_negatives is True
