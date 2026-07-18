#!/usr/bin/env python3
"""Unit tests for ORCA variant configuration and construction (tasks 2.12, 6.1).

Component under test: ``orca/config.py`` — ``OrcaConfig`` (variant resolution,
``validate``, and the retained ``validate_mvp`` helper) and its wiring into
``orca/factory.py``'s ``make_loss_engine``.

History: Milestone 4 (task 6.1) ends the MVP-only restriction. Milestones 2
(adaptive sampling) and 3 (alignment) are complete, so all six recognized
variants are now valid and the factory constructs the appropriate engine/model
configuration for each. Only a truly *unrecognized* ``orca_variant`` raises
(Requirement 6.8). ``validate_mvp`` is retained (it is still the single source
of truth for the historical MVP surface and some callers/tests reference it),
so its direct behavior is still covered here — but it is no longer applied as a
hard gate inside ``make_loss_engine``.

With ``orca_enabled`` false the config guards never run and the byte-identical
career path is untouched (Requirement 7.1).

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_mvp_guard.py

Requirements: 6.1, 6.4, 6.7, 6.8, 7.1, 11.2, 11.4
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contrastive_learning.loss_engine import ContrastiveLossEngine  # noqa: E402
from orca.config import (  # noqa: E402
    MVP_VARIANT,
    NUM_ONTOLOGY_SCALARS,
    RECOGNIZED_VARIANTS,
    OrcaConfig,
    OrcaConfigError,
)
from orca.factory import make_loss_engine  # noqa: E402
from orca.loss_engine import OrcaLossEngine  # noqa: E402


class _Cfg:
    """Minimal stand-in exposing ``orca_*`` attributes via defaults + overrides."""

    def __init__(self, **overrides):
        self.__dict__.update(overrides)


def _orca_config(**overrides) -> OrcaConfig:
    """Build an ``OrcaConfig`` from an attribute stand-in (uses field defaults)."""
    return OrcaConfig.from_training_config(_Cfg(**overrides))


# --------------------------------------------------------------------------- #
# MVP variant is "denominator"
# --------------------------------------------------------------------------- #
def test_mvp_variant_is_denominator():
    """The MVP variant constant is ORCA-Denominator (Requirement 11.2)."""
    assert MVP_VARIANT == "denominator"


# --------------------------------------------------------------------------- #
# Valid MVP config passes (Requirement 11.2)
# --------------------------------------------------------------------------- #
def test_valid_mvp_config_passes():
    """The exact MVP config validates without raising (Requirement 11.2)."""
    cfg = _orca_config(
        orca_variant="denominator",
        orca_use_history=False,
        orca_use_alignment=False,
        orca_adaptive_sampling=False,
    )
    # Must not raise.
    cfg.validate_mvp()


# --------------------------------------------------------------------------- #
# Each non-MVP condition raises, naming the offending field (Requirement 11.4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides, offending_field",
    [
        (dict(orca_variant="full", orca_adaptive_sampling=False), "orca_variant"),
        (
            dict(orca_variant="external_weight", orca_adaptive_sampling=False),
            "orca_variant",
        ),
        (
            dict(orca_variant="denominator", orca_adaptive_sampling=True),
            "orca_adaptive_sampling",
        ),
        (
            dict(
                orca_variant="denominator",
                orca_adaptive_sampling=False,
                orca_use_alignment=True,
            ),
            "orca_use_alignment",
        ),
        (
            dict(
                orca_variant="denominator",
                orca_adaptive_sampling=False,
                orca_use_history=True,
            ),
            "orca_use_history",
        ),
    ],
)
def test_non_mvp_condition_raises_naming_field(overrides, offending_field):
    """Every non-MVP feature raises OrcaConfigError identifying it (Req 11.4)."""
    cfg = _orca_config(**overrides)
    with pytest.raises(OrcaConfigError) as exc_info:
        cfg.validate_mvp()
    assert offending_field in str(exc_info.value)


def test_multiple_offending_conditions_all_named():
    """When several non-MVP features are on, all are reported (Req 11.4)."""
    cfg = _orca_config(
        orca_variant="full",
        orca_adaptive_sampling=True,
        orca_use_alignment=True,
        orca_use_history=True,
    )
    with pytest.raises(OrcaConfigError) as exc_info:
        cfg.validate_mvp()
    message = str(exc_info.value)
    for field in (
        "orca_variant",
        "orca_adaptive_sampling",
        "orca_use_alignment",
        "orca_use_history",
    ):
        assert field in message


# --------------------------------------------------------------------------- #
# Factory wiring: golden path returns the unchanged engine
# --------------------------------------------------------------------------- #
def test_factory_golden_path_returns_unchanged_engine():
    """With ORCA off the factory returns the unchanged engine (Requirement 7.1).

    A non-default variant is set alongside ``orca_enabled=False`` to prove the
    ORCA config path is not consulted on the byte-identical golden path.
    """
    cfg = _Cfg(
        orca_enabled=False,
        orca_variant="full",
        orca_adaptive_sampling=True,
        projection_dim=16,
        temperature=0.05,
    )
    engine = make_loss_engine(cfg)
    assert isinstance(engine, ContrastiveLossEngine)


# --------------------------------------------------------------------------- #
# All six recognized variants are now valid and construct an OrcaLossEngine
# (task 6.1 ends the MVP-only restriction; Requirements 6.1, 6.2, 6.4-6.7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", list(RECOGNIZED_VARIANTS))
def test_factory_constructs_engine_for_every_recognized_variant(variant):
    """Each of the six recognized variants builds an ``OrcaLossEngine``."""
    cfg = _Cfg(
        orca_enabled=True,
        orca_variant=variant,
        # ``full`` implies alignment on; other variants leave it off. Adaptive
        # sampling is a variant-gated concern handled by the orchestrator, so it
        # is safe to leave at its default here.
        orca_use_alignment=(variant == "full"),
        projection_dim=16,
        temperature=0.05,
        orca_feature_dim=5,
    )
    engine = make_loss_engine(cfg)
    assert isinstance(engine, OrcaLossEngine)
    assert engine.variant == variant


def test_factory_no_ontology_excludes_ontology_scalar_features():
    """ORCA-NoOntology builds the ReliabilityMLP without the ontology scalars.

    Requirement 6.4 / design B.2 variant hook: the feature layout drops the five
    ESCO/ISCO/OT scalars, so the reliability model's ``feature_dim`` shrinks by
    exactly ``NUM_ONTOLOGY_SCALARS`` relative to a variant that keeps them.
    """
    common = dict(orca_enabled=True, projection_dim=16, temperature=0.05,
                  orca_feature_dim=5 + 4)  # 5 ontology scalars + 4 coverage dims

    denom_engine = make_loss_engine(_Cfg(orca_variant="denominator", **common))
    no_ont_engine = make_loss_engine(_Cfg(orca_variant="no_ontology", **common))

    assert denom_engine.reliability_model.feature_dim == 5 + 4
    assert no_ont_engine.reliability_model.feature_dim == 4
    assert (
        denom_engine.reliability_model.feature_dim
        - no_ont_engine.reliability_model.feature_dim
        == NUM_ONTOLOGY_SCALARS
    )


def test_factory_no_future_forces_history_off():
    """ORCA-NoFuture forces the weak-target history signal off (Requirement 6.5)."""
    cfg = _Cfg(
        orca_enabled=True,
        orca_variant="no_future",
        orca_use_history=True,      # user asks for history…
        projection_dim=16,
        temperature=0.05,
    )
    engine = make_loss_engine(cfg)
    # …but the variant redistributes its weight by forcing it off.
    assert engine.weak_builder.use_history is False


def test_factory_full_variant_enables_alignment():
    """ORCA-Full includes the OntologyAlignmentLoss (Requirement 6.7)."""
    cfg = _Cfg(
        orca_enabled=True,
        orca_variant="full",
        projection_dim=16,
        temperature=0.05,
    )
    engine = make_loss_engine(cfg)
    assert engine.alignment is not None


def test_factory_denominator_excludes_alignment():
    """ORCA-Denominator has no alignment term (Requirements 4.4, 6.6 family)."""
    cfg = _Cfg(
        orca_enabled=True,
        orca_variant="denominator",
        projection_dim=16,
        temperature=0.05,
    )
    engine = make_loss_engine(cfg)
    assert engine.alignment is None


def test_factory_external_weight_disables_adaptive_and_denominator_calibration():
    """ORCA-ExternalWeight resolves to the external-loss site (Requirement 6.1)."""
    orca_cfg = _orca_config(orca_variant="external_weight")
    assert orca_cfg.reliability_site() == "external_weight"
    assert orca_cfg.uses_adaptive_sampling() is False


# --------------------------------------------------------------------------- #
# Only a truly unrecognized variant raises (Requirement 6.8)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_variant", ["", "denom", "ORCA-Full", "future", "none"])
def test_factory_raises_for_unrecognized_variant(bad_variant):
    """An unrecognized ``orca_variant`` raises OrcaConfigError (Requirement 6.8)."""
    cfg = _Cfg(
        orca_enabled=True,
        orca_variant=bad_variant,
        projection_dim=16,
        temperature=0.05,
    )
    with pytest.raises(OrcaConfigError) as exc_info:
        make_loss_engine(cfg)
    # The error identifies the offending variant value (Requirement 6.8).
    assert bad_variant in str(exc_info.value) or "orca_variant" in str(exc_info.value)


def test_factory_rejects_alignment_on_non_full_variant():
    """Enabling alignment on a non-``full`` variant is inconsistent (Req 6.6/6.7)."""
    cfg = _Cfg(
        orca_enabled=True,
        orca_variant="no_align",
        orca_use_alignment=True,     # contradicts the variant's semantics
        projection_dim=16,
        temperature=0.05,
    )
    with pytest.raises(OrcaConfigError):
        make_loss_engine(cfg)
