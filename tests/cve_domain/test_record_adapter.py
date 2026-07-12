"""Unit tests for CVERecordAdapter (task 7.3).

Component under test: ``CVERecordAdapter`` in ``cve_domain/record_adapter.py``.

The CVE Record_Adapter plugs into the additive ``Domain_Adapter_Seam`` and maps
one anchor ``CVE_View_Record`` (paired by the ``CVEPositiveSelector`` with a
selected ontology-related positive ``CVE_View_Record`` attached under the
``"positive"`` key) onto the shared ``TrainingSample`` contract:

- ``sample.resume`` holds the anchor ``{cve, encoder_view}`` (first view slot).
- ``sample.job`` holds the positive ``{cve, encoder_view}`` (second view slot).
- ``sample.label`` is ``"positive"`` (the anchor/positive pair is a positive).
- ``metadata['resume_id']`` is the anchor ``cve`` (group-id injection).
- ``metadata['cve_labels']`` carries the anchor's present supervised labels.

These example-based tests cover the four behaviors called out by task 7.3:

- Group-id injection: ``metadata['resume_id'] == anchor cve``.
- Label passthrough: ``metadata['cve_labels']`` carries the present labels.
- Two-slot anchor/positive encoder views: anchor in ``resume``, positive in ``job``.
- Validation: empty/missing ``encoder_view`` or ``cve`` -> ``validate`` returns
  ``False`` (and a missing/invalid positive -> ``build_sample`` returns ``None``).

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_record_adapter.py

Requirements: 7.4
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from contrastive_learning.data_structures import TrainingConfig, TrainingSample
from contrastive_learning.domain_adapters import get_domain_adapter
from cve_domain.record_adapter import CVERecordAdapter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> TrainingConfig:
    """A CVE-domain TrainingConfig (all other fields keep career-safe defaults)."""
    return TrainingConfig(domain_adapter="cve")


@pytest.fixture
def adapter(config: TrainingConfig) -> CVERecordAdapter:
    return CVERecordAdapter(config)


def _anchor_record(
    *,
    cve: str = "CVE-2024-27199",
    encoder_view: str = (
        "CVE-2024-27199. JetBrains TeamCity path traversal. CWE-22, CWE-23. "
        "CVSS 7.3 HIGH. In CISA KEV. Affects JetBrains TeamCity."
    ),
    cve_labels: Optional[Dict[str, Any]] = None,
    positive: Optional[Dict[str, Any]] = "__default__",
) -> Dict[str, Any]:
    """Build an anchor CVE_View_Record with a paired positive attached."""
    if cve_labels is None:
        cve_labels = {
            "priority_score": 99.05,
            "priority_band": "critical",
            "in_kev": True,
            "ransomware": True,
        }
    record: Dict[str, Any] = {
        "cve": cve,
        "encoder_view": encoder_view,
        "nvd_published": "2024-03-04T18:15:09.377",
        "cve_labels": cve_labels,
        "ontology": {"cwes": ["CWE-22", "CWE-23"], "cpes": [], "vendors": ["jetbrains"]},
    }
    if positive == "__default__":
        positive = {
            "cve": "CVE-2023-42793",
            "encoder_view": (
                "CVE-2023-42793. JetBrains TeamCity auth bypass. CWE-288. "
                "CVSS 9.8 CRITICAL. In CISA KEV. Affects JetBrains TeamCity."
            ),
        }
    if positive is not None:
        record["positive"] = positive
    return record


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_adapter_registered_under_cve(config: TrainingConfig) -> None:
    """The adapter is resolvable through the seam under the name "cve"."""
    resolved = get_domain_adapter("cve", config)
    assert isinstance(resolved, CVERecordAdapter)
    assert resolved.name == "cve"


# ---------------------------------------------------------------------------
# Two-slot anchor/positive encoder views
# ---------------------------------------------------------------------------


def test_two_slot_anchor_and_positive_views(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """anchor -> sample.resume (slot 1); positive -> sample.job (slot 2)."""
    record = _anchor_record()
    sample = adapter.build_sample(record, line_number=0, config=config)

    assert isinstance(sample, TrainingSample)
    # Anchor occupies the first view slot.
    assert sample.resume == {
        "cve": "CVE-2024-27199",
        "encoder_view": record["encoder_view"],
    }
    # Positive occupies the second view slot.
    assert sample.job == {
        "cve": "CVE-2023-42793",
        "encoder_view": record["positive"]["encoder_view"],
    }
    # The two slots hold genuinely different profile texts (a real pair).
    assert sample.resume["cve"] != sample.job["cve"]
    assert sample.resume["encoder_view"] != sample.job["encoder_view"]
    # The pair is a positive.
    assert sample.label == "positive"


def test_encoder_views_are_trimmed(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """Surrounding whitespace on cve / encoder_view is trimmed into the slots."""
    record = _anchor_record(cve="  CVE-2024-0001  ", encoder_view="  padded view text  ")
    sample = adapter.build_sample(record, line_number=1, config=config)

    assert sample is not None
    assert sample.resume["cve"] == "CVE-2024-0001"
    assert sample.resume["encoder_view"] == "padded view text"


# ---------------------------------------------------------------------------
# Group-id injection
# ---------------------------------------------------------------------------


def test_group_id_injection_uses_anchor_cve(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """metadata['resume_id'] == the anchor cve (each anchor is its own group)."""
    record = _anchor_record(cve="CVE-2024-27199")
    sample = adapter.build_sample(record, line_number=0, config=config)

    assert sample is not None
    assert sample.metadata["resume_id"] == "CVE-2024-27199"


def test_group_id_uses_trimmed_anchor_cve(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """The injected group id is the trimmed anchor cve, not the positive's cve."""
    record = _anchor_record(cve="  CVE-2024-0002  ")
    sample = adapter.build_sample(record, line_number=0, config=config)

    assert sample is not None
    assert sample.metadata["resume_id"] == "CVE-2024-0002"
    assert sample.metadata["resume_id"] != sample.job["cve"]


def test_existing_metadata_preserved_without_mutating_record(
    adapter: CVERecordAdapter, config: TrainingConfig
) -> None:
    """Existing metadata keys are preserved and the source record is not mutated."""
    record = _anchor_record()
    record["metadata"] = {"source": "nvd"}
    sample = adapter.build_sample(record, line_number=0, config=config)

    assert sample is not None
    assert sample.metadata["source"] == "nvd"
    assert sample.metadata["resume_id"] == "CVE-2024-27199"
    # Original record metadata must not gain a resume_id (no shared-dict mutation).
    assert "resume_id" not in record["metadata"]


# ---------------------------------------------------------------------------
# Label passthrough
# ---------------------------------------------------------------------------


def test_label_passthrough_all_present(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """metadata['cve_labels'] carries all present anchor labels verbatim."""
    labels = {
        "priority_score": 99.05,
        "priority_band": "critical",
        "in_kev": True,
        "ransomware": True,
    }
    record = _anchor_record(cve_labels=labels)
    sample = adapter.build_sample(record, line_number=0, config=config)

    assert sample is not None
    assert sample.metadata["cve_labels"] == labels


def test_label_passthrough_drops_none_values(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """Present-keys-only: None-valued labels are dropped from cve_labels."""
    record = _anchor_record(
        cve_labels={
            "priority_score": None,   # omitted / invalid upstream
            "priority_band": None,    # omitted / invalid upstream
            "in_kev": False,
            "ransomware": True,
        }
    )
    sample = adapter.build_sample(record, line_number=0, config=config)

    assert sample is not None
    assert sample.metadata["cve_labels"] == {"in_kev": False, "ransomware": True}


def test_label_passthrough_missing_labels_yields_empty_dict(
    adapter: CVERecordAdapter, config: TrainingConfig
) -> None:
    """A record with no cve_labels yields an empty labels dict (not an error)."""
    record = _anchor_record()
    del record["cve_labels"]
    sample = adapter.build_sample(record, line_number=0, config=config)

    assert sample is not None
    assert sample.metadata["cve_labels"] == {}


# ---------------------------------------------------------------------------
# build_sample skip behavior (missing / invalid positive or anchor)
# ---------------------------------------------------------------------------


def test_missing_positive_returns_none(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """No paired positive -> anchor excluded from Stage 1 (build_sample -> None)."""
    record = _anchor_record(positive=None)
    assert adapter.build_sample(record, line_number=0, config=config) is None


def test_positive_missing_encoder_view_returns_none(
    adapter: CVERecordAdapter, config: TrainingConfig
) -> None:
    """A positive lacking an encoder_view is invalid -> build_sample returns None."""
    record = _anchor_record(positive={"cve": "CVE-2023-42793"})
    assert adapter.build_sample(record, line_number=0, config=config) is None


def test_positive_empty_cve_returns_none(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """A positive with an empty cve is invalid -> build_sample returns None."""
    record = _anchor_record(positive={"cve": "   ", "encoder_view": "some text"})
    assert adapter.build_sample(record, line_number=0, config=config) is None


def test_anchor_missing_encoder_view_returns_none(
    adapter: CVERecordAdapter, config: TrainingConfig
) -> None:
    """An anchor lacking an encoder_view is invalid -> build_sample returns None."""
    record = _anchor_record(encoder_view="")
    assert adapter.build_sample(record, line_number=0, config=config) is None


def test_anchor_empty_cve_returns_none(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """An anchor with an empty cve is invalid -> build_sample returns None."""
    record = _anchor_record(cve="   ")
    assert adapter.build_sample(record, line_number=0, config=config) is None


def test_non_mapping_record_returns_none(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """A non-mapping record is skipped gracefully."""
    assert adapter.build_sample("not a dict", line_number=0, config=config) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_true_for_well_formed_anchor(adapter: CVERecordAdapter, config: TrainingConfig) -> None:
    """A sample whose anchor slot has non-empty cve + encoder_view validates True."""
    record = _anchor_record()
    sample = adapter.build_sample(record, line_number=0, config=config)
    assert sample is not None
    assert adapter.validate(sample) is True


def test_validate_false_for_empty_anchor_encoder_view(adapter: CVERecordAdapter) -> None:
    """Empty anchor encoder_view -> validate returns False."""
    sample = TrainingSample(
        resume={"cve": "CVE-2024-0001", "encoder_view": ""},
        job={"cve": "CVE-2024-0002", "encoder_view": "positive text"},
        label="positive",
        sample_id="cve_test_empty_view",
    )
    assert adapter.validate(sample) is False


def test_validate_false_for_empty_anchor_cve(adapter: CVERecordAdapter) -> None:
    """Empty anchor cve -> validate returns False."""
    sample = TrainingSample(
        resume={"cve": "   ", "encoder_view": "anchor text"},
        job={"cve": "CVE-2024-0002", "encoder_view": "positive text"},
        label="positive",
        sample_id="cve_test_empty_cve",
    )
    assert adapter.validate(sample) is False


def test_validate_false_for_missing_anchor_fields(adapter: CVERecordAdapter) -> None:
    """Anchor slot missing both cve and encoder_view -> validate returns False."""
    sample = TrainingSample(
        resume={},
        job={"cve": "CVE-2024-0002", "encoder_view": "positive text"},
        label="positive",
        sample_id="cve_test_missing_fields",
    )
    assert adapter.validate(sample) is False
