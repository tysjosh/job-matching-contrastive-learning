"""Maps a GO/PPI split record to a ``TrainingSample`` (``domain_adapter="go_ppi"``).

Structurally identical to ``trials_domain/record_adapter.py``; only the two id keys
differ (``protein_id`` / ``partner_id`` instead of ``topic_id`` / ``nct_id``).

``encoder_view`` is the load-bearing key. It bypasses the career-specific text
serialization in both ``trainer._encode_content_to_text_embedding`` and
``BatchEfficientEncoder._content_to_text``, and it is already in the embedding
cache's essential-fields allow-list. Without it every protein would collapse onto
one cache key and the encoder would see identical text for all of them.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from contrastive_learning.data_structures import TrainingConfig, TrainingSample
from contrastive_learning.domain_adapters import register_domain_adapter

ENCODER_VIEW_KEY = "encoder_view"


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _slot(raw: Any, id_key: str) -> Optional[Dict[str, Any]]:
    """Validate and copy one view slot, or ``None`` when unusable."""
    if not isinstance(raw, Mapping):
        return None
    identifier = _clean(raw.get(id_key))
    view = _clean(raw.get(ENCODER_VIEW_KEY))
    if not identifier or not view:
        return None
    slot: Dict[str, Any] = {
        id_key: identifier,
        ENCODER_VIEW_KEY: view,
        # Both facets default to empty lists rather than being omitted, so
        # downstream ``.get('skill_uris', [])`` reads are uniform and a protein
        # with no usable GO terms degrades to "no ontology signal" rather than
        # raising.
        "skill_uris": list(raw.get("skill_uris") or []),
        "coarse_uris": list(raw.get("coarse_uris") or []),
    }
    if raw.get("title"):
        slot["title"] = _clean(raw["title"])
    if raw.get("grade") is not None:
        slot["grade"] = raw["grade"]
    if raw.get("original_label"):
        slot["original_label"] = _clean(raw["original_label"])
    return slot


class GoPpiRecordAdapter:
    """Maps an (anchor protein, partner protein) record to a ``TrainingSample``.

    Instantiated via ``get_domain_adapter("go_ppi", config)``.
    """

    name = "go_ppi"

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def build_sample(
        self,
        record: Dict[str, Any],
        line_number: int,
        config: TrainingConfig,
    ) -> Optional[TrainingSample]:
        if not isinstance(record, Mapping):
            return None

        anchor = _slot(record.get("resume"), "protein_id")
        candidate = _slot(record.get("job"), "partner_id")
        if anchor is None or candidate is None:
            return None

        raw_metadata = record.get("metadata")
        metadata: Dict[str, Any] = (
            dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        protein_id = anchor["protein_id"]
        # Query-group identity for grouped / ordinal batching. The anchor protein
        # is the query, so it is the group key -- the analogue of resume identity.
        metadata["resume_id"] = metadata.get("resume_id") or protein_id
        metadata.setdefault("protein_id", protein_id)
        metadata.setdefault("partner_id", candidate["partner_id"])

        # Graded relevance carried through for the ordinal/graded loss paths.
        # Grade 2 records are the positives; grade 1 and 0 negatives arrive
        # through the negative-selection seam, not as training records.
        grade = metadata.get("grade", candidate.get("grade"))
        metadata["grade"] = grade
        metadata.setdefault(
            "original_label", candidate.get("original_label", "good_fit"))

        label = "positive" if record.get("label", 1) in (1, "1", True) else "negative"

        return TrainingSample(
            resume=anchor,
            job=candidate,
            label=label,
            sample_id=f"{protein_id}__{candidate['partner_id']}",
            metadata=metadata,
        )

    def validate(self, sample: TrainingSample) -> bool:
        """Valid when both slots carry an id and a non-empty encoder view.

        Career-shaped fields (``role``, ``experience``, ``title`` + ``description``)
        are deliberately not required.
        """
        anchor, candidate = sample.resume, sample.job
        if not isinstance(anchor, Mapping) or not isinstance(candidate, Mapping):
            return False
        return bool(
            _clean(anchor.get("protein_id"))
            and _clean(anchor.get(ENCODER_VIEW_KEY))
            and _clean(candidate.get("partner_id"))
            and _clean(candidate.get(ENCODER_VIEW_KEY))
        )


# Import-time registration, mirroring the career, trials and CVE adapters.
# Additive: adds a registry entry, never overrides the default "career".
register_domain_adapter("go_ppi", GoPpiRecordAdapter)
