"""TrialsRecordAdapter — the Domain_Adapter for the TREC Clinical Trials domain.

Plugs into the additive ``Domain_Adapter_Seam`` in
``contrastive_learning/domain_adapters.py``, mapping one split record emitted by
:mod:`trials_domain.data_splitter` onto the shared ``TrainingSample`` contract.

Slot mapping
------------
The shared ``TrainingSample`` names its two view slots ``resume`` and ``job``.
Those names are structural, not semantic — the CVE domain already reuses them for
two vulnerability texts. Here:

  * ``sample.resume`` -> the **patient topic** (the query / anchor)
  * ``sample.job``    -> a **clinical trial** (the candidate)

Each slot carries a pre-serialized ``encoder_view`` plus both ontology facets
(``skill_uris`` = all MeSH descriptors, ``coarse_uris`` = condition descriptors
only). Emitting ``encoder_view`` is what bypasses the career-specific
role/experience/title/description text branches in the encoder: both
``trainer._encode_content_to_text_embedding`` and
``BatchEfficientEncoder._content_to_text`` check for it first.

No embedding-cache change is needed. ``encoder_view`` is already in the cache's
essential-fields allow-list (added for the CVE domain), and it fully determines
the encoded text, so distinct topics and trials hash to distinct content keys.
Had it not been present, every trials record would have normalized to an empty
dict and collapsed onto a single shared cache key.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from contrastive_learning.data_structures import TrainingConfig, TrainingSample
from contrastive_learning.domain_adapters import register_domain_adapter

#: Key carrying each slot's pre-serialized encoder text.
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
        # downstream ``.get('skill_uris', [])`` reads are uniform and a record
        # with no resolvable MeSH terms degrades to "no ontology signal" instead
        # of raising.
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


class TrialsRecordAdapter:
    """Maps a converted (topic, trial) record to a :class:`TrainingSample`.

    Instantiated via ``get_domain_adapter("trials", config)``.
    """

    name = "trials"

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def build_sample(
        self,
        record: Dict[str, Any],
        line_number: int,
        config: TrainingConfig,
    ) -> Optional[TrainingSample]:
        """Build a sample from one split record, or ``None`` to skip it."""
        if not isinstance(record, Mapping):
            return None

        anchor = _slot(record.get("resume"), "topic_id")
        candidate = _slot(record.get("job"), "nct_id")
        if anchor is None or candidate is None:
            return None

        raw_metadata = record.get("metadata")
        metadata: Dict[str, Any] = (
            dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        topic_id = anchor["topic_id"]
        # Query-group identity for grouped / ordinal batching. The topic is the
        # query, so it is the group key — the direct analogue of resume identity.
        metadata["resume_id"] = metadata.get("resume_id") or topic_id
        metadata.setdefault("topic_id", topic_id)
        metadata.setdefault("nct_id", candidate["nct_id"])

        # Graded relevance carried through for the ordinal/graded loss paths.
        # Grade 2 records are the positives; the graded negatives arrive through
        # the negative-selection seam, not as records.
        grade = metadata.get("grade", candidate.get("grade"))
        metadata["grade"] = grade
        metadata.setdefault(
            "original_label", candidate.get("original_label", "eligible")
        )

        label = "positive" if record.get("label", 1) in (1, "1", True) else "negative"

        return TrainingSample(
            resume=anchor,
            job=candidate,
            label=label,
            sample_id=f"{topic_id}__{candidate['nct_id']}",
            metadata=metadata,
        )

    def validate(self, sample: TrainingSample) -> bool:
        """A sample is valid when both slots carry an id and an encoder view.

        Career-shaped fields (``role``, ``experience``, ``title`` + ``description``)
        are deliberately not required.
        """
        anchor, candidate = sample.resume, sample.job
        if not isinstance(anchor, Mapping) or not isinstance(candidate, Mapping):
            return False
        return bool(
            _clean(anchor.get("topic_id"))
            and _clean(anchor.get(ENCODER_VIEW_KEY))
            and _clean(candidate.get("nct_id"))
            and _clean(candidate.get(ENCODER_VIEW_KEY))
        )


# Import-time registration, mirroring the career and CVE adapters. Additive: it
# only adds a registry entry and never overrides the default "career" adapter.
register_domain_adapter("trials", TrialsRecordAdapter)
