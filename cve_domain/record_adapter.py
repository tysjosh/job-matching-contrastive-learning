"""CVERecordAdapter — the Record_Adapter for the CVE domain (task 7.2).

This adapter plugs into the additive ``Domain_Adapter_Seam`` defined in
``contrastive_learning/domain_adapters.py``. It maps one anchor
``CVE_View_Record`` — paired by the ``CVEPositiveSelector`` (task 7.10) with a
selected *ontology-related* positive ``CVE_View_Record`` — onto the shared
``TrainingSample`` contract the ``DataLoader`` consumes.

Unlike the career domain, there is **no augmentation** and **no resume/job
mirror shim**. The anchor CVE and its ontology-related positive CVE are two
genuinely different vulnerability profile texts that form a real contrastive
pair — the direct analogue of the resume/job pair in the career domain. The two
``TrainingSample`` view slots (``resume`` and ``job``) therefore hold the anchor
and positive ``Profile_Text_View`` strings respectively:

- ``sample.resume`` -> ``{"cve": <anchor cve>, "encoder_view": <anchor text>}``
- ``sample.job``    -> ``{"cve": <positive cve>, "encoder_view": <positive text>}``
- ``label``         -> ``"positive"`` (the anchor/positive pair is a positive)
- ``metadata['resume_id']`` -> the anchor ``cve`` (each anchor is its own query
  group, so grouped/ordinal batching keeps working unchanged)
- ``metadata['cve_labels']`` -> the anchor's present supervised labels, carried
  through for Stage 2 supervised fine-tuning.

The paired positive is supplied on the anchor record under the ``"positive"``
key (a full ``CVE_View_Record`` dict) by the ``CVEPositiveSelector``. When no
positive is available (an anchor with no in-split ontology sibling), the anchor
is excluded from Stage 1 by returning ``None`` — no positive is fabricated
(Req 13.5).

Requirements: 7.4
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from contrastive_learning.data_structures import TrainingConfig, TrainingSample
from contrastive_learning.domain_adapters import register_domain_adapter


#: Key under which the paired positive ``CVE_View_Record`` is attached to the
#: anchor record by the ``CVEPositiveSelector``.
POSITIVE_KEY = "positive"

#: Key under which each view-slot dict carries its ``Profile_Text_View`` string.
ENCODER_VIEW_KEY = "encoder_view"

#: Key under which each record carries its ontology object ``{cwes, cpes, vendors}``.
ONTOLOGY_KEY = "ontology"

#: Default facet weights for the anchor↔positive ontology-overlap signal
#: (Requirement: CVE ontological signal for sample-level loss weighting). CWE is
#: the strongest ontology signal (matches the positive-selector cascade priority),
#: with CPE and vendor contributing the remainder. Weights are renormalized over
#: whichever facets carry data on at least one side, so sparse records are not
#: penalized for missing a facet. Overridable via ``config.cve_ontology_signal_weights``.
DEFAULT_ONTOLOGY_SIGNAL_WEIGHTS: Dict[str, float] = {
    "cwes": 0.5,
    "cpes": 0.25,
    "vendors": 0.25,
}

#: Neutral ontology-overlap signal (0.5 -> no modulation under the reused formula
#: ``weight = base * (1 + ontology_weight*(2*signal - 1))``). Injected when the
#: opt-in ontology-overlap ablation is disabled, so the per-sample weight is driven
#: purely by the independent label-completeness quality tier below.
CVE_NEUTRAL_ONTOLOGY_SIGNAL = 0.5

#: The two optional supervised targets whose presence defines a CVE anchor's
#: label-completeness quality tier (an ontology-independent data-quality signal).
#: ``in_kev`` / ``ransomware`` are always present (defaulted) so they carry no
#: completeness information and are excluded.
_COMPLETENESS_LABELS = ("priority_score", "priority_band")

#: Map the count of present completeness labels (0..2) to a career quality tier,
#: reusing the loss engine's ``tier_weights`` bases (A=1.0, C=0.75, F=0.5). A CVE
#: with both a valid priority_score and a priority_band is a fully-scored, higher
#: quality training anchor; one with neither is the weakest.
_COMPLETENESS_TIER = {2: "A", 1: "C", 0: "F"}


def _label_completeness_tier(present_labels: Mapping[str, Any]) -> str:
    """Return the quality tier from how many completeness labels are present.

    Independent of the ontology pairing (avoids the self-referential circularity
    of weighting by the same overlap that selected the positive): it reflects only
    how fully the CVE anchor is supervised/curated.
    """
    n = sum(1 for key in _COMPLETENESS_LABELS if key in present_labels)
    return _COMPLETENESS_TIER.get(n, "F")


def _clean_str(value: Any) -> str:
    """Return ``value`` as a whitespace-trimmed string (``""`` when falsy)."""
    if value is None:
        return ""
    return str(value).strip()


def _view_slot(view_record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a view-slot dict from a ``CVE_View_Record``.

    Returns a dict carrying the record's ``cve`` and ``encoder_view`` when both
    are present (non-empty), or ``None`` when either is missing so the caller
    can skip the record.
    """
    if not isinstance(view_record, Mapping):
        return None
    cve = _clean_str(view_record.get("cve"))
    encoder_view = _clean_str(view_record.get(ENCODER_VIEW_KEY))
    if not cve or not encoder_view:
        return None
    return {"cve": cve, ENCODER_VIEW_KEY: encoder_view}


def _present_labels(cve_labels: Any) -> Dict[str, Any]:
    """Return the present supervised labels from a record's ``cve_labels``.

    ``priority_score`` / ``priority_band`` are present-keys-only in a
    ``CVE_View_Record`` (omitted when invalid/absent), and ``in_kev`` /
    ``ransomware`` are always-present booleans. Any ``None`` value is dropped so
    only genuinely present labels are carried through.
    """
    if not isinstance(cve_labels, Mapping):
        return {}
    return {k: v for k, v in cve_labels.items() if v is not None}


def _token_set(ontology: Any, facet: str) -> set:
    """Return the trimmed, non-empty token set for one ontology facet.

    Reads ``ontology[facet]`` (a list on a ``CVE_View_Record``); tolerates a
    stray scalar and drops blank tokens so set overlap is well defined.
    """
    if not isinstance(ontology, Mapping):
        return set()
    value = ontology.get(facet)
    if value is None:
        return set()
    if isinstance(value, str):
        token = value.strip()
        return {token} if token else set()
    result: set = set()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            token = _clean_str(item)
            if token:
                result.add(token)
    return result


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity |a∩b| / |a∪b| (0.0 when the union is empty)."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def ontology_overlap_signal(
    anchor_ontology: Any,
    positive_ontology: Any,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Weighted-Jaccard ontology-overlap signal in ``[0, 1]`` for a CVE pair.

    Measures how strongly the anchor and its selected ontology-related positive
    share ontology structure — the CVE analogue of the career domain's
    ``ontology_similarity``. For each facet (``cwes``/``cpes``/``vendors``) it
    computes the Jaccard overlap of the two token sets, then averages the facets
    by their configured weights, **renormalizing over only the facets where at
    least one side carries data** (so a pair that shares a CWE but has no CPE/
    vendor data is not dragged down by the empty facets).

    Returns ``0.5`` (neutral -> weight 1.0 under the reused formula) when neither
    record carries any ontology facet, so a missing ontology never changes the
    loss. Higher values upweight strongly-overlapping pairs; lower values
    downweight weakly-related pairs.
    """
    facet_weights = dict(weights or DEFAULT_ONTOLOGY_SIGNAL_WEIGHTS)

    total_weight = 0.0
    accumulated = 0.0
    for facet, weight in facet_weights.items():
        if weight <= 0:
            continue
        anchor_tokens = _token_set(anchor_ontology, facet)
        positive_tokens = _token_set(positive_ontology, facet)
        if not anchor_tokens and not positive_tokens:
            # Neither side has data for this facet -> exclude from the average.
            continue
        accumulated += weight * _jaccard(anchor_tokens, positive_tokens)
        total_weight += weight

    if total_weight <= 0:
        # No facet had data on either side -> neutral (no weighting effect).
        return 0.5
    return accumulated / total_weight


class CVERecordAdapter:
    """Maps an anchor+positive CVE pair to a :class:`TrainingSample`.

    Instantiated by ``get_domain_adapter("cve", config)`` through the seam.
    """

    name = "cve"

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def build_sample(
        self,
        record: Dict[str, Any],
        line_number: int,
        config: TrainingConfig,
    ) -> Optional[TrainingSample]:
        """Build a positive contrastive sample from an anchor+positive CVE pair.

        ``record`` is the anchor ``CVE_View_Record`` with its selected positive
        ``CVE_View_Record`` attached under ``record["positive"]``. Returns
        ``None`` (skip) when the anchor is malformed or when no valid positive is
        paired with it (the anchor is then excluded from Stage 1 — Req 13.5).
        """
        if not isinstance(record, Mapping):
            return None

        # Anchor view slot (first slot): the anchor CVE's Profile_Text_View.
        anchor_slot = _view_slot(record)
        if anchor_slot is None:
            return None

        # Positive view slot (second slot): the ontology-related positive CVE.
        positive_record = record.get(POSITIVE_KEY)
        positive_slot = _view_slot(positive_record) if positive_record else None
        if positive_slot is None:
            # No in-split ontology sibling -> exclude the anchor, do not fabricate.
            return None

        # Grouping identifier: the anchor cve is its own query group, injected
        # into metadata['resume_id'] so grouped/ordinal batching is unchanged.
        anchor_cve = anchor_slot["cve"]
        raw_metadata = record.get("metadata")
        metadata: Dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata["resume_id"] = anchor_cve

        # Carry the anchor's present supervised labels for Stage 2.
        metadata["cve_labels"] = _present_labels(record.get("cve_labels"))

        # Sample-level loss weighting metadata (only active when the Run_Config
        # sets ontology_weight > 0; otherwise the loss engine returns weight 1.0
        # and these fields are ignored — fully opt-in / backward compatible).
        #
        # PRINCIPLED, ontology-INDEPENDENT signal: the quality tier reflects how
        # fully the CVE anchor is supervised (label completeness), the direct
        # analogue of the career "data quality" tier. This is the default weight.
        metadata.setdefault("quality_tier", _label_completeness_tier(metadata["cve_labels"]))

        # OPT-IN ABLATION: the anchor↔positive ontology overlap (weighted Jaccard
        # over CWE/CPE/vendor). This is self-referential — it is the same signal
        # that selected the positive — so it is OFF by default (neutral 0.5, no
        # modulation) and only enabled via config.cve_ontology_overlap_weighting.
        if getattr(config, "cve_ontology_overlap_weighting", False):
            signal_weights = getattr(config, "cve_ontology_signal_weights", None)
            metadata["ontology_similarity"] = ontology_overlap_signal(
                record.get(ONTOLOGY_KEY),
                positive_record.get(ONTOLOGY_KEY) if isinstance(positive_record, Mapping) else None,
                signal_weights,
            )
        else:
            metadata["ontology_similarity"] = CVE_NEUTRAL_ONTOLOGY_SIGNAL

        sample_id = record.get("sample_id") or f"cve_{anchor_cve}_{positive_slot['cve']}"

        return TrainingSample(
            resume=anchor_slot,
            job=positive_slot,
            label="positive",
            sample_id=str(sample_id),
            metadata=metadata,
        )

    def validate(self, sample: TrainingSample) -> bool:
        """Validate that the anchor slot carries a non-empty view and cve.

        The anchor lives in ``sample.resume`` (the first view slot). A CVE sample
        is valid when the anchor's ``encoder_view`` and ``cve`` are both
        non-empty (Req 1.6, 7.4). Resume/job-shaped fields are NOT required.
        """
        anchor = sample.resume
        if not isinstance(anchor, Mapping):
            return False
        return bool(_clean_str(anchor.get("cve"))) and bool(
            _clean_str(anchor.get(ENCODER_VIEW_KEY))
        )


# Register the adapter under "cve" as an import-time side effect (task 7.2).
# This is additive: it only adds a new entry to the registry and never mutates
# or overrides the default "career" adapter, so the career path is unaffected.
register_domain_adapter("cve", CVERecordAdapter)
