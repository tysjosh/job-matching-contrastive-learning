"""CVE Data_Converter — Profile_Text_View serializer (Requirement 2).

This module implements the deterministic ``Profile_Text_View`` serializer used to
turn a single ``CVE_Record`` (a CSV row, optionally joined to its
``Ontology_Profile``) into the single-string encoder input consumed by the frozen
sentence-transformer.

The serializer is a pure function of its inputs: given the same ``CVE_Record`` it
produces byte-identical output (Req 2.1). It follows a fixed segment order, a
presence rule (a field is present iff it exists and its whitespace-trimmed length
is at least 1), prefers ``vulnerability_name`` over ``short_description`` (Req 2.3),
splits/trims/drops-empty CWEs in source order (Req 2.5), renders at most a
configured number of ``cpes_sample`` values in source order (Req 2.6), emits no
placeholder text and no dangling delimiter for omitted segments (Req 2.4), always
includes the ``cve`` segment for a record that has a ``cve`` identifier (Req 2.7),
and truncates at a segment boundary when the untruncated view would exceed 2000
characters, reporting whether truncation occurred (Req 2.8).

The serializer (task 4.1) and the supervised-label extractor ``extract_labels``
(task 4.5) live here; ``CVEDataConverter.convert`` (task 4.7) is added by a later
task.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

# Maximum length of a Profile_Text_View (Req 2.8).
MAX_VIEW_CHARS = 2000

# Default maximum number of cpes_sample values rendered in the affected-product
# segment (Req 2.6).
DEFAULT_MAX_CPES = 5

# Delimiter that joins segments. The full view is the segments joined by ``". "``
# and terminated by ``"."`` — e.g. "seg1. seg2. seg3.".
_SEGMENT_JOINER = ". "
_SEGMENT_TERMINATOR = "."

# Recognized truthy / falsy tokens for boolean-style CSV fields such as
# ``in_kev`` and ``known_ransomware_campaign_use`` (e.g. "True"/"False" and
# "Known"/"Unknown"). Matching is done on the trimmed, lower-cased token.
_TRUTHY_TOKENS = frozenset({"true", "1", "yes", "y", "t", "known"})
_FALSY_TOKENS = frozenset({"false", "0", "no", "n", "f", "unknown", ""})

# Inclusive bounds for a valid Priority_Score value (Req 3.2).
PRIORITY_SCORE_MIN = 0.0
PRIORITY_SCORE_MAX = 100.0


@dataclass(frozen=True)
class ProfileTextView:
    """Result of serializing a CVE_Record into a Profile_Text_View.

    Attributes:
        text: The serialized, deterministic, length-bounded profile text view.
        truncated: True when the untruncated view would have exceeded
            ``MAX_VIEW_CHARS`` and the text was truncated at a segment boundary
            (Req 2.8). The Data_Converter aggregates this into the conversion
            report's ``truncation_count``.
    """

    text: str
    truncated: bool


def _present(value: Any) -> bool:
    """Return True when a field is present per Req 2.2.

    A field is present iff it exists (is not None) and its string form has a
    whitespace-trimmed length of at least 1 character.
    """
    if value is None:
        return False
    return len(str(value).strip()) >= 1


def _clean(value: Any) -> str:
    """Return the whitespace-trimmed string form of a value."""
    return str(value).strip()


def _is_truthy(value: Any) -> bool:
    """Return True when a boolean-style field holds a recognized truthy token."""
    if value is None:
        return False
    return _clean(value).lower() in _TRUTHY_TOKENS


def _split_tokens(value: Any, delimiter: str) -> List[str]:
    """Split on ``delimiter``, trim each token, and drop empty tokens.

    Preserves source order (Req 2.5 for CWEs, Req 2.6 for CPEs).
    """
    if value is None:
        return []
    tokens = []
    for raw in str(value).split(delimiter):
        token = raw.strip()
        if token:
            tokens.append(token)
    return tokens


def _get(row: Mapping[str, Any], profile: Optional[Mapping[str, Any]], key: str) -> Any:
    """Fetch a field from the CSV row, falling back to the ontology profile.

    The CSV row is authoritative; the joined ontology profile only supplies a
    value when the row does not carry a present one.
    """
    value = row.get(key) if row is not None else None
    if _present(value):
        return value
    if profile is not None:
        return profile.get(key)
    return None


def _build_segments(
    row: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]],
    max_cpes: int,
    exclude_leakage: bool = False,
) -> List[str]:
    """Build the ordered, present-only segment list for the Profile_Text_View.

    Fixed segment order (Req 2.2):
        1. cve identifier (always present when the record has a cve — Req 2.7)
        2. vulnerability_name OR short_description (prefer name; never both — Req 2.3)
        3. cwes (";"-split, trimmed, empties dropped, source order — Req 2.5)
        4. CVSS "CVSS {cvss_base_score} {cvss_base_severity}"
        5. epss "EPSS {epss}"
        6. in_kev -> "In CISA KEV" (only when truthy)
        7. known_ransomware_campaign_use -> "Known ransomware use" (only when truthy)
        8. affected product: vendor_project / product + up to max_cpes cpes_sample
    Missing/empty fields contribute no segment (Req 2.4).

    Leakage control: ``priority_score`` (the ranking target) is a deterministic
    formula dominated by EPSS (~0.70 weight) plus the CISA-KEV and ransomware
    flags. Segments 4–7 (CVSS, EPSS, "In CISA KEV", "Known ransomware use")
    therefore leak the label's own inputs into the encoder's input text. When
    ``exclude_leakage`` is True these four segments are dropped, leaving a purely
    *semantic* description (identifier, name/description, CWEs, affected product)
    so the model must predict priority from meaning rather than by copying the
    EPSS/KEV tokens it was handed. Default False preserves the original view.
    """
    segments: List[str] = []

    # 1. cve identifier
    cve = _get(row, profile, "cve")
    if _present(cve):
        segments.append(_clean(cve))

    # 2. vulnerability_name preferred over short_description; never both (Req 2.3)
    name = _get(row, profile, "vulnerability_name")
    if _present(name):
        segments.append(_clean(name))
    else:
        short_desc = _get(row, profile, "short_description")
        if _present(short_desc):
            segments.append(_clean(short_desc))

    # 3. cwes — split/trim/drop-empty in source order (Req 2.5)
    cwe_tokens = _split_tokens(_get(row, profile, "cwes"), ";")
    if cwe_tokens:
        segments.append(", ".join(cwe_tokens))

    # Segments 4–7 carry the label's own inputs (EPSS dominates priority_score,
    # plus the KEV / ransomware flags). Skip them entirely when excluding leakage
    # so the view is a purely semantic description (Req: no-leakage ablation).
    if not exclude_leakage:
        # 4. CVSS — combine the present score/severity parts, no dangling delimiter
        cvss_score = _get(row, profile, "cvss_base_score")
        cvss_severity = _get(row, profile, "cvss_base_severity")
        cvss_parts: List[str] = []
        if _present(cvss_score):
            cvss_parts.append(_clean(cvss_score))
        if _present(cvss_severity):
            cvss_parts.append(_clean(cvss_severity))
        if cvss_parts:
            segments.append("CVSS " + " ".join(cvss_parts))

        # 5. epss
        epss = _get(row, profile, "epss")
        if _present(epss):
            segments.append("EPSS " + _clean(epss))

        # 6. in_kev -> only when truthy
        if _is_truthy(_get(row, profile, "in_kev")):
            segments.append("In CISA KEV")

        # 7. known_ransomware_campaign_use -> only when truthy
        if _is_truthy(_get(row, profile, "known_ransomware_campaign_use")):
            segments.append("Known ransomware use")

    # 8. affected product: vendor_project / product + up to max_cpes cpes_sample
    vendor = _get(row, profile, "vendor_project")
    product = _get(row, profile, "product")
    cpe_tokens = _split_tokens(_get(row, profile, "cpes_sample"), ";")
    if max_cpes >= 0:
        cpe_tokens = cpe_tokens[:max_cpes]
    affected_parts: List[str] = []
    if _present(vendor):
        affected_parts.append(_clean(vendor))
    if _present(product):
        affected_parts.append(_clean(product))
    if affected_parts or cpe_tokens:
        affected = "Affects"
        if affected_parts:
            affected += " " + " ".join(affected_parts)
        if cpe_tokens:
            affected += " " + ", ".join(cpe_tokens)
        segments.append(affected)

    return segments


def _render(segments: List[str]) -> str:
    """Join segments with ``". "`` and terminate with ``"."`` (empty -> "")."""
    if not segments:
        return ""
    return _SEGMENT_JOINER.join(segments) + _SEGMENT_TERMINATOR


def _truncate_at_segment_boundary(segments: List[str], max_chars: int) -> str:
    """Return the longest rendered prefix of ``segments`` within ``max_chars``.

    Truncation happens only at a segment boundary (Req 2.8): whole segments are
    kept or dropped, never split mid-segment. At least the first segment (the cve
    identifier) is retained so the view stays non-empty and carries the cve
    segment (Req 2.7), even in the pathological case where a single segment alone
    exceeds ``max_chars``.
    """
    if not segments:
        return ""

    kept = 0
    for count in range(1, len(segments) + 1):
        candidate = _render(segments[:count])
        if len(candidate) <= max_chars:
            kept = count
        else:
            break

    # Always keep at least the first segment (Req 2.7).
    if kept == 0:
        kept = 1

    return _render(segments[:kept])


def build_profile_text_view_report(
    row: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]] = None,
    *,
    max_cpes: int = DEFAULT_MAX_CPES,
    max_chars: int = MAX_VIEW_CHARS,
    exclude_leakage: bool = False,
) -> ProfileTextView:
    """Serialize a CVE_Record into a Profile_Text_View with a truncation flag.

    Args:
        row: The CSV-derived CVE_Record fields.
        profile: The optionally-joined Ontology_Profile; supplies a value only
            when the row lacks a present one.
        max_cpes: Maximum number of cpes_sample values to render (Req 2.6).
        max_chars: Maximum view length before segment-boundary truncation (Req 2.8).
        exclude_leakage: When True, drop the CVSS / EPSS / KEV / ransomware
            segments that leak the label's own inputs (see ``_build_segments``).

    Returns:
        A ``ProfileTextView`` carrying the deterministic, length-bounded text and
        whether the untruncated view would have exceeded ``max_chars``.
    """
    segments = _build_segments(row, profile, max_cpes, exclude_leakage=exclude_leakage)
    full = _render(segments)

    if len(full) <= max_chars:
        return ProfileTextView(text=full, truncated=False)

    truncated_text = _truncate_at_segment_boundary(segments, max_chars)
    return ProfileTextView(text=truncated_text, truncated=True)


def build_profile_text_view(
    row: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]] = None,
    *,
    max_cpes: int = DEFAULT_MAX_CPES,
    max_chars: int = MAX_VIEW_CHARS,
    exclude_leakage: bool = False,
) -> str:
    """Serialize a CVE_Record into its Profile_Text_View string (Req 2).

    Thin wrapper over :func:`build_profile_text_view_report` that returns just the
    text, matching the design's ``build_profile_text_view`` signature.
    """
    return build_profile_text_view_report(
        row, profile, max_cpes=max_cpes, max_chars=max_chars, exclude_leakage=exclude_leakage
    ).text


@dataclass(frozen=True)
class ExtractedLabels:
    """Result of extracting the four supervised labels from a CVE_Record (Req 3).

    Attributes:
        labels: The supervised-label dict attached to a CVE_View_Record's
            ``cve_labels``. ``priority_score`` and ``priority_band`` are
            present-keys-only (included only when valid/present — Req 3.2, 3.3,
            3.6, 3.7), while ``in_kev`` and ``ransomware`` are always present
            booleans (Req 3.1, 3.4, 3.5).
        priority_score_omitted: True when Priority_Score was missing, non-numeric,
            or outside ``[0, 100]`` and therefore omitted (Req 3.3). The
            Data_Converter aggregates this into ``priority_score_omitted_count``.
        priority_band_omitted: True when Priority_Band was missing or empty and
            therefore omitted (Req 3.7). Aggregated into
            ``priority_band_omitted_count``.
        defaulted_label_count: Number of boolean labels (0, 1, or 2) that were
            missing or unparseable and defaulted to False (Req 3.5). Aggregated
            into the report's ``defaulted_label_count``.
    """

    labels: dict
    priority_score_omitted: bool
    priority_band_omitted: bool
    defaulted_label_count: int


def _parse_bool(value: Any) -> Optional[bool]:
    """Parse a boolean-style field into True / False / None (Req 3.4, 3.5).

    Returns True for a recognized truthy token, False for a recognized falsy
    token, and None when the value is missing or cannot be parsed as a boolean
    (the caller defaults None to False and counts it).
    """
    if value is None:
        return None
    token = _clean(value).lower()
    if token in _TRUTHY_TOKENS:
        return True
    if token in _FALSY_TOKENS:
        return False
    return None


def _parse_priority_score(value: Any) -> Optional[float]:
    """Parse a Priority_Score into a float within ``[0, 100]`` or None (Req 3.2).

    Returns the numeric value only when it is present, numeric, and within the
    inclusive range ``[0, 100]``; otherwise returns None so the caller omits and
    counts it (Req 3.3). Booleans are rejected (a bool is not a Priority_Score),
    and NaN / infinity are rejected as out of range.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        num = float(value)
    else:
        token = _clean(value)
        if not token:
            return None
        try:
            num = float(token)
        except (TypeError, ValueError):
            return None

    # Reject NaN (which compares unequal to itself) and any out-of-range value;
    # infinities fall outside the range and are rejected here too.
    if num != num:
        return None
    if num < PRIORITY_SCORE_MIN or num > PRIORITY_SCORE_MAX:
        return None
    return num


def _parse_cvss_score(value: Any) -> Optional[float]:
    """Parse a CVSS base score into a float within ``[0, 10]`` or None.

    Used only for the alternate CVSS target. Booleans and NaN/inf are rejected;
    out-of-range values are treated as absent so the caller omits + counts them,
    mirroring :func:`_parse_priority_score`.
    """
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    try:
        num = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    if num < 0.0 or num > 10.0:
        return None
    return num


# Valid target labels for extract_labels_report. "priority" is the default
# EPSS-derived priority_score/priority_band; "cvss" swaps in the CVSS base
# score/severity (an independent ordinal target) via the same label slots.
TARGET_LABEL_PRIORITY = "priority"
TARGET_LABEL_CVSS = "cvss"


def extract_labels_report(row: Mapping[str, Any], target_label: str = TARGET_LABEL_PRIORITY) -> ExtractedLabels:
    """Extract the four supervised labels from a CVE_Record with accounting (Req 3).

    ``target_label`` selects what fills the ordinal ``priority_score`` /
    ``priority_band`` slots:

    * ``"priority"`` (default): the EPSS-derived ``priority_score`` (0–100) and
      its ``priority_band`` bucketing — the original behavior.
    * ``"cvss"``: the CVSS base score (0–10, scaled ×10 to the 0–100 range so the
      Stage 2 regression normalization and evaluation cut points are unchanged)
      and the lower-cased CVSS ``cvss_base_severity`` as the band. This is an
      ordinal target independent of the EPSS formula, used to show the method is
      not tuned to one label definition. ``in_kev`` / ``ransomware`` are unchanged.
    

    Produces the ``cve_labels`` dict attached to each CVE_View_Record plus the
    per-record counts the conversion report aggregates:

    - ``priority_score``: stored unchanged only when present, numeric, and within
      ``[0, 100]`` (Req 3.2); otherwise omitted and flagged (Req 3.3).
    - ``priority_band``: stored as the trimmed categorical value when present
      (Req 3.6); otherwise omitted and flagged (Req 3.7).
    - ``in_kev`` / ``ransomware``: booleans parsed from ``in_kev`` and
      ``known_ransomware_campaign_use``, always present, defaulting to False and
      counted when missing or unparseable (Req 3.4, 3.5).

    Args:
        row: The CSV-derived CVE_Record fields.

    Returns:
        An ``ExtractedLabels`` carrying the labels dict and the omission /
        defaulting counts.
    """
    labels: dict = {}

    # Select the source of the ordinal score/band by target (default: priority).
    if target_label == TARGET_LABEL_CVSS:
        cvss_num = _parse_cvss_score(row.get("cvss_base_score"))
        score = None if cvss_num is None else cvss_num * 10.0  # 0–10 -> 0–100
        band_source = row.get("cvss_base_severity")
        band_is_cvss = True
    else:
        score = _parse_priority_score(row.get("priority_score"))
        band_source = row.get("priority_band")
        band_is_cvss = False

    # priority_score slot — present-key-only, numeric within [0, 100] (Req 3.2, 3.3)
    priority_score_omitted = score is None
    if score is not None:
        labels["priority_score"] = score

    # priority_band slot — present-key-only categorical (Req 3.6, 3.7). CVSS
    # severities are lower-cased so they match the configured band order.
    priority_band_omitted = not _present(band_source)
    if not priority_band_omitted:
        labels["priority_band"] = _clean(band_source).lower() if band_is_cvss else _clean(band_source)

    # in_kev / ransomware — always-present booleans, default False + count (Req 3.4, 3.5)
    defaulted_label_count = 0

    in_kev = _parse_bool(row.get("in_kev"))
    if in_kev is None:
        in_kev = False
        defaulted_label_count += 1
    labels["in_kev"] = in_kev

    ransomware = _parse_bool(row.get("known_ransomware_campaign_use"))
    if ransomware is None:
        ransomware = False
        defaulted_label_count += 1
    labels["ransomware"] = ransomware

    return ExtractedLabels(
        labels=labels,
        priority_score_omitted=priority_score_omitted,
        priority_band_omitted=priority_band_omitted,
        defaulted_label_count=defaulted_label_count,
    )


def extract_labels(row: Mapping[str, Any], target_label: str = TARGET_LABEL_PRIORITY) -> dict:
    """Extract the four supervised labels from a CVE_Record (Req 3).

    Thin wrapper over :func:`extract_labels_report` that returns just the
    ``cve_labels`` dict (present-keys-only for ``priority_score`` /
    ``priority_band``, always-present booleans for ``in_kev`` / ``ransomware``),
    matching the design's ``extract_labels`` signature. ``target_label`` selects
    the ordinal target (``"priority"`` default or ``"cvss"``).
    """
    return extract_labels_report(row, target_label).labels


# ---------------------------------------------------------------------------
# CVEDataConverter — join, dedup, validate, emit JSONL + conversion report (Req 1)
# ---------------------------------------------------------------------------


@dataclass
class CVEConvertConfig:
    """Configuration for :class:`CVEDataConverter` (task 4.7).

    Attributes:
        domain_adapter: Name of the configured Domain_Adapter whose ``validate()``
            hook gates each emitted record. For the CVE domain this is ``"cve"``
            (the ``CVERecordAdapter``), whose contract is a non-empty
            ``encoder_view`` and a non-empty ``cve`` — not the legacy resume/job
            validators (Req 1.6). When the named adapter is not registered, the
            converter falls back to the equivalent built-in CVE predicate
            (:func:`validate_cve_view_record`), so conversion never depends on a
            not-yet-wired adapter.
        max_cpes: Maximum number of ``cpes_sample`` values rendered in the
            Profile_Text_View affected-product segment (Req 2.6).
        max_view_chars: Maximum Profile_Text_View length before segment-boundary
            truncation (Req 2.8).
        exclude_leakage_segments: When True, the Profile_Text_View omits the
            CVSS / EPSS / "In CISA KEV" / "Known ransomware use" segments, which
            leak the label's own inputs (priority_score is an EPSS+KEV formula).
            Used for the no-leakage ablation so the model predicts priority from
            the semantic description alone. Default False keeps the original view.
    """

    domain_adapter: str = "cve"
    max_cpes: int = DEFAULT_MAX_CPES
    max_view_chars: int = MAX_VIEW_CHARS
    exclude_leakage_segments: bool = False
    cve_target_label: str = "priority"


@dataclass
class ConversionReport:
    """Exact accounting of a conversion run (design "ConversionReport", Req 1.10).

    The four partitioning counts satisfy the invariant (Property 2):

        ``input_rows == emitted_records + skipped_row_count
                        + duplicate_cve_count + validation_failure_count``

    Attributes:
        input_rows: Total CSV data rows read (excludes the header).
        emitted_records: CVE_View_Records written to the output JSONL.
        missing_or_empty_field_count: Rows skipped specifically because the ``cve``
            identifier was missing or empty after trimming (Req 1.8); a descriptive
            subset of ``skipped_row_count``.
        missing_profile_count: Emitted records that had no matching Ontology_Profile
            and were built from CSV-derived fields only (Req 1.5).
        duplicate_cve_count: Later CSV occurrences of an already-seen ``cve``
            identifier that were discarded (Req 1.7).
        skipped_row_count: Rows skipped because they could not be parsed or had a
            missing/empty ``cve`` identifier (Req 1.8).
        validation_failure_count: Constructed records that failed the configured
            Domain_Adapter's ``validate()`` hook and were excluded (Req 1.9).
        truncation_count: Records whose Profile_Text_View was truncated at a segment
            boundary because it would have exceeded ``max_view_chars`` (Req 2.8).
        priority_score_omitted_count: Records whose Priority_Score was missing,
            non-numeric, or out of ``[0, 100]`` and therefore omitted (Req 3.3).
        priority_band_omitted_count: Records whose Priority_Band was missing/empty
            and therefore omitted (Req 3.7).
        defaulted_label_count: Boolean labels (``in_kev`` / ``ransomware``) that were
            missing/unparseable and defaulted to False (Req 3.5).
    """

    input_rows: int = 0
    emitted_records: int = 0
    missing_or_empty_field_count: int = 0
    missing_profile_count: int = 0
    duplicate_cve_count: int = 0
    skipped_row_count: int = 0
    validation_failure_count: int = 0
    truncation_count: int = 0
    priority_score_omitted_count: int = 0
    priority_band_omitted_count: int = 0
    defaulted_label_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "emitted_records": self.emitted_records,
            "missing_or_empty_field_count": self.missing_or_empty_field_count,
            "missing_profile_count": self.missing_profile_count,
            "duplicate_cve_count": self.duplicate_cve_count,
            "skipped_row_count": self.skipped_row_count,
            "validation_failure_count": self.validation_failure_count,
            "truncation_count": self.truncation_count,
            "priority_score_omitted_count": self.priority_score_omitted_count,
            "priority_band_omitted_count": self.priority_band_omitted_count,
            "defaulted_label_count": self.defaulted_label_count,
        }


def validate_cve_view_record(record: Mapping[str, Any]) -> bool:
    """Built-in CVE-domain validation predicate (Req 1.6).

    Mirrors ``CVERecordAdapter.validate`` (task 7.2): a CVE_View_Record is valid iff
    it carries a non-empty ``encoder_view`` and a non-empty ``cve`` identifier. This
    is deliberately *not* the legacy resume/job validation — the CVE record does not
    carry resume/job-shaped fields.
    """
    return _present(record.get("cve")) and _present(record.get("encoder_view"))


def _build_ontology(
    row: Mapping[str, Any], profile: Optional[Mapping[str, Any]]
) -> Dict[str, List[str]]:
    """Build the ``ontology`` object ``{cwes, cpes, vendors}`` for a CVE_View_Record.

    Values come from the joined Ontology_Profile when present (its ``cwes`` /
    ``cpes`` / ``vendors`` are already list-valued). For a missing-profile join
    (Req 1.5), the ontology is derived from CSV fields only: ``cwes`` and
    ``cpes_sample`` are ``;``-split/trimmed in source order and ``vendor_project``
    supplies the single vendor. The ``CVEPositiveSelector`` (task 7.10) reads this
    object to find ontology-related positives.
    """
    if profile is not None:
        def _as_str_list(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [_clean(item) for item in value if _present(item)]
            # Fall back to a ;-split for a scalar/string profile value.
            return _split_tokens(value, ";")

        return {
            "cwes": _as_str_list(profile.get("cwes")),
            "cpes": _as_str_list(profile.get("cpes")),
            "vendors": _as_str_list(profile.get("vendors")),
        }

    # CSV-only derivation.
    vendors: List[str] = []
    if _present(row.get("vendor_project")):
        vendors.append(_clean(row.get("vendor_project")))
    return {
        "cwes": _split_tokens(row.get("cwes"), ";"),
        "cpes": _split_tokens(row.get("cpes_sample"), ";"),
        "vendors": vendors,
    }


class CVEDataConverter:
    """Convert the CVE CSV + Ontology_Profiles into CDCL-compatible view records.

    Implements the join / dedup / validate / emit algorithm of Requirement 1
    (design section 4 "Data_Converter"):

    1. Index profiles by ``cve`` (first-wins per unique id).
    2. Stream CSV rows; skip rows with a missing/empty ``cve`` (count), and discard
       later duplicates of an already-emitted ``cve`` (count).
    3. Join each row to its profile by ``cve``; a missing profile yields a CSV-only
       view and is counted.
    4. Build the Profile_Text_View (Req 2) and the four supervised labels (Req 3).
    5. Assemble the CVE_View_Record (``cve``, ``encoder_view``, ``nvd_published``,
       ``cve_labels``, ``ontology`` + ``metadata['resume_id'] = cve``) and validate
       it through the configured Domain_Adapter's ``validate()`` hook; excluded +
       counted on failure.
    6. Emit UTF-8 JSONL, one object per line.
    7. Write ``conversion_report.json`` with every required count.
    """

    def __init__(self, config: Optional[CVEConvertConfig] = None):
        self.config = config or CVEConvertConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def convert(self, csv_path: str, profiles_path: str, out_path: str) -> ConversionReport:
        """Convert ``csv_path`` + ``profiles_path`` into ``out_path`` JSONL.

        Writes the emitted CVE_View_Records to ``out_path`` (UTF-8, one JSON object
        per line, Req 1.2) and a sibling ``conversion_report.json`` enumerating every
        required count (Req 1.10). Returns the :class:`ConversionReport`.
        """
        report = ConversionReport()
        profiles = self._index_profiles(profiles_path)
        validator = self._resolve_validator()

        out_file = Path(out_path)
        if out_file.parent and str(out_file.parent):
            out_file.parent.mkdir(parents=True, exist_ok=True)

        seen: set = set()

        with open(csv_path, "r", encoding="utf-8", newline="") as csv_handle, open(
            out_file, "w", encoding="utf-8"
        ) as out_handle:
            reader = csv.DictReader(csv_handle)
            for row in reader:
                report.input_rows += 1

                # Req 1.8: skip rows with a missing/empty cve identifier.
                cve_raw = row.get("cve")
                if not _present(cve_raw):
                    report.skipped_row_count += 1
                    report.missing_or_empty_field_count += 1
                    continue
                cve = _clean(cve_raw)

                # Req 1.7: discard later duplicates, keep the first occurrence.
                if cve in seen:
                    report.duplicate_cve_count += 1
                    continue

                # Req 1.4 / 1.5: join to the profile; missing profile → CSV-only.
                profile = profiles.get(cve)
                if profile is None:
                    report.missing_profile_count += 1

                record = self._build_record(cve, row, profile, report)

                # Req 1.6 / 1.9: validate via the configured adapter's hook.
                if not validator(record):
                    report.validation_failure_count += 1
                    continue

                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(cve)
                report.emitted_records += 1

        self._write_report(out_file, report)
        logger.info(
            "Conversion complete: %d/%d rows emitted (%d skipped, %d duplicates, "
            "%d validation failures, %d missing profiles)",
            report.emitted_records,
            report.input_rows,
            report.skipped_row_count,
            report.duplicate_cve_count,
            report.validation_failure_count,
            report.missing_profile_count,
        )
        return report

    def build_profile_text_view(
        self, row: Mapping[str, Any], profile: Optional[Mapping[str, Any]] = None
    ) -> str:
        """Serialize a CVE_Record into its Profile_Text_View (design signature)."""
        return build_profile_text_view(
            row, profile, max_cpes=self.config.max_cpes, max_chars=self.config.max_view_chars,
            exclude_leakage=self.config.exclude_leakage_segments,
        )

    def extract_labels(self, row: Mapping[str, Any]) -> dict:
        """Extract the four supervised labels from a CVE_Record (design signature)."""
        return extract_labels(row, self.config.cve_target_label)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_record(
        self,
        cve: str,
        row: Mapping[str, Any],
        profile: Optional[Mapping[str, Any]],
        report: ConversionReport,
    ) -> Dict[str, Any]:
        """Assemble one CVE_View_Record and fold its per-record counts into report."""
        view = build_profile_text_view_report(
            row, profile, max_cpes=self.config.max_cpes, max_chars=self.config.max_view_chars,
            exclude_leakage=self.config.exclude_leakage_segments,
        )
        if view.truncated:
            report.truncation_count += 1

        labels = extract_labels_report(row, self.config.cve_target_label)
        if labels.priority_score_omitted:
            report.priority_score_omitted_count += 1
        if labels.priority_band_omitted:
            report.priority_band_omitted_count += 1
        report.defaulted_label_count += labels.defaulted_label_count

        published = row.get("nvd_published")
        nvd_published = _clean(published) if _present(published) else None

        record: Dict[str, Any] = {
            "cve": cve,
            "encoder_view": view.text,
            "nvd_published": nvd_published,
            "cve_labels": labels.labels,
            "ontology": _build_ontology(row, profile),
            "metadata": {"resume_id": cve},
        }
        return record

    def _resolve_validator(self) -> Callable[[Mapping[str, Any]], bool]:
        """Resolve the record validator, honoring the Domain_Adapter_Seam.

        Prefers a record-level ``validate_view_record`` hook exposed by the
        configured Domain_Adapter, so validation routes through the adapter once the
        ``CVERecordAdapter`` (task 7.2) is registered. Falls back to the equivalent
        built-in CVE predicate when the adapter is not registered or exposes no
        record-level hook, keeping conversion functional and independent of the
        not-yet-wired adapter.
        """
        try:
            from contrastive_learning.domain_adapters import get_domain_adapter

            adapter = get_domain_adapter(self.config.domain_adapter, self.config)
        except Exception:  # not registered / needs a different config — use fallback
            return validate_cve_view_record

        hook = getattr(adapter, "validate_view_record", None)
        if callable(hook):
            return hook
        return validate_cve_view_record

    @staticmethod
    def _index_profiles(profiles_path: str) -> Dict[str, Dict[str, Any]]:
        """Index Ontology_Profiles by ``cve`` (first-wins per unique id).

        Skips blank and unparseable JSONL lines and any profile without a present
        ``cve`` identifier. The first profile seen for a ``cve`` wins.
        """
        profiles: Dict[str, Dict[str, Any]] = {}
        with open(profiles_path, "r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid profile JSON at line %d: %s", line_num, exc)
                    continue
                if not isinstance(obj, Mapping):
                    continue
                cve = obj.get("cve")
                if not _present(cve):
                    continue
                key = _clean(cve)
                if key not in profiles:  # first-wins
                    profiles[key] = dict(obj)
        return profiles

    @staticmethod
    def _write_report(out_file: Path, report: ConversionReport) -> None:
        """Write ``conversion_report.json`` alongside the emitted JSONL (Req 1.10)."""
        report_path = out_file.parent / "conversion_report.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
