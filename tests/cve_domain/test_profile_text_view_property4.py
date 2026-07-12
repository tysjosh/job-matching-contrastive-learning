"""Property-based tests for the CVE Profile_Text_View serializer (Property 4).

# Feature: cve-vulnerability-ranking, Property 4: View construction is deterministic and length-bounded

Property 4 (design.md): *For any* CVE_Record, converting it repeatedly yields
byte-identical Profile_Text_View output, and the view's length is at most 2000
characters, truncated at a segment boundary with the truncation counted when the
untruncated view would have exceeded 2000 characters.

**Validates: Requirements 2.1, 2.8**
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

# Make the repository root importable when pytest is invoked from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cve_domain.data_converter import (  # noqa: E402
    MAX_VIEW_CHARS,
    _build_segments,
    _render,
    build_profile_text_view,
    build_profile_text_view_report,
)


# --------------------------------------------------------------------------- #
# Generators — CVE_Records including some with very long fields to force
# segment-boundary truncation (Req 2.8).
# --------------------------------------------------------------------------- #

# CVE identifiers are always short in the real data; keep the mandatory first
# segment bounded so the length guarantee is meaningful (a single mandatory
# segment longer than the cap is a documented pathological exception, not a
# realistic CVE id).
_cve_id = st.from_regex(r"CVE-[0-9]{4}-[0-9]{1,7}", fullmatch=True)

# A short optional text field (present ~ often, missing/blank sometimes).
_short_field = st.one_of(
    st.none(),
    st.text(max_size=0),                     # empty -> treated as absent (Req 2.2)
    st.text(min_size=1, max_size=120),
)

# A field that may occasionally be very long, to push the untruncated view past
# the 2000-char cap and exercise the truncation path.
_long_field = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=120),
    st.text(min_size=1500, max_size=5000),   # long enough to trigger truncation
)

# Semicolon-joined multi-token fields (cwes, cpes_sample), some long.
_semicolon_field = st.one_of(
    st.none(),
    st.lists(st.text(min_size=1, max_size=40), max_size=6).map(";".join),
    st.lists(st.text(min_size=40, max_size=200), min_size=20, max_size=60).map(";".join),
)

_bool_token = st.sampled_from(
    ["true", "false", "True", "False", "1", "0", "yes", "no", "known", "unknown", "", None]
)


@st.composite
def cve_records(draw):
    """Generate a CVE_Record dict spanning present, absent, and very-long fields."""
    record = {
        "cve": draw(_cve_id),
        "vulnerability_name": draw(_long_field),
        "short_description": draw(_long_field),
        "cwes": draw(_semicolon_field),
        "cvss_base_score": draw(st.one_of(st.none(), st.floats(0, 10).map(str))),
        "cvss_base_severity": draw(
            st.one_of(st.none(), st.sampled_from(["LOW", "MEDIUM", "HIGH", "CRITICAL"]))
        ),
        "epss": draw(st.one_of(st.none(), st.floats(0, 1).map(str))),
        "in_kev": draw(_bool_token),
        "known_ransomware_campaign_use": draw(_bool_token),
        "vendor_project": draw(_short_field),
        "product": draw(_short_field),
        "cpes_sample": draw(_semicolon_field),
    }
    return record


# --------------------------------------------------------------------------- #
# Property 4
# --------------------------------------------------------------------------- #


@settings(max_examples=200)
@given(record=cve_records())
def test_view_is_deterministic_and_length_bounded(record):
    # --- Determinism (Req 2.1): repeated conversion is byte-identical. ---
    first = build_profile_text_view(record)
    second = build_profile_text_view(record)
    assert first == second, "Profile_Text_View is not deterministic for the same record"

    report_a = build_profile_text_view_report(record)
    report_b = build_profile_text_view_report(record)
    assert report_a.text == report_b.text
    assert report_a.truncated == report_b.truncated
    # The wrapper and the report agree on the produced text.
    assert first == report_a.text

    # --- Length bound (Req 2.8): view is at most MAX_VIEW_CHARS. ---
    assert len(report_a.text) <= MAX_VIEW_CHARS, (
        f"view length {len(report_a.text)} exceeds cap {MAX_VIEW_CHARS}"
    )

    # --- Truncation flag reflects the untruncated length (Req 2.8). ---
    # Rebuild the full, untruncated view by rendering all segments.
    segments = _build_segments(record, None, 5)
    full_untruncated = _render(segments)

    if report_a.truncated:
        assert len(full_untruncated) > MAX_VIEW_CHARS, (
            "truncated flag set but the untruncated view fits within the cap"
        )
    else:
        # Not truncated -> the text is exactly the full untruncated render.
        assert report_a.text == full_untruncated
        assert len(full_untruncated) <= MAX_VIEW_CHARS

    # --- Truncation happens only at a segment boundary (Req 2.8). ---
    # The produced text must be the render of some whole-segment prefix.
    prefix_renders = {_render(segments[:k]) for k in range(1, len(segments) + 1)}
    if segments:
        assert report_a.text in prefix_renders, (
            "truncated view was not cut at a segment boundary"
        )


@settings(max_examples=100)
@given(record=cve_records())
def test_truncated_view_keeps_longest_fitting_prefix(record):
    """When truncated, the kept text is the longest whole-segment prefix <= cap.

    Reinforces Req 2.8: truncation drops whole trailing segments, never splits a
    segment, and keeps as many leading segments as fit within the cap (retaining
    at least the mandatory first segment).
    """
    report = build_profile_text_view_report(record)
    if not report.truncated:
        return

    segments = _build_segments(record, None, 5)
    # Determine the longest prefix that fits within the cap.
    longest_fitting = 0
    for k in range(1, len(segments) + 1):
        if len(_render(segments[:k])) <= MAX_VIEW_CHARS:
            longest_fitting = k
        else:
            break
    expected_kept = max(longest_fitting, 1)  # always keep at least the first segment
    assert report.text == _render(segments[:expected_kept])
