"""Property-based tests for the Profile_Text_View serializer (Requirement 2).

Task 4.4 — Property 5: CWE and CPE rendering follows split/trim/cap rules in
source order.

These tests exercise ``cve_domain.data_converter.build_profile_text_view`` with
``cwes`` / ``cpes_sample`` strings containing varying token counts, empty tokens,
surrounding whitespace, and more than five CPE values, and assert that:

- the CWEs rendered in the view are exactly the ``;``-split, whitespace-trimmed,
  non-empty tokens in source order (Req 2.5), and
- at most five ``cpes_sample`` values appear, taken in source order (Req 2.6).

Tokens are drawn from distinctive namespaces (``CWE-<n>`` and ``CPE-<n>``) so the
rendered tokens can be recovered from the view unambiguously via a regex, while all
other record fields are populated from CWE/CPE-pattern-free alphabets so they never
collide with the tokens under test.
"""

from __future__ import annotations

import re

from hypothesis import given, settings, strategies as st

from cve_domain.data_converter import build_profile_text_view

# Distinctive token namespaces so rendered tokens are recoverable from the view.
_CWE_RE = re.compile(r"CWE-\d+")
_CPE_RE = re.compile(r"CPE-\d+")

# Cap applied to cpes_sample values (Req 2.6 default of 5).
_MAX_CPES = 5

# Whitespace variants injected around/between tokens to exercise trim/drop-empty.
_WS = ["", " ", "  ", "\t", " \t ", "\n"]


def _reference_tokens(raw: str) -> list[str]:
    """Independent reference for the ;-split / trim / drop-empty rule (Req 2.5)."""
    return [tok.strip() for tok in raw.split(";") if tok.strip()]


@st.composite
def _token_field(draw, prefix: str, min_count: int, max_count: int):
    """Build a raw ``;``-joined field plus its expected split/trim/non-empty tokens.

    Real tokens look like ``{prefix}-{n}`` (duplicates allowed). The raw string
    interleaves optional surrounding whitespace and injected whitespace-only /
    empty segments so the trim + drop-empty behavior is exercised. The expected
    token list is recomputed from the raw string via the independent reference.
    """
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    ids = draw(
        st.lists(st.integers(min_value=0, max_value=9999), min_size=count, max_size=count)
    )
    pieces = [draw(st.sampled_from(_WS)) + f"{prefix}-{i}" + draw(st.sampled_from(_WS)) for i in ids]

    # Inject empty / whitespace-only segments at random positions (Req 2.5 drop-empty).
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        pos = draw(st.integers(min_value=0, max_value=len(pieces)))
        pieces.insert(pos, draw(st.sampled_from(["", " ", "  ", "\t"])))

    raw = ";".join(pieces)
    return raw, _reference_tokens(raw)


# Safe values for unrelated record fields: letters/spaces only (no "-<digit>"),
# so they can never match the CWE-<n> / CPE-<n> patterns under test.
_safe_text = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=20)
_safe_num = st.sampled_from(["7.3", "9.8", "0.0", "5.5", "10.0"])
_bool_tok = st.sampled_from(["true", "false", "Known", "Unknown", "yes", "no"])


@st.composite
def _cve_record(draw):
    """Generate a CVE_Record with varied cwes/cpes and varied unrelated fields."""
    cwes_raw, cwes_expected = draw(_token_field("CWE", 0, 6))
    # cpes: allow more than five so the cap (Req 2.6) is exercised.
    cpes_raw, cpes_expected = draw(_token_field("CPE", 0, 9))

    row: dict = {"cve": f"CVE-{draw(st.integers(0, 9999))}", "cwes": cwes_raw, "cpes_sample": cpes_raw}

    # Optionally populate unrelated fields from CWE/CPE-pattern-free alphabets.
    if draw(st.booleans()):
        row["vulnerability_name"] = draw(_safe_text)
    if draw(st.booleans()):
        row["short_description"] = draw(_safe_text)
    if draw(st.booleans()):
        row["cvss_base_score"] = draw(_safe_num)
    if draw(st.booleans()):
        row["cvss_base_severity"] = draw(st.sampled_from(["LOW", "MEDIUM", "HIGH", "CRITICAL"]))
    if draw(st.booleans()):
        row["epss"] = draw(_safe_num)
    if draw(st.booleans()):
        row["in_kev"] = draw(_bool_tok)
    if draw(st.booleans()):
        row["known_ransomware_campaign_use"] = draw(_bool_tok)
    if draw(st.booleans()):
        row["vendor_project"] = draw(_safe_text)
    if draw(st.booleans()):
        row["product"] = draw(_safe_text)

    return row, cwes_expected, cpes_expected


# Feature: cve-vulnerability-ranking, Property 5: CWE and CPE rendering follows split/trim/cap rules in source order
@settings(max_examples=200)
@given(_cve_record())
def test_cwe_and_cpe_rendering_follows_split_trim_cap_rules(data):
    """Property 5: CWEs are the ;-split/trim/non-empty tokens in source order, and
    at most 5 cpes_sample values appear, taken in source order.

    Validates: Requirements 2.5, 2.6
    """
    row, expected_cwes, expected_cpes = data
    view = build_profile_text_view(row)

    # Req 2.5: rendered CWEs are exactly the split/trimmed/non-empty tokens in order.
    rendered_cwes = _CWE_RE.findall(view)
    assert rendered_cwes == expected_cwes

    # Req 2.6: at most 5 cpes appear, taken in source order.
    rendered_cpes = _CPE_RE.findall(view)
    assert len(rendered_cpes) <= _MAX_CPES
    assert rendered_cpes == expected_cpes[:_MAX_CPES]
