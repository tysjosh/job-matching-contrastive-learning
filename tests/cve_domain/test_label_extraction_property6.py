"""Property-based tests for CVE supervised-label extraction (Property 6).

# Feature: cve-vulnerability-ranking, Property 6: Supervised label extraction respects validity and defaulting rules

Property 6 (design.md): *For any* CVE_Record, ``priority_score`` is stored unchanged
when it is numeric within ``[0, 100]`` and omitted (and counted) otherwise;
``priority_band`` is stored when present and omitted (and counted) when missing or
empty; and ``in_kev`` / ``ransomware`` are stored as booleans from recognized
truthy/falsy tokens, defaulting to false (and counted) when unparseable.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

# Make the repository root importable when pytest is invoked from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cve_domain.data_converter import (  # noqa: E402
    PRIORITY_SCORE_MAX,
    PRIORITY_SCORE_MIN,
    _FALSY_TOKENS,
    _TRUTHY_TOKENS,
    extract_labels_report,
)


# --------------------------------------------------------------------------- #
# Generators — CVE_Record fields spanning every branch of the label rules.
# The generators deliberately constrain to the input space the extractor cares
# about (in-range / out-of-range / NaN / non-numeric / missing scores;
# present / empty / missing bands; recognized truthy / falsy / unparseable
# boolean tokens) so each property assertion exercises a meaningful case.
# --------------------------------------------------------------------------- #

# priority_score values across every documented category (Req 3.2, 3.3).
_in_range_score = st.one_of(
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=100),
    # In-range values presented as strings (CSV cells are strings on disk).
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False).map(str),
    st.integers(min_value=0, max_value=100).map(str),
)

_out_of_range_score = st.one_of(
    st.floats(min_value=100.0001, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e9, max_value=-0.0001, allow_nan=False, allow_infinity=False),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.just("100.5"),
    st.just("-3"),
)

_nan_score = st.one_of(st.just(float("nan")), st.just("nan"), st.just("NaN"))

_non_numeric_score = st.one_of(
    st.just("high"),
    st.just("critical"),
    st.just("abc"),
    st.just("N/A"),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=10),
    st.booleans(),  # a bool is not a valid Priority_Score (Req 3.2)
)

_missing_score = st.one_of(st.none(), st.just(""), st.just("   "))

_any_score = st.one_of(
    _in_range_score, _out_of_range_score, _nan_score, _non_numeric_score, _missing_score
)

# priority_band values (Req 3.6, 3.7).
_present_band = st.sampled_from(["low", "medium", "high", "critical", " High ", "band-x"])
_missing_band = st.one_of(st.none(), st.just(""), st.just("   "))
_any_band = st.one_of(_present_band, _missing_band)

# Boolean-style tokens (Req 3.4, 3.5). Cover recognized truthy, recognized
# falsy, and unparseable tokens (which must default to False and be counted).
_recognized_truthy = st.sampled_from(sorted(_TRUTHY_TOKENS))
# Falsy tokens include "" (empty). Draw from the recognized falsy set as-is.
_recognized_falsy = st.sampled_from(sorted(_FALSY_TOKENS))
_case_variants = st.sampled_from(["True", "FALSE", "Yes", "No", "Known", "Unknown"])
_unparseable_bool = st.one_of(
    st.just("maybe"),
    st.just("2"),
    st.just("kinda"),
    st.just("n/a"),
    st.text(alphabet="ghjklmpqrvwxz", min_size=2, max_size=6),  # never a token
)
_any_bool = st.one_of(
    st.none(), _recognized_truthy, _recognized_falsy, _case_variants, _unparseable_bool
)


@st.composite
def cve_records(draw):
    """Generate a CVE_Record dict spanning every label branch."""
    return {
        "priority_score": draw(_any_score),
        "priority_band": draw(_any_band),
        "in_kev": draw(_any_bool),
        "known_ransomware_campaign_use": draw(_any_bool),
    }


# --------------------------------------------------------------------------- #
# Reference helpers mirroring the specification's rules (independent of the
# implementation internals) so the property is a genuine oracle.
# --------------------------------------------------------------------------- #


def _expected_score_kept(raw):
    """Return the numeric value the spec says to keep, or None to omit."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        num = float(raw)
    else:
        token = str(raw).strip()
        if not token:
            return None
        try:
            num = float(token)
        except (TypeError, ValueError):
            return None
    if math.isnan(num) or math.isinf(num):
        return None
    if num < PRIORITY_SCORE_MIN or num > PRIORITY_SCORE_MAX:
        return None
    return num


def _expected_bool(raw):
    """Return (value, was_defaulted) per the recognized-token rules."""
    if raw is None:
        return False, True
    token = str(raw).strip().lower()
    if token in _TRUTHY_TOKENS:
        return True, False
    if token in _FALSY_TOKENS:
        return False, False
    return False, True


# --------------------------------------------------------------------------- #
# Property 6
# --------------------------------------------------------------------------- #


@settings(max_examples=300)
@given(record=cve_records())
def test_label_extraction_respects_validity_and_defaulting(record):
    result = extract_labels_report(record)
    labels = result.labels

    # --- priority_score (Req 3.2, 3.3) --------------------------------- #
    expected_score = _expected_score_kept(record["priority_score"])
    if expected_score is None:
        assert "priority_score" not in labels, (
            "invalid priority_score should be omitted from cve_labels"
        )
        assert result.priority_score_omitted is True, (
            "omitted priority_score must be counted (Req 3.3)"
        )
    else:
        assert "priority_score" in labels, "valid priority_score should be stored"
        # Stored unchanged (as a float within range).
        assert labels["priority_score"] == expected_score
        assert PRIORITY_SCORE_MIN <= labels["priority_score"] <= PRIORITY_SCORE_MAX
        assert result.priority_score_omitted is False

    # --- priority_band (Req 3.6, 3.7) ---------------------------------- #
    raw_band = record["priority_band"]
    band_present = raw_band is not None and len(str(raw_band).strip()) >= 1
    if band_present:
        assert "priority_band" in labels, "present priority_band should be stored"
        assert labels["priority_band"] == str(raw_band).strip()
        assert result.priority_band_omitted is False
    else:
        assert "priority_band" not in labels, (
            "missing/empty priority_band should be omitted"
        )
        assert result.priority_band_omitted is True, (
            "omitted priority_band must be counted (Req 3.7)"
        )

    # --- in_kev / ransomware booleans (Req 3.1, 3.4, 3.5) -------------- #
    exp_kev, kev_defaulted = _expected_bool(record["in_kev"])
    exp_rw, rw_defaulted = _expected_bool(record["known_ransomware_campaign_use"])

    # Both boolean labels are ALWAYS present (Req 3.1) and are real booleans.
    assert "in_kev" in labels and isinstance(labels["in_kev"], bool)
    assert "ransomware" in labels and isinstance(labels["ransomware"], bool)
    assert labels["in_kev"] is exp_kev
    assert labels["ransomware"] is exp_rw

    # Defaulted-label count equals the number of unparseable/missing booleans (Req 3.5).
    expected_defaulted = int(kev_defaulted) + int(rw_defaulted)
    assert result.defaulted_label_count == expected_defaulted


@settings(max_examples=100)
@given(
    score=_in_range_score,
    band=_present_band,
    kev=_recognized_truthy,
    rw=_recognized_falsy,
)
def test_all_valid_record_stores_every_label_with_no_omissions(score, band, kev, rw):
    """A fully-valid record stores all four labels with zero omission/default counts.

    Reinforces Req 3.1: all four labels attached; Req 3.2/3.6: valid score and band
    stored; Req 3.5: no defaulting when booleans are recognized tokens.
    """
    record = {
        "priority_score": score,
        "priority_band": band,
        "in_kev": kev,
        "known_ransomware_campaign_use": rw,
    }
    result = extract_labels_report(record)

    assert "priority_score" in result.labels
    assert "priority_band" in result.labels
    assert result.labels["in_kev"] is True
    # rw drawn from the recognized falsy set -> parsed False, never defaulted.
    assert result.labels["ransomware"] is False
    assert result.priority_score_omitted is False
    assert result.priority_band_omitted is False
    assert result.defaulted_label_count == 0
