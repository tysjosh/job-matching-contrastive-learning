"""Property-based tests for the CVE Profile_Text_View serializer (Requirement 2).

Function under test: ``build_profile_text_view`` (and
``build_profile_text_view_report``) in ``cve_domain/data_converter.py``.

These tests use Hypothesis to generate random ``CVE_Record`` inputs with varying
presence/absence of each field and assert the well-formedness / presence-faithful
properties of the resulting Profile_Text_View.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_profile_text_view_property.py
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from cve_domain.data_converter import build_profile_text_view


# ---------------------------------------------------------------------------
# Generation helpers
#
# Free-text field values are drawn from a "safe" alphabet (letters, digits and
# hyphens, no "." and no spaces) and carry distinctive prefixes so that
# substring-membership assertions are unambiguous. Because no generated value
# contains a "." or a run of spaces, every "." in the rendered view is a
# structural delimiter and every " " is a single intra-segment space — which
# lets the structural (no dangling / doubled delimiter) checks be exact.
# ---------------------------------------------------------------------------

# Truthy tokens recognized by the serializer for status fields (mirrors
# ``_TRUTHY_TOKENS`` in data_converter). Matching is on the trimmed, lower-cased
# token, so " True " renders and "false"/"unknown"/"maybe"/"" do not.
TRUTHY_TOKENS = frozenset({"true", "1", "yes", "y", "t", "known"})

# Values that represent an ABSENT field: None, empty, and whitespace-only all
# have a trimmed length < 1 and must therefore contribute no segment (Req 2.2/2.4).
ABSENT_VALUES = [None, "", "   ", "\t", " \n "]

# Lowercase letters + digits + hyphen only. Free-text bodies use this alphabet
# while field markers ("VN", "SD", "VENDOR", "PROD", ...) are uppercase, so an
# uppercase marker can only originate from its intended field — never from a
# randomly generated body of another field (which was the collision the first
# run surfaced, e.g. a cpe body "cpeVN").
_SAFE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"


def _tagged_text(prefix: str) -> st.SearchStrategy:
    """A present, distinctively-prefixed token containing no '.' and no spaces."""
    body = st.text(alphabet=_SAFE_ALPHABET, min_size=1, max_size=12)
    return body.map(lambda s: prefix + s)


def _optional(prefix: str) -> st.SearchStrategy:
    """A field that is either absent (None/empty/whitespace) or a present token."""
    return st.one_of(st.sampled_from(ABSENT_VALUES), _tagged_text(prefix))


def _cve_ids() -> st.SearchStrategy:
    """Always-present, non-empty cve identifiers, e.g. 'CVE-2024-1234'.

    Some are wrapped in surrounding whitespace to exercise the trim rule while
    still being present (trimmed length >= 1).
    """
    base = st.tuples(
        st.integers(min_value=1990, max_value=2035),
        st.integers(min_value=1, max_value=999999),
    ).map(lambda t: f"CVE-{t[0]}-{t[1]}")
    return st.one_of(base, base.map(lambda s: f"  {s}  "))


def _cwes() -> st.SearchStrategy:
    """A ';'-joined CWE string, possibly with empty/whitespace pieces, or absent.

    CWE tokens carry the distinctive 'CWE-' prefix so their presence/absence in
    the view is unambiguous (and never collides with the 'CVE-' identifier).
    """
    token = st.integers(min_value=1, max_value=999).map(lambda n: f"CWE-{n}")
    pieces = st.lists(st.one_of(token, st.just(""), st.just("  ")), min_size=1, max_size=5)
    joined = pieces.map(lambda ps: ";".join(ps))
    return st.one_of(st.sampled_from(ABSENT_VALUES), joined)


def _cpes() -> st.SearchStrategy:
    """A ';'-joined cpes_sample string (lowercase 'cpe' prefix) or absent."""
    token = st.text(alphabet=_SAFE_ALPHABET, min_size=1, max_size=8).map(lambda s: "cpe" + s)
    pieces = st.lists(st.one_of(token, st.just("")), min_size=1, max_size=6)
    joined = pieces.map(lambda ps: ";".join(ps))
    return st.one_of(st.sampled_from(ABSENT_VALUES), joined)


def _status() -> st.SearchStrategy:
    """A status field: absent, a recognized truthy/falsy token, or an unknown one."""
    return st.one_of(
        st.sampled_from(ABSENT_VALUES),
        st.sampled_from(["true", "True", " true ", "yes", "1", "known", "t", "y"]),
        st.sampled_from(["false", "no", "0", "unknown", "maybe", "nope"]),
    )


@st.composite
def cve_records(draw) -> dict:
    """Generate a CVE_Record with varying presence/absence of each field.

    The ``cve`` identifier is always present (Req 2.7 concerns records that carry
    at least a cve). Every other field independently may be absent, present, or —
    for status fields — truthy/falsy/unrecognized.
    """
    return {
        "cve": draw(_cve_ids()),
        "vulnerability_name": draw(_optional("VN")),
        "short_description": draw(_optional("SD")),
        "cwes": draw(_cwes()),
        "cvss_base_score": draw(_optional("SCORE")),
        "cvss_base_severity": draw(_optional("SEV")),
        "epss": draw(_optional("EP")),
        "in_kev": draw(_status()),
        "known_ransomware_campaign_use": draw(_status()),
        "vendor_project": draw(_optional("VENDOR")),
        "product": draw(_optional("PROD")),
        "cpes_sample": draw(_cpes()),
    }


# ---------------------------------------------------------------------------
# Presence / expectation helpers (independent re-derivation of the spec rules)
# ---------------------------------------------------------------------------

def _present(value) -> bool:
    """Req 2.2 presence rule: exists and trimmed length >= 1."""
    return value is not None and len(str(value).strip()) >= 1


def _is_truthy(value) -> bool:
    return value is not None and str(value).strip().lower() in TRUTHY_TOKENS


def _cwe_tokens(value) -> list:
    if value is None:
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]


def _cpe_tokens(value) -> list:
    if value is None:
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]


# ---------------------------------------------------------------------------
# Property 3
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 3: The Profile_Text_View is well-formed and presence-faithful
@settings(max_examples=200, deadline=None)
@given(record=cve_records())
def test_profile_text_view_well_formed_and_presence_faithful(record):
    """Property 3 (Validates: Requirements 2.2, 2.3, 2.4, 2.7).

    For any CVE_Record the Profile_Text_View is non-empty, contains the cve
    identifier segment, includes a field's segment iff that field is present
    (trimmed length >= 1), prefers vulnerability_name over short_description
    (never both), and contains no placeholder text and no dangling or doubled
    delimiter for any omitted segment.
    """
    view = build_profile_text_view(record)

    cve = str(record["cve"]).strip()

    # --- Req 2.7: non-empty and carries the cve identifier segment -----------
    assert view, "view must be non-empty for a record carrying a cve"
    assert cve in view, "the cve identifier must appear in the view"
    assert view.startswith(cve), "the cve identifier must be the first segment"

    # --- Structural well-formedness: no dangling / doubled delimiter (Req 2.4)
    # No generated value contains '.', so every '.' is a structural delimiter.
    assert view.endswith("."), "view must terminate with a single '.'"
    assert not view.startswith("."), "view must not start with a delimiter"
    assert ".." not in view, "no doubled '.' delimiter (would imply an empty tail)"
    assert ". ." not in view, "no empty segment between delimiters"
    # No generated value contains spaces, so any run of spaces would be a
    # dangling join around an omitted segment.
    assert "  " not in view, "no doubled space from an omitted segment join"

    # --- Req 2.4: no placeholder text for omitted fields ---------------------
    # The serializer emits NO placeholder for a missing field (it omits the
    # segment entirely — already asserted by the presence-faithful "iff" checks
    # below). These guard against it accidentally stringifying a Python ``None``.
    # Only capitalized / punctuated placeholder tokens are checked, because the
    # generated field bodies use a lowercase alnum alphabet and can legitimately
    # contain lowercase substrings like "null"/"nan" (e.g. a product named
    # "PRODnull"); a lowercase substring check would false-match real content.
    assert "None" not in view
    assert "N/A" not in view

    # --- Req 2.3: prefer vulnerability_name over short_description -----------
    name_present = _present(record["vulnerability_name"])
    desc_present = _present(record["short_description"])
    if name_present:
        assert str(record["vulnerability_name"]).strip() in view
        # short_description must be omitted entirely (never both). The 'SD'
        # marker is disjoint from every other token/keyword in the view.
        assert "SD" not in view
    elif desc_present:
        assert str(record["short_description"]).strip() in view
        assert "VN" not in view
    else:
        assert "VN" not in view
        assert "SD" not in view

    # --- Presence-faithful "iff" for the remaining segments (Req 2.2/2.4) ----
    # cwes: the 'CWE-' marked segment appears iff there is >=1 non-empty token.
    cwe_tokens = _cwe_tokens(record["cwes"])
    if cwe_tokens:
        for tok in cwe_tokens:
            assert tok in view
    else:
        assert "CWE-" not in view

    # CVSS segment appears iff score or severity is present.
    cvss_present = _present(record["cvss_base_score"]) or _present(
        record["cvss_base_severity"]
    )
    assert ("CVSS " in view) == cvss_present

    # EPSS segment appears iff epss is present.
    assert ("EPSS " in view) == _present(record["epss"])

    # in_kev / ransomware status segments appear iff the field is truthy.
    assert ("In CISA KEV" in view) == _is_truthy(record["in_kev"])
    assert ("Known ransomware use" in view) == _is_truthy(
        record["known_ransomware_campaign_use"]
    )

    # Affected-product segment appears iff vendor, product, or a cpe token exists.
    affected_present = (
        _present(record["vendor_project"])
        or _present(record["product"])
        or bool(_cpe_tokens(record["cpes_sample"]))
    )
    assert ("Affects" in view) == affected_present
    if _present(record["vendor_project"]):
        assert str(record["vendor_project"]).strip() in view
    if _present(record["product"]):
        assert str(record["product"]).strip() in view
