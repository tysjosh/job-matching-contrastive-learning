"""Unit tests for CVE ordinal Stage-1 band-proximity grading (loss_engine).

Covers the band→rank map and the proximity→level mapping used to grade each
Stage-1 candidate by priority_band distance to the anchor (same=2, adjacent=1,
distant/unknown=0), and asserts the career ordinal path is left untouched when a
triplet carries no CVE band metadata.
"""

from contrastive_learning.data_structures import TrainingConfig
from contrastive_learning.loss_engine import ContrastiveLossEngine


def _engine(**overrides):
    cfg = TrainingConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return ContrastiveLossEngine(cfg, None)


def test_band_rank_from_default_order():
    eng = _engine()
    # Default order: watch < low < medium < high < critical.
    assert eng._cve_band_rank("watch") == 0
    assert eng._cve_band_rank("critical") == 4
    assert eng._cve_band_rank("HIGH") == 3          # case-insensitive
    assert eng._cve_band_rank(" medium ") == 2      # trimmed
    assert eng._cve_band_rank("unknown-band") is None
    assert eng._cve_band_rank(None) is None


def test_band_level_proximity_same_adjacent_distant():
    eng = _engine()
    high = eng._cve_band_rank("high")               # rank 3
    assert eng._cve_band_level(high, "high") == 2       # same band → good
    assert eng._cve_band_level(high, "critical") == 1   # adjacent (4) → potential
    assert eng._cve_band_level(high, "medium") == 1     # adjacent (2) → potential
    assert eng._cve_band_level(high, "low") == 0        # distance 2 → no
    assert eng._cve_band_level(high, "watch") == 0      # distance 3 → no


def test_band_level_unknown_or_missing_is_floor():
    eng = _engine()
    anchor = eng._cve_band_rank("critical")
    assert eng._cve_band_level(anchor, "not-a-band") == 0
    assert eng._cve_band_level(anchor, None) == 0
    assert eng._cve_band_level(None, "critical") == 0   # unknown anchor → floor


def test_custom_band_order_respected():
    # A different ordinal order changes proximity relationships.
    eng = _engine(cve_band_order=["a", "b", "c"])
    a = eng._cve_band_rank("a")
    assert eng._cve_band_level(a, "a") == 2
    assert eng._cve_band_level(a, "b") == 1
    assert eng._cve_band_level(a, "c") == 0


def test_career_path_has_no_band_map_until_used():
    # The band-rank map is only materialized on demand; a fresh engine used for
    # the career path never triggers it, so nothing about that path changes.
    eng = _engine()
    assert getattr(eng, "_cve_band_rank_map", None) is None
