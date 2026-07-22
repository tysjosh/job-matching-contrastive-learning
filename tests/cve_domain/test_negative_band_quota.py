"""Tests for CVE ordinal Stage-1 band-quota negative selection (ontology-guided).

Band-quota mode grades the ONTOLOGY-selected negatives (the anchor's cyber-KG
hard/medium/easy pools) by priority_band distance to the anchor — adjacent →
graded level 1, distant → level 0 — using same-band ontology negatives only as
fill. Candidates always come from the ontology pool, so selection stays
ontology-guided even while enforcing band gradedness.
"""

from cve_domain.negative_selector import CVENegativeSelector


class _Pool:
    """Minimal DenominatorPool stand-in with tier id lists."""

    def __init__(self, hard=None, medium=None, easy=None):
        self.hard_negatives = list(hard or [])
        self.medium_negatives = list(medium or [])
        self.easy_negatives = list(easy or [])


class _Adapter:
    """Ontology adapter stub returning a fixed pool for every anchor."""

    def __init__(self, pool):
        self._pool = pool

    def get_pool(self, cve):
        return self._pool


def _selector(pool, max_n=8, ratio=0.5):
    return CVENegativeSelector(
        _Adapter(pool),
        max_negatives_per_anchor=max_n,
        seed=42,
        band_quota=True,
        band_quota_adjacent_ratio=ratio,
        band_order=["watch", "low", "medium", "high", "critical"],
    )


def test_prefers_graded_bands_over_same_band_within_ontology_pool():
    # Ontology pool for the anchor holds a glut of same-band (high) candidates
    # plus a few adjacent (medium/critical) and far (watch/low) ones.
    hard = (
        [f"H{i}" for i in range(20)]      # same band (high)
        + [f"M{i}" for i in range(4)]     # adjacent (medium)
        + [f"C{i}" for i in range(4)]     # adjacent (critical)
        + [f"W{i}" for i in range(4)]     # far (watch)
        + [f"L{i}" for i in range(4)]     # far (low)
    )
    s = _selector(_Pool(hard=hard), max_n=8)
    band_by_id = {"A": "high"}
    band_by_id.update({f"H{i}": "high" for i in range(20)})
    band_by_id.update({f"M{i}": "medium" for i in range(4)})
    band_by_id.update({f"C{i}": "critical" for i in range(4)})
    band_by_id.update({f"W{i}": "watch" for i in range(4)})
    band_by_id.update({f"L{i}": "low" for i in range(4)})
    s.set_band_lookup(band_by_id)
    present = set(band_by_id)

    negs = s.select_negatives("A", list(present), present)

    assert len(negs) == 8
    assert "A" not in negs
    # Same-band 'high' ontology candidates must NOT be chosen while graded remain.
    assert all(not n.startswith("H") for n in negs), negs
    assert any(n[0] in ("M", "C") for n in negs)   # adjacent represented
    assert any(n[0] in ("W", "L") for n in negs)   # far represented
    assert s.report.band_quota_samefill_count == 0


def test_same_band_fill_when_ontology_pool_lacks_band_diversity():
    # A watch anchor whose ontology pool is all watch -> only same-band fill.
    hard = [f"W{i}" for i in range(10)]
    s = _selector(_Pool(hard=hard), max_n=5)
    band_by_id = {"A": "watch"}
    band_by_id.update({f"W{i}": "watch" for i in range(10)})
    s.set_band_lookup(band_by_id)
    present = set(band_by_id)

    negs = s.select_negatives("A", list(present), present)

    assert len(negs) == 5
    assert all(n.startswith("W") for n in negs)
    assert s.report.band_quota_samefill_count >= 5


def test_random_fallback_only_when_pool_too_small():
    # Ontology pool has just 2 candidates; the rest tops up from the split.
    s = _selector(_Pool(hard=["M0", "C0"]), max_n=6)
    band_by_id = {"A": "high", "M0": "medium", "C0": "critical"}
    band_by_id.update({f"X{i}": "low" for i in range(10)})   # only in the split
    s.set_band_lookup(band_by_id)
    present = set(band_by_id)

    negs = s.select_negatives("A", list(present), present)

    assert "M0" in negs and "C0" in negs          # ontology candidates used first
    assert s.report.random_fallback_anchor_count >= 1
    assert len(negs) == 6


def test_deterministic_under_seed():
    hard = [f"H{i}" for i in range(6)] + [f"C{i}" for i in range(6)] + [f"W{i}" for i in range(6)]
    band_by_id = {"A": "medium"}
    band_by_id.update({f"H{i}": "high" for i in range(6)})
    band_by_id.update({f"C{i}": "critical" for i in range(6)})
    band_by_id.update({f"W{i}": "watch" for i in range(6)})

    def run():
        s = _selector(_Pool(hard=hard), max_n=6)
        s.set_band_lookup(band_by_id)
        present = set(band_by_id)
        return s.select_negatives("A", list(present), present)

    assert run() == run()


def test_no_op_without_band_order_uses_standard_ontology_path():
    # band_quota on but no band_order -> guard falls through to the standard
    # ontology tier path (no crash, anchor excluded).
    s = CVENegativeSelector(
        _Adapter(_Pool(hard=["H0"])), max_negatives_per_anchor=4, seed=42,
        band_quota=True,
    )
    s.set_band_lookup({"A": "high", "H0": "high"})
    negs = s.select_negatives("A", ["A", "H0"], {"A", "H0"})
    assert "A" not in negs
