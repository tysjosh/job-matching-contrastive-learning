"""Guards the in-memory memoization of ``OntologySkillMatcher.ontology_set_similarity``.

The matcher normally loads a large ESCO knowledge graph in ``__init__``; these
tests bypass that by constructing a bare instance and injecting a deterministic
stub ``_skill_distance``. That isolates the memoization logic so we can assert
the cached result is bit-for-bit identical to the uncached computation, the
function is symmetric, and repeated calls are served from the cache.
"""
import math

from contrastive_learning.ontology_skill_matcher import OntologySkillMatcher


def _make_matcher(distances):
    """Build a matcher without loading a graph, with a stub distance function.

    ``distances`` maps an unordered pair (frozenset of two URIs) to an integer
    hop distance; missing pairs are treated as disconnected (None).
    """
    m = OntologySkillMatcher.__new__(OntologySkillMatcher)
    m.alpha = 0.7
    m.reuse_weights = {}
    m.skill_reuse_level = {}
    m.transversal_skills = set()
    m.transversal_weight = 0.3
    m._set_sim_cache = {}
    m._set_sim_cache_max = 1000

    def _skill_distance(u, v):
        if u == v:
            return 0
        return distances.get(frozenset((u, v)))

    m._skill_distance = _skill_distance
    return m


def test_cached_matches_uncached_and_is_symmetric():
    dists = {
        frozenset(("a", "x")): 1,
        frozenset(("a", "y")): 3,
        frozenset(("b", "x")): 2,
        frozenset(("b", "y")): 1,
    }
    m = _make_matcher(dists)
    A = ["a", "b"]
    B = ["x", "y"]

    # Direct (uncached) computation, then the public (cached) path.
    expected = m._compute_set_similarity(frozenset(A), frozenset(B))
    first = m.ontology_set_similarity(A, B)

    assert math.isclose(first, expected, rel_tol=0.0, abs_tol=0.0)
    # Cache populated after first call.
    assert len(m._set_sim_cache) == 1
    # Second call is served from cache and identical.
    assert m.ontology_set_similarity(A, B) == first
    # Symmetric: sim(A, B) == sim(B, A), and no extra cache entry (order-normalized key).
    assert m.ontology_set_similarity(B, A) == first
    assert len(m._set_sim_cache) == 1


def test_empty_sets_return_zero_without_caching():
    m = _make_matcher({})
    assert m.ontology_set_similarity([], ["x"]) == 0.0
    assert m.ontology_set_similarity(["a"], []) == 0.0
    assert len(m._set_sim_cache) == 0


def test_cache_is_bounded():
    m = _make_matcher({})
    m._set_sim_cache_max = 2
    # Three distinct set-pairs, but the cache must stop growing at the bound.
    m.ontology_set_similarity(["a"], ["p"])
    m.ontology_set_similarity(["b"], ["q"])
    m.ontology_set_similarity(["c"], ["r"])
    assert len(m._set_sim_cache) <= 2
