"""Gene Ontology index and ``OntologySkillMatcher``-compatible similarity.

Mirrors ``trials_domain/mesh_ontology.py`` method-for-method so the shared
``BatchProcessor`` call sites work unchanged:

    skill_distance(u, v)            -> Optional[int]    GO DAG hops
    skill_sim(u, v)                 -> float            exp(-alpha * hops)
    ontology_set_similarity(A, B)   -> float            simGIC  (the s_esco analogue)
    branch_distance(A, B)           -> float            coarse dissimilarity (d_isco analogue)
    ot_distance(A, B)               -> Optional[float]  Sinkhorn OT over DAG distances

WHY simGIC AND NOT BEST-MATCH-AVERAGE
-------------------------------------
The MeSH matcher uses a symmetric best-match average over pairwise term
similarities. That is O(|A|x|B|) per call, and a protein carries a median of 31
propagated GO terms against a trial's handful of MeSH descriptors, so BMA here
would cost ~1000 term comparisons per (anchor, negative) pair per batch.

simGIC is set-based and O(|A| + |B|):

    simGIC(A, B) = sum(IC of shared ancestors) / sum(IC of union of ancestors)

It is also the measure the ceiling was established with (0.8195 on the hard
contrast), so training consumes exactly the signal that was measured rather than
a correlate of it. Substituting BMA here would silently break that link --
``Resnik_BMA`` scored 0.7666 and ``Lin_BMA`` 0.7814 on the same data, close but
not the same number.

Information content is frequency-based over the annotation corpus:

    IC(t) = -log( n_genes_annotated_to_t_or_a_descendant / n_genes_total )

so a term that nearly every protein carries contributes almost nothing and a
specific term dominates the numerator when it is shared. This is what makes
simGIC insensitive to the "protein-containing complex" style of vacuous overlap
that plain Jaccard rewards.

CIRCULARITY GUARDS (carried over from the ceiling measurement)
-------------------------------------------------------------
The label is STRING's ``experimental`` channel -- physical assay evidence. Three
GO annotation sources are derived from that same evidence and would leak it:

  * ``GO:0005515`` "protein binding" and ``GO:0005488`` "binding" are annotated
    directly from interaction assays. Excluded.
  * ``IPI`` evidence ("inferred from physical interaction"). Excluded.
  * ``IEA`` (uncurated electronic annotation) and ``ND`` (no data) are excluded
    for quality, not circularity.

Defaulting to the biological_process aspect also keeps molecular_function's
binding terms out structurally. Dropping all the guards moved the measured
ceiling by only -0.017, so they are cheap insurance rather than load-bearing --
but they must match the diagnostic's settings for the ceiling number to describe
the signal training actually sees.
"""

from __future__ import annotations

import gzip
import logging
import math
import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: Exponential decay on GO DAG hops for the term-level ``skill_sim``. Only used
#: by ``skill_distance``/``skill_sim``/``ot_distance``; simGIC does not use it.
DEFAULT_ALPHA = 0.5

#: Hops beyond which two terms count as unrelated.
DEFAULT_MAX_HOPS = 12

#: Returned by ``branch_distance`` when either side has no coarse term, matching
#: ``_isco_distance``'s 0.5-for-unknown convention.
NEUTRAL_DISTANCE = 0.5

#: OT cost for a pair of terms with no common ancestor in the aspect.
DEFAULT_DISCONNECTED_COST = 16.0

ASPECT_NAMESPACE = {
    "P": "biological_process",
    "F": "molecular_function",
    "C": "cellular_component",
}

#: Evidence codes dropped by default. IPI is the circularity risk; IEA and ND are
#: quality exclusions.
DEFAULT_EXCLUDED_EVIDENCE = frozenset({"IPI", "IEA", "ND"})

#: Terms annotated straight from interaction assays -- would leak the label.
DEFAULT_EXCLUDED_TERMS = frozenset({"GO:0005515", "GO:0005488"})

#: Ancestors at or below this depth form the COARSE facet (``coarse_uris``), the
#: ``d_isco`` analogue: "which broad biological area", deliberately decorrelated
#: from the fine-grained simGIC over the full closure.
DEFAULT_COARSE_MAX_DEPTH = 3


class GoOntologyError(RuntimeError):
    """Raised when the GO sources cannot be parsed into a usable index."""


# --------------------------------------------------------------------- parsing
def parse_obo(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Parse ``go-basic.obo`` into (parents, namespace, alt_to_main, name).

    Follows ``is_a`` and ``part_of``, which is the standard closure for GO
    similarity work; obsolete terms are dropped.
    """
    parents: Dict[str, Set[str]] = defaultdict(set)
    namespace: Dict[str, str] = {}
    alt_to_main: Dict[str, str] = {}
    name: Dict[str, str] = {}

    cur = cur_ns = cur_name = None
    cur_parents: Set[str] = set()
    cur_alts: Set[str] = set()
    obsolete = False
    in_term = False

    def flush() -> None:
        if cur and not obsolete:
            parents[cur] |= cur_parents
            if cur_ns:
                namespace[cur] = cur_ns
            if cur_name:
                name[cur] = cur_name
            for a in cur_alts:
                alt_to_main[a] = cur

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("["):
                flush()
                in_term = line == "[Term]"
                cur = cur_ns = cur_name = None
                cur_parents = set()
                cur_alts = set()
                obsolete = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif line.startswith("namespace: "):
                cur_ns = line[11:].strip()
            elif line.startswith("name: "):
                cur_name = line[6:].strip()
            elif line.startswith("alt_id: GO:"):
                cur_alts.add(line[8:].strip())
            elif line.startswith("is_a: GO:"):
                cur_parents.add(line[6:].split("!")[0].strip())
            elif line.startswith("relationship: part_of GO:"):
                cur_parents.add(line[22:].split("!")[0].strip())
            elif line.startswith("is_obsolete: true"):
                obsolete = True
    flush()
    return parents, namespace, alt_to_main, name


def parse_gaf(
    path: Path,
    aspect: str,
    excluded_evidence: Iterable[str] = DEFAULT_EXCLUDED_EVIDENCE,
    excluded_terms: Iterable[str] = DEFAULT_EXCLUDED_TERMS,
    taxon: str = "taxon:9606",
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Parse a GAF into ``{gene_symbol: {direct GO terms}}`` plus drop counters."""
    excluded_evidence = frozenset(excluded_evidence)
    excluded_terms = frozenset(excluded_terms)
    direct: Dict[str, Set[str]] = defaultdict(set)
    stats: Dict[str, int] = defaultdict(int)

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 15:
                continue
            symbol, qualifier, go_id, evidence, asp, tax = c[2], c[3], c[4], c[6], c[8], c[12]
            stats["lines"] += 1
            if "NOT" in qualifier:
                stats["drop_not"] += 1
                continue
            if aspect != "A" and asp != aspect:
                stats["drop_aspect"] += 1
                continue
            if evidence in excluded_evidence:
                stats[f"drop_ev_{evidence}"] += 1
                continue
            if go_id in excluded_terms:
                stats["drop_excluded_term"] += 1
                continue
            if not tax.startswith(taxon):
                stats["drop_taxon"] += 1
                continue
            direct[symbol].add(go_id)
            stats["kept"] += 1
    return dict(direct), dict(stats)


class GoIndex:
    """Parsed GO DAG plus the human annotation corpus and its information content.

    Attributes:
        ancestors: ``{term: frozenset(term + all ancestors)}`` within the aspect.
        depth: ``{term: min hops from an aspect root}``.
        direct: ``{gene: frozenset(direct terms)}`` after the guards.
        closure: ``{gene: frozenset(propagated terms)}``.
        coarse: ``{gene: frozenset(ancestors with depth <= coarse_max_depth)}``.
        ic: ``{term: information content}``.
    """

    def __init__(
        self,
        ancestors: Dict[str, FrozenSet[str]],
        depth: Dict[str, int],
        direct: Dict[str, FrozenSet[str]],
        closure: Dict[str, FrozenSet[str]],
        coarse: Dict[str, FrozenSet[str]],
        ic: Dict[str, float],
        meta: Dict[str, Any],
    ) -> None:
        self.ancestors = ancestors
        self.depth = depth
        self.direct = direct
        self.closure = closure
        self.coarse = coarse
        self.ic = ic
        self.meta = meta
        self.max_ic = max(ic.values()) if ic else 1.0

    def __len__(self) -> int:
        """Number of GO terms in the aspect (``len(matcher.index)`` in summaries)."""
        return len(self.ancestors)

    def __contains__(self, term: object) -> bool:
        return term in self.ancestors

    def genes(self) -> List[str]:
        return sorted(self.closure)

    def terms_for(self, gene: str) -> FrozenSet[str]:
        return self.direct.get(gene, frozenset())

    def closure_for(self, gene: str) -> FrozenSet[str]:
        return self.closure.get(gene, frozenset())

    def coarse_for(self, gene: str) -> FrozenSet[str]:
        return self.coarse.get(gene, frozenset())

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        obo_path: str | Path,
        gaf_path: str | Path,
        aspect: str = "P",
        excluded_evidence: Iterable[str] = DEFAULT_EXCLUDED_EVIDENCE,
        excluded_terms: Iterable[str] = DEFAULT_EXCLUDED_TERMS,
        coarse_max_depth: int = DEFAULT_COARSE_MAX_DEPTH,
    ) -> "GoIndex":
        obo_path, gaf_path = Path(obo_path), Path(gaf_path)
        if not obo_path.exists():
            raise GoOntologyError(f"GO OBO not found: {obo_path}")
        if not gaf_path.exists():
            raise GoOntologyError(f"GO annotation file not found: {gaf_path}")

        parents, namespace, alt_to_main, _name = parse_obo(obo_path)
        if aspect == "A":
            keep = set(namespace)
        else:
            ns = ASPECT_NAMESPACE[aspect]
            keep = {t for t, v in namespace.items() if v == ns}
        if not keep:
            raise GoOntologyError(f"no GO terms in aspect {aspect!r}")

        # Ancestor closure, iterative so a deep DAG cannot blow the stack.
        ancestors: Dict[str, FrozenSet[str]] = {}
        for t in keep:
            acc: Set[str] = {t}
            stack = [t]
            while stack:
                n = stack.pop()
                for p in parents.get(n, ()):
                    if p in keep and p not in acc:
                        acc.add(p)
                        stack.append(p)
            ancestors[t] = frozenset(acc)

        # Depth = shortest path to a root (a term whose only ancestor is itself).
        roots = {t for t, a in ancestors.items() if len(a) == 1}
        children: Dict[str, Set[str]] = defaultdict(set)
        for t in keep:
            for p in parents.get(t, ()):
                if p in keep:
                    children[p].add(t)
        depth: Dict[str, int] = {r: 0 for r in roots}
        frontier = list(roots)
        while frontier:
            nxt: List[str] = []
            for n in frontier:
                for ch in children.get(n, ()):
                    if ch not in depth:
                        depth[ch] = depth[n] + 1
                        nxt.append(ch)
            frontier = nxt

        direct_raw, gaf_stats = parse_gaf(
            gaf_path, aspect, excluded_evidence, excluded_terms)

        direct: Dict[str, FrozenSet[str]] = {}
        closure: Dict[str, FrozenSet[str]] = {}
        coarse: Dict[str, FrozenSet[str]] = {}
        unmapped: Set[str] = set()
        for gene, terms in direct_raw.items():
            mapped, acc = set(), set()
            for t in terms:
                t = alt_to_main.get(t, t)
                a = ancestors.get(t)
                if a is None:
                    unmapped.add(t)
                    continue
                mapped.add(t)
                acc |= a
            if not acc:
                continue
            direct[gene] = frozenset(mapped)
            closure[gene] = frozenset(acc)
            coarse[gene] = frozenset(
                x for x in acc if depth.get(x, 99) <= coarse_max_depth)

        if not closure:
            raise GoOntologyError("no genes retained after annotation filtering")

        # IC over the propagated corpus.
        freq: Dict[str, int] = defaultdict(int)
        for acc in closure.values():
            for t in acc:
                freq[t] += 1
        total = len(closure)
        ic = {t: -math.log(n / total) for t, n in freq.items() if n > 0}

        meta = {
            "obo_path": str(obo_path),
            "gaf_path": str(gaf_path),
            "aspect": aspect,
            "aspect_namespace": ASPECT_NAMESPACE.get(aspect, "all_aspects"),
            "excluded_evidence": sorted(excluded_evidence),
            "excluded_terms": sorted(excluded_terms),
            "coarse_max_depth": coarse_max_depth,
            "terms_in_aspect": len(ancestors),
            "genes_annotated": len(closure),
            "gaf_stats": gaf_stats,
            "unmapped_terms": len(unmapped),
        }
        logger.info(
            "GoIndex built: aspect=%s, %d terms, %d genes, median closure %d terms",
            aspect, len(ancestors), len(closure),
            sorted(len(v) for v in closure.values())[len(closure) // 2],
        )
        return cls(ancestors, depth, direct, closure, coarse, ic, meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "ancestors": self.ancestors, "depth": self.depth,
                    "direct": self.direct, "closure": self.closure,
                    "coarse": self.coarse, "ic": self.ic, "meta": self.meta,
                },
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "GoIndex":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["ancestors"], d["depth"], d["direct"], d["closure"],
                   d["coarse"], d["ic"], d["meta"])

    @classmethod
    def build_or_load(
        cls,
        obo_path: str | Path,
        gaf_path: str | Path,
        cache_path: str | Path,
        aspect: str = "P",
        **kwargs,
    ) -> "GoIndex":
        """Load the pickled index when present, else build and cache it.

        The OBO parse plus closure is ~30s, so caching matters when every
        learning-curve point constructs its own matcher.
        """
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                index = cls.load(cache_path)
                if index.meta.get("aspect") == aspect:
                    logger.info("GoIndex loaded from cache: %s", cache_path)
                    return index
                logger.warning(
                    "GoIndex cache %s is for aspect %r, need %r; rebuilding",
                    cache_path, index.meta.get("aspect"), aspect)
            except Exception as exc:
                logger.warning("GoIndex cache %s unreadable (%s); rebuilding",
                               cache_path, exc)
        index = cls.build(obo_path, gaf_path, aspect=aspect, **kwargs)
        index.save(cache_path)
        logger.info("GoIndex cached to %s", cache_path)
        return index


class GoMatcher:
    """simGIC similarity over GO terms, shaped like ``OntologySkillMatcher``.

    Args:
        index: A built :class:`GoIndex`.
        alpha: Decay for the term-level ``skill_sim`` (not used by simGIC).
        max_hops: Hops beyond which two terms are unrelated.
        set_sim_cache_size: Bound on the memoized set-similarity map.
    """

    def __init__(
        self,
        index: GoIndex,
        alpha: float = DEFAULT_ALPHA,
        max_hops: int = DEFAULT_MAX_HOPS,
        ot_reg: float = 0.4,
        disconnected_cost: float = DEFAULT_DISCONNECTED_COST,
        cache_size: int = 500_000,
        set_sim_cache_size: int = 4_000_000,
        similarity_mode: str = "simgic",
    ) -> None:
        if similarity_mode not in {"exact", "ancestor", "simgic"}:
            raise ValueError(f"unknown GO similarity mode: {similarity_mode}")
        self.index = index
        self.similarity_mode = similarity_mode
        self.alpha = alpha
        self.max_hops = max_hops
        self.ot_reg = ot_reg
        self.disconnected_cost = disconnected_cost
        self._set_sim_cache: Dict[Tuple[Any, Any], float] = {}
        self._set_sim_cache_max = set_sim_cache_size
        self._closure_cache: Dict[FrozenSet[str], FrozenSet[str]] = {}

        ancestors = index.ancestors
        depth = index.depth

        @lru_cache(maxsize=cache_size)
        def _distance(u: str, v: str) -> Optional[int]:
            """Hops u -> LCA -> v, minimised over common ancestors."""
            if u == v:
                return 0
            au, av = ancestors.get(u), ancestors.get(v)
            if not au or not av:
                return None
            common = au & av
            if not common:
                return None
            du, dv = depth.get(u), depth.get(v)
            if du is None or dv is None:
                return None
            best: Optional[int] = None
            for c in common:
                dc = depth.get(c)
                if dc is None:
                    continue
                hops = (du - dc) + (dv - dc)
                if hops >= 0 and (best is None or hops < best):
                    best = hops
            if best is None or best > max_hops:
                return None
            return best

        self._term_distance = _distance

    # ------------------------------------------------------------- public API
    def skill_distance(self, u: str, v: str) -> Optional[int]:
        """GO DAG hops through the lowest common ancestor (``None`` if unrelated)."""
        return self._term_distance(u, v)

    def skill_sim(self, u: str, v: str) -> float:
        """``exp(-alpha * hops)`` in ``[0, 1]``; ``0.0`` when unrelated."""
        d = self._term_distance(u, v)
        return 0.0 if d is None else math.exp(-self.alpha * d)

    def _closure(self, terms: FrozenSet[str]) -> FrozenSet[str]:
        """Ancestor closure of a term set, memoised on the input set."""
        hit = self._closure_cache.get(terms)
        if hit is not None:
            return hit
        anc = self.index.ancestors
        acc: Set[str] = set()
        for t in terms:
            a = anc.get(t)
            if a:
                acc |= a
        out = frozenset(acc)
        if len(self._closure_cache) < 200_000:
            self._closure_cache[terms] = out
        return out

    def ontology_set_similarity(self, A: Sequence[str], B: Sequence[str]) -> float:
        """simGIC: IC-weighted Jaccard over ancestor closures. The ``s_esco`` analogue.

        Returns ``0.0`` when either side is empty (no signal), matching the MeSH
        and ESCO matchers. Symmetric and deterministic, so memoising is exact.

        Accepts either direct or already-propagated term lists: the closure is
        idempotent, so passing propagated terms costs a little time but changes
        nothing.
        """
        fa, fb = frozenset(A), frozenset(B)
        if not fa or not fb:
            return 0.0

        ha, hb = hash(fa), hash(fb)
        key = (fa, fb) if ha <= hb else (fb, fa)
        cached = self._set_sim_cache.get(key)
        if cached is not None:
            return cached

        # The exact arm uses only directly annotated terms. The other arms
        # propagate through the same pinned GO DAG, then differ only in IC.
        ca, cb = (fa, fb) if self.similarity_mode == "exact" else (
            self._closure(fa), self._closure(fb))
        if not ca or not cb:
            return 0.0
        inter = ca & cb
        union = ca | cb
        if self.similarity_mode != "simgic":
            val = len(inter) / len(union)
            if len(self._set_sim_cache) < self._set_sim_cache_max:
                self._set_sim_cache[key] = val
            return val
        ic = self.index.ic
        denom = sum(ic.get(t, 0.0) for t in union)
        if denom <= 0.0:
            return 0.0
        num = 0.0
        for t in inter:
            num += ic.get(t, 0.0)
        val = num / denom
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0

        if len(self._set_sim_cache) < self._set_sim_cache_max:
            self._set_sim_cache[key] = val
        return val

    def terms_for_gene(self, gene: str) -> FrozenSet[str]:
        return self.index.terms_for(gene)

    def coarse_for_gene(self, gene: str) -> FrozenSet[str]:
        return self.index.coarse_for(gene)

    def branch_distance(self, A: Sequence[str], B: Sequence[str]) -> float:
        """Coarse dissimilarity in ``[0, 1]`` -- the ``d_isco`` analogue.

        Jaccard *distance* over the two sides' shallow ancestors (depth <=
        ``coarse_max_depth``), i.e. "are these proteins in the same broad
        biological area". Deliberately unweighted and deliberately coarse: it must
        be decorrelated from :meth:`ontology_set_similarity`, which is IC-weighted
        over the full closure and therefore dominated by specific terms.

        Callers pass ``coarse_uris`` (already shallow, written by the converter).
        If given full term lists it still works -- the sets are intersected as
        given -- but the signal is then no longer coarse.

        Returns :data:`NEUTRAL_DISTANCE` when either side is empty, matching
        ``_isco_distance``'s 0.5-for-unknown convention.
        """
        fa, fb = frozenset(A), frozenset(B)
        if not fa or not fb:
            return NEUTRAL_DISTANCE
        union = len(fa | fb)
        if union == 0:
            return NEUTRAL_DISTANCE
        return 1.0 - (len(fa & fb) / union)

    def ot_distance(self, A: Sequence[str], B: Sequence[str]) -> Optional[float]:
        """Sinkhorn OT over pairwise GO DAG distances. ``None`` when either side is empty.

        Only computed when ``orca_capture_ot_distance`` is set; it is the most
        expensive signal here because a protein's term list is long.
        """
        import numpy as np

        A = list(dict.fromkeys(A))
        B = list(dict.fromkeys(B))
        if not A or not B:
            return None

        n, m = len(A), len(B)
        a = np.ones(n, dtype=np.float32) / n
        b = np.ones(m, dtype=np.float32) / m
        C = np.zeros((n, m), dtype=np.float32)
        for i, u in enumerate(A):
            for j, v in enumerate(B):
                d = self._term_distance(u, v)
                C[i, j] = float(d if d is not None else self.disconnected_cost)
        return self._sinkhorn(a, b, C, reg=self.ot_reg)

    @staticmethod
    def _sinkhorn(a, b, C, reg: float = 0.4, num_iters: int = 200, tol: float = 1e-6) -> float:
        """Entropic-regularised OT cost (same routine as the MeSH/ESCO matchers)."""
        import numpy as np

        K = np.exp(-C / reg)
        u = np.ones_like(a)
        v = np.ones_like(b)
        for _ in range(num_iters):
            u_prev = u
            Kv = K @ v
            Kv[Kv < 1e-30] = 1e-30
            u = a / Kv
            KTu = K.T @ u
            KTu[KTu < 1e-30] = 1e-30
            v = b / KTu
            if np.max(np.abs(u - u_prev)) < tol:
                break
        P = np.diag(u) @ K @ np.diag(v)
        return float(np.sum(P * C))

    def get_cache_stats(self) -> Dict[str, Any]:
        info = self._term_distance.cache_info()
        return {
            "distance_hits": info.hits,
            "distance_misses": info.misses,
            "distance_cached": info.currsize,
            "set_sim_cached": len(self._set_sim_cache),
            "closure_cached": len(self._closure_cache),
        }


class GoMultiAspectMatcher:
    """Mean of separately normalized GO aspect similarities.

    Missing annotations on either protein remove that aspect from the pair's
    denominator. No shared annotations within a covered aspect still score zero.
    Each branch keeps its own IC distribution and evidence filters.
    """

    def __init__(self, matchers: Dict[str, GoMatcher],
                 weights: Optional[Dict[str, float]] = None) -> None:
        if set(matchers) != {"P", "F", "C"}:
            raise ValueError("the GO hybrid requires P, F and C matchers")
        self.matchers = matchers
        self.weights = {a: float((weights or {}).get(a, 1.0)) for a in matchers}
        if any(not math.isfinite(w) or w <= 0 for w in self.weights.values()):
            raise ValueError("GO aspect weights must be finite and positive")
        self.similarity_mode = next(iter(matchers.values())).similarity_mode
        if any(m.similarity_mode != self.similarity_mode for m in matchers.values()):
            raise ValueError("GO aspects must use the same similarity mode")
        self.index = self  # existing logging code asks for len(matcher.index)
        self.meta = {"aspect": "A", "aspects": list(matchers)}
        self.alpha = next(iter(matchers.values())).alpha
        self.max_hops = next(iter(matchers.values())).max_hops
        self._term_aspect = {
            term: aspect for aspect, matcher in matchers.items()
            for term in matcher.index.ancestors
        }

    def __len__(self) -> int:
        return len(self._term_aspect)

    def terms_for_gene(self, gene: str) -> FrozenSet[str]:
        return frozenset().union(*(m.terms_for_gene(gene) for m in self.matchers.values()))

    def coarse_for_gene(self, gene: str) -> FrozenSet[str]:
        return frozenset().union(*(m.coarse_for_gene(gene) for m in self.matchers.values()))

    def aspect_similarities(self, A: Sequence[str], B: Sequence[str]) -> Dict[str, Optional[float]]:
        aa: Dict[str, Set[str]] = {a: set() for a in self.matchers}
        bb: Dict[str, Set[str]] = {a: set() for a in self.matchers}
        for terms, target in ((A, aa), (B, bb)):
            for term in terms:
                aspect = self._term_aspect.get(term)
                if aspect:
                    target[aspect].add(term)
        return {
            aspect: (matcher.ontology_set_similarity(aa[aspect], bb[aspect])
                     if aa[aspect] and bb[aspect] else None)
            for aspect, matcher in self.matchers.items()
        }

    def ontology_set_similarity(self, A: Sequence[str], B: Sequence[str]) -> float:
        scores = self.aspect_similarities(A, B)
        used = {a: score for a, score in scores.items() if score is not None}
        if not used:
            return 0.0
        return sum(self.weights[a] * score for a, score in used.items()) / sum(
            self.weights[a] for a in used)

    def branch_distance(self, A: Sequence[str], B: Sequence[str]) -> float:
        a, b = set(A), set(B)
        return NEUTRAL_DISTANCE if not a or not b else 1.0 - len(a & b) / len(a | b)

    def skill_distance(self, u: str, v: str) -> Optional[int]:
        aspect = self._term_aspect.get(u)
        if aspect is None or self._term_aspect.get(v) != aspect:
            return None
        return self.matchers[aspect].skill_distance(u, v)

    def skill_sim(self, u: str, v: str) -> float:
        d = self.skill_distance(u, v)
        return 0.0 if d is None else math.exp(-self.alpha * d)

    def ot_distance(self, A: Sequence[str], B: Sequence[str]) -> Optional[float]:
        # There is no defensible shared transport cost across GO aspects.
        raise ValueError("GO hybrid does not define cross-aspect OT distance")
