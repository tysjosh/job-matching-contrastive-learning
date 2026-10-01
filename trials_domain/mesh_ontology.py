"""MeSH 2021 ontology index + an ``OntologySkillMatcher``-compatible matcher.

This is the clinical-trials analogue of
``contrastive_learning/ontology_skill_matcher.py``. It exposes the same public
surface the CDCL negative-selection path and ORCA's per-negative feature capture
already call — ``skill_distance``, ``skill_sim``, ``ontology_set_similarity``,
``ot_distance`` — so ``BatchProcessor._compute_negative_ontology_features`` and
``_select_ontology_negatives`` work unchanged against MeSH descriptors.

Why tree numbers instead of a graph
-----------------------------------
ESCO needs NetworkX shortest paths (and a precomputed ``skill_distances.pkl``)
because its skill graph has no closed-form ancestry. MeSH encodes ancestry
directly in dotted **tree numbers** (``C18.452.394.750.149``), so distance is a
string-prefix operation: O(depth), no graph, no precompute, no cache file. This
makes the trials ontology strictly cheaper than the career one.

Distance semantics
------------------
Each of the 16 MeSH top-level letters (A, B, C, D, ...) is its own tree. Two
tree numbers under the same letter are connected through a virtual per-letter
root; two under different letters are treated as disconnected. For paths ``p``
and ``q`` sharing a longest common component prefix ``lca``:

    hops(p, q) = (depth(p) - depth(lca)) + (depth(q) - depth(lca))

A descriptor is **polyhierarchical** — 52% of the 29,917 descriptors occupy more
than one position, up to 20 — so the distance between two *descriptors* is the
minimum ``hops`` over all pairs of their tree numbers. Taking the minimum (not
the mean) matches the "are these concepts close in any sense" reading that the
ESCO shortest-path distance also has.

Two deliberate deviations from the career matcher
------------------------------------------------
1. ``alpha`` defaults to ``0.5`` rather than ESCO's ``0.7``. The decay is applied
   to a *different distance scale*: MeSH tree hops run 0-26 across a 13-level
   hierarchy, whereas ESCO skill-graph hops are capped at 8. Reusing 0.7 would
   drive every non-sibling pair to a similarity of ~0 and flatten the signal.
   This is a domain-calibrated constant, not a tuned hyperparameter.

2. The ISCO analogue (:meth:`branch_distance`) uses a Wu-Palmer-style depth ratio
   instead of ISCO's discrete 4/3/2/1-digit bands. ISCO codes are fixed-width, so
   discrete prefix bands are well defined there; MeSH tree numbers vary from 1 to
   13 components, so a fixed band table would mean different things at different
   depths. The ratio ``1 - 2*|lca| / (|p| + |q|)`` is depth-normalized and stays
   in ``[0, 1]``, preserving the "coarse hierarchy agreement" semantics that
   ``d_isco`` contributes to the ORCA feature vector.
"""

from __future__ import annotations

import gzip
import logging
import math
import pickle
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Exponential-decay sharpness for tree-hop -> similarity. See the module
#: docstring for why this differs from the ESCO matcher's 0.7.
DEFAULT_ALPHA = 0.5

#: Tree hops beyond which two descriptors are treated as unrelated. The deepest
#: possible within-letter distance is 26 (two depth-13 leaves under one letter);
#: 12 keeps anything past "same level-1 branch, different subtree" at zero
#: similarity while leaving genuine subtree relationships intact.
DEFAULT_MAX_HOPS = 12

#: Cost assigned to a disconnected descriptor pair in the OT cost matrix,
#: mirroring the ESCO matcher's ``disconnected_cost``.
DEFAULT_DISCONNECTED_COST = 20.0

#: Neutral value returned when an ontology signal cannot be computed at all.
#: Matches the career path's convention (``_isco_distance`` returns 0.5 for an
#: unknown code) so ORCA's feature vector sees the same "signal absent" encoding.
NEUTRAL_DISTANCE = 0.5

#: The two MeSH descriptors that ship with an EMPTY ``TreeNumberList``
#: (``Female``, ``Male``). They are demographic terms with no hierarchy
#: position; a tree-number index that assumes every descriptor lands somewhere
#: would silently drop them. Tracked explicitly so callers can special-case
#: them against a trial's structured ``gender`` eligibility field.
KNOWN_TREELESS_DESCRIPTORS = ("D005260", "D008297")

_WS = re.compile(r"\s+")


def normalize_term(text: str) -> str:
    """Normalize a term string for entry-vocabulary lookup.

    Lowercases, collapses internal whitespace, and strips surrounding
    punctuation/space so ``"Diabetes Mellitus, Type 2"`` and
    ``"diabetes mellitus, type 2"`` resolve identically. Kept deliberately
    simple (no stemming, no synonym expansion) so resolution is deterministic
    and reproducible under a fixed ``training_seed``.
    """
    if not text:
        return ""
    return _WS.sub(" ", text.strip().lower()).strip(" .;:")


class MeshOntologyError(Exception):
    """Raised when the MeSH descriptor file cannot be read or parsed.

    Surfaces before training (mirroring ``CVEOntologyLoadError``) so a bad or
    missing ontology never degrades silently into a neutral-signal run.
    """


class MeshIndex:
    """Parsed MeSH 2021 descriptor hierarchy.

    Holds four maps built in a single streaming pass over ``desc<year>.gz``:

      * ``ui_to_name``   — ``D003924`` -> ``"Diabetes Mellitus, Type 2"``
      * ``ui_to_trees``  — ``D003924`` -> ``("C18.452.394.750.149", "C19.246.300")``
      * ``tree_to_ui``   — inverse of the above
      * ``term_to_ui``   — every normalized entry term -> its descriptor UI

    ``term_to_ui`` spans all ~252k terms (preferred plus entry synonyms), not
    just the 29,917 preferred names. This matters: 6.4% of the corpus's
    ``<mesh_term>`` annotations (``Infection``, ``Gemcitabine``,
    ``Liposomal doxorubicin``, ...) are entry terms rather than preferred
    descriptor names, and would otherwise fail to resolve.
    """

    __slots__ = ("ui_to_name", "ui_to_trees", "tree_to_ui", "term_to_ui", "source")

    def __init__(
        self,
        ui_to_name: Dict[str, str],
        ui_to_trees: Dict[str, Tuple[str, ...]],
        tree_to_ui: Dict[str, str],
        term_to_ui: Dict[str, str],
        source: str = "",
    ) -> None:
        self.ui_to_name = ui_to_name
        self.ui_to_trees = ui_to_trees
        self.tree_to_ui = tree_to_ui
        self.term_to_ui = term_to_ui
        self.source = source

    # ------------------------------------------------------------------ build
    @classmethod
    def from_descriptor_file(cls, path: str | Path) -> "MeshIndex":
        """Stream-parse ``desc2021.gz`` (or an uncompressed ``.xml``) into an index.

        Uses ``iterparse`` with ``el.clear()`` so the 16 MB gzip (a ~300 MB XML
        document) never materializes in memory.

        Raises:
            MeshOntologyError: If the file is missing, unreadable, or not
                parseable as a MeSH ``DescriptorRecordSet``.
        """
        path = Path(path)
        if not path.exists():
            raise MeshOntologyError(f"MeSH descriptor file not found: {path}")

        ui_to_name: Dict[str, str] = {}
        ui_to_trees: Dict[str, Tuple[str, ...]] = {}
        tree_to_ui: Dict[str, str] = {}
        term_to_ui: Dict[str, str] = {}

        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rb") as handle:
                for _event, el in ET.iterparse(handle, events=("end",)):
                    if el.tag != "DescriptorRecord":
                        continue

                    ui = (el.findtext("DescriptorUI") or "").strip()
                    name = (el.findtext("DescriptorName/String") or "").strip()
                    if not ui:
                        el.clear()
                        continue

                    ui_to_name[ui] = name

                    trees = tuple(
                        t.text.strip()
                        for t in el.iter("TreeNumber")
                        if t.text and t.text.strip()
                    )
                    ui_to_trees[ui] = trees
                    for tree in trees:
                        tree_to_ui[tree] = ui

                    # Index every lexical variant, preferred name included.
                    for term in el.iter("Term"):
                        raw = term.findtext("String")
                        key = normalize_term(raw or "")
                        # First writer wins: the preferred descriptor for a
                        # surface form beats a later homograph entry term.
                        if key and key not in term_to_ui:
                            term_to_ui[key] = ui
                    key = normalize_term(name)
                    if key:
                        term_to_ui[key] = ui  # preferred name always authoritative

                    el.clear()
        except MeshOntologyError:
            raise
        except (OSError, ET.ParseError) as exc:
            raise MeshOntologyError(f"Failed to parse {path}: {exc}") from exc

        if not ui_to_name:
            raise MeshOntologyError(
                f"{path} contained no DescriptorRecord entries; is it a MeSH "
                "DescriptorRecordSet?"
            )

        index = cls(ui_to_name, ui_to_trees, tree_to_ui, term_to_ui, source=str(path))
        treeless = [u for u in KNOWN_TREELESS_DESCRIPTORS if not ui_to_trees.get(u)]
        logger.info(
            "MeshIndex built from %s: %d descriptors, %d tree numbers, "
            "%d entry terms (%d known treeless: %s)",
            path.name,
            len(ui_to_name),
            len(tree_to_ui),
            len(term_to_ui),
            len(treeless),
            ", ".join(ui_to_name.get(u, u) for u in treeless) or "none",
        )
        return index

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Pickle the index so later runs skip the XML parse (~30 s)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "ui_to_name": self.ui_to_name,
                    "ui_to_trees": self.ui_to_trees,
                    "tree_to_ui": self.tree_to_ui,
                    "term_to_ui": self.term_to_ui,
                    "source": self.source,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("MeshIndex cached to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MeshIndex":
        """Load a pickled index written by :meth:`save`."""
        path = Path(path)
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise MeshOntologyError(f"Failed to load MeSH index {path}: {exc}") from exc
        return cls(
            payload["ui_to_name"],
            payload["ui_to_trees"],
            payload["tree_to_ui"],
            payload["term_to_ui"],
            source=payload.get("source", str(path)),
        )

    @classmethod
    def build_or_load(
        cls, descriptor_path: str | Path, cache_path: Optional[str | Path] = None
    ) -> "MeshIndex":
        """Load ``cache_path`` when present, else parse and populate it."""
        if cache_path is not None and Path(cache_path).exists():
            return cls.load(cache_path)
        index = cls.from_descriptor_file(descriptor_path)
        if cache_path is not None:
            index.save(cache_path)
        return index

    # ------------------------------------------------------------------ lookup
    def resolve(self, term: str) -> Optional[str]:
        """Resolve a term string (preferred name or entry synonym) to its UI."""
        return self.term_to_ui.get(normalize_term(term))

    def resolve_all(self, terms: Iterable[str]) -> List[str]:
        """Resolve many terms, dropping unresolvable ones, preserving order.

        Deduplicates while preserving first-seen order so the resulting descriptor
        list is deterministic for a given input ordering.
        """
        out: List[str] = []
        seen = set()
        for term in terms:
            ui = self.resolve(term)
            if ui is not None and ui not in seen:
                seen.add(ui)
                out.append(ui)
        return out

    def trees(self, ui: str) -> Tuple[str, ...]:
        """Tree numbers for ``ui`` (empty for unknown or treeless descriptors)."""
        return self.ui_to_trees.get(ui, ())

    def name(self, ui: str) -> str:
        """Preferred descriptor name for ``ui`` (``""`` when unknown)."""
        return self.ui_to_name.get(ui, "")

    def branches(self, ui: str) -> FrozenSet[str]:
        """Top-level letters (``{"C", "D"}``) this descriptor appears under."""
        return frozenset(t[0] for t in self.trees(ui) if t)

    def in_subtree(self, ui: str, prefixes: Sequence[str]) -> bool:
        """Whether ``ui`` occupies any position under one of ``prefixes``.

        Accepts a bare branch letter (``"C"``) or a deeper tree prefix
        (``"F03"``, ``"C18.452"``). See :func:`tree_matches_prefix` for the
        component-boundary matching rule.
        """
        trees = self.trees(ui)
        return any(
            tree_matches_prefix(tree, prefix) for tree in trees for prefix in prefixes
        )

    def __len__(self) -> int:
        return len(self.ui_to_name)

    def __contains__(self, ui: object) -> bool:
        return ui in self.ui_to_name


# --------------------------------------------------------------------- geometry
def tree_matches_prefix(tree: str, prefix: str) -> bool:
    """Whether ``tree`` lies at or under ``prefix``, respecting component boundaries.

    A bare letter matches its whole branch (``"C"`` matches ``"C18.452"``).
    A deeper prefix must align on a dot boundary, so ``"C1"`` does **not** match
    ``"C18.452"`` and ``"F03"`` matches ``"F03.600.300"`` but not ``"F03X"``.
    """
    if not tree or not prefix:
        return False
    if len(prefix) == 1:
        return tree[0] == prefix
    return tree == prefix or tree.startswith(prefix + ".")


def tree_path_hops(p: str, q: str) -> Optional[int]:
    """Hops between two tree numbers through their lowest common ancestor.

    Each top-level letter is a separate tree joined by a virtual root, so
    ``C18.452`` and ``C19.246`` are 4 hops apart (up two, down two) while
    ``C18.452`` and ``D02.241`` are disconnected.

    Returns ``None`` when the two paths sit under different letters.
    """
    if not p or not q:
        return None
    if p == q:
        return 0
    if p[0] != q[0]:
        return None

    pp = p.split(".")
    qq = q.split(".")
    shared = 0
    for a, b in zip(pp, qq):
        if a != b:
            break
        shared += 1
    return (len(pp) - shared) + (len(qq) - shared)


def tree_path_wu_palmer(p: str, q: str) -> float:
    """Depth-normalized dissimilarity between two tree numbers, in ``[0, 1]``.

    ``1 - 2*|lca| / (|p| + |q|)``: ``0.0`` for identical paths, ``1.0`` for paths
    under different letters (no shared ancestor). Depth-normalized so it means the
    same thing for a 2-component path and a 13-component one — the reason this
    replaces ISCO's fixed-width digit bands. See the module docstring.
    """
    if not p or not q:
        return 1.0
    if p == q:
        return 0.0
    if p[0] != q[0]:
        return 1.0

    pp = p.split(".")
    qq = q.split(".")
    shared = 0
    for a, b in zip(pp, qq):
        if a != b:
            break
        shared += 1
    return 1.0 - (2.0 * shared) / (len(pp) + len(qq))


class MeshMatcher:
    """``OntologySkillMatcher``-compatible similarity over MeSH descriptors.

    Method names and return contracts mirror the ESCO matcher so the existing
    ``BatchProcessor`` call sites work unchanged:

      * :meth:`skill_distance` -> ``Optional[int]`` tree hops
      * :meth:`skill_sim` -> ``exp(-alpha * hops)``, ``0.0`` when disconnected
      * :meth:`ontology_set_similarity` -> symmetric best-match average (``s_esco``)
      * :meth:`ot_distance` -> Sinkhorn OT over the tree-distance cost matrix

    Args:
        index: A built :class:`MeshIndex`.
        alpha: Exponential decay applied to tree hops (see :data:`DEFAULT_ALPHA`).
        max_hops: Hops beyond which similarity is 0 (see :data:`DEFAULT_MAX_HOPS`).
        ot_reg: Sinkhorn entropic regularization.
        disconnected_cost: OT cost for a cross-branch descriptor pair.
        cache_size: LRU size for pairwise descriptor distance.
        set_sim_cache_size: Bound on the memoized set-similarity map.
    """

    def __init__(
        self,
        index: MeshIndex,
        alpha: float = DEFAULT_ALPHA,
        max_hops: int = DEFAULT_MAX_HOPS,
        ot_reg: float = 0.4,
        disconnected_cost: float = DEFAULT_DISCONNECTED_COST,
        cache_size: int = 500_000,
        set_sim_cache_size: int = 4_000_000,
        similarity_mode: str = "hierarchy",
    ) -> None:
        if similarity_mode not in {"exact", "hierarchy"}:
            raise ValueError(f"unknown MeSH similarity mode: {similarity_mode}")
        self.index = index
        self.similarity_mode = similarity_mode
        self.alpha = alpha
        self.max_hops = max_hops
        self.ot_reg = ot_reg
        self.disconnected_cost = disconnected_cost

        self._set_sim_cache: Dict[Tuple[Any, Any], float] = {}
        self._set_sim_cache_max = set_sim_cache_size

        trees = index.ui_to_trees

        @lru_cache(maxsize=cache_size)
        def _distance(u: str, v: str) -> Optional[int]:
            """Min hops over every tree-number pair of two descriptors."""
            if u == v:
                return 0
            best: Optional[int] = None
            for p in trees.get(u, ()):
                for q in trees.get(v, ()):
                    hops = tree_path_hops(p, q)
                    if hops is not None and (best is None or hops < best):
                        best = hops
            if best is None or best > max_hops:
                return None
            return best

        self._descriptor_distance = _distance

    # ------------------------------------------------------------- public API
    def skill_distance(self, u: str, v: str) -> Optional[int]:
        """Tree-hop distance between two descriptor UIs (``None`` if unrelated)."""
        return self._descriptor_distance(u, v)

    def skill_sim(self, u: str, v: str) -> float:
        """Exponential-decay similarity ``exp(-alpha * hops)`` in ``[0, 1]``."""
        d = self._descriptor_distance(u, v)
        if d is None:
            return 0.0
        return math.exp(-self.alpha * d)

    def ontology_set_similarity(self, A: Sequence[str], B: Sequence[str]) -> float:
        """Symmetric best-match average similarity between two descriptor sets.

        This is the ``s_esco`` analogue. Pure, deterministic, and symmetric in
        ``A``/``B`` given a fixed index, so the memoized value is identical to
        recomputing it — the same justification the ESCO matcher uses for its
        cache. Returns ``0.0`` when either side is empty (no signal).
        """
        fa = frozenset(A)
        fb = frozenset(B)
        if not fa or not fb:
            return 0.0

        ha, hb = hash(fa), hash(fb)
        key = (fa, fb) if ha <= hb else (fb, fa)
        cached = self._set_sim_cache.get(key)
        if cached is not None:
            return cached

        if self.similarity_mode == "exact":
            overlap = len(fa & fb)
            val = 0.5 * (overlap / len(fa) + overlap / len(fb))
        else:
            val = self._compute_set_similarity(fa, fb)
        if len(self._set_sim_cache) < self._set_sim_cache_max:
            self._set_sim_cache[key] = val
        return val

    def _compute_set_similarity(self, A: FrozenSet[str], B: FrozenSet[str]) -> float:
        """Uncached symmetric best-match average (mirrors the ESCO formulation)."""

        def dir_score(X: FrozenSet[str], Y: FrozenSet[str]) -> float:
            total = 0.0
            for x in X:
                best = 0.0
                for y in Y:
                    s = self.skill_sim(x, y)
                    if s > best:
                        best = s
                        if best >= 0.999:
                            break
                total += best
            return total / len(X) if X else 0.0

        return 0.5 * (dir_score(A, B) + dir_score(B, A))

    def branch_distance(self, A: Sequence[str], B: Sequence[str]) -> float:
        """Coarse hierarchy dissimilarity in ``[0, 1]`` — the ``d_isco`` analogue.

        Minimum Wu-Palmer path dissimilarity over every tree-number pair drawn
        from the two descriptor sets. Fed the trial's *condition* descriptors only
        (not its interventions), this plays the role ISCO occupation groups play
        in the career domain: a coarse "are these in the same region of the
        hierarchy" signal that is deliberately decorrelated from the fine-grained
        set similarity in :meth:`ontology_set_similarity`.

        Returns :data:`NEUTRAL_DISTANCE` when either side has no positioned
        descriptor, matching ``_isco_distance``'s 0.5-for-unknown convention.
        """
        a_trees = [t for ui in A for t in self.index.trees(ui)]
        b_trees = [t for ui in B for t in self.index.trees(ui)]
        if not a_trees or not b_trees:
            return NEUTRAL_DISTANCE

        best = 1.0
        for p in a_trees:
            for q in b_trees:
                d = tree_path_wu_palmer(p, q)
                if d < best:
                    best = d
                    if best <= 0.0:
                        return 0.0
        return best

    def ot_distance(self, A: Sequence[str], B: Sequence[str]) -> Optional[float]:
        """Sinkhorn OT distance between two descriptor sets over tree distances.

        Mirrors the ESCO matcher: uniform marginals, cost matrix of pairwise tree
        hops with ``disconnected_cost`` for cross-branch pairs. Returns ``None``
        when either side is empty.
        """
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
                d = self._descriptor_distance(u, v)
                C[i, j] = float(d if d is not None else self.disconnected_cost)
        return self._sinkhorn(a, b, C, reg=self.ot_reg)

    @staticmethod
    def _sinkhorn(
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        reg: float = 0.4,
        num_iters: int = 200,
        tol: float = 1e-6,
    ) -> float:
        """Entropic-regularized OT cost (same routine as the ESCO matcher)."""
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
        """Cache diagnostics, mirroring the ESCO matcher's reporting surface."""
        info = self._descriptor_distance.cache_info()
        return {
            "distance_hits": info.hits,
            "distance_misses": info.misses,
            "distance_cached": info.currsize,
            "set_sim_cached": len(self._set_sim_cache),
        }
