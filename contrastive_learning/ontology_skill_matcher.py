"""
On-the-fly ontology-aware skill similarity for negative selection.

Uses the full ESCO KG undirected graph (skill↔occupation↔skill paths)
to compute resume-to-job skill distances during batch processing.

The skill_distance results are cached across the entire training run,
so repeated skill pairs are essentially free after the first lookup.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class OntologySkillMatcher:
    """Computes ontology-aware skill similarity between resume and job skill URI sets."""

    def __init__(
        self,
        esco_graph_path: str,
        alpha: float = 0.7,
        max_hops: int = 8,
        ot_reg: float = 0.4,
        disconnected_cost: float = 10.0,
        cache_size: int = 500_000,
        esco_skills_path: str = None,
        reuse_weights: dict = None,
    ):
        logger.info(f"Loading ESCO graph from {esco_graph_path} for skill matching...")
        if esco_graph_path.endswith('.gpickle'):
            import pickle
            with open(esco_graph_path, 'rb') as f:
                directed = pickle.load(f)
        else:
            directed = nx.read_gexf(esco_graph_path)
        self._graph = directed.to_undirected()
        logger.info(
            f"OntologySkillMatcher ready: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

        self.alpha = alpha
        self.max_hops = max_hops
        self.ot_reg = ot_reg
        self.disconnected_cost = disconnected_cost

        # Skill reuse level weights
        self.skill_reuse_level = {}  # skill_uri -> reuse level string
        self.reuse_weights = reuse_weights or {}
        if esco_skills_path and reuse_weights:
            self._load_skill_reuse_levels(esco_skills_path)

        # Transversal skill mask (curated list, overrides reuse level)
        self.transversal_skills = set()
        self.transversal_weight = 0.3  # default weight for transversal skills

        # Build the cached distance function bound to this graph
        @lru_cache(maxsize=cache_size)
        def _cached_distance(u: str, v: str) -> Optional[int]:
            if u == v:
                return 0
            try:
                d = nx.shortest_path_length(self._graph, u, v)
                return d if d <= max_hops else None
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        self._skill_distance = _cached_distance

        # Build the cached distance function bound to this graph
        @lru_cache(maxsize=cache_size)
        def _cached_distance(u: str, v: str) -> Optional[int]:
            if u == v:
                return 0
            try:
                d = nx.shortest_path_length(self._graph, u, v)
                return d if d <= max_hops else None
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        self._skill_distance = _cached_distance

    def _load_skill_reuse_levels(self, csv_path: str) -> None:
        """Load skill reuse levels from skills CSV."""
        import csv
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = row.get('conceptUri', '')
                reuse = row.get('reuseLevel', '')
                if uri and reuse:
                    self.skill_reuse_level[uri] = reuse
                    count += 1
        logger.info(f"Loaded {count} skill reuse levels")

    def load_transversal_skills(self, csv_path: str, weight: float = 0.3) -> None:
        """Load curated transversal skills list for downweighting."""
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = row.get('conceptUri', '')
                if uri:
                    self.transversal_skills.add(uri)
        self.transversal_weight = weight
        logger.info(f"Loaded {len(self.transversal_skills)} transversal skills (weight={weight})")

    def load_precomputed_distances(self, cache_path: str) -> None:
        """Load precomputed skill-pair distances to seed the LRU cache.
        Replaces the cached function with a dict-backed version."""
        import pickle
        with open(cache_path, 'rb') as f:
            precomputed = pickle.load(f)

        # Replace the LRU-cached function with a dict-backed one that falls back to graph
        original_fn = self._skill_distance.__wrapped__ if hasattr(self._skill_distance, '__wrapped__') else None
        _graph = self._graph
        _max_hops = self.max_hops
        _precomputed = precomputed

        @lru_cache(maxsize=500_000)
        def _cached_with_precomputed(u: str, v: str) -> Optional[int]:
            key = (u, v)
            if key in _precomputed:
                return _precomputed[key]
            rev_key = (v, u)
            if rev_key in _precomputed:
                return _precomputed[rev_key]
            # Fall back to graph computation
            if u == v:
                return 0
            try:
                d = nx.shortest_path_length(_graph, u, v)
                return d if d <= _max_hops else None
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        self._skill_distance = _cached_with_precomputed
        logger.info(f"Loaded {len(precomputed):,} precomputed skill distances")

    # ── Public API ──

    def skill_distance(self, u: str, v: str) -> Optional[int]:
        """Shortest path distance between two skill URIs (cached)."""
        return self._skill_distance(u, v)

    def skill_sim(self, u: str, v: str) -> float:
        """Exponential decay similarity between two skill URIs."""
        d = self._skill_distance(u, v)
        if d is None:
            return 0.0
        return math.exp(-self.alpha * d)

    def ontology_set_similarity(self, A: List[str], B: List[str]) -> float:
        """Symmetric best-match average similarity between two skill URI sets.
        When reuse weights are configured, downweights transversal skills."""
        A = list(set(A))
        B = list(set(B))
        if not A or not B:
            return 0.0

        def skill_weight(uri: str) -> float:
            # Transversal mask takes priority (curated list)
            if self.transversal_skills and uri in self.transversal_skills:
                return self.transversal_weight
            # Fall back to reuse level weights
            if not self.reuse_weights:
                return 1.0
            reuse = self.skill_reuse_level.get(uri, '')
            return self.reuse_weights.get(reuse, 1.0)

        def dir_score(X, Y):
            total_weighted_sim = 0.0
            total_weight = 0.0
            for x in X:
                best = 0.0
                for y in Y:
                    best = max(best, self.skill_sim(x, y))
                    if best >= 0.999:
                        break
                w = skill_weight(x)
                total_weighted_sim += w * best
                total_weight += w
            return total_weighted_sim / total_weight if total_weight > 0 else 0.0

        return 0.5 * (dir_score(A, B) + dir_score(B, A))

    def ot_distance(self, A: List[str], B: List[str]) -> Optional[float]:
        """Sinkhorn OT distance between two skill URI sets using graph-distance cost."""
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
                d = self._skill_distance(u, v)
                C[i, j] = float(d if d is not None else self.disconnected_cost)
        return self._sinkhorn(a, b, C)

    def compute_resume_job_distance(
        self,
        resume: Dict[str, Any],
        job: Dict[str, Any],
    ) -> float:
        """Compute ontology distance between a resume and a job.

        Extracts skill URIs from metadata and computes ontology_set_similarity.
        Returns 0.0 (most similar) to 1.0 (least similar / no data).
        """
        r_uris = self._extract_skill_uris(resume, is_resume=True)
        j_uris = self._extract_skill_uris(job, is_resume=False)

        if not r_uris or not j_uris:
            return 0.5  # neutral — no ontology signal

        sim = self.ontology_set_similarity(r_uris, j_uris)
        return 1.0 - sim  # convert similarity to distance

    def get_cache_stats(self) -> dict:
        return dict(self._skill_distance.cache_info()._asdict())

    # ── Private helpers ──

    @staticmethod
    def _sinkhorn(
        a: np.ndarray, b: np.ndarray, C: np.ndarray,
        reg: float = 0.4, num_iters: int = 200, tol: float = 1e-6,
    ) -> float:
        K = np.exp(-C / reg)
        u = np.ones_like(a)
        v = np.ones_like(b)
        for _ in range(num_iters):
            u_prev = u
            u = a / (K @ v + 1e-12)
            v = b / (K.T @ u + 1e-12)
            if np.linalg.norm(u - u_prev, 1) < tol:
                break
        P = np.outer(u, v) * K
        return float(np.sum(P * C))

    @staticmethod
    def _extract_skill_uris(content: Dict[str, Any], is_resume: bool) -> List[str]:
        """Extract skill URIs from a resume or job dict.

        Looks in metadata first (from v3 enrichment), then falls back to
        extracting skill names (no URIs available).
        """
        # Check metadata for precomputed URIs
        metadata = content.get("_metadata", {})
        if is_resume:
            uris = metadata.get("resume_skill_uris", [])
        else:
            uris = metadata.get("job_skill_uris", [])
        if uris:
            return uris

        # Check if URIs are stored directly on the content
        uris = content.get("skill_uris", [])
        if uris:
            return uris

        return []
