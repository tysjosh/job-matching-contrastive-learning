#!/usr/bin/env python3
"""
Precompute ontology-aware similarity and optimal transport scores
for each record in the v3 enriched dataset.

Adds to esco_enrichment_v3.scores:
  - ontology_similarity: symmetric best-match average (0-1, higher = more similar)
  - ot_distance: Sinkhorn optimal transport cost (lower = more similar)

Uses the full ESCO KG undirected graph (has_skill edges create
skill↔occupation↔skill paths) — NOT the skill_parent_of hierarchy
which only covers ISCED-F education categories.

Usage:
    python3 scripts/compute_ontology_scores.py \
        --input preprocess/Combined_Structured_V1.2_esco_enriched_v3.jsonl \
        --output preprocess/Combined_Structured_V1.2_esco_enriched_v3_scored.jsonl \
        --graph dataset/esco/esco_kg.gexf
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Graph distance primitives
# ──────────────────────────────────────────────────────────────

UG: nx.Graph = None  # module-level undirected graph, set in main


@lru_cache(maxsize=500_000)
def skill_distance(u: str, v: str, max_hops: int = 8) -> int | None:
    if u == v:
        return 0
    try:
        dist = nx.shortest_path_length(UG, u, v)
        return dist if dist <= max_hops else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ──────────────────────────────────────────────────────────────
# Ontology-aware set similarity
# ──────────────────────────────────────────────────────────────

def skill_sim(u: str, v: str, alpha: float = 0.7, max_hops: int = 8) -> float:
    d = skill_distance(u, v, max_hops=max_hops)
    if d is None:
        return 0.0
    return math.exp(-alpha * d)


def ontology_set_similarity(
    A: list[str], B: list[str], alpha: float = 0.7, max_hops: int = 8,
) -> float:
    """Symmetric best-match average similarity between two skill URI sets."""
    A = list(set(A))
    B = list(set(B))
    if not A or not B:
        return 0.0

    def dir_score(X, Y):
        s = 0.0
        for x in X:
            best = 0.0
            for y in Y:
                best = max(best, skill_sim(x, y, alpha=alpha, max_hops=max_hops))
                if best >= 0.999:
                    break
            s += best
        return s / len(X)

    return 0.5 * (dir_score(A, B) + dir_score(B, A))


# ──────────────────────────────────────────────────────────────
# Optimal Transport with graph-distance cost
# ──────────────────────────────────────────────────────────────

def sinkhorn(
    a: np.ndarray, b: np.ndarray, C: np.ndarray,
    reg: float = 0.4, num_iters: int = 200, tol: float = 1e-6,
) -> tuple[float, np.ndarray]:
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
    return float(np.sum(P * C)), P


def ot_graph_distance(
    resume_skills: list[str], job_skills: list[str],
    reg: float = 0.4, max_hops: int = 8, disconnected_cost: float = 10.0,
) -> float | None:
    resume_skills = list(dict.fromkeys(resume_skills))
    job_skills = list(dict.fromkeys(job_skills))
    if not resume_skills or not job_skills:
        return None
    n, m = len(resume_skills), len(job_skills)
    a = np.ones(n, dtype=np.float32) / n
    b = np.ones(m, dtype=np.float32) / m
    C = np.zeros((n, m), dtype=np.float32)
    for i, u in enumerate(resume_skills):
        for j, v in enumerate(job_skills):
            d = skill_distance(u, v, max_hops=max_hops)
            C[i, j] = float(d if d is not None else disconnected_cost)
    dist, _ = sinkhorn(a, b, C, reg=reg)
    return dist


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute ontology scores for v3 enriched data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph", default="dataset/esco/esco_kg.gexf",
                        help="Path to ESCO KG .gexf file")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Decay factor for ontology similarity")
    parser.add_argument("--ot-reg", type=float, default=0.4,
                        help="Sinkhorn regularization")
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--disconnected-cost", type=float, default=10.0)
    args = parser.parse_args()

    global UG
    logger.info(f"Loading ESCO graph from {args.graph}...")
    directed = nx.read_gexf(args.graph)
    UG = directed.to_undirected()
    logger.info(f"Undirected graph: {UG.number_of_nodes()} nodes, {UG.number_of_edges()} edges")

    total = 0
    computed = 0
    skipped_empty = 0
    start = time.time()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            rec = json.loads(line)
            total += 1
            e = rec.get("esco_enrichment_v3", {})
            r_uris = e.get("resume_skill_uris", [])
            j_uris = e.get("job_skill_uris", [])

            if not r_uris or not j_uris:
                e.setdefault("scores", {})["ontology_similarity"] = None
                e["scores"]["ot_distance"] = None
                skipped_empty += 1
            else:
                ont_sim = ontology_set_similarity(
                    r_uris, j_uris, alpha=args.alpha, max_hops=args.max_hops)
                ot_dist = ot_graph_distance(
                    r_uris, j_uris, reg=args.ot_reg,
                    max_hops=args.max_hops, disconnected_cost=args.disconnected_cost)

                e.setdefault("scores", {})["ontology_similarity"] = round(ont_sim, 6)
                e["scores"]["ot_distance"] = round(ot_dist, 6) if ot_dist is not None else None
                computed += 1

            rec["esco_enrichment_v3"] = e
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total % 1000 == 0:
                elapsed = time.time() - start
                logger.info(f"  {total} records ({elapsed:.1f}s) ...")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"Done: {total} records in {elapsed:.1f}s")
    logger.info(f"  Computed: {computed}, Skipped (empty): {skipped_empty}")
    logger.info(f"  Cache: {skill_distance.cache_info()}")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
