#!/usr/bin/env python3
"""
Compute φ (skill coverage) scores for training data.

φ(c, j) measures how well candidate c's skills cover job j's requirements,
with partial credit for nearby skills in the ESCO graph.

Uses Strategy D (tight + reduced credit) — best Cohen's d per ablation:
  - Candidate has s exactly (URI match):        credit = 1.0
  - Candidate has s' with graph distance ≤ 2:   credit = 0.5  (1-hop logical)
  - No match or path found:                     credit = 0.0

φ(c, j) = Σ credit(s, c) / |skills(j)|

Usage:
    python3 scripts/compute_phi_scores.py \
        --input preprocess/data_splits_v4/train.jsonl \
        --output preprocess/data_splits_v4/train_phi.jsonl \
        --graph dataset/esco/esco_kg.gexf

    # Process all splits at once:
    python3 scripts/compute_phi_scores.py \
        --input preprocess/data_splits_v4/train.jsonl \
               preprocess/data_splits_v4/validation.jsonl \
               preprocess/data_splits_v4/test.jsonl \
        --graph dataset/esco/esco_kg.gexf
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from functools import lru_cache
from pathlib import Path

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Module-level undirected graph
UG: nx.Graph = None


# ──────────────────────────────────────────────────────────────
# Graph distance primitive (reused from compute_ontology_scores.py)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=500_000)
def skill_distance(u: str, v: str, max_hops: int = 4) -> int | None:
    """Shortest path length between two URIs in the undirected ESCO graph.
    Returns None if no path exists within max_hops."""
    if u == v:
        return 0
    try:
        dist = nx.shortest_path_length(UG, u, v)
        return dist if dist <= max_hops else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ──────────────────────────────────────────────────────────────
# φ computation
# ──────────────────────────────────────────────────────────────

def skill_credit(job_skill_uri: str, candidate_skill_uris: list[str]) -> float:
    """Compute credit for a single job skill given candidate's skill set.

    Uses Strategy D (tight + reduced credit) — best Cohen's d per ablation in
    compare_phi_strategies.py (d=0.292 good_fit vs no_fit).

    Returns:
        1.0 if exact match (URI match)
        0.5 if candidate has a skill within 2 graph hops (1 logical hop)
        0.0 otherwise

    The 0.5 credit (vs 0.7 in Strategy B) gives tighter stdevs and better
    effect sizes across all three label pairs.
    """
    best_distance = None
    for c_uri in candidate_skill_uris:
        d = skill_distance(job_skill_uri, c_uri, max_hops=2)
        if d is not None:
            if d == 0:
                return 1.0  # Exact match, no need to check further
            if best_distance is None or d < best_distance:
                best_distance = d

    if best_distance is None:
        return 0.0
    elif best_distance <= 2:
        return 0.5  # 1-hop logical (skill → occ → skill)
    else:
        return 0.0


def compute_phi(
    candidate_skill_uris: list[str],
    job_skill_uris: list[str],
) -> float | None:
    """Compute φ(c, j) — directional skill coverage score.

    Iterates over job skills and checks how well the candidate covers each one.
    Returns None if job has no skill URIs.
    """
    job_skills = list(set(job_skill_uris))
    candidate_skills = list(set(candidate_skill_uris))

    if not job_skills:
        return None
    if not candidate_skills:
        return 0.0

    total_credit = 0.0
    for j_uri in job_skills:
        total_credit += skill_credit(j_uri, candidate_skills)

    return total_credit / len(job_skills)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str, fallback_phi: float) -> dict:
    """Process a single JSONL file, adding phi to metadata. Returns stats."""
    total = 0
    computed = 0
    skipped_no_job = 0
    skipped_no_candidate = 0
    credit_distribution = {1.0: 0, 0.7: 0, 0.5: 0, 0.0: 0}
    phi_by_label = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            rec = json.loads(line)
            total += 1

            r_uris = rec["resume"].get("skill_uris", [])
            j_uris = rec["job"].get("skill_uris", [])
            label = rec.get("metadata", {}).get("original_label", "unknown")

            phi = compute_phi(r_uris, j_uris)

            if phi is None:
                # No job skills — use fallback
                phi = fallback_phi
                skipped_no_job += 1
            elif not r_uris:
                skipped_no_candidate += 1
            else:
                computed += 1

                # Track per-skill credit distribution for diagnostics
                for j_uri in set(j_uris):
                    c = skill_credit(j_uri, list(set(r_uris)))
                    credit_distribution[c] = credit_distribution.get(c, 0) + 1

            rec.setdefault("metadata", {})["phi"] = round(phi, 6)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Track phi by label
            phi_by_label.setdefault(label, []).append(phi)

            if total % 500 == 0:
                logger.info(f"  {total} records processed...")

    return {
        "total": total,
        "computed": computed,
        "skipped_no_job": skipped_no_job,
        "skipped_no_candidate": skipped_no_candidate,
        "credit_distribution": credit_distribution,
        "phi_by_label": phi_by_label,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute φ skill coverage scores")
    parser.add_argument("--input", required=True, nargs="+",
                        help="Input JSONL file(s)")
    parser.add_argument("--graph", default="dataset/esco/esco_kg.gexf",
                        help="Path to ESCO KG .gexf file")
    parser.add_argument("--fallback-phi", type=float, default=0.5,
                        help="Fallback φ for samples without job skill URIs")
    args = parser.parse_args()

    # Load graph
    global UG
    logger.info(f"Loading ESCO graph from {args.graph}...")
    directed = nx.read_gexf(args.graph)
    UG = directed.to_undirected()
    logger.info(f"Undirected graph: {UG.number_of_nodes()} nodes, {UG.number_of_edges()} edges")

    start = time.time()

    for input_path in args.input:
        # Output goes next to input with _phi suffix
        p = Path(input_path)
        output_path = str(p.parent / (p.stem + "_phi" + p.suffix))

        logger.info(f"\nProcessing {input_path} -> {output_path}")
        stats = process_file(input_path, output_path, args.fallback_phi)

        # Report
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Computed: {stats['computed']}")
        logger.info(f"  Skipped (no job URIs): {stats['skipped_no_job']}")
        logger.info(f"  Skipped (no candidate URIs): {stats['skipped_no_candidate']}")

        # Credit distribution
        cd = stats["credit_distribution"]
        total_credits = sum(cd.values())
        if total_credits > 0:
            logger.info(f"  Per-skill credit distribution:")
            logger.info(f"    1.0 (exact):  {cd.get(1.0, 0):5d} ({100*cd.get(1.0, 0)/total_credits:.1f}%)")
            logger.info(f"    0.5 (1-hop):  {cd.get(0.5, 0):5d} ({100*cd.get(0.5, 0)/total_credits:.1f}%)")
            logger.info(f"    0.0 (none):   {cd.get(0.0, 0):5d} ({100*cd.get(0.0, 0)/total_credits:.1f}%)")

        # φ by label
        import statistics
        logger.info(f"  φ by label:")
        for label in ["good_fit", "potential_fit", "no_fit"]:
            vals = stats["phi_by_label"].get(label, [])
            if vals:
                logger.info(
                    f"    {label:15s}: mean={statistics.mean(vals):.4f}, "
                    f"median={statistics.median(vals):.4f}, "
                    f"stdev={statistics.stdev(vals):.4f}, "
                    f"n={len(vals)}"
                )

    elapsed = time.time() - start
    logger.info(f"\nDone in {elapsed:.1f}s")
    logger.info(f"Cache: {skill_distance.cache_info()}")


if __name__ == "__main__":
    main()
