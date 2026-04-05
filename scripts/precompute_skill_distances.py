#!/usr/bin/env python3
"""
Precompute skill-pair distances for the training data to warm the LRU cache.

Collects all unique skill URIs from the training JSONL and the global job pool,
then computes pairwise distances for all resume-skill × job-skill combinations.
Saves the cache to disk for fast loading during training.

Usage:
    python3 scripts/precompute_skill_distances.py \
        --input preprocess/alitianchi_splits/train.jsonl \
        --graph dataset/esco/esco_kg_enriched.gpickle \
        --output embedding_cache/skill_distances.pkl \
        --pool-size 1000
"""
import argparse, json, logging, pickle, time
from pathlib import Path
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graph", default="dataset/esco/esco_kg_enriched.gpickle")
    parser.add_argument("--output", default="embedding_cache/skill_distances.pkl")
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--max-hops", type=int, default=8)
    args = parser.parse_args()

    import networkx as nx

    # Load graph
    logger.info(f"Loading graph from {args.graph}")
    if args.graph.endswith('.gpickle'):
        with open(args.graph, 'rb') as f:
            G = pickle.load(f)
    else:
        G = nx.read_gexf(args.graph)
    UG = G.to_undirected()
    logger.info(f"Graph: {UG.number_of_nodes()} nodes, {UG.number_of_edges()} edges")

    # Collect skill URIs from training data
    resume_skills = set()
    job_pool_skills = set()
    pool_jobs_seen = 0

    with open(args.input) as f:
        for line in f:
            d = json.loads(line)
            r_uris = d["resume"].get("skill_uris", [])
            j_uris = d["job"].get("skill_uris", [])
            resume_skills.update(r_uris)
            if pool_jobs_seen < args.pool_size:
                job_pool_skills.update(j_uris)
                pool_jobs_seen += 1
            # Also add all job skills (they appear as negatives)
            job_pool_skills.update(j_uris)

    logger.info(f"Resume skills: {len(resume_skills)}, Job pool skills: {len(job_pool_skills)}")
    logger.info(f"Pairs to compute: {len(resume_skills)} x {len(job_pool_skills)} = {len(resume_skills) * len(job_pool_skills):,}")

    # Compute distances
    cache = {}
    total_pairs = len(resume_skills) * len(job_pool_skills)
    computed = 0
    start = time.time()

    for i, r_uri in enumerate(resume_skills):
        for j_uri in job_pool_skills:
            if r_uri == j_uri:
                cache[(r_uri, j_uri)] = 0
            else:
                try:
                    d = nx.shortest_path_length(UG, r_uri, j_uri)
                    cache[(r_uri, j_uri)] = d if d <= args.max_hops else None
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    cache[(r_uri, j_uri)] = None
            computed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = computed / elapsed
            remaining = (total_pairs - computed) / rate / 60
            logger.info(f"  {i+1}/{len(resume_skills)} resume skills, "
                        f"{computed:,}/{total_pairs:,} pairs ({elapsed:.0f}s, ~{remaining:.0f}min remaining)")

    elapsed = time.time() - start
    logger.info(f"Computed {len(cache):,} distances in {elapsed:.1f}s")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(cache, f)
    logger.info(f"Saved to {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
