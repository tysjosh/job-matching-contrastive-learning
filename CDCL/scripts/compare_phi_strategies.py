#!/usr/bin/env python3
"""
Compare different φ computation strategies to find the best separation.

Strategy A (original/loose): 0 hops=1.0, ≤2 hops=0.7, ≤4 hops=0.5, else=0.0
Strategy B (tight):          0 hops=1.0, ≤2 hops=0.7, else=0.0
Strategy C (exact only):     0 hops=1.0, else=0.0

Usage:
    python3 scripts/compare_phi_strategies.py
"""
import json
import math
import statistics
import time
from collections import Counter
from functools import lru_cache

import networkx as nx

# Load graph
print("Loading ESCO graph...")
directed = nx.read_gexf("../dataset/esco/esco_kg.gexf")
UG = directed.to_undirected()
print(f"Graph: {UG.number_of_nodes()} nodes, {UG.number_of_edges()} edges")


@lru_cache(maxsize=500_000)
def skill_distance(u: str, v: str) -> int | None:
    """Shortest path in undirected ESCO graph. Returns None if >6 or no path."""
    if u == v:
        return 0
    try:
        dist = nx.shortest_path_length(UG, u, v)
        return dist if dist <= 6 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ── Strategy credit functions ──

def credit_A(job_uri: str, candidate_uris: list[str]) -> float:
    """Strategy A (original): 0=1.0, ≤2=0.7, ≤4=0.5, else=0.0"""
    best = None
    for c_uri in candidate_uris:
        d = skill_distance(job_uri, c_uri)
        if d is not None:
            if d == 0:
                return 1.0
            if best is None or d < best:
                best = d
    if best is None:
        return 0.0
    if best <= 2:
        return 0.7
    if best <= 4:
        return 0.5
    return 0.0


def credit_B(job_uri: str, candidate_uris: list[str]) -> float:
    """Strategy B (tight): 0=1.0, ≤2=0.7, else=0.0"""
    best = None
    for c_uri in candidate_uris:
        d = skill_distance(job_uri, c_uri)
        if d is not None:
            if d == 0:
                return 1.0
            if best is None or d < best:
                best = d
    if best is None:
        return 0.0
    if best <= 2:
        return 0.7
    return 0.0


def credit_C(job_uri: str, candidate_uris: list[str]) -> float:
    """Strategy C (exact only): 0=1.0, else=0.0"""
    for c_uri in candidate_uris:
        if job_uri == c_uri:
            return 1.0
    return 0.0


def credit_D(job_uri: str, candidate_uris: list[str]) -> float:
    """Strategy D (tight + reduced credit): 0=1.0, ≤2=0.5, else=0.0"""
    best = None
    for c_uri in candidate_uris:
        d = skill_distance(job_uri, c_uri)
        if d is not None:
            if d == 0:
                return 1.0
            if best is None or d < best:
                best = d
    if best is None:
        return 0.0
    if best <= 2:
        return 0.5
    return 0.0


def compute_phi(job_uris: list[str], candidate_uris: list[str], credit_fn) -> float | None:
    """Compute φ using the given credit function."""
    job_skills = list(set(job_uris))
    cand_skills = list(set(candidate_uris))
    if not job_skills:
        return None
    if not cand_skills:
        return 0.0
    total = sum(credit_fn(j, cand_skills) for j in job_skills)
    return total / len(job_skills)


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Effect size: Cohen's d."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = statistics.mean(group1), statistics.mean(group2)
    s1, s2 = statistics.stdev(group1), statistics.stdev(group2)
    pooled = math.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else 0.0


# ── Load training data ──
print("\nLoading training data...")
records = []
with open("../preprocess/data_splits_v4/train.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
print(f"Loaded {len(records)} records")

# ── Compute φ for each strategy ──
strategies = {
    "A (loose: ≤2→0.7, ≤4→0.5)": credit_A,
    "B (tight: ≤2→0.7, else→0)": credit_B,
    "C (exact only)": credit_C,
    "D (tight: ≤2→0.5, else→0)": credit_D,
}

print("\nComputing φ scores for all strategies...")
start = time.time()

results = {}  # strategy_name -> {label -> [phi_values]}

for name, credit_fn in strategies.items():
    phi_by_label: dict[str, list[float]] = {}
    credit_dist = Counter()
    computed = 0

    for rec in records:
        j_uris = rec["job"].get("skill_uris", [])
        r_uris = rec["resume"].get("skill_uris", [])
        label = rec.get("metadata", {}).get("original_label", "unknown")

        phi = compute_phi(j_uris, r_uris, credit_fn)
        if phi is not None:
            phi_by_label.setdefault(label, []).append(phi)
            computed += 1

    results[name] = phi_by_label
    print(f"  {name}: {computed} computed")

elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s")

# ── Report ──
print("\n" + "=" * 80)
print("STRATEGY COMPARISON")
print("=" * 80)

for name, phi_by_label in results.items():
    print(f"\n{'─' * 60}")
    print(f"Strategy: {name}")
    print(f"{'─' * 60}")

    for label in ["good_fit", "potential_fit", "no_fit"]:
        vals = phi_by_label.get(label, [])
        if vals:
            print(
                f"  {label:15s}: mean={statistics.mean(vals):.4f}, "
                f"stdev={statistics.stdev(vals):.4f}, "
                f"median={statistics.median(vals):.4f}, "
                f"n={len(vals)}"
            )

    gf = phi_by_label.get("good_fit", [])
    pf = phi_by_label.get("potential_fit", [])
    nf = phi_by_label.get("no_fit", [])

    if gf and nf:
        gap_gf_nf = statistics.mean(gf) - statistics.mean(nf)
        d_gf_nf = cohens_d(gf, nf)
        print(f"  Gap (good-no):       {gap_gf_nf:+.4f}  Cohen's d={d_gf_nf:.3f}")
    if gf and pf:
        gap_gf_pf = statistics.mean(gf) - statistics.mean(pf)
        d_gf_pf = cohens_d(gf, pf)
        print(f"  Gap (good-potential): {gap_gf_pf:+.4f}  Cohen's d={d_gf_pf:.3f}")
    if pf and nf:
        gap_pf_nf = statistics.mean(pf) - statistics.mean(nf)
        d_pf_nf = cohens_d(pf, nf)
        print(f"  Gap (potential-no):   {gap_pf_nf:+.4f}  Cohen's d={d_pf_nf:.3f}")

# ── Also compare with ontology_similarity from the data ──
print(f"\n{'─' * 60}")
print("Reference: ontology_similarity (from data)")
print(f"{'─' * 60}")

ont_by_label: dict[str, list[float]] = {}
for rec in records:
    ont = rec.get("metadata", {}).get("ontology_similarity")
    label = rec.get("metadata", {}).get("original_label", "unknown")
    if ont is not None:
        ont_by_label.setdefault(label, []).append(ont)

for label in ["good_fit", "potential_fit", "no_fit"]:
    vals = ont_by_label.get(label, [])
    if vals:
        print(
            f"  {label:15s}: mean={statistics.mean(vals):.4f}, "
            f"stdev={statistics.stdev(vals):.4f}, "
            f"n={len(vals)}"
        )

gf = ont_by_label.get("good_fit", [])
nf = ont_by_label.get("no_fit", [])
if gf and nf:
    print(f"  Gap (good-no):       {statistics.mean(gf) - statistics.mean(nf):+.4f}  Cohen's d={cohens_d(gf, nf):.3f}")

print(f"\nCache: {skill_distance.cache_info()}")
