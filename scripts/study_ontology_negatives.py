#!/usr/bin/env python3
"""Study: does the ESCO KG actually yield usable hard/medium/easy negative tiers
on the v7 data?

This is the empirical crux of "how ontology helps InfoNCE". Ontology's only real
lever on the InfoNCE embedding is hard-negative mining: `_select_ontology_negatives`
scores each candidate negative job by ESCO skill-set distance to the anchor resume
(distance = 1 - ontology_set_similarity), buckets hard(<=0.3)/medium/easy(>0.6),
and samples a fixed ratio (0.34/0.33/0.33) from each. That only helps if the
candidate pool (a RANDOM global pool of ~1000 jobs) actually contains hard
(skill-similar) negatives.

This script reproduces that scoring faithfully on v7 train pairs + esco_kg.gexf and
reports the realized bucket distribution and how often the "hard" quota can be
filled. If hard is ~0%, ontology-tiered negatives collapse to (mostly) random.

Usage:
    .venv/bin/python scripts/study_ontology_negatives.py \
        --data preprocess/data_splits_v7/train.jsonl \
        --kg dataset/esco/esco_kg.gexf --anchors 25 --pool 150
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

HARD_MAX = 0.3      # distance <= 0.3  -> hard  (>=~70% skill-set similarity)
MED_MAX = 0.6       # 0.3 < d <= 0.6   -> medium; d > 0.6 -> easy
MAX_NEG = 7
HARD_RATIO, MED_RATIO = 0.34, 0.33


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocess/data_splits_v7/train.jsonl")
    ap.add_argument("--kg", default="dataset/esco/esco_kg.gexf")
    ap.add_argument("--precomputed", default="embedding_cache/skill_distances.pkl")
    ap.add_argument("--anchors", type=int, default=25)
    ap.add_argument("--pool", type=int, default=150, help="candidate jobs scored per anchor")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from contrastive_learning.ontology_skill_matcher import OntologySkillMatcher

    rng = random.Random(args.seed)
    records = [json.loads(l) for l in open(args.data) if l.strip()]
    # Jobs pool: (occupation_uri, skill_uris) from every record's job slot.
    jobs = [r["job"] for r in records if (r.get("job") or {}).get("skill_uris")]
    resumes = [r["resume"] for r in records if (r.get("resume") or {}).get("skill_uris")]
    print(f"records={len(records)} jobs_with_skills={len(jobs)} resumes_with_skills={len(resumes)}")

    m = OntologySkillMatcher(args.kg)
    if Path(args.precomputed).exists():
        m.load_precomputed_distances(args.precomputed)
        print(f"loaded precomputed distances from {args.precomputed}")

    hard_needed = int(MAX_NEG * HARD_RATIO)   # = 2
    med_needed = int(MAX_NEG * MED_RATIO)     # = 2

    anchors = rng.sample(resumes, min(args.anchors, len(resumes)))
    all_d = []
    per_anchor = []
    t0 = time.time()
    for i, res in enumerate(anchors):
        r_uris = res.get("skill_uris") or []
        pool = rng.sample(jobs, min(args.pool, len(jobs)))
        dists = []
        for job in pool:
            sim = m.ontology_set_similarity(r_uris, job.get("skill_uris") or [])
            dists.append(1.0 - sim)
        all_d.extend(dists)
        hard = sum(1 for d in dists if d <= HARD_MAX)
        med = sum(1 for d in dists if HARD_MAX < d <= MED_MAX)
        easy = sum(1 for d in dists if d > MED_MAX)
        per_anchor.append((hard, med, easy))
        print(f"  anchor {i+1}/{len(anchors)}: pool={len(dists)} "
              f"hard={hard} med={med} easy={easy} "
              f"min_d={min(dists):.3f} median_d={sorted(dists)[len(dists)//2]:.3f}", flush=True)

    n = len(all_d)
    hard = sum(1 for d in all_d if d <= HARD_MAX)
    med = sum(1 for d in all_d if HARD_MAX < d <= MED_MAX)
    easy = sum(1 for d in all_d if d > MED_MAX)
    print(f"\n=== overall distance-bucket distribution (n={n} pairs, {time.time()-t0:.0f}s) ===")
    print(f"  hard  (d<=0.30): {hard:6d}  ({100*hard/n:5.2f}%)")
    print(f"  medium(0.3-0.6): {med:6d}  ({100*med/n:5.2f}%)")
    print(f"  easy  (d>0.60):  {easy:6d}  ({100*easy/n:5.2f}%)")

    # How often can the hard/medium quotas actually be filled from the pool?
    hard_ok = sum(1 for h, _, _ in per_anchor if h >= hard_needed)
    med_ok = sum(1 for _, mm, _ in per_anchor if mm >= med_needed)
    A = len(per_anchor)
    print(f"\n=== quota fillability (per anchor, pool={args.pool}) ===")
    print(f"  need {hard_needed} hard negatives: {hard_ok}/{A} anchors can fill "
          f"({100*hard_ok/A:.0f}%)")
    print(f"  need {med_needed} medium negatives: {med_ok}/{A} anchors can fill "
          f"({100*med_ok/A:.0f}%)")
    print("\nInterpretation: if 'hard' is a tiny % and few anchors can fill the hard "
          "quota, the ontology-tiered selection collapses to medium/easy + random "
          "fill — i.e. ontology injects few/no genuinely hard negatives into InfoNCE.")


if __name__ == "__main__":
    main()
