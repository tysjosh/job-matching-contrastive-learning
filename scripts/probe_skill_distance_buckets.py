#!/usr/bin/env python3
"""Measure the realized ESCO skill-distance distribution against the ABSOLUTE tiers.

Tests the claim in ``BatchProcessor._select_ontology_negatives`` that on career v7
the absolute "hard" cut point (``d <= 0.3``) is unreachable, so the hard bucket is
always empty and ontology-tiered negative selection silently degrades to
medium/easy plus random fill.

That claim matters for interpreting the April 2026 learning-curve runs: they predate
``ontology_negative_rank_tiers`` (added 2026-07-23), so they used absolute tiers. If
the hard bucket really was empty, their "ontology-guided negatives" never delivered a
single hard negative to the loss, and whatever effect they showed cannot be credited
to hard-negative mining.

Distances are computed exactly as the batch processor computes them — same matcher,
same ``ontology_set_similarity``, same ``1 - similarity`` convention — so the numbers
are the ones selection actually saw, not a re-derivation.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: The absolute cut points hard-coded in the pre-fix bucketing.
HARD_MAX = 0.3
MEDIUM_MAX = 0.6


def load(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-file", type=Path,
                    default=ROOT / "preprocess/learning_curve_v7/frac_20/train.jsonl")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config/lc_career_ontfull.json")
    ap.add_argument("--anchors", type=int, default=60,
                    help="resumes to sample")
    ap.add_argument("--candidates", type=int, default=200,
                    help="candidate jobs scored per anchor (mirrors the global pool)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.ontology_skill_matcher import OntologySkillMatcher

    config = TrainingConfig.from_json(str(args.config))
    # Positional, matching how BatchProcessor constructs it (the first parameter is
    # named esco_graph_path but receives the full KG path when one is configured).
    kg_path = getattr(config, "esco_kg_path", None) or config.esco_graph_path
    matcher = OntologySkillMatcher(kg_path)

    records = load(args.train_file)
    rng = random.Random(args.seed)

    # Candidate jobs = the pool selection would draw from.
    jobs = [r["job"] for r in records if (r.get("job") or {}).get("skill_uris")]
    resumes = [r["resume"] for r in records if (r.get("resume") or {}).get("skill_uris")]
    print(f"{args.train_file.parent.name}: {len(records)} records, "
          f"{len(resumes)} resumes with skill_uris, {len(jobs)} jobs with skill_uris")
    if not resumes or not jobs:
        print("no skill URIs available — ontology selection could not run at all")
        return 1

    anchors = rng.sample(resumes, min(args.anchors, len(resumes)))
    all_d: List[float] = []
    per_anchor_min: List[float] = []
    empty_hard = 0
    bucket_totals = Counter()

    for resume in anchors:
        r_uris = resume.get("skill_uris", [])
        pool = rng.sample(jobs, min(args.candidates, len(jobs)))
        ds = []
        for job in pool:
            j_uris = job.get("skill_uris", [])
            if not j_uris:
                continue
            try:
                sim = float(matcher.ontology_set_similarity(r_uris, j_uris))
            except Exception:
                continue
            ds.append(1.0 - sim)
        if not ds:
            continue
        all_d.extend(ds)
        per_anchor_min.append(min(ds))
        hard = sum(1 for d in ds if d <= HARD_MAX)
        med = sum(1 for d in ds if HARD_MAX < d <= MEDIUM_MAX)
        easy = sum(1 for d in ds if d > MEDIUM_MAX)
        bucket_totals.update({"hard": hard, "medium": med, "easy": easy})
        if hard == 0:
            empty_hard += 1

    n = len(all_d)
    ordered = sorted(all_d)

    def pct(p):
        return ordered[min(n - 1, int(p * n))]

    print("\nrealized skill distance d = 1 - ontology_set_similarity")
    print("-" * 62)
    print(f"  pairs scored     {n}")
    print(f"  min              {min(all_d):.4f}")
    print(f"  p01 / p05 / p25  {pct(0.01):.4f} / {pct(0.05):.4f} / {pct(0.25):.4f}")
    print(f"  median           {st.median(all_d):.4f}")
    print(f"  p75 / max        {pct(0.75):.4f} / {max(all_d):.4f}")
    print(f"  mean             {st.fmean(all_d):.4f}")

    print("\nABSOLUTE bucket occupancy (the pre-fix scheme)")
    print("-" * 62)
    total = sum(bucket_totals.values())
    for name, lo, hi in (("hard", 0.0, HARD_MAX),
                         ("medium", HARD_MAX, MEDIUM_MAX),
                         ("easy", MEDIUM_MAX, 1.0)):
        c = bucket_totals[name]
        print(f"  {name:7s} (d in {lo:.1f}-{hi:.1f}]  {c:7d} pairs  {c/total:6.2%}")

    print(f"\n  anchors whose HARD bucket was EMPTY: {empty_hard}/{len(per_anchor_min)}"
          f"  ({empty_hard/max(1,len(per_anchor_min)):.0%})")
    print(f"  hardest available candidate per anchor: "
          f"min={min(per_anchor_min):.4f} median={st.median(per_anchor_min):.4f} "
          f"max={max(per_anchor_min):.4f}")

    verdict = (bucket_totals["hard"] == 0)
    print("\nVERDICT")
    print("-" * 62)
    if verdict:
        print("  The absolute hard bucket is EMPTY for every pair scored. Under the")
        print("  pre-fix scheme the ontology contributed NO hard negatives, so the")
        print("  April arm's 'ontology-guided negatives' were inert and its effect")
        print("  cannot be attributed to hard-negative mining.")
    elif empty_hard == len(per_anchor_min):
        print("  Every ANCHOR had an empty hard bucket even though some pairs fall")
        print("  under the threshold globally — selection is per-anchor, so the")
        print("  ontology still contributed no hard negatives.")
    else:
        print(f"  The hard bucket is reachable: {bucket_totals['hard']} pairs "
              f"({bucket_totals['hard']/total:.2%}) sit at d<={HARD_MAX}, and "
              f"{len(per_anchor_min)-empty_hard} anchors had a non-empty hard bucket.")
        print("  The code comment's premise does NOT hold on this split, so absolute")
        print("  vs rank tiering is a weaker distinction than assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
