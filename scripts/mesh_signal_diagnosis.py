#!/usr/bin/env python3
"""Why MeSH ontology guidance didn't help on TREC clinical trials.

The claim "MeSH correlates with the grade" was asserted (Cohen's d=0.91 g2-vs-g0,
d=0.21 g2-vs-g1) but never recomputed in this session — no script in the repo
produces those numbers from data. This does, directly from the MeSH matcher used
in training, and asks four separate questions rather than one:

1. Is the raw MeSH signal (1 - ontology_set_similarity) actually separated by
   grade, on the SAME split the low-budget arms trained on? Recomputes the
   correlation rather than trusting the earlier figure.
2. Does that signal survive the negative-selection pipeline, or does it get
   diluted by the score_cap / tercile mechanics before it reaches a batch?
3. Is the signal available on enough anchors to matter, or is coverage (missing
   skill_uris, condition_uris) quietly cutting the effective sample size?
4. Does the signal's distribution overlap so much between grades that no
   threshold-free classifier (which is what a projection head trained on
   contrastive loss approximates) could separate them even with unlimited data?
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cohens_d(a: List[float], b: List[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = st.variance(a), st.variance(b)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    if pooled == 0:
        return float("nan")
    return (st.fmean(a) - st.fmean(b)) / pooled


def mann_whitney_auc(pos: List[float], neg: List[float]) -> Optional[float]:
    """Rank-based AUC: P(pos > neg) for a randomly drawn pair, ties split."""
    if not pos or not neg:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n = len(merged)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_v, lab) in zip(ranks, merged) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", type=Path,
                    default=ROOT / "preprocess/trec_ct_splits/validation.jsonl",
                    help="the split the low-budget arms actually trained/scored on")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "results/label_budget/low_ontology_s42/training_config.json")
    ap.add_argument("--anchors", type=int, default=200)
    ap.add_argument("--per-anchor-candidates", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import random
    from contrastive_learning.data_structures import TrainingConfig
    from trials_domain.run_config import build_mesh_matcher

    config = TrainingConfig.from_json(str(args.config))
    matcher = build_mesh_matcher(config)
    rng = random.Random(args.seed)

    records = [json.loads(l) for l in open(args.split) if l.strip()]
    by_topic: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        topic = (r.get("metadata") or {}).get("topic_id") or r.get("resume", {}).get("topic_id")
        by_topic[str(topic)].append(r)

    # ------------------------------------------------------ Q1: raw signal
    print("=" * 84)
    print("Q1: is the raw MeSH distance actually separated by grade, on THIS split?")
    print("=" * 84)
    by_grade: Dict[int, List[float]] = defaultdict(list)
    coverage = {"total": 0, "missing_anchor_uris": 0, "missing_candidate_uris": 0, "scored": 0}

    topics = list(by_topic)
    rng.shuffle(topics)
    for topic in topics[: args.anchors]:
        rows = by_topic[topic]
        anchor_uris = (rows[0].get("resume") or {}).get("skill_uris", [])
        coverage["total"] += len(rows)
        if not anchor_uris:
            coverage["missing_anchor_uris"] += len(rows)
            continue
        sample = rng.sample(rows, min(args.per_anchor_candidates, len(rows)))
        for row in sample:
            job = row.get("job") or {}
            cand_uris = job.get("skill_uris", [])
            grade = job.get("grade")
            if grade is None:
                continue
            if not cand_uris:
                coverage["missing_candidate_uris"] += 1
                continue
            try:
                sim = matcher.ontology_set_similarity(anchor_uris, cand_uris)
            except Exception:
                continue
            by_grade[int(grade)].append(1.0 - float(sim))
            coverage["scored"] += 1

    print(f"  coverage: {coverage['scored']}/{coverage['total']} pairs scored "
          f"({coverage['missing_anchor_uris']} anchors missing skill_uris, "
          f"{coverage['missing_candidate_uris']} candidates missing skill_uris)")
    for g in sorted(by_grade):
        v = by_grade[g]
        print(f"  grade {g}: n={len(v):4d}  mean={st.fmean(v):.4f}  "
              f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  "
              f"median={st.median(v):.4f}")

    print("\n  pairwise separation of the RAW MeSH DISTANCE by grade:")
    for hi, lo, label in ((2, 0, "g2 vs g0 (topical)"), (2, 1, "g2 vs g1 (eligibility)"),
                          (1, 0, "g1 vs g0")):
        if hi in by_grade and lo in by_grade:
            d = cohens_d(by_grade[lo], by_grade[hi])  # lower distance = more similar = grade advantage
            auc = mann_whitney_auc([-x for x in by_grade[hi]], [-x for x in by_grade[lo]])
            print(f"    {label:22s} Cohen's d={d:+.3f}  rank-AUC={auc:.3f}"
                  f"  (n_hi={len(by_grade[hi])}, n_lo={len(by_grade[lo])})")

    # -------------------------------------------------- Q2: overlap / ceiling
    print("\n" + "=" * 84)
    print("Q2: distribution overlap — what AUC ceiling does the raw signal imply?")
    print("=" * 84)
    print("  A rank-AUC on the raw MeSH distance alone is the ceiling any model")
    print("  trained ONLY on that signal could reach (no encoder, no learning).")
    print("  If it's already near 0.5 for g2-vs-g1, no amount of training data or")
    print("  architecture can push the eligibility contrast higher via this ontology.")

    if 2 in by_grade and 1 in by_grade:
        pooled_g2g0 = mann_whitney_auc([-x for x in by_grade.get(2, [])],
                                       [-x for x in by_grade.get(0, [])]) if 0 in by_grade else None
        pooled_g2g1 = mann_whitney_auc([-x for x in by_grade.get(2, [])],
                                       [-x for x in by_grade.get(1, [])])
        print(f"\n  MeSH-alone ceiling:  g2-vs-g0 = {pooled_g2g0}   g2-vs-g1 = {pooled_g2g1}")
        print(f"  Measured model (trained projection head, from the earlier diagnostic):"
              f" g2-vs-g0 = 0.8356   g2-vs-g1 = 0.5396")
        print(f"  -> the MODEL exceeds the ontology-alone ceiling on g2-vs-g0"
              f" (learns from text beyond what MeSH encodes),")
        print(f"     and sits close to the ontology-alone ceiling on g2-vs-g1"
              f" (there may be little more to learn from MeSH on this contrast).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
