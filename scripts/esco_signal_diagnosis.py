#!/usr/bin/env python3
"""Why ESCO ontology guidance didn't help on career resume-job matching.

Direct analogue of scripts/mesh_signal_diagnosis.py. Asks the same question with
the roles reversed: career's "hard" contrast is good_fit vs potential_fit (the
model must separate genuinely qualified candidates from plausible-but-not-quite
ones), analogous to trials' eligible-vs-ineligible. good_fit vs no_fit is the easy
contrast, analogous to trials' eligible-vs-not-relevant.

If ESCO shows the same pattern MeSH did -- strong separation on the easy contrast,
near-chance on the hard one -- both nulls share one mechanism: the ontology's own
discriminative power predicts, before any training, which contrast it can help
with, and neither ontology has signal on the contrast each task actually grades.
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


LABEL_ORDER = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}
LABEL_NAME = {2: "good_fit", 1: "potential_fit", 0: "no_fit"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "preprocess/data_splits_v7/train.jsonl",
                    help="the same source the career learning-curve arms trained on")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config/lc_career_ontfull.json")
    ap.add_argument("--sample", type=int, default=4000,
                    help="records to score (0 = all)")
    args = ap.parse_args(argv)

    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.ontology_skill_matcher import OntologySkillMatcher

    config = TrainingConfig.from_json(str(args.config))
    kg_path = getattr(config, "esco_kg_path", None) or config.esco_graph_path
    matcher = OntologySkillMatcher(kg_path)

    records = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.sample and len(records) > args.sample:
        import random
        records = random.Random(42).sample(records, args.sample)

    print("=" * 84)
    print("Q1: is the raw ESCO skill distance actually separated by grade?")
    print("=" * 84)
    by_grade: Dict[int, List[float]] = defaultdict(list)
    coverage = {"total": len(records), "missing_resume_uris": 0, "missing_job_uris": 0, "scored": 0}

    for r in records:
        resume_uris = (r.get("resume") or {}).get("skill_uris", [])
        job_uris = (r.get("job") or {}).get("skill_uris", [])
        label = (r.get("metadata") or {}).get("original_label")
        grade = LABEL_ORDER.get(label)
        if grade is None:
            continue
        if not resume_uris:
            coverage["missing_resume_uris"] += 1
            continue
        if not job_uris:
            coverage["missing_job_uris"] += 1
            continue
        try:
            sim = matcher.ontology_set_similarity(resume_uris, job_uris)
        except Exception:
            continue
        by_grade[grade].append(1.0 - float(sim))
        coverage["scored"] += 1

    print(f"  coverage: {coverage['scored']}/{coverage['total']} pairs scored "
          f"({coverage['missing_resume_uris']} resumes missing skill_uris, "
          f"{coverage['missing_job_uris']} jobs missing skill_uris)")
    for g in sorted(by_grade, reverse=True):
        v = by_grade[g]
        print(f"  {LABEL_NAME[g]:15s} (grade {g}): n={len(v):5d}  mean={st.fmean(v):.4f}  "
              f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  median={st.median(v):.4f}")

    print("\n  pairwise separation of the RAW ESCO DISTANCE by grade "
          "(lower distance = more similar = expected for the higher grade):")
    for hi, lo, label in ((2, 0, "good_fit vs no_fit (topical/'easy')"),
                          (2, 1, "good_fit vs potential_fit ('hard' -- analogue of eligibility)"),
                          (1, 0, "potential_fit vs no_fit")):
        if hi in by_grade and lo in by_grade:
            d = cohens_d(by_grade[lo], by_grade[hi])
            auc = mann_whitney_auc([-x for x in by_grade[hi]], [-x for x in by_grade[lo]])
            print(f"    {label:48s} Cohen's d={d:+.3f}  rank-AUC={auc:.3f}"
                  f"  (n_hi={len(by_grade[hi])}, n_lo={len(by_grade[lo])})")

    print("\n" + "=" * 84)
    print("Q2: what AUC ceiling does the raw ESCO signal imply on its own?")
    print("=" * 84)
    if 2 in by_grade and 0 in by_grade and 1 in by_grade:
        easy = mann_whitney_auc([-x for x in by_grade[2]], [-x for x in by_grade[0]])
        hard = mann_whitney_auc([-x for x in by_grade[2]], [-x for x in by_grade[1]])
        print(f"  ESCO-alone ceiling:  good_fit-vs-no_fit = {easy:.4f}   "
              f"good_fit-vs-potential_fit = {hard:.4f}")
        print(f"\n  For comparison, the TREC-CT result (scripts/mesh_signal_diagnosis.py):")
        print(f"  MeSH-alone ceiling:  g2-vs-g0 = 0.7647   g2-vs-g1 (eligibility) = 0.5256")
        print()
        if hard < 0.55:
            print(f"  -> SAME PATTERN as trials: ESCO ceiling on the hard contrast ({hard:.3f})")
            print(f"     is near chance, just as MeSH's was on eligibility (0.526).")
            print(f"     One mechanism explains both nulls: the ontology has no signal on")
            print(f"     the contrast each task actually grades.")
        else:
            print(f"  -> DIFFERENT from trials: ESCO's ceiling on the hard contrast ({hard:.3f})")
            print(f"     is well above chance. The null on career therefore needs a")
            print(f"     different explanation than 'no ontology signal' -- e.g. the")
            print(f"     measured signal not reaching training (absolute-tier bucket")
            print(f"     empty, or the training pipeline diluting it).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
