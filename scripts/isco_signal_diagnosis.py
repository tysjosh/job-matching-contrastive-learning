#!/usr/bin/env python3
"""Does ISCO's occupation hierarchy separate career grades any better than ESCO skills did?

Companion to scripts/esco_signal_diagnosis.py and scripts/mesh_signal_diagnosis.py.
ESCO skill distance came back uniformly weak (rank-AUC 0.558-0.628 across all three
contrasts). ISCO is a coarser, four-level occupational hierarchy rather than a
skill graph, so it is a genuinely different signal, not a rescaling of the same one
-- worth checking on its own rather than assuming it inherits ESCO's weakness.

Uses BatchProcessor._isco_distance's exact tiering (same 4-digit=0.0, 3-digit=0.2,
2-digit=0.4, 1-digit=0.7, different=1.0) so the measured signal matches what
training actually sees.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
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


def isco_distance(a: str, b: str, occ_to_isco: Dict[str, str]) -> Optional[float]:
    """Mirrors BatchProcessor._isco_distance exactly."""
    isco_a, isco_b = occ_to_isco.get(a, ""), occ_to_isco.get(b, "")
    if not isco_a or not isco_b:
        return None  # unlike training's 0.5 fallback, exclude so coverage is visible
    if isco_a == isco_b:
        return 0.0
    if len(isco_a) >= 3 and len(isco_b) >= 3 and isco_a[:3] == isco_b[:3]:
        return 0.2
    if len(isco_a) >= 2 and len(isco_b) >= 2 and isco_a[:2] == isco_b[:2]:
        return 0.4
    if isco_a[:1] == isco_b[:1]:
        return 0.7
    return 1.0


LABEL_ORDER = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}
LABEL_NAME = {2: "good_fit", 1: "potential_fit", 0: "no_fit"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "preprocess/data_splits_v7/train_with_resume_occ.jsonl",
                    help="use the good_fit-job-proxy path (train.jsonl) to reproduce "
                         "the earlier, pre-fix measurement")
    ap.add_argument("--occupations-csv", type=Path,
                    default=ROOT / "dataset/esco/occupations_en.csv")
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args(argv)

    occ_to_isco: Dict[str, str] = {}
    with open(args.occupations_csv, "r") as f:
        for row in csv.DictReader(f):
            uri, isco = row.get("conceptUri", ""), row.get("iscoGroup", "")
            if uri and isco:
                occ_to_isco[uri] = isco
    print(f"loaded {len(occ_to_isco)} occupation -> ISCO group mappings")

    isco_groups = Counter(occ_to_isco.values())
    print(f"distinct 4-digit ISCO groups referenced: {len(isco_groups)}")
    print(f"distinct 1-digit ISCO majors:            "
          f"{len({g[:1] for g in occ_to_isco.values() if g})}")

    records = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.sample and len(records) > args.sample:
        import random
        records = random.Random(42).sample(records, args.sample)

    print("\n" + "=" * 84)
    print("Q1: does ISCO occupation-group distance separate career grades?")
    print("=" * 84)
    by_grade: Dict[int, List[float]] = defaultdict(list)
    coverage = {"total": len(records), "missing_resume_occ": 0, "missing_job_occ": 0,
                "unmapped": 0, "scored": 0}

    # Anchor occupation source, in priority order:
    #   1. resume.occupation_uri, if the input file carries it (added by
    #      scripts/add_occupation_uri.py, which resolves resume.role/skills
    #      against the same ESCO occupation index used for jobs). This is the
    #      resume's OWN signal, not a proxy.
    #   2. Fallback: the PAIRED POSITIVE JOB's occupation_uri, exactly what
    #      training does (batch_processor.py:911) when resume-side occupation is
    #      unavailable. Guarded against self-comparison: a resume's own good_fit
    #      record would trivially score distance=0 against an occupation derived
    #      FROM that same record, so the source record is excluded from scoring.
    has_real_resume_occ = any((r.get("resume") or {}).get("occupation_uri") for r in records)
    print(f"  resume.occupation_uri present in input: {has_real_resume_occ}")

    good_fit_occ_by_resume: Dict[str, str] = {}
    good_fit_source_ids: set = set()
    if not has_real_resume_occ:
        for idx, r in enumerate(records):
            if (r.get("metadata") or {}).get("original_label") == "good_fit":
                resume_key = json.dumps(r.get("resume") or {}, sort_keys=True)
                occ = (r.get("job") or {}).get("occupation_uri", "")
                if occ and resume_key not in good_fit_occ_by_resume:
                    good_fit_occ_by_resume[resume_key] = occ
                    good_fit_source_ids.add(idx)

    for idx, r in enumerate(records):
        if idx in good_fit_source_ids:
            continue  # the record that DEFINED the anchor occupation (proxy mode only)
        resume_occ_real = (r.get("resume") or {}).get("occupation_uri", "")
        if resume_occ_real:
            anchor_occ = resume_occ_real
        else:
            resume_key = json.dumps(r.get("resume") or {}, sort_keys=True)
            anchor_occ = good_fit_occ_by_resume.get(resume_key, "")
        job_occ = (r.get("job") or {}).get("occupation_uri", "")
        label = (r.get("metadata") or {}).get("original_label")
        grade = LABEL_ORDER.get(label)
        if grade is None:
            continue
        if not anchor_occ:
            coverage["missing_resume_occ"] += 1
            continue
        if not job_occ:
            coverage["missing_job_occ"] += 1
            continue
        d = isco_distance(anchor_occ, job_occ, occ_to_isco)
        if d is None:
            coverage["unmapped"] += 1
            continue
        by_grade[grade].append(d)
        coverage["scored"] += 1

    print(f"  coverage: {coverage['scored']}/{coverage['total']} pairs scored")
    print(f"    missing resume.occupation_uri: {coverage['missing_resume_occ']}")
    print(f"    missing job.occupation_uri:    {coverage['missing_job_occ']}")
    print(f"    occupation_uri not in ISCO map: {coverage['unmapped']}")
    for g in sorted(by_grade, reverse=True):
        v = by_grade[g]
        print(f"  {LABEL_NAME[g]:15s} (grade {g}): n={len(v):5d}  mean={st.fmean(v):.4f}  "
              f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  median={st.median(v):.4f}")

    print("\n  pairwise separation of ISCO DISTANCE by grade "
          "(lower distance = same/closer occupation group):")
    for hi, lo, label in ((2, 0, "good_fit vs no_fit ('easy')"),
                          (2, 1, "good_fit vs potential_fit ('hard')"),
                          (1, 0, "potential_fit vs no_fit")):
        if hi in by_grade and lo in by_grade:
            d = cohens_d(by_grade[lo], by_grade[hi])
            auc = mann_whitney_auc([-x for x in by_grade[hi]], [-x for x in by_grade[lo]])
            print(f"    {label:32s} Cohen's d={d:+.3f}  rank-AUC={auc:.3f}"
                  f"  (n_hi={len(by_grade[hi])}, n_lo={len(by_grade[lo])})")

    print("\n" + "=" * 84)
    print("Q2: ceiling comparison across all three ontologies measured this session")
    print("=" * 84)
    if 2 in by_grade and 0 in by_grade and 1 in by_grade:
        easy = mann_whitney_auc([-x for x in by_grade[2]], [-x for x in by_grade[0]])
        hard = mann_whitney_auc([-x for x in by_grade[2]], [-x for x in by_grade[1]])
        print(f"\n  {'ontology':10s} {'domain':8s} {'easy contrast':>16s} {'hard contrast':>16s}")
        print(f"  {'-'*10} {'-'*8} {'-'*16} {'-'*16}")
        print(f"  {'MeSH':10s} {'trials':8s} {0.7647:16.4f} {0.5256:16.4f}   <- eligibility")
        print(f"  {'ESCO skill':10s} {'career':8s} {0.6275:16.4f} {0.5581:16.4f}   <- potential_fit")
        print(f"  {'ISCO occ.':10s} {'career':8s} {easy:16.4f} {hard:16.4f}   <- potential_fit")
        print()
        if hard > 0.6275:
            print(f"  -> ISCO is a STRONGER signal than ESCO skill on career's hard contrast.")
            print(f"     A hybrid or ISCO-weighted arm may outperform pure ESCO skill selection")
            print(f"     even though ESCO-skill-only and ESCO-full both came back null.")
        elif hard < 0.53:
            print(f"  -> ISCO collapses to near-chance, matching MeSH's eligibility pattern.")
            print(f"     Coarse occupational codes carry even less fine-grained signal than")
            print(f"     ESCO's skill graph, consistent with the isco_only arm's weaker (or")
            print(f"     more negative) delta already observed in the unseeded v7 sweep.")
        else:
            print(f"  -> ISCO sits in the same weak-but-nonzero range as ESCO skill.")
            print(f"     Both career ontologies are consistently weak; the finding is about")
            print(f"     career's signal strength overall, not about which ontology is used.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
