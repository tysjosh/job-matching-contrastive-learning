#!/usr/bin/env python3
"""Does combining ESCO skill + ISCO occupation beat either alone on career?

Measured separately: ESCO skill ceiling on the hard contrast (good_fit vs
potential_fit) = 0.558; ISCO occupation ceiling on the same contrast = 0.503
(chance). This measures the hybrid BatchProcessor._select_ontology_negatives
actually computes:

    distance = (1 - isco_weight) * skill_distance + isco_weight * isco_distance

using the exact formula and the config's own isco_weight (default 0.4), rather
than assuming the combination helps or hurts. Two ways it could go:
  * ISCO's chance-level noise, mixed in at weight w, dilutes the skill signal by
    roughly (1-w) -- an uninformative average that predicts a WORSE ceiling than
    ESCO alone.
  * The two signals are weakly anti-correlated or capture different failure
    cases, in which case the blend could do no worse, or slightly better, than
    either alone. Only measurement settles which.
"""

from __future__ import annotations

import argparse
import csv
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
    isco_a, isco_b = occ_to_isco.get(a, ""), occ_to_isco.get(b, "")
    if not isco_a or not isco_b:
        return None
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
                    default=ROOT / "preprocess/data_splits_v7/train.jsonl")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config/lc_career_ontfull.json")
    ap.add_argument("--occupations-csv", type=Path,
                    default=ROOT / "dataset/esco/occupations_en.csv")
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    help="isco_weight values to sweep; 0.4 is the training default")
    args = ap.parse_args(argv)

    from contrastive_learning.data_structures import TrainingConfig
    from contrastive_learning.ontology_skill_matcher import OntologySkillMatcher

    config = TrainingConfig.from_json(str(args.config))
    kg_path = getattr(config, "esco_kg_path", None) or config.esco_graph_path
    matcher = OntologySkillMatcher(kg_path)
    configured_w = getattr(config, "isco_weight", 0.4)

    occ_to_isco: Dict[str, str] = {}
    with open(args.occupations_csv, "r") as f:
        for row in csv.DictReader(f):
            uri, isco = row.get("conceptUri", ""), row.get("iscoGroup", "")
            if uri and isco:
                occ_to_isco[uri] = isco

    records = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.sample and len(records) > args.sample:
        import random
        records = random.Random(42).sample(records, args.sample)

    # Same anchor-occupation proxy as scripts/isco_signal_diagnosis.py: training
    # uses the paired positive JOB's occupation_uri as the anchor's occupation
    # (resume.occupation_uri is None throughout). Excludes each resume's own
    # good_fit source record from scoring to avoid a trivial self-match.
    good_fit_occ_by_resume: Dict[str, str] = {}
    good_fit_source_ids: set = set()
    for idx, r in enumerate(records):
        if (r.get("metadata") or {}).get("original_label") == "good_fit":
            key = json.dumps(r.get("resume") or {}, sort_keys=True)
            occ = (r.get("job") or {}).get("occupation_uri", "")
            if occ and key not in good_fit_occ_by_resume:
                good_fit_occ_by_resume[key] = occ
                good_fit_source_ids.add(idx)

    # Per-record raw components, computed ONCE.
    rows = []
    coverage = {"total": len(records), "no_skill": 0, "no_isco": 0, "usable": 0}
    for idx, r in enumerate(records):
        if idx in good_fit_source_ids:
            continue
        label = (r.get("metadata") or {}).get("original_label")
        grade = LABEL_ORDER.get(label)
        if grade is None:
            continue
        resume_uris = (r.get("resume") or {}).get("skill_uris", [])
        job_uris = (r.get("job") or {}).get("skill_uris", [])
        skill_d = None
        if resume_uris and job_uris:
            try:
                skill_d = 1.0 - float(matcher.ontology_set_similarity(resume_uris, job_uris))
            except Exception:
                skill_d = None
        key = json.dumps(r.get("resume") or {}, sort_keys=True)
        anchor_occ = good_fit_occ_by_resume.get(key, "")
        job_occ = (r.get("job") or {}).get("occupation_uri", "")
        isco_d = isco_distance(anchor_occ, job_occ, occ_to_isco) if anchor_occ and job_occ else None

        if skill_d is None:
            coverage["no_skill"] += 1
        if isco_d is None:
            coverage["no_isco"] += 1
        # A record needs skill_d to be usable AT ALL (training's own gate:
        # _select_ontology_negatives only runs when resume_uris is non-empty;
        # ISCO blending only applies on top of that, never replaces it).
        if skill_d is not None:
            coverage["usable"] += 1
            rows.append((grade, skill_d, isco_d))

    print(f"coverage: {coverage['usable']}/{coverage['total']} records have a skill "
          f"distance (the training-required signal); of those, "
          f"{sum(1 for _, _, i in rows if i is not None)} also have ISCO coverage")
    print(f"configured isco_weight in {args.config.name}: {configured_w}")

    print("\n" + "=" * 92)
    print("HYBRID CEILING vs isco_weight  (distance = (1-w)*skill + w*isco; "
          "records lacking ISCO fall back to skill-only, matching training's own fallback)")
    print("=" * 92)
    print(f"{'isco_weight':>12s} {'easy (g2vg0)':>14s} {'hard (g2vg1)':>14s} {'n_g2':>6s} {'n_g1':>6s} {'n_g0':>6s}")
    print("-" * 92)

    best = {"easy": (None, -1), "hard": (None, -1)}
    for w in args.weights:
        by_grade: Dict[int, List[float]] = defaultdict(list)
        for grade, skill_d, isco_d in rows:
            if isco_d is None:
                d = skill_d  # training's fallback: no ISCO signal, use skill only
            else:
                d = (1.0 - w) * skill_d + w * isco_d
            by_grade[grade].append(d)
        easy = mann_whitney_auc([-x for x in by_grade.get(2, [])], [-x for x in by_grade.get(0, [])])
        hard = mann_whitney_auc([-x for x in by_grade.get(2, [])], [-x for x in by_grade.get(1, [])])
        marker = "  <- config default" if abs(w - configured_w) < 1e-9 else ""
        print(f"{w:12.2f} {easy:14.4f} {hard:14.4f} "
              f"{len(by_grade.get(2,[])):6d} {len(by_grade.get(1,[])):6d} {len(by_grade.get(0,[])):6d}{marker}")
        if easy and easy > best["easy"][1]:
            best["easy"] = (w, easy)
        if hard and hard > best["hard"][1]:
            best["hard"] = (w, hard)

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    w0_easy = mann_whitney_auc([-x for x in [r[1] for r in rows if r[0]==2]],
                               [-x for x in [r[1] for r in rows if r[0]==0]])
    w0_hard = mann_whitney_auc([-x for x in [r[1] for r in rows if r[0]==2]],
                               [-x for x in [r[1] for r in rows if r[0]==1]])
    print(f"  ESCO skill alone   (w=0.0):  easy={w0_easy:.4f}  hard={w0_hard:.4f}")
    print(f"  best hybrid, easy:  w={best['easy'][0]:.2f} -> {best['easy'][1]:.4f}  "
          f"(delta vs skill-alone: {best['easy'][1]-w0_easy:+.4f})")
    print(f"  best hybrid, hard:  w={best['hard'][0]:.2f} -> {best['hard'][1]:.4f}  "
          f"(delta vs skill-alone: {best['hard'][1]-w0_hard:+.4f})")
    at_default_easy = None
    at_default_hard = None
    for w in args.weights:
        if abs(w - configured_w) < 1e-9:
            by_grade = defaultdict(list)
            for grade, skill_d, isco_d in rows:
                d = skill_d if isco_d is None else (1 - w) * skill_d + w * isco_d
                by_grade[grade].append(d)
            at_default_easy = mann_whitney_auc([-x for x in by_grade[2]], [-x for x in by_grade[0]])
            at_default_hard = mann_whitney_auc([-x for x in by_grade[2]], [-x for x in by_grade[1]])
    if at_default_hard is not None:
        print(f"\n  AT THE CONFIG DEFAULT (w={configured_w}), used by lc_career_ontfull.json:")
        print(f"    easy={at_default_easy:.4f} (skill-alone {w0_easy:.4f}, delta {at_default_easy-w0_easy:+.4f})")
        print(f"    hard={at_default_hard:.4f} (skill-alone {w0_hard:.4f}, delta {at_default_hard-w0_hard:+.4f})")
        if at_default_hard < w0_hard:
            print(f"    -> the configured blend DILUTES the hard-contrast signal versus ESCO alone.")
            print(f"       Consistent with ISCO's near-chance ceiling (0.503) pulling the average down.")
        else:
            print(f"    -> the configured blend does not measurably hurt the hard-contrast ceiling.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
