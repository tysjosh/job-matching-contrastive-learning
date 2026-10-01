#!/usr/bin/env python3
"""Trivial patient-independent baselines for within-topic eligibility ranking.

Why this exists
---------------
``probe_signed_criteria_ontology.py`` found that signed criteria-derived MeSH features
improve the within-topic hard contrast by +0.0201 (6/6 checkpoints) where the published
score fusion moved it by roughly nothing. The ``signed_repartition`` control -- which
pools the two criteria sections and re-splits them into pseudo-sections of the true
sizes, destroying only the inclusion/exclusion assignment -- matched and slightly beat
that result (+0.0255). So the gain does not come from criterion polarity, and what
survives the repartition is the SET SIZES.

``ontology_set_similarity`` is a symmetric best-match average and therefore depends on
set cardinality, so ``sim(pt, inc) - sim(pt, exc)`` correlates with ``n_inc - n_exc``.
This script tests that directly by scoring trial features that use no patient
information, no ontology similarity and no embeddings at all.

The result: ``-n_exc`` reaches 0.6056 within-topic hard AUC, beating the trained text
encoder (0.5683) and every ontology variant. Trials with more exclusion criteria give a
patient more ways to fail, so within a patient they skew toward grade 1. That is a
trial-restrictiveness prior, not biomedical knowledge.

Any future knowledge feature on this dataset must clear these baselines before it is
reported as a knowledge result.

Usage
    .venv/bin/python3 scripts/probe_criteria_count_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rank_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Tie-corrected rank AUC. Ties matter here: these features are integer counts."""
    if not len(pos) or not len(neg):
        return float("nan")
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n = len(merged)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rsum = sum(r for r, (_v, lab) in zip(ranks, merged) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rsum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


#: Trial features. Each takes (n_inc, n_exc, criteria_chars) and returns a score.
#: None of them can see the patient.
FEATURES: Dict[str, Callable[[int, int, int], float]] = {
    "n_exc": lambda inc, exc, chars: float(exc),
    "n_inc": lambda inc, exc, chars: float(inc),
    "criteria_chars": lambda inc, exc, chars: float(chars),
    "n_inc - n_exc": lambda inc, exc, chars: float(inc - exc),
    "-criteria_chars": lambda inc, exc, chars: float(-chars),
    "-n_exc": lambda inc, exc, chars: float(-exc),
}


def topic_bootstrap(per_topic: List[float], replicates: int, seed: int
                    ) -> Tuple[float, float]:
    """Percentile interval resampling whole topics.

    Topics are the unit of independence here: the 50 test topics contribute many
    pairs each, so resampling pairs would understate uncertainty. This mirrors the
    resampling policy ``ONTOLOGY_ALIGNMENT_AUDIT.md`` requires for MeSH.
    """
    values = np.asarray([v for v in per_topic if not np.isnan(v)], dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
             for _ in range(replicates)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path,
                    default=ROOT / "preprocess" / "trec_ct_splits" / "test.jsonl")
    ap.add_argument("--concepts", type=Path,
                    default=ROOT / "embedding_cache"
                    / "trials_criteria_concepts_fullcriteria.json")
    ap.add_argument("--trials", type=Path,
                    default=ROOT / "preprocess" / "trec_ct" / "trials.jsonl")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "ontology_ceiling"
                    / "criteria_count_baseline.json")
    args = ap.parse_args(argv)

    concepts = json.loads(args.concepts.read_text())
    chars: Dict[str, int] = {}
    with open(args.trials, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                chars[record["nct_id"]] = len(
                    (record.get("eligibility") or {}).get("criteria") or "")

    by_topic: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    total = 0
    with open(args.split, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            job = record.get("job") or {}
            grade = job.get("grade")
            if grade not in (0, 1, 2):
                continue
            topic = (record.get("metadata") or {}).get("topic_id")
            if topic is None:
                continue
            nct = job.get("nct_id")
            sections = concepts.get(nct, {"inc": [], "exc": []})
            by_topic[str(topic)].append(
                (int(grade), len(sections["inc"]), len(sections["exc"]),
                 chars.get(nct, 0)))
            total += 1

    print(f"split {args.split.name}: {total} judged pairs across {len(by_topic)} topics")
    print()
    print("PATIENT-INDEPENDENT trial features")
    print("within-topic macro hard AUC (eligible=2 vs ineligible=1), "
          f"topic bootstrap {args.replicates} replicates")
    print()
    print(f"  {'feature':22}{'AUC':>9}{'95% topic interval':>26}{'topics':>9}")

    results: Dict[str, Dict[str, float]] = {}
    for name, fn in FEATURES.items():
        per_topic = []
        for rows in by_topic.values():
            pos = [fn(inc, exc, ch) for g, inc, exc, ch in rows if g == 2]
            neg = [fn(inc, exc, ch) for g, inc, exc, ch in rows if g == 1]
            if pos and neg:
                per_topic.append(rank_auc(pos, neg))
        auc = float(np.nanmean(per_topic))
        lo, hi = topic_bootstrap(per_topic, args.replicates, args.seed)
        results[name] = {"auc": auc, "ci_low": lo, "ci_high": hi,
                         "topics": len(per_topic)}
        print(f"  {name:22}{auc:>9.4f}   [{lo:.4f}, {hi:.4f}]{len(per_topic):>12}")

    print()
    print("  reference points from probe_signed_criteria_ontology.py --full-criteria:")
    print("    text encoder (w=0)        0.5683")
    print("    signed                    0.5594")
    print("    signed_repartition        0.5693")
    print("    browse                    0.5288")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "split": str(args.split),
        "pairs": total,
        "topics": len(by_topic),
        "replicates": args.replicates,
        "seed": args.seed,
        "results": results,
    }, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
