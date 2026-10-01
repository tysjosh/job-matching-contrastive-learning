#!/usr/bin/env python3
"""Does the signed criteria ontology feature add anything BEYOND criteria cardinality?

Context
-------
``probe_signed_criteria_ontology.py`` found signed criteria features improve the
within-topic hard contrast by +0.0201 (6/6 checkpoints). Two controls then showed the
gain is not ontological:

  * ``signed_repartition`` -- pool both criteria sections, re-split into pseudo-sections
    of the TRUE sizes -- matched it (+0.0255). Randomising which concepts are inclusion
    versus exclusion does not reduce the effect, so criterion polarity is not the
    mechanism.
  * ``probe_criteria_count_baseline.py`` -- ``-n_exc``, a patient-independent count,
    reaches 0.6056 within-topic hard AUC, above every ontology variant and above the
    trained text encoder.

``ontology_set_similarity`` is a symmetric best-match average and therefore scales with
set cardinality, so ``sim(pt, inc) - sim(pt, exc)`` is partly a restated
``n_inc - n_exc``. This script asks the remaining question directly: once the count
features are regressed out, does ANY ontological signal survive in the residual?

If the residual AUC is at chance, MeSH contributes nothing beyond criteria length for
within-topic eligibility ranking, and the confound raised against the original null is
fully closed.

Protocol
--------
The count model is fitted on VALIDATION and applied unchanged to TEST, matching the
z-normalisation and weight-selection discipline used everywhere else in this project.
Fitting it on test would let test statistics define the thing being removed.

Uncertainty resamples whole topics, per ``ONTOLOGY_ALIGNMENT_AUDIT.md``.

Usage
    .venv/bin/python3 scripts/probe_ontology_residual.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rank_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
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


def load_rows(path: Path, concepts: Dict, chars: Dict[str, int], matcher
              ) -> List[dict]:
    def set_sim(a, b) -> float:
        if not a or not b:
            return 0.0
        try:
            return float(matcher.ontology_set_similarity(list(a), list(b)))
        except Exception:
            return 0.0

    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
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
            patient = (record.get("resume") or {}).get("skill_uris") or []
            s_inc = set_sim(patient, sections["inc"])
            s_exc = set_sim(patient, sections["exc"])
            rows.append({
                "topic": str(topic), "grade": int(grade),
                "signed": s_inc - s_exc,
                "n_inc": len(sections["inc"]), "n_exc": len(sections["exc"]),
                "chars": chars.get(nct, 0),
            })
    return rows


def design(rows: List[dict]) -> np.ndarray:
    return np.column_stack([
        np.ones(len(rows)),
        [r["n_inc"] for r in rows],
        [r["n_exc"] for r in rows],
        [r["chars"] for r in rows],
    ])


def within_topic(rows: List[dict], values: np.ndarray) -> Tuple[float, List[float]]:
    by_topic = defaultdict(list)
    for row, value in zip(rows, values):
        by_topic[row["topic"]].append((row["grade"], value))
    per = []
    for entries in by_topic.values():
        pos = [v for g, v in entries if g == 2]
        neg = [v for g, v in entries if g == 1]
        if pos and neg:
            per.append(rank_auc(pos, neg))
    return float(np.nanmean(per)), per


def bootstrap(per_topic: List[float], replicates: int, seed: int) -> Tuple[float, float]:
    vals = np.asarray([v for v in per_topic if not np.isnan(v)], dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
             for _ in range(replicates)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ontology_ceiling"
                    / "ontology_residual.json")
    args = ap.parse_args(argv)

    from contrastive_learning.data_structures import TrainingConfig
    from trials_domain.run_config import build_mesh_matcher

    concepts = json.loads((ROOT / "embedding_cache"
                           / "trials_criteria_concepts_fullcriteria.json").read_text())
    chars: Dict[str, int] = {}
    with open(ROOT / "preprocess" / "trec_ct" / "trials.jsonl", "r",
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                chars[record["nct_id"]] = len(
                    (record.get("eligibility") or {}).get("criteria") or "")

    config = TrainingConfig.from_json(ROOT / "config" / "lc_trials_ontneg_only.json")
    matcher = build_mesh_matcher(config)

    split_dir = ROOT / "preprocess" / "trec_ct_splits"
    print("scoring validation ...", flush=True)
    val = load_rows(split_dir / "validation.jsonl", concepts, chars, matcher)
    print("scoring test ...", flush=True)
    test = load_rows(split_dir / "test.jsonl", concepts, chars, matcher)
    print(f"validation {len(val)} pairs, test {len(test)} pairs", flush=True)

    # count model fitted on validation ONLY
    x_val, y_val = design(val), np.asarray([r["signed"] for r in val])
    beta, *_ = np.linalg.lstsq(x_val, y_val, rcond=None)
    r2 = 1.0 - float(np.var(y_val - x_val @ beta) / np.var(y_val))

    y_test = np.asarray([r["signed"] for r in test])
    residual = y_test - design(test) @ beta

    raw_auc, _ = within_topic(test, y_test)
    res_auc, res_per = within_topic(test, residual)
    lo, hi = bootstrap(res_per, args.replicates, args.seed)

    print()
    print(f"count model (n_inc, n_exc, criteria_chars) fitted on validation, "
          f"R^2 = {r2:.4f}")
    print()
    print("within-topic macro hard AUC on test")
    print(f"  signed, raw                                   {raw_auc:.4f}")
    print(f"  signed, residualised on the count features    {res_auc:.4f}"
          f"   95% topic CI [{lo:.4f}, {hi:.4f}]")
    print()
    print("  chance 0.5000    -n_exc alone 0.6056    text encoder 0.5683")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "count_model_r2_validation": r2,
        "count_model_beta": beta.tolist(),
        "within_hard_signed_raw": raw_auc,
        "within_hard_signed_residual": res_auc,
        "residual_ci": [lo, hi],
        "topics": len(res_per),
        "replicates": args.replicates,
        "seed": args.seed,
    }, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
