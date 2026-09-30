#!/usr/bin/env python3
"""Within-trials replication of the career ESCO-vs-ISCO finding: does a
HIERARCHY carry less usable signal than a SET/GRAPH view of the same ontology?

Motivation
----------
On career, the two ontology views split cleanly on the hard contrast:

    ESCO skill graph (set/graph view)   hard-contrast rank-AUC 0.5581
    ISCO occupation codes (hierarchy)   hard-contrast rank-AUC 0.4851  <- below chance

That is a single observation on a single dataset, and ISCO has a known
confound: a fixed-width 4-digit band structure, where "one digit differs" means
something different at each level. So the career result could be about ISCO's
particular encoding rather than about hierarchies in general.

MeSH lets us separate those two explanations WITHIN the trials dataset, with no
cross-dataset comparison (which this project does not do). MeSH carries both
views over the SAME descriptors and the SAME labels:

    set view        ontology_set_similarity -- the matcher used in training
    hierarchy view  tree numbers (C18.452.394), with a DEPTH-NORMALISED
                    Wu-Palmer distance that does not have ISCO's fixed-width
                    problem, plus raw LCA hops and the coarse top-level branch
                    overlap that training's branch_distance uses

If the hierarchy view underperforms the set view here too, the career
ESCO-vs-ISCO gap is not an ISCO encoding artefact. If it matches or beats the
set view, then it is: depth-normalisation fixes it, and ISCO should be re-encoded
rather than dropped.

Held-out data
-------------
Defaults to the 2022 TEST split, never the validation split. Validation is the
checkpoint-selection set (trainer.py:503 picks best_checkpoint.pt by lowest
validation loss), and evaluating on it was measured to inflate results by
+0.0122 to +0.0208 AUC elsewhere in this project. The ceiling measured here is
label-vs-ontology only -- no model, no checkpoint -- so selection bias cannot
enter mechanically, but using test keeps every reported number on the same
footing.

Contrasts (parallel to mesh_signal_diagnosis.py)
    easy  g2 vs g0   topical relevance
    hard  g2 vs g1   eligibility: eligible vs "relevant but not eligible",
                     the contrast negative selection actually has to get right

Usage
    .venv/bin/python3 scripts/mesh_hierarchy_vs_set_diagnosis.py
    .venv/bin/python3 scripts/mesh_hierarchy_vs_set_diagnosis.py --split preprocess/trec_ct_splits/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = st.variance(a), st.variance(b)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    if pooled == 0:
        return float("nan")
    return (st.fmean(a) - st.fmean(b)) / pooled


def rank_auc(pos, neg):
    """P(pos > neg), ties split. Feed SIMILARITIES so >0.5 always means signal."""
    if not pos or not neg:
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
    np_, nn = len(pos), len(neg)
    return (rsum - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def boot_ci(pos, neg, rng, n_boot=400, alpha=0.05):
    if not pos or not neg:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        p = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        q = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        vals.append(rank_auc(p, q))
    vals.sort()
    return (vals[int(alpha / 2 * len(vals))],
            vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))])


def tie_fraction(vals):
    if not vals:
        return float("nan")
    c = defaultdict(int)
    for v in vals:
        c[round(v, 6)] += 1
    return max(c.values()) / len(vals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path,
                    default=ROOT / "preprocess/trec_ct_splits/test.jsonl")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "results/label_budget/low_ontology_s42/training_config.json")
    ap.add_argument("--anchors", type=int, default=250)
    ap.add_argument("--per-anchor-candidates", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    # accept both absolute and repo-relative --split/--config
    args.split = args.split if args.split.is_absolute() else (ROOT / args.split)
    args.config = args.config if args.config.is_absolute() else (ROOT / args.config)

    from contrastive_learning.data_structures import TrainingConfig
    from trials_domain.mesh_ontology import tree_path_hops, tree_path_wu_palmer
    from trials_domain.run_config import build_mesh_matcher

    config = TrainingConfig.from_json(str(args.config))
    matcher = build_mesh_matcher(config)
    index = matcher.index
    rng = random.Random(args.seed)

    records = [json.loads(l) for l in open(args.split) if l.strip()]
    by_topic = defaultdict(list)
    for r in records:
        t = (r.get("metadata") or {}).get("topic_id") or (r.get("resume") or {}).get("topic_id")
        by_topic[str(t)].append(r)

    print("=" * 88)
    print("MeSH: hierarchy view vs set view, same descriptors, same labels, one dataset")
    print("=" * 88)
    print(f"split   : {args.split.relative_to(ROOT)}  ({len(records)} pairs, {len(by_topic)} topics)")
    print(f"held-out: {'YES (test)' if 'test' in args.split.name else 'NO -- this is a selection set'}")

    # ---------------------------------------------------------------- measures
    def m_set_sim(a_uris, b_uris):
        """Set view: exactly what training consumes."""
        try:
            return float(matcher.ontology_set_similarity(a_uris, b_uris))
        except Exception:
            return None

    def m_branch_sim(a_uris, b_uris):
        """Coarse hierarchy: top-level branch overlap (training's branch_distance)."""
        try:
            return 1.0 - float(matcher.branch_distance(a_uris, b_uris))
        except Exception:
            return None

    def _trees(uris):
        out = []
        for u in uris:
            tr = index.trees(u)
            if tr:
                out.append(tr)
        return out

    def m_wu_palmer_sim(a_uris, b_uris):
        """Depth-normalised hierarchy, best-match-average over descriptor pairs.

        This is the measure that removes ISCO's fixed-width-band confound: a
        shared prefix counts relative to path depth, not as a fixed digit count.
        """
        ta, tb = _trees(a_uris), _trees(b_uris)
        if not ta or not tb:
            return None
        rows = []
        for pa in ta:
            best = 0.0
            for pb in tb:
                for x in pa:
                    for y in pb:
                        s = 1.0 - tree_path_wu_palmer(x, y)
                        if s > best:
                            best = s
            rows.append(best)
        for pb in tb:
            best = 0.0
            for pa in ta:
                for y in pb:
                    for x in pa:
                        s = 1.0 - tree_path_wu_palmer(x, y)
                        if s > best:
                            best = s
            rows.append(best)
        return st.fmean(rows)

    def m_hops_sim(a_uris, b_uris):
        """Raw LCA hop count, best-match-average. Not depth-normalised, so this
        is the closest analogue of what ISCO digit-band distance does."""
        ta, tb = _trees(a_uris), _trees(b_uris)
        if not ta or not tb:
            return None
        MAXH = 26.0
        rows = []
        for pa in ta:
            best = 0.0
            for pb in tb:
                for x in pa:
                    for y in pb:
                        h = tree_path_hops(x, y)
                        s = 0.0 if h is None else 1.0 - min(h, MAXH) / MAXH
                        if s > best:
                            best = s
            rows.append(best)
        return st.fmean(rows)

    measures = [
        ("set: ontology_set_similarity", "set", m_set_sim),
        ("hier: Wu-Palmer (depth-norm)", "hierarchy", m_wu_palmer_sim),
        ("hier: LCA hops (raw)", "hierarchy", m_hops_sim),
        ("hier: top-level branch overlap", "hierarchy", m_branch_sim),
    ]

    by_measure = {name: defaultdict(list) for name, _, _ in measures}
    cov = {"pairs_seen": 0, "no_anchor_uris": 0, "no_cand_uris": 0, "scored": 0}

    topics = list(by_topic)
    rng.shuffle(topics)
    for topic in topics[: args.anchors]:
        rows = by_topic[topic]
        a_uris = (rows[0].get("resume") or {}).get("skill_uris") or []
        cov["pairs_seen"] += len(rows)
        if not a_uris:
            cov["no_anchor_uris"] += len(rows)
            continue
        sample = rng.sample(rows, min(args.per_anchor_candidates, len(rows)))
        for row in sample:
            job = row.get("job") or {}
            b_uris = job.get("skill_uris") or []
            grade = job.get("grade")
            if grade is None:
                continue
            if not b_uris:
                cov["no_cand_uris"] += 1
                continue
            g = int(grade)
            any_scored = False
            for name, _, fn in measures:
                v = fn(a_uris, b_uris)
                if v is not None and not math.isnan(v):
                    by_measure[name][g].append(v)
                    any_scored = True
            if any_scored:
                cov["scored"] += 1

    print(f"coverage: {cov['scored']}/{cov['pairs_seen']} pairs scored  "
          f"(anchors missing URIs: {cov['no_anchor_uris']}, candidates missing URIs: {cov['no_cand_uris']})")

    results = {
        "split": str(args.split.relative_to(ROOT)),
        "held_out": "test" in args.split.name,
        "coverage": cov,
        "seed": args.seed,
        "measures": {},
    }

    contrasts = [("easy  g2 vs g0", 2, 0), ("hard  g2 vs g1", 2, 1), ("      g1 vs g0", 1, 0)]

    for name, kind, _ in measures:
        bg = by_measure[name]
        if not bg:
            continue
        print()
        print("-" * 88)
        print(f"{name}    [{kind} view]")
        print("-" * 88)
        allv = [v for vs in bg.values() for v in vs]
        for g in sorted(bg):
            v = bg[g]
            print(f"  grade {g}: n={len(v):5d}  mean={st.fmean(v):.4f}  "
                  f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  median={st.median(v):.4f}")
        tf = tie_fraction(allv)
        print(f"  most common single value covers {tf * 100:.1f}% of pairs"
              + ("   <- coarse/degenerate, little to rank on" if tf > 0.5 else ""))
        mres = {"kind": kind, "tie_fraction": tf,
                "groups": {str(g): {"n": len(v), "mean": st.fmean(v),
                                    "sd": st.stdev(v) if len(v) > 1 else 0.0,
                                    "median": st.median(v)} for g, v in sorted(bg.items())},
                "contrasts": {}}
        for label, hi, lo in contrasts:
            if hi not in bg or lo not in bg:
                continue
            auc = rank_auc(bg[hi], bg[lo])
            d = cohens_d(bg[hi], bg[lo])
            ci = boot_ci(bg[hi], bg[lo], rng)
            key = label.strip().split()[0] if label.strip().split()[0] in ("easy", "hard") else "g1_vs_g0"
            mres["contrasts"][key] = {"rank_auc": auc, "cohens_d": d, "auc_ci95": list(ci),
                                      "n_pos": len(bg[hi]), "n_neg": len(bg[lo])}
            flag = ""
            if not math.isnan(ci[0]) and ci[0] <= 0.5 <= ci[1]:
                flag = "   (CI spans 0.5 -- indistinguishable from chance)"
            print(f"    {label:16s} rank-AUC={auc:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}{flag}")
        results["measures"][name] = mres

    # ------------------------------------------------------------ verdict
    print()
    print("=" * 88)
    print("hierarchy vs set, hard contrast (g2 vs g1), within trials")
    print("=" * 88)
    set_hard = results["measures"].get("set: ontology_set_similarity", {}).get("contrasts", {}).get("hard", {}).get("rank_auc")
    hier = {n: r["contrasts"].get("hard", {}).get("rank_auc")
            for n, r in results["measures"].items() if r["kind"] == "hierarchy"}
    print(f"  {'view':<34} {'hard AUC':>9}")
    print(f"  {'-'*34} {'-'*9}")
    if set_hard is not None:
        print(f"  {'set: ontology_set_similarity':<34} {set_hard:>9.4f}")
    for n, v in hier.items():
        if v is not None:
            print(f"  {n:<34} {v:>9.4f}")

    best_hier = max((v for v in hier.values() if v is not None), default=None)
    if set_hard is not None and best_hier is not None:
        gap = set_hard - best_hier
        print()
        print(f"  set - best hierarchy = {gap:+.4f}")
        print()
        print("  career reference (same style of contrast, its own dataset/label):")
        print("    ESCO skill graph (set)      0.5581")
        print("    ISCO occupation (hierarchy) 0.4851     gap +0.0730")
        print()
        if gap > 0.02:
            print("  On THIS split the set view leads.")
        elif gap < -0.02:
            print("  On THIS split the depth-normalised hierarchy leads.")
        else:
            print("  On THIS split the two views are within noise of each other.")

        print()
        print("  DO NOT read a general hierarchy-vs-set conclusion off one split. Running")
        print("  this script on both trials splits gives OPPOSITE orderings:")
        print("      validation (2021 topics)  set 0.5256   best hier 0.4870   set leads")
        print("      test       (2022 topics)  set 0.6073   best hier 0.6263   hier leads")
        print("  The hard contrast has only ~220-270 grade-1 pairs per split, so its CI is")
        print("  roughly +/-0.05 and the two splits' intervals barely separate. The ordering")
        print("  is not resolvable at this sample size, and the set view's own hard AUC is")
        print("  itself unstable across splits (0.5256 vs 0.6073).")
        print()
        print("  For the ISCO question specifically, this comparison is the WRONG instrument")
        print("  and is not needed: scripts/isco_encoding_diagnosis.py settles it directly and")
        print("  analytically. ISCO is fixed-depth and single-position, so every")
        print("  prefix-length-based distance (fixed bands, Wu-Palmer, hops) is a monotone")
        print("  transform of the same integer and yields IDENTICAL rank-AUC; and the oracle")
        print("  over the five prefix buckets caps ISCO's hard contrast at 0.5192. No")
        print("  re-encoding of ISCO can help, whatever this MeSH comparison shows.")
        print()
        print("  NOTE: this is a within-dataset comparison of two VIEWS of one ontology")
        print("  against one label set. It is not a comparison against the career numbers,")
        print("  which are printed only to show the pattern being tested for replication.")

    out = args.out or (ROOT / "results" / "ontology_ceiling" / f"mesh_hier_vs_set_{args.split.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
