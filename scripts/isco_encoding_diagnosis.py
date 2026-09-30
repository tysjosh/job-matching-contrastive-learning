#!/usr/bin/env python3
"""Is ISCO's 0.485 hard-contrast ceiling a property of the HIERARCHY, or of the
fixed-band ENCODING that batch_processor.py uses to read it?

Background
----------
BatchProcessor._isco_distance (batch_processor.py:1100) maps occupation pairs to
five fixed values:

    same 4-digit -> 0.0    3-digit -> 0.2    2-digit -> 0.4
    1-digit      -> 0.7    different -> 1.0

The trials diagnostic (scripts/mesh_hierarchy_vs_set_diagnosis.py) found that on
MeSH, a depth-normalised Wu-Palmer hierarchy distance (hard AUC 0.6263) clearly
beat a raw hop-count distance (0.5615) over the SAME tree. That raised an
obvious question for ISCO, whose fixed bands resemble the raw hop count.

An analytical result settles half of it before any measurement
--------------------------------------------------------------
Rank-AUC depends only on the ORDERING of scores, so it is invariant under any
strictly monotone transformation of the score. Every ISCO measure that is a
function of "number of shared leading digits" alone -- the fixed bands, a
Wu-Palmer normalisation (1 - shared/4), a linear 1 - shared/4, a hop count -- is
monotone in that one integer. They are therefore all monotone transforms of each
other and MUST produce identical rank-AUC.

So re-encoding ISCO with Wu-Palmer cannot move 0.485 by even one digit. The
MeSH gain did not come from normalisation per se: MeSH descriptors sit at
VARIABLE depths and in MULTIPLE tree positions (polyhierarchy), so normalising
changes the ordering there. ISCO gives every occupation exactly one code at
exactly one depth, so there is no ordering left to change.

This script verifies that invariance empirically (as a check on the reasoning),
then tests the one family of encodings that CAN change the ordering.

What can actually change the ordering
-------------------------------------
A measure that distinguishes pairs sharing the same NUMBER of digits by WHICH
digits they share. Sharing "2" (Professionals, a huge major group) is far less
informative than sharing a small, specific group. That is precisely the
information-content idea that GO similarity uses:

    IC(prefix) = -log( P(occupation falls under prefix) )
    Resnik     = IC(longest common prefix)
    Lin        = 2*IC(LCA) / (IC(code_a) + IC(code_b))

These are NOT functions of shared-digit-count alone, so they can reorder pairs
and can in principle beat the fixed bands.

Also reported: the ORACLE ceiling for any prefix-length-only measure. Given the
per-grade counts in each of the five buckets, the best achievable AUC is
obtained by ordering buckets by P(good_fit | bucket). No fixed-band retuning,
no normalisation, no monotone rescaling can exceed that number -- it is the hard
ceiling on ISCO-as-digit-prefix, and worth knowing before anyone tries to tune
the five constants.

Usage
    .venv/bin/python3 scripts/isco_encoding_diagnosis.py
    .venv/bin/python3 scripts/isco_encoding_diagnosis.py --ic-source esco
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABEL_ORDER = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}
LABEL_NAME = {2: "good_fit", 1: "potential_fit", 0: "no_fit"}


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
    """P(pos > neg), ties split. Feed SIMILARITIES: >0.5 means signal."""
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


def boot_ci(pos, neg, rng, n_boot=500, alpha=0.05):
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


def shared_digits(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "preprocess/data_splits_v7/train_with_resume_occ.jsonl")
    ap.add_argument("--occupations-csv", type=Path,
                    default=ROOT / "dataset/esco/occupations_en.csv")
    ap.add_argument("--ic-source", choices=["data", "esco"], default="data",
                    help="frequency base for information content: the empirical "
                         "occupation distribution in the dataset (data), or the "
                         "full ESCO occupation list (esco)")
    ap.add_argument("--sample", type=int, default=0, help="0 = use all records")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)

    occ_to_isco = {}
    with open(args.occupations_csv) as f:
        for row in csv.DictReader(f):
            uri, isco = row.get("conceptUri", ""), row.get("iscoGroup", "")
            if uri and isco:
                occ_to_isco[uri] = isco

    records = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.sample and len(records) > args.sample:
        records = rng.sample(records, args.sample)

    print("=" * 92)
    print("ISCO: is the 0.485 ceiling the hierarchy, or the fixed-band encoding?")
    print("=" * 92)
    print(f"dataset : {args.dataset.relative_to(ROOT)}  ({len(records)} records)")
    print(f"ISCO map: {len(occ_to_isco)} occupation URIs -> 4-digit groups")

    # ---------------------------------------------------------- pair extraction
    has_real = any((r.get("resume") or {}).get("occupation_uri") for r in records)
    print(f"resume.occupation_uri present: {has_real}")
    if not has_real:
        sys.exit("this diagnostic needs resume-side occupation_uri; "
                 "run scripts/add_occupation_uri.py first")

    pairs = []  # (isco_a, isco_b, grade)
    cov = defaultdict(int)
    for r in records:
        a = (r.get("resume") or {}).get("occupation_uri", "")
        b = (r.get("job") or {}).get("occupation_uri", "")
        grade = LABEL_ORDER.get((r.get("metadata") or {}).get("original_label"))
        if grade is None:
            cov["no_grade"] += 1
            continue
        if not a:
            cov["no_resume_occ"] += 1
            continue
        if not b:
            cov["no_job_occ"] += 1
            continue
        ia, ib = occ_to_isco.get(a, ""), occ_to_isco.get(b, "")
        if not ia or not ib:
            cov["unmapped"] += 1
            continue
        pairs.append((ia, ib, grade))
        cov["scored"] += 1
    print(f"coverage: scored={cov['scored']}  no_resume_occ={cov['no_resume_occ']}  "
          f"no_job_occ={cov['no_job_occ']}  unmapped={cov['unmapped']}  no_grade={cov['no_grade']}")
    if cov["scored"] < 100:
        sys.exit("too few scored pairs")

    # ------------------------------------------------------------- IC over prefixes
    if args.ic_source == "esco":
        codes = list(occ_to_isco.values())
    else:
        codes = [c for a, b, _ in pairs for c in (a, b)]
    total = len(codes)
    prefix_count = Counter()
    for c in codes:
        for k in range(1, len(c) + 1):
            prefix_count[c[:k]] += 1
    ic = {p: -math.log(n / total) for p, n in prefix_count.items()}
    max_ic = max(ic.values()) if ic else 1.0
    print(f"IC base : {args.ic_source}  ({total} occupation mentions, "
          f"{len(prefix_count)} distinct prefixes, max IC={max_ic:.3f})")
    depth_ic = defaultdict(list)
    for p, v in ic.items():
        depth_ic[len(p)].append(v)
    print("          mean IC by prefix depth: " + "  ".join(
        f"{d}:{st.fmean(v):.2f}" for d, v in sorted(depth_ic.items())))

    # ------------------------------------------------------------------ measures
    def m_fixed_bands(a, b):
        """Exactly BatchProcessor._isco_distance, returned as a similarity."""
        if a == b:
            d = 0.0
        elif len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3]:
            d = 0.2
        elif len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]:
            d = 0.4
        elif a[:1] == b[:1]:
            d = 0.7
        else:
            d = 1.0
        return 1.0 - d

    def m_wu_palmer(a, b):
        """Depth-normalised. ISCO is fixed-depth-4, so this is 1 - shared/4."""
        s = shared_digits(a, b)
        return (2.0 * s) / (len(a) + len(b)) if (len(a) + len(b)) else 0.0

    def m_linear(a, b):
        return shared_digits(a, b) / 4.0

    def m_resnik(a, b):
        s = shared_digits(a, b)
        if s == 0:
            return 0.0
        return ic.get(a[:s], 0.0) / max_ic

    def m_lin(a, b):
        s = shared_digits(a, b)
        if s == 0:
            return 0.0
        num = 2.0 * ic.get(a[:s], 0.0)
        den = ic.get(a, 0.0) + ic.get(b, 0.0)
        return num / den if den > 0 else 0.0

    measures = [
        ("fixed bands (training code)", "prefix-length only", m_fixed_bands),
        ("Wu-Palmer (depth-normalised)", "prefix-length only", m_wu_palmer),
        ("linear shared/4", "prefix-length only", m_linear),
        ("Resnik IC(LCA prefix)", "IC-weighted", m_resnik),
        ("Lin IC-normalised", "IC-weighted", m_lin),
    ]

    by_measure = {name: defaultdict(list) for name, _, _ in measures}
    for ia, ib, g in pairs:
        for name, _, fn in measures:
            by_measure[name][g].append(fn(ia, ib))

    contrasts = [("easy  g2 vs g0", 2, 0), ("hard  g2 vs g1", 2, 1), ("      g1 vs g0", 1, 0)]
    results = {"dataset": str(args.dataset.relative_to(ROOT)), "coverage": dict(cov),
               "ic_source": args.ic_source, "measures": {}}

    for name, kind, _ in measures:
        bg = by_measure[name]
        print()
        print("-" * 92)
        print(f"{name}   [{kind}]")
        print("-" * 92)
        allv = [v for vs in bg.values() for v in vs]
        n_distinct = len(set(round(v, 9) for v in allv))
        for g in sorted(bg, reverse=True):
            v = bg[g]
            print(f"  {LABEL_NAME[g]:14s} n={len(v):5d}  mean={st.fmean(v):.4f}  "
                  f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  median={st.median(v):.4f}")
        c = Counter(round(v, 9) for v in allv)
        print(f"  distinct values={n_distinct}   most common covers "
              f"{max(c.values()) / len(allv) * 100:.1f}% of pairs")
        mres = {"kind": kind, "n_distinct_values": n_distinct, "contrasts": {}}
        for label, hi, lo in contrasts:
            if hi not in bg or lo not in bg:
                continue
            auc = rank_auc(bg[hi], bg[lo])
            d = cohens_d(bg[hi], bg[lo])
            ci = boot_ci(bg[hi], bg[lo], rng)
            key = "easy" if "g2 vs g0" in label else ("hard" if "g2 vs g1" in label else "g1_vs_g0")
            mres["contrasts"][key] = {"rank_auc": auc, "cohens_d": d, "auc_ci95": list(ci),
                                      "n_pos": len(bg[hi]), "n_neg": len(bg[lo])}
            flag = "   (CI spans 0.5)" if (not math.isnan(ci[0]) and ci[0] <= 0.5 <= ci[1]) else ""
            print(f"    {label:16s} rank-AUC={auc:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}{flag}")
        results["measures"][name] = mres

    # ---------------------------------------------------- oracle over the 5 buckets
    print()
    print("=" * 92)
    print("ORACLE: best AUC any prefix-length-only measure can reach")
    print("=" * 92)
    bucket = defaultdict(lambda: defaultdict(int))
    for ia, ib, g in pairs:
        bucket[shared_digits(ia, ib)][g] += 1
    print(f"  {'shared digits':>13} {'n_g2':>7} {'n_g1':>7} {'n_g0':>7}   P(g2|bucket vs g1)")
    for s in sorted(bucket, reverse=True):
        b = bucket[s]
        n2, n1, n0 = b.get(2, 0), b.get(1, 0), b.get(0, 0)
        p = n2 / (n2 + n1) if (n2 + n1) else float("nan")
        print(f"  {s:>13} {n2:>7} {n1:>7} {n0:>7}   {p:.3f}")

    oracle = {}
    for cname, hi, lo in (("easy", 2, 0), ("hard", 2, 1)):
        order = sorted(bucket, key=lambda s: (bucket[s].get(hi, 0) /
                                              (bucket[s].get(hi, 0) + bucket[s].get(lo, 0))
                                              if (bucket[s].get(hi, 0) + bucket[s].get(lo, 0)) else -1))
        score = {s: i for i, s in enumerate(order)}
        pos = [score[shared_digits(a, b)] for a, b, g in pairs if g == hi]
        neg = [score[shared_digits(a, b)] for a, b, g in pairs if g == lo]
        auc = rank_auc(pos, neg)
        oracle[cname] = auc
        print(f"  oracle {cname:5s} rank-AUC = {auc:.4f}   "
              f"(optimal bucket order, best->worst: {list(reversed(order))})")
    results["oracle_prefix_length_only"] = oracle

    # ----------------------------------------------------------------- verdict
    print()
    print("=" * 92)
    print("verdict")
    print("=" * 92)
    fb = results["measures"]["fixed bands (training code)"]["contrasts"]
    wp = results["measures"]["Wu-Palmer (depth-normalised)"]["contrasts"]
    rk = results["measures"]["Resnik IC(LCA prefix)"]["contrasts"]
    ln = results["measures"]["Lin IC-normalised"]["contrasts"]

    same = all(abs(fb[k]["rank_auc"] - wp[k]["rank_auc"]) < 1e-9 for k in fb if k in wp)
    print(f"  1. monotone-invariance check: fixed bands vs Wu-Palmer identical on every "
          f"contrast = {same}")
    if same:
        print("     Confirms the analytical argument. Re-encoding ISCO with a depth-normalised")
        print("     distance is mathematically incapable of changing its AUC, because ISCO is")
        print("     fixed-depth and single-position: every such measure is a monotone function")
        print("     of shared-digit-count. The MeSH Wu-Palmer gain came from MeSH's variable")
        print("     depth and polyhierarchy, neither of which ISCO has.")
    else:
        print("     UNEXPECTED: they differ, so some pairs have unequal code lengths.")

    best_ic_hard = max(rk["hard"]["rank_auc"], ln["hard"]["rank_auc"])
    print(f"\n  2. IC weighting, hard contrast: fixed bands {fb['hard']['rank_auc']:.4f} "
          f"-> best IC {best_ic_hard:.4f} ({best_ic_hard - fb['hard']['rank_auc']:+.4f})")
    print(f"     oracle ceiling for ANY prefix-length-only measure: {oracle['hard']:.4f}")
    if best_ic_hard > oracle["hard"] + 0.005:
        print("     IC weighting EXCEEDS the prefix-length oracle, so it is extracting real")
        print("     information that the five fixed bands structurally cannot represent.")
    elif best_ic_hard > fb["hard"]["rank_auc"] + 0.005:
        print("     IC weighting helps but stays under the prefix-length oracle: most of the")
        print("     gain is available by simply reordering/retuning the five bands.")
    else:
        print("     IC weighting does NOT help. ISCO's ceiling is a property of the hierarchy's")
        print("     information content on this label, not of how the hierarchy is encoded.")

    if oracle["hard"] < 0.55:
        print(f"\n  3. The oracle itself is only {oracle['hard']:.4f}. Even a perfectly tuned")
        print("     prefix-based ISCO distance cannot separate good_fit from potential_fit.")
        print("     That closes the encoding question: ISCO's weakness on the hard contrast is")
        print("     not a bug in _isco_distance's five constants. Occupation-group identity")
        print("     simply does not carry the good_fit/potential_fit distinction, which is")
        print("     about degree of fit WITHIN a plausible occupation match.")
        print("     Actionable: isco_weight=0.0 stands. Retuning the bands is not worth a run.")

    out = args.out or (ROOT / "results" / "ontology_ceiling" / f"isco_encoding_{args.ic_source}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
