#!/usr/bin/env python3
"""Does GO semantic similarity separate high-confidence PPIs from low-confidence
ones? Same discipline as the career/trials diagnostics: measure the ontology's
OWN discriminative power against the real graded label, before any training.

Data
----
dataset/go_ppi/network_sample.tsv       -- real STRING API pulls (species=9606,
    physical network around TP53/EGFR/MYC/AKT1), graded confidence score in
    [0.161, 0.999], the label this diagnostic tests against. 43 pairs, 29 genes.
dataset/go_ppi/go_annotations_sample.tsv -- real UniProt REST pulls, GO term
    sets for the same 29 genes.

CAVEAT: this is a small, non-random plausibility probe (four seed proteins and
their immediate neighbors), not a publication-grade measurement. It answers
"does GO clear the bar worth a real bulk pull", not "what is GO's true ceiling".
A real measurement needs the full STRING human interactome bulk download and a
proper information-content-weighted GO similarity measure (Resnik/Lin), not the
raw Jaccard used here for parity with the career/trials scripts.

Grading
-------
STRING scores are continuous, not discrete like career/trials' 0/1/2. Binned into
tertiles (low/med/high confidence) so the same "hard vs easy contrast" framing
applies: high-vs-low is the coarse contrast; high-vs-med is the harder one, the
one closest to a real discrimination task (distinguishing a real high-confidence
interaction from a merely plausible one).
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "go_ppi"


def jaccard_distance(a: set, b: set) -> float:
    if not a or not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return None
    return 1.0 - inter / union


def mann_whitney_auc(pos, neg):
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
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_v, lab) in zip(ranks, merged) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = st.variance(a), st.variance(b)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    if pooled == 0:
        return float("nan")
    return (st.fmean(a) - st.fmean(b)) / pooled


# ---------------------------------------------------------------- load GO sets
network_path = DATA_DIR / (sys.argv[1] if len(sys.argv) > 1 else "network_sample.tsv")
go_path = DATA_DIR / (sys.argv[2] if len(sys.argv) > 2 else "go_annotations_sample.tsv")

go_by_gene = {}
with open(go_path) as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        terms = set(t for t in row["GO"].split(";") if t)
        go_by_gene[row["Gene"]] = terms

print(f"loaded GO annotations for {len(go_by_gene)} genes "
      f"({sum(1 for v in go_by_gene.values() if v)} with >=1 term)")

# ------------------------------------------------------------- load network
pairs = []
with open(network_path) as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        ga, gb = row["preferredName_A"], row["preferredName_B"]
        score = float(row["score"])
        pairs.append((ga, gb, score))

print(f"loaded {len(pairs)} scored protein pairs")

# ------------------------------------------------------- tertile the STRING score
scores_sorted = sorted(s for _, _, s in pairs)
n = len(scores_sorted)
t1 = scores_sorted[n // 3]
t2 = scores_sorted[(2 * n) // 3]
print(f"\nSTRING score tertile cut points: low<={t1:.3f}  med<=({t1:.3f},{t2:.3f}]  high>{t2:.3f}")


def tier(score):
    if score <= t1:
        return 0  # low confidence
    if score <= t2:
        return 1  # medium confidence
    return 2  # high confidence


# ---------------------------------------------------- score GO Jaccard per pair
by_tier = defaultdict(list)
coverage = {"total": len(pairs), "missing_go": 0, "scored": 0}
for ga, gb, score in pairs:
    sa, sb = go_by_gene.get(ga), go_by_gene.get(gb)
    if not sa or not sb:
        coverage["missing_go"] += 1
        continue
    d = jaccard_distance(sa, sb)
    if d is None:
        coverage["missing_go"] += 1
        continue
    by_tier[tier(score)].append(d)
    coverage["scored"] += 1

print(f"\ncoverage: {coverage['scored']}/{coverage['total']} pairs scored "
      f"({coverage['missing_go']} missing GO on one side)")

NAME = {2: "high-confidence", 1: "medium-confidence", 0: "low-confidence"}
print("\n" + "=" * 84)
print("Q1: does GO Jaccard distance separate STRING confidence tiers?")
print("=" * 84)
for t in sorted(by_tier, reverse=True):
    v = by_tier[t]
    print(f"  {NAME[t]:18s} n={len(v):3d}  mean_dist={st.fmean(v):.4f}  "
          f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  median={st.median(v):.4f}")

print("\n  pairwise separation of GO JACCARD DISTANCE by STRING confidence tier "
      "(lower distance = more GO overlap = expected for higher confidence):")
for hi, lo, label in ((2, 0, "high vs low ('easy')"),
                      (2, 1, "high vs medium ('hard' -- discriminating real vs plausible)"),
                      (1, 0, "medium vs low")):
    if hi in by_tier and lo in by_tier:
        d = cohens_d(by_tier[lo], by_tier[hi])
        auc = mann_whitney_auc([-x for x in by_tier[hi]], [-x for x in by_tier[lo]])
        print(f"    {label:58s} Cohen's d={d:+.3f}  rank-AUC={auc:.3f}  "
              f"(n_hi={len(by_tier[hi])}, n_lo={len(by_tier[lo])})")

print("\n" + "=" * 84)
print("Q2: ceiling comparison against the career/trials ontologies measured earlier")
print("=" * 84)
if 2 in by_tier and 1 in by_tier and 0 in by_tier:
    easy = mann_whitney_auc([-x for x in by_tier[2]], [-x for x in by_tier[0]])
    hard = mann_whitney_auc([-x for x in by_tier[2]], [-x for x in by_tier[1]])
    print(f"\n  {'ontology':12s} {'domain':8s} {'easy contrast':>16s} {'hard contrast':>16s}")
    print(f"  {'-'*12} {'-'*8} {'-'*16} {'-'*16}")
    print(f"  {'MeSH':12s} {'trials':8s} {0.7647:16.4f} {0.5256:16.4f}   <- eligibility")
    print(f"  {'ESCO skill':12s} {'career':8s} {0.6275:16.4f} {0.5581:16.4f}   <- potential_fit")
    print(f"  {'ISCO occ.':12s} {'career':8s} {0.5639:16.4f} {0.4851:16.4f}   <- potential_fit (real occ.)")
    print(f"  {'GO terms':12s} {'PPI':8s} {easy:16.4f} {hard:16.4f}   <- high-vs-medium confidence")
    print(f"\n  NOTE: n={coverage['scored']} pairs from ONE small network pull (TP53/EGFR/MYC/AKT1")
    print(f"  neighborhoods). This is a plausibility probe, not a publication-grade")
    print(f"  measurement -- the literature's validated GO-PPI correlation was measured")
    print(f"  on much larger, curated benchmarks. Treat this as 'does it clear the bar")
    print(f"  worth a real pull', not as the final number.")
    if hard > 0.60:
        print(f"\n  -> GO's hard-contrast ceiling ({hard:.3f}) is well above what MeSH or ISCO")
        print(f"     showed on their hard contrasts (0.526, 0.485). Worth a real pull.")
    elif hard > 0.55:
        print(f"\n  -> GO's hard-contrast ceiling ({hard:.3f}) is comparable to ESCO's (0.558) --")
        print(f"     weak but real, not dead like MeSH/ISCO. Marginal case for a real pull.")
    else:
        print(f"\n  -> GO's hard-contrast ceiling ({hard:.3f}) is in the same dead range as")
        print(f"     MeSH/ISCO's hard contrasts. On this small sample it does NOT clear the bar.")
