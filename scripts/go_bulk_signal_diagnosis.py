#!/usr/bin/env python3
"""
Publication-grade GO ontology-signal ceiling on the full human interactome.

This is the upgraded version of scripts/go_signal_diagnosis.py, which was a
43-pair plausibility probe using raw Jaccard over four seed proteins. Three
methodological weaknesses of that probe are fixed here.

1. SCALE.  Full STRING human network instead of four ego-networks.

2. LABEL INDEPENDENCE.  The probe scored pairs with STRING's `combined_score`,
   which fuses a `database` channel (curated pathways) and a `textmining`
   channel. Both partly derive from the same co-annotation evidence that GO
   encodes, so GO-vs-combined_score is circular. Here the label is the
   `experiments` channel alone: physical assay evidence (Y2H, affinity
   capture, co-IP). That channel is built from experimental interaction
   databases and does not consume GO.

3. SIMILARITY MEASURE.  Raw Jaccard treats "protein-containing complex" and
   "positive regulation of mitotic sister chromatid separation" as equally
   informative. Here similarity is information-content weighted:
     IC(t)   = -log( n_genes(t) / n_genes_total )   over the ancestor closure
     simGIC  = sum IC(intersection) / sum IC(union)          [Pesquita 2007]
     Resnik  = IC of most informative common ancestor, best-match-average
     Lin     = 2*IC(MICA) / (IC(t1)+IC(t2)), best-match-average
   simGIC is the primary number: it is set-based, has no best-match asymmetry,
   and is the measure that holds up best in GO similarity benchmarks.

CIRCULARITY GUARDS (the failure mode that inflates every naive GO-PPI result)
  - Aspect defaults to biological_process, so GO:0005515 "protein binding"
    (molecular_function) cannot enter. That single term is annotated straight
    from interaction assays; leaving it in would leak the label outright.
  - IPI ("inferred from physical interaction") evidence is dropped by default.
    Same leak, one level less obvious.
  - IEA (electronic annotation, uncurated) is dropped by default.
  Run with --keep-ipi / --keep-iea to see how much those guards cost; the gap
  is itself the measure of how much of a naive result is circular.

TIERING: WHY --tier-mode absolute IS THE DEFAULT
  The obvious design is to split interacting pairs into confidence tertiles and
  contrast top vs middle. On STRING that is actively misleading, and it produced
  a wrong answer here before being caught. The experimental channel is extremely
  right-skewed: of 5.85M nonzero scores, the median is 102 and the 90th
  percentile is 292, while STRING itself calls 400 "medium confidence" and 700
  "high confidence". So tertile cuts land at roughly 84 and 134 and the entire
  contrast lives inside the low-confidence band. Measured that way GO looks
  near-chance (hard-contrast AUC 0.528), but the reason is that the comparison
  is between "a little weak evidence" and "slightly more weak evidence" -- which
  tracks how much assay attention a protein pair has received, i.e. study bias,
  not whether the interaction is real. GO has no reason to predict that, and
  does not.

  Absolute mode uses STRING's own thresholds instead, which restores the
  intended semantics:
      high   experimental >= 700     established interaction
      medium experimental in [1,150) weak / ambiguous evidence
      low    experimental in [150,700) intermediate
      random non-interacting pairs
  Under absolute bands the same data gives hard-contrast AUC 0.819. The tertile
  number was measuring the wrong construct, not a weak ontology.

CONTRASTS (absolute mode; kept parallel in spirit to the career/trials
diagnostics, where easy = clear positive vs clear negative and hard = the
distinction that negative selection has to get right)
  easy = high confidence vs random non-interacting
  hard = high confidence vs weak-evidence interaction -- "established" vs
         "plausible but unconfirmed", the analogue of good_fit vs potential_fit
         and of eligible vs relevant-but-ineligible
  intermediate vs weak-evidence is also reported; GO is at chance on it, which
  localises GO's signal to the high-confidence end rather than making it a
  smooth monotone predictor of confidence.

Inputs (dataset/go_ppi/bulk/, fetched from STRING v12.0 and current GOA)
  9606.protein.links.detailed.v12.0.txt.gz   per-channel STRING scores
  9606.protein.info.v12.0.txt.gz             ENSP -> gene symbol
  goa_human.gaf.gz                           human GO annotations
  go-basic.obo                               GO DAG (is_a + part_of)

Usage
  .venv/bin/python3 scripts/go_bulk_signal_diagnosis.py
  .venv/bin/python3 scripts/go_bulk_signal_diagnosis.py --aspect C --n-per-tier 3000
  .venv/bin/python3 scripts/go_bulk_signal_diagnosis.py --keep-ipi --keep-iea
"""

import argparse
import gzip
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "dataset" / "go_ppi" / "bulk"

ASPECT_NAME = {
    "P": "biological_process",
    "F": "molecular_function",
    "C": "cellular_component",
    "A": "all_aspects",
}

# Evidence codes dropped by default. IPI and IEA are the circularity risks;
# ND ("no biological data") carries no information at all.
DEFAULT_EXCLUDED_EVIDENCE = {"IPI", "IEA", "ND"}

# Annotated straight from interaction assays -- would leak the label.
BLACKLIST_TERMS = {"GO:0005515", "GO:0005488"}  # protein binding, binding


# --------------------------------------------------------------------- GO DAG
def parse_obo(path):
    """Return (parents, namespace, alt_to_main, name) for non-obsolete terms."""
    parents = defaultdict(set)
    namespace = {}
    alt_to_main = {}
    name = {}
    cur = None
    cur_ns = None
    cur_name = None
    cur_parents = set()
    cur_alts = set()
    obsolete = False
    in_term = False

    def flush():
        if cur and not obsolete:
            parents[cur] |= cur_parents
            if cur_ns:
                namespace[cur] = cur_ns
            if cur_name:
                name[cur] = cur_name
            for a in cur_alts:
                alt_to_main[a] = cur

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("["):
                flush()
                in_term = line == "[Term]"
                cur = None
                cur_ns = None
                cur_name = None
                cur_parents = set()
                cur_alts = set()
                obsolete = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif line.startswith("namespace: "):
                cur_ns = line[11:].strip()
            elif line.startswith("name: "):
                cur_name = line[6:].strip()
            elif line.startswith("alt_id: GO:"):
                cur_alts.add(line[8:].strip())
            elif line.startswith("is_a: GO:"):
                cur_parents.add(line[6:].split("!")[0].strip())
            elif line.startswith("relationship: part_of GO:"):
                cur_parents.add(line[22:].split("!")[0].strip())
            elif line.startswith("is_obsolete: true"):
                obsolete = True
    flush()
    return parents, namespace, alt_to_main, name


def build_ancestors(parents, keep):
    """Iterative DFS ancestor closure, restricted to `keep` (one aspect)."""
    anc = {}

    def get(t):
        if t in anc:
            return anc[t]
        out = set()
        stack = [t]
        seen = {t}
        while stack:
            n = stack.pop()
            for p in parents.get(n, ()):
                if p in keep and p not in seen:
                    seen.add(p)
                    out.add(p)
                    stack.append(p)
        out.add(t)
        anc[t] = out
        return out

    for t in keep:
        get(t)
    return anc


# ---------------------------------------------------------------- annotations
def parse_gaf(path, aspect, excluded_evidence, blacklist):
    """gene symbol -> set of direct GO terms, plus dropped-line counters."""
    direct = defaultdict(set)
    stats = defaultdict(int)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 15:
                continue
            symbol, qualifier, go_id, evidence, asp, taxon = c[2], c[3], c[4], c[6], c[8], c[12]
            stats["lines"] += 1
            if "NOT" in qualifier:
                stats["drop_not"] += 1
                continue
            if aspect != "A" and asp != aspect:
                stats["drop_aspect"] += 1
                continue
            if evidence in excluded_evidence:
                stats[f"drop_ev_{evidence}"] += 1
                continue
            if go_id in blacklist:
                stats["drop_blacklist"] += 1
                continue
            if not taxon.startswith("taxon:9606"):
                stats["drop_taxon"] += 1
                continue
            direct[symbol].add(go_id)
            stats["kept"] += 1
    return direct, stats


def propagate(direct, ancestors, alt_to_main):
    """Replace each gene's direct terms with its full ancestor closure."""
    out = {}
    unmapped = set()
    for gene, terms in direct.items():
        acc = set()
        for t in terms:
            t = alt_to_main.get(t, t)
            a = ancestors.get(t)
            if a is None:
                unmapped.add(t)
                continue
            acc |= a
        if acc:
            out[gene] = acc
    return out, unmapped


def information_content(gene_terms):
    """IC(t) = -log(P(t)); P from gene frequency over the ancestor closure."""
    freq = defaultdict(int)
    for terms in gene_terms.values():
        for t in terms:
            freq[t] += 1
    total = len(gene_terms)
    return {t: -math.log(n / total) for t, n in freq.items()}, freq, total


# ----------------------------------------------------------------- similarity
def sim_jaccard(a, b):
    """Unweighted Jaccard -- what the n=42 probe used. Included so the probe's
    headline can be reproduced and the measure isolated as a factor."""
    u = len(a | b)
    return (len(a & b) / u) if u else None


def sim_gic(a, b, ic):
    inter = a & b
    union = a | b
    d = sum(ic.get(t, 0.0) for t in union)
    if d <= 0:
        return None
    return sum(ic.get(t, 0.0) for t in inter) / d


def _mica_ic(t1, t2, ancestors, ic, cache):
    key = (t1, t2) if t1 <= t2 else (t2, t1)
    v = cache.get(key)
    if v is not None:
        return v
    common = ancestors.get(t1, set()) & ancestors.get(t2, set())
    v = max((ic.get(t, 0.0) for t in common), default=0.0)
    cache[key] = v
    return v


def bma_resnik_lin(a_leaf, b_leaf, ancestors, ic, cache):
    """Best-match-average Resnik and Lin over the two most-specific term sets."""
    if not a_leaf or not b_leaf:
        return None, None
    res_rows, lin_rows = [], []
    for t1 in a_leaf:
        best_r, best_l = 0.0, 0.0
        ic1 = ic.get(t1, 0.0)
        for t2 in b_leaf:
            m = _mica_ic(t1, t2, ancestors, ic, cache)
            if m > best_r:
                best_r = m
            denom = ic1 + ic.get(t2, 0.0)
            l = (2.0 * m / denom) if denom > 0 else 0.0
            if l > best_l:
                best_l = l
        res_rows.append(best_r)
        lin_rows.append(best_l)
    for t2 in b_leaf:
        best_r, best_l = 0.0, 0.0
        ic2 = ic.get(t2, 0.0)
        for t1 in a_leaf:
            m = _mica_ic(t1, t2, ancestors, ic, cache)
            if m > best_r:
                best_r = m
            denom = ic2 + ic.get(t1, 0.0)
            l = (2.0 * m / denom) if denom > 0 else 0.0
            if l > best_l:
                best_l = l
        res_rows.append(best_r)
        lin_rows.append(best_l)
    return st.mean(res_rows), st.mean(lin_rows)


def most_specific(terms, ic, cap):
    return sorted(terms, key=lambda t: ic.get(t, 0.0), reverse=True)[:cap]


# ----------------------------------------------------------------- statistics
def rank_auc(pos, neg):
    """P(pos > neg), ties at 0.5. Mann-Whitney U normalised."""
    if not pos or not neg:
        return float("nan")
    merged = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    ranks = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rsum = sum(ranks[k] for k, (_, g) in enumerate(merged) if g == 0)
    n1, n2 = len(pos), len(neg)
    u = rsum - n1 * (n1 + 1) / 2.0
    return u / (n1 * n2)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = st.variance(a), st.variance(b)
    n1, n2 = len(a), len(b)
    pooled = ((n1 - 1) * va + (n2 - 1) * vb) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    return (st.mean(a) - st.mean(b)) / math.sqrt(pooled)


def boot_auc_ci(pos, neg, rng, n_boot=400, alpha=0.05):
    if not pos or not neg:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        p = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        n = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        vals.append(rank_auc(p, n))
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (lo, hi)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspect", default="P", choices=["P", "F", "C", "A"],
                    help="P/F/C = single GO aspect; A = all three pooled (what the "
                         "naive probe did -- lets GO:0005515 protein binding leak in "
                         "unless --keep-binding is off)")
    ap.add_argument("--n-per-tier", type=int, default=2000)
    ap.add_argument("--leaf-cap", type=int, default=30,
                    help="cap on most-specific terms per gene for BMA Resnik/Lin")
    ap.add_argument("--min-experiments", type=int, default=1,
                    help="minimum label-channel score to count as an interaction")
    ap.add_argument("--tier-mode", choices=["tertile", "absolute"], default="absolute",
                    help="tertile = split interacting pairs into equal thirds. That is "
                         "MISLEADING on STRING: 90%% of nonzero experimental scores are "
                         "below 292, so tertile cuts (~84/134) all sit inside STRING's "
                         "low-confidence band and the 'high' tier is not high confidence. "
                         "absolute = use STRING's own confidence thresholds, so 'high' "
                         "means an established interaction and 'medium' means weak "
                         "evidence. absolute is the analogue of good_fit vs potential_fit.")
    ap.add_argument("--high-cut", type=int, default=700,
                    help="absolute mode: experimental >= this is 'high' (STRING high confidence)")
    ap.add_argument("--medium-cut", type=int, default=150,
                    help="absolute mode: experimental < this (and >= min) is 'medium' (weak evidence)")
    ap.add_argument("--label-column", default="experimental",
                    help="STRING channel used as the label. `experimental` is "
                         "GO-independent; `combined_score` fuses database+textmining "
                         "and is circular -- use it only to reproduce the naive probe.")
    ap.add_argument("--restrict-to-genes", default=None,
                    help="comma-separated gene symbols; keep only pairs where BOTH sides are "
                         "in the ego-network of these seeds (reproduces the n=42 probe's sampling)")
    ap.add_argument("--keep-ipi", action="store_true", help="do NOT drop IPI evidence (circularity test)")
    ap.add_argument("--keep-iea", action="store_true", help="do NOT drop IEA evidence")
    ap.add_argument("--keep-binding", action="store_true", help="do NOT drop protein-binding terms")
    ap.add_argument("--no-propagate", action="store_true",
                    help="use DIRECT annotations only, no ancestor closure -- what the "
                         "n=42 probe did. Propagation is standard practice; this exists "
                         "to isolate it as a factor in reproducing that probe.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    aspect_ns = ASPECT_NAME[args.aspect]

    excluded = set(DEFAULT_EXCLUDED_EVIDENCE)
    if args.keep_ipi:
        excluded.discard("IPI")
    if args.keep_iea:
        excluded.discard("IEA")
    blacklist = set() if args.keep_binding else BLACKLIST_TERMS

    obo = BULK / "go-basic.obo"
    gaf = BULK / "goa_human.gaf.gz"
    info = BULK / "9606.protein.info.v12.0.txt.gz"
    links = BULK / "9606.protein.links.detailed.v12.0.txt.gz"
    for p in (obo, gaf, info, links):
        if not p.exists():
            sys.exit(f"missing input: {p}")

    print("=" * 84)
    print("GO ontology-signal ceiling -- bulk human interactome")
    print("=" * 84)
    print(f"aspect            : {args.aspect} ({aspect_ns})")
    print(f"evidence excluded : {sorted(excluded) or 'none'}")
    print(f"terms excluded    : {sorted(blacklist) or 'none'}")
    print(f"label channel     : STRING `experiments` (>= {args.min_experiments}), GO-independent")

    parents, namespace, alt_to_main, _ = parse_obo(obo)
    if args.aspect == "A":
        keep = set(namespace)
    else:
        keep = {t for t, ns in namespace.items() if ns == aspect_ns}
    ancestors = build_ancestors(parents, keep)
    print(f"GO DAG            : {len(namespace)} terms total, {len(keep)} in aspect")

    direct, gstats = parse_gaf(gaf, args.aspect, excluded, blacklist)
    if args.no_propagate:
        gene_terms = {g: {alt_to_main.get(t, t) for t in ts} for g, ts in direct.items() if ts}
        unmapped = set()
        print("propagation       : OFF (direct annotations only)")
    else:
        gene_terms, unmapped = propagate(direct, ancestors, alt_to_main)
    ic, freq, n_genes = information_content(gene_terms)
    print(f"GAF               : {gstats['lines']} lines, {gstats['kept']} kept, "
          f"{len(gene_terms)} genes annotated")
    print(f"                    dropped: aspect={gstats['drop_aspect']} "
          f"NOT={gstats['drop_not']} blacklist={gstats['drop_blacklist']} "
          + " ".join(f"{k[8:]}={v}" for k, v in sorted(gstats.items()) if k.startswith("drop_ev_")))
    if unmapped:
        print(f"                    {len(unmapped)} annotated terms absent from DAG aspect (skipped)")
    med_terms = st.median(len(v) for v in gene_terms.values())
    print(f"                    median propagated terms/gene = {med_terms:.0f}, "
          f"max IC = {max(ic.values()):.2f}")

    # ENSP -> symbol
    ensp2sym = {}
    with gzip.open(info, "rt") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2:
                ensp2sym[c[0]] = c[1]
    print(f"STRING info       : {len(ensp2sym)} proteins")

    # interacting pairs, experiments channel as label
    pairs = []
    genes_in_net = set()
    with gzip.open(links, "rt") as f:
        header = next(f).split()
        i_exp = None
        cands = [args.label_column]
        if args.label_column == "experimental":
            cands.append("experiments")
        for cand in cands:
            if cand in header:
                i_exp = header.index(cand)
                exp_col = cand
                break
        if i_exp is None:
            sys.exit(f"no `{args.label_column}` column in {links.name}: {header}")
        circular = exp_col not in ("experimental", "experiments")
        print(f"label column      : `{exp_col}` (index {i_exp})"
              + ("   [CIRCULAR: fuses GO-derived channels]" if circular else "   [GO-independent]"))
        for line in f:
            c = line.split()
            if len(c) <= i_exp:
                continue
            exp = int(c[i_exp])
            if exp < args.min_experiments:
                continue
            a = ensp2sym.get(c[0])
            b = ensp2sym.get(c[1])
            if not a or not b or a == b:
                continue
            if a not in gene_terms or b not in gene_terms:
                continue
            if a > b:
                a, b = b, a
            pairs.append((a, b, exp))
            genes_in_net.add(a)
            genes_in_net.add(b)
    # STRING lists both directions
    pairs = list({(a, b): (a, b, e) for a, b, e in pairs}.values())
    print(f"STRING links      : {len(pairs)} unique undirected pairs with "
          f"{exp_col}>={args.min_experiments} and GO on both sides "
          f"({len(genes_in_net)} genes)")

    if args.restrict_to_genes:
        seeds = {g.strip().upper() for g in args.restrict_to_genes.split(",") if g.strip()}
        ego = set(seeds)
        for a, b, _ in pairs:
            if a in seeds:
                ego.add(b)
            if b in seeds:
                ego.add(a)
        pairs = [(a, b, e) for a, b, e in pairs if a in ego and b in ego]
        genes_in_net = {g for a, b, _ in pairs for g in (a, b)}
        print(f"ego restriction   : seeds={sorted(seeds)} -> {len(ego)} genes in ego-network, "
              f"{len(pairs)} pairs retained ({len(genes_in_net)} genes present)")
    if len(pairs) < 30:
        sys.exit(f"only {len(pairs)} pairs after filtering; nothing to measure")
    tiers = {"high": [], "medium": [], "low": []}
    if args.tier_mode == "tertile":
        scores = sorted(e for _, _, e in pairs)
        q33 = scores[len(scores) // 3]
        q67 = scores[2 * len(scores) // 3]
        print(f"{exp_col} TERTILES: low<={q33}  medium=({q33},{q67}]  high>{q67}")
        if q67 < 400:
            print(f"  WARNING: the 'high' tertile starts at {q67}, below STRING's "
                  f"medium-confidence threshold of 400. This contrast is entirely "
                  f"inside the low-confidence band -- it measures evidence QUANTITY, "
                  f"not whether the interaction is real. Use --tier-mode absolute.")
        for a, b, e in pairs:
            if e > q67:
                tiers["high"].append((a, b, e))
            elif e > q33:
                tiers["medium"].append((a, b, e))
            else:
                tiers["low"].append((a, b, e))
    else:
        q33, q67 = args.medium_cut, args.high_cut
        print(f"{exp_col} ABSOLUTE (STRING confidence bands): "
              f"high>={args.high_cut} (high conf)  "
              f"medium=[{args.min_experiments},{args.medium_cut}) (weak evidence)  "
              f"low=[{args.medium_cut},{args.high_cut}) (intermediate)")
        for a, b, e in pairs:
            if e >= args.high_cut:
                tiers["high"].append((a, b, e))
            elif e < args.medium_cut:
                tiers["medium"].append((a, b, e))
            else:
                tiers["low"].append((a, b, e))
        print(f"  tier populations before sampling: "
              + "  ".join(f"{k}={len(v)}" for k, v in tiers.items()))
        for k in ("high", "medium"):
            if len(tiers[k]) < 200:
                print(f"  WARNING: only {len(tiers[k])} pairs in '{k}'")
    for k in tiers:
        rng.shuffle(tiers[k])
        tiers[k] = tiers[k][: args.n_per_tier]
    print("tier sample sizes : " + "  ".join(f"{k}={len(v)}" for k, v in tiers.items()))

    # random non-interacting control
    interacting = {(a, b) for a, b, _ in pairs}
    net_genes = sorted(genes_in_net)
    random_pairs = []
    tries = 0
    target = min(args.n_per_tier, len(tiers["high"]))
    while len(random_pairs) < target and tries < target * 200:
        tries += 1
        a = net_genes[rng.randrange(len(net_genes))]
        b = net_genes[rng.randrange(len(net_genes))]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) in interacting:
            continue
        random_pairs.append((a, b, 0))
    print(f"random control    : {len(random_pairs)} non-interacting pairs")

    # score every sampled pair
    leaf_cache = {}

    def leaves(g):
        v = leaf_cache.get(g)
        if v is None:
            v = most_specific(gene_terms[g], ic, args.leaf_cap)
            leaf_cache[g] = v
        return v

    mica_cache = {}
    measures = ("Jaccard", "simGIC", "Resnik_BMA", "Lin_BMA")
    vals = {m: defaultdict(list) for m in measures}
    groups = list(tiers.items()) + [("random", random_pairs)]
    for gname, plist in groups:
        for a, b, _ in plist:
            ta, tb = gene_terms[a], gene_terms[b]
            j = sim_jaccard(ta, tb)
            if j is not None:
                vals["Jaccard"][gname].append(j)
            g = sim_gic(ta, tb, ic)
            if g is not None:
                vals["simGIC"][gname].append(g)
            r, l = bma_resnik_lin(leaves(a), leaves(b), ancestors, ic, mica_cache)
            if r is not None:
                vals["Resnik_BMA"][gname].append(r)
                vals["Lin_BMA"][gname].append(l)
        print(f"  scored {gname}: n={len(vals['simGIC'][gname])}")

    if args.tier_mode == "absolute":
        contrasts = [
            ("easy    (high vs random)", "high", "random"),
            ("hard    (high vs weak-ev)", "high", "medium"),
            ("intermediate vs weak-ev", "low", "medium"),
        ]
    else:
        contrasts = [
            ("easy    (high vs low)", "high", "low"),
            ("hard    (high vs medium)", "high", "medium"),
            ("trivial (high vs random)", "high", "random"),
            ("medium vs low", "medium", "low"),
        ]

    results = {
        "config": {
            "aspect": args.aspect,
            "aspect_namespace": aspect_ns,
            "excluded_evidence": sorted(excluded),
            "excluded_terms": sorted(blacklist),
            "label": f"STRING {exp_col} channel >= {args.min_experiments}",
            "label_column": exp_col,
            "label_is_circular": circular,
            "restrict_to_genes": args.restrict_to_genes,
            "n_per_tier": args.n_per_tier,
            "leaf_cap": args.leaf_cap,
            "seed": args.seed,
            "tier_mode": args.tier_mode,
            "tier_cuts": {"medium_below": q33, "high_at_or_above": q67},
            "n_pairs_total": len(pairs),
            "n_genes_annotated": len(gene_terms),
            "n_genes_in_network": len(genes_in_net),
        },
        "measures": {},
    }

    for m in measures:
        print()
        print("=" * 84)
        print(f"{m}: does GO similarity separate experimental-confidence tiers?")
        print("=" * 84)
        per_group = {}
        for gname, _ in groups:
            v = vals[m][gname]
            if not v:
                continue
            per_group[gname] = {
                "n": len(v),
                "mean": st.mean(v),
                "sd": st.stdev(v) if len(v) > 1 else 0.0,
                "median": st.median(v),
            }
            print(f"  {gname:<8} n={len(v):<6} mean={st.mean(v):.4f}  "
                  f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  median={st.median(v):.4f}")
        cres = {}
        for label, hi, lo in contrasts:
            a, b = vals[m].get(hi, []), vals[m].get(lo, [])
            if not a or not b:
                continue
            auc = rank_auc(a, b)
            d = cohens_d(a, b)
            ci = boot_auc_ci(a, b, rng)
            cres[label.split()[0] + ("_" + label.split()[1] if label.startswith("medium") else "")] = {
                "rank_auc": auc, "cohens_d": d, "auc_ci95": list(ci),
                "n_pos": len(a), "n_neg": len(b),
            }
            print(f"    {label:<26} rank-AUC={auc:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}")
        results["measures"][m] = {"groups": per_group, "contrasts": cres}

    # comparison table, primary measure simGIC / hard contrast
    hard_auc = results["measures"]["simGIC"]["contrasts"].get("hard", {}).get("rank_auc", float("nan"))
    easy_auc = results["measures"]["simGIC"]["contrasts"].get("easy", {}).get("rank_auc", float("nan"))
    print()
    print("=" * 84)
    print("ceiling comparison (within-domain contrasts; NOT comparable across datasets)")
    print("=" * 84)
    print(f"  {'ontology':<12} {'domain':<8} {'easy':>10} {'hard':>10}   label")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10}   {'-'*34}")
    print(f"  {'MeSH':<12} {'trials':<8} {0.7647:>10.4f} {0.5256:>10.4f}   eligibility grade")
    print(f"  {'ESCO skill':<12} {'career':<8} {0.6275:>10.4f} {0.5581:>10.4f}   potential_fit grade")
    print(f"  {'ISCO occ.':<12} {'career':<8} {0.5639:>10.4f} {0.4851:>10.4f}   potential_fit grade")
    print(f"  {'GO (Jaccard)':<12} {'PPI':<8} {0.7959:>10.4f} {0.7628:>10.4f}   STRING combined, n=42 probe")
    print(f"  {'GO (simGIC)':<12} {'PPI':<8} {easy_auc:>10.4f} {hard_auc:>10.4f}   STRING experiments, bulk")
    print()
    print("  Each row's easy/hard AUCs are within-dataset contrasts against that")
    print("  dataset's own label. Rows are not commensurable with each other; the")
    print("  table shows whether each ontology clears chance on ITS OWN task.")
    print()
    if not math.isnan(hard_auc):
        if hard_auc >= 0.65:
            verdict = ("GO retains strong hard-contrast signal under a GO-independent label and "
                       "IC weighting. The probe's headline survives the rigour upgrade.")
        elif hard_auc >= 0.58:
            verdict = ("GO's hard-contrast signal is real but materially weaker than the n=42 "
                       "Jaccard probe suggested. Part of that probe was circularity and small-sample noise.")
        else:
            verdict = ("GO's hard-contrast signal largely collapses once the label is GO-independent "
                       "and IC weighting replaces Jaccard. The probe's 0.76 was mostly artefact.")
        print(f"  VERDICT: {verdict}")
        print(f"           Jaccard probe hard-contrast 0.7628 -> bulk simGIC {hard_auc:.4f} "
              f"({hard_auc - 0.7628:+.4f})")

    tag = f"go_bulk_{args.aspect}_{exp_col}_{args.tier_mode}"
    if args.restrict_to_genes:
        tag += "_ego"
    if args.keep_ipi:
        tag += "_keepIPI"
    if args.keep_iea:
        tag += "_keepIEA"
    if args.keep_binding:
        tag += "_keepBinding"
    if args.no_propagate:
        tag += "_noProp"
    out = Path(args.out) if args.out else (ROOT / "results" / "ontology_ceiling" / f"{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
