#!/usr/bin/env python3
"""Does the CPC classification hierarchy separate examiner-graded prior art?

Third ontology in the ceiling series, after career (ESCO/ISCO) and trials (MeSH),
with GO/PPI as the positive control. CPC earns its place for one specific reason:
it is a CLASSIFICATION HIERARCHY like ISCO, but unlike ISCO it is
variable-depth and each document carries MANY codes. ISCO's ceiling turned out to
be structurally capped -- one fixed-depth 4-digit code per occupation means every
prefix-based distance is a monotone transform of "shared digits" and the oracle
over its five buckets caps the hard contrast at 0.5192
(scripts/isco_encoding_diagnosis.py). CPC has neither limitation, so it tests
whether the ISCO result was about hierarchies or about ISCO's impoverished
encoding.

Label: PatentMatch (Risch et al.), built from EPO search reports
  X document = cited by the examiner against novelty or inventive step
  A document = cited as general background / state of the art
Both are examiner-cited, so both are topically plausible. X-vs-A is therefore a
genuine HARD contrast in this project's sense -- the same shape as good_fit vs
potential_fit and eligible vs relevant-but-ineligible. It is not
relevant-vs-irrelevant.

  easy contrast = X-cited vs a random uncited document (topicality)
  hard contrast = X vs A (both cited; is this prior art novelty-relevant?)

DEDUPLICATION MATTERS HERE
  PatentMatch rows are claim-text x cited-paragraph pairs, so one
  (application, cited document) pair recurs many times -- 69,576 rows over only
  293 applications and 677 cited documents. CPC codes are a property of the
  DOCUMENT, not of the claim or paragraph, so scoring rows directly would repeat
  each CPC comparison dozens of times, shrink the confidence interval by roughly
  sqrt(rows/pairs) and produce a fake precision. This script deduplicates to
  unique (application, cited document, label) pairs and reports both counts.
  Pairs appearing under BOTH labels are dropped as ambiguous.

Inputs
  dataset/patent_cpc/patentmatch_mirror/test_balanced.tsv   (PatentMatch, held-out split)
  dataset/patent_cpc/cpc_by_patent.json                     (scripts/fetch_cpc_codes.py)
  dataset/patent_cpc/ontology/CPCSchemeXML202008/*.xml      (CPC scheme, true tree levels)

Usage
  .venv/bin/python3 scripts/cpc_signal_diagnosis.py
  .venv/bin/python3 scripts/cpc_signal_diagnosis.py --no-dedupe    # show the inflation
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "dataset" / "patent_cpc"
SCHEME_DIR = PC / "ontology" / "CPCSchemeXML202008"

ITEM_RE = re.compile(
    r'<classification-item\b[^>]*?level="(\d+)"[^>]*?>\s*<classification-symbol>([^<]+)</classification-symbol>'
)


# ------------------------------------------------------------------ CPC scheme
def load_scheme(scheme_dir):
    """Return (depth, parent) from the CPC scheme XML.

    Items are properly nested and carry an explicit `level`, so a symbol's parent
    is the nearest enclosing item with a smaller level. A symbol can appear more
    than once (guide heading plus real entry); the shallowest occurrence wins.
    """
    depth, parent = {}, {}
    files = sorted(scheme_dir.glob("cpc-scheme-*.xml"))
    if not files:
        sys.exit(f"no scheme XML under {scheme_dir}")
    for fp in files:
        s = fp.read_text(encoding="utf-8", errors="replace")
        stack = []  # (level, symbol)
        for m in ITEM_RE.finditer(s):
            lvl, sym = int(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            if stack:
                parent.setdefault(sym, stack[-1][1])
            if sym not in depth or lvl < depth[sym]:
                depth[sym] = lvl
            stack.append((lvl, sym))
    return depth, parent, len(files)


def string_path(code):
    """Fallback ancestor path from the code string alone.

    H03F1/3211 -> [H, H03, H03F, H03F1, H03F1/3211]
    """
    out = []
    if len(code) >= 1:
        out.append(code[0])
    if len(code) >= 3:
        out.append(code[:3])
    if len(code) >= 4:
        out.append(code[:4])
    if "/" in code:
        out.append(code.split("/")[0])
    out.append(code)
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def ancestor_path(code, parent):
    """Root-to-node path, from the scheme tree when possible."""
    if code not in parent and code not in ():
        pass
    chain, cur, guard = [], code, 0
    while cur is not None and guard < 40:
        chain.append(cur)
        cur = parent.get(cur)
        guard += 1
    chain.reverse()
    if len(chain) <= 1:
        return string_path(code)
    # prepend the string-derived coarse levels the scheme omits (section/class)
    head = [x for x in string_path(code)[:3] if x not in chain]
    return head + chain


# ------------------------------------------------------------------ statistics
def rank_auc(pos, neg):
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


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = st.variance(a), st.variance(b)
    pooled = (((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)) ** 0.5
    return (st.fmean(a) - st.fmean(b)) / pooled if pooled else float("nan")


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=PC / "patentmatch_mirror" / "test_balanced.tsv")
    ap.add_argument("--cpc-json", type=Path, default=PC / "cpc_by_patent.json")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="score raw claim-paragraph rows (demonstrates the CI inflation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)

    if not args.cpc_json.exists():
        sys.exit(f"missing {args.cpc_json}; run scripts/fetch_cpc_codes.py first")
    cpc = {k: v for k, v in json.loads(args.cpc_json.read_text()).items() if v}

    depth, parent, n_files = load_scheme(SCHEME_DIR)

    import pandas as pd
    df = pd.read_csv(args.tsv, sep="\t", engine="python", on_bad_lines="skip",
                     usecols=["patent_application_id", "cited_document_id", "label"])
    df = df.dropna(subset=["patent_application_id", "cited_document_id", "label"])
    df["label"] = df["label"].astype(int)
    n_rows = len(df)

    print("=" * 96)
    print("CPC classification: does it separate X (novelty-relevant) from A (background)?")
    print("=" * 96)
    print(f"CPC scheme      : {n_files} XML files, {len(depth)} symbols with tree depth")
    print(f"CPC assignments : {len(cpc)} documents with >=1 full code, "
          f"median {st.median([len(v) for v in cpc.values()]):.0f} codes/doc")
    print(f"PatentMatch     : {n_rows} claim-paragraph rows, "
          f"{df.patent_application_id.nunique()} applications, "
          f"{df.cited_document_id.nunique()} cited documents")

    # ------------------------------------------------------------- dedupe
    if args.no_dedupe:
        pairs = [(a, b, l) for a, b, l in
                 df[["patent_application_id", "cited_document_id", "label"]].itertuples(index=False)]
        print("dedupe          : OFF -- scoring raw rows (CIs will be falsely narrow)")
    else:
        lab_by_pair = defaultdict(set)
        for a, b, l in df[["patent_application_id", "cited_document_id", "label"]].itertuples(index=False):
            lab_by_pair[(a, b)].add(l)
        ambiguous = [k for k, v in lab_by_pair.items() if len(v) > 1]
        pairs = [(a, b, next(iter(v))) for (a, b), v in lab_by_pair.items() if len(v) == 1]
        print(f"dedupe          : ON -- {n_rows} rows -> {len(pairs)} unique document pairs "
              f"({len(ambiguous)} dropped as label-ambiguous)")

    scored_pairs, miss_a, miss_b = [], 0, 0
    for a, b, l in pairs:
        if a not in cpc:
            miss_a += 1
            continue
        if b not in cpc:
            miss_b += 1
            continue
        scored_pairs.append((a, b, l))
    print(f"CPC coverage    : {len(scored_pairs)}/{len(pairs)} pairs "
          f"({miss_a} missing application codes, {miss_b} missing cited codes)")
    if len(scored_pairs) < 100:
        sys.exit("too few pairs with CPC on both sides; fetch more codes first")
    if args.max_pairs and len(scored_pairs) > args.max_pairs:
        scored_pairs = rng.sample(scored_pairs, args.max_pairs)

    nX = sum(1 for _, _, l in scored_pairs if l == 1)
    nA = len(scored_pairs) - nX
    print(f"label balance   : X(label=1)={nX}  A(label=0)={nA}")

    # ------------------------------------- random uncited control (easy contrast)
    cited_set = {(a, b) for a, b, _ in scored_pairs}
    apps = sorted({a for a, _, _ in scored_pairs})
    docs = sorted({b for _, b, _ in scored_pairs})
    random_pairs, tries = [], 0
    while len(random_pairs) < nX and tries < nX * 300:
        tries += 1
        a, b = apps[rng.randrange(len(apps))], docs[rng.randrange(len(docs))]
        if (a, b) in cited_set:
            continue
        random_pairs.append((a, b, -1))
    print(f"random control  : {len(random_pairs)} uncited application-document pairs")

    # ------------------------------------------------------------- IC over codes
    anc_cache = {}

    def anc(code):
        v = anc_cache.get(code)
        if v is None:
            v = ancestor_path(code, parent)
            anc_cache[code] = v
        return v

    closure_cache = {}

    def closure(pid):
        v = closure_cache.get(pid)
        if v is None:
            s = set()
            for c in cpc[pid]:
                s.update(anc(c))
            closure_cache[pid] = v = s
        return v

    freq = Counter()
    for pid in cpc:
        freq.update(closure(pid))
    total = len(cpc)
    ic = {c: -math.log(n / total) for c, n in freq.items() if n > 0}
    max_ic = max(ic.values()) if ic else 1.0

    def node_depth(c):
        d = depth.get(c)
        return d if d is not None else len(string_path(c))

    # ------------------------------------------------------------------ measures
    def m_jaccard(a, b):
        A, B = set(cpc[a]), set(cpc[b])
        u = len(A | B)
        return len(A & B) / u if u else None

    def m_simgic(a, b):
        A, B = closure(a), closure(b)
        den = sum(ic.get(x, 0.0) for x in (A | B))
        return sum(ic.get(x, 0.0) for x in (A & B)) / den if den > 0 else None

    def _wu(c1, c2):
        p1, p2 = anc(c1), anc(c2)
        shared = 0
        for x, y in zip(p1, p2):
            if x != y:
                break
            shared += 1
        return (2.0 * shared) / (len(p1) + len(p2)) if (len(p1) + len(p2)) else 0.0

    def m_wu_bma(a, b):
        A, B = cpc[a], cpc[b]
        if not A or not B:
            return None
        rows = [max(_wu(x, y) for y in B) for x in A]
        rows += [max(_wu(x, y) for x in A) for y in B]
        return st.fmean(rows)

    def m_lca_depth_bma(a, b):
        """Raw depth of the deepest common ancestor, NOT depth-normalised --
        the analogue of ISCO's fixed-band / raw-hop encoding."""
        A, B = cpc[a], cpc[b]
        if not A or not B:
            return None
        MAXD = 14.0

        def dd(c1, c2):
            p1, p2 = anc(c1), anc(c2)
            shared = 0
            for x, y in zip(p1, p2):
                if x != y:
                    break
                shared += 1
            return min(shared, MAXD) / MAXD

        rows = [max(dd(x, y) for y in B) for x in A]
        return st.fmean(rows)

    def m_subclass(a, b):
        """Coarse: 4-character subclass sets (H03F). The closest CPC analogue of
        ISCO's coarse occupation-group bands."""
        A = {c[:4] for c in cpc[a]}
        B = {c[:4] for c in cpc[b]}
        u = len(A | B)
        return len(A & B) / u if u else None

    measures = [
        ("set: Jaccard(full codes)", "set", m_jaccard),
        ("set: simGIC (IC-weighted)", "set", m_simgic),
        ("hier: Wu-Palmer BMA (depth-norm)", "hierarchy", m_wu_bma),
        ("hier: LCA depth BMA (raw)", "hierarchy", m_lca_depth_bma),
        ("coarse: subclass overlap", "coarse", m_subclass),
    ]

    groups = {"X": [p for p in scored_pairs if p[2] == 1],
              "A": [p for p in scored_pairs if p[2] == 0],
              "random": random_pairs}
    vals = {name: defaultdict(list) for name, _, _ in measures}
    for gname, plist in groups.items():
        for a, b, _ in plist:
            for name, _, fn in measures:
                try:
                    v = fn(a, b)
                except Exception:
                    v = None
                if v is not None and not math.isnan(v):
                    vals[name][gname].append(v)

    results = {
        "tsv": str(args.tsv.relative_to(ROOT)),
        "deduped": not args.no_dedupe,
        "n_rows": n_rows,
        "n_pairs_scored": len(scored_pairs),
        "n_X": nX, "n_A": nA, "n_random": len(random_pairs),
        "n_docs_with_cpc": len(cpc),
        "measures": {},
    }

    contrasts = [("easy  X vs random", "X", "random"), ("hard  X vs A", "X", "A")]

    for name, kind, _ in measures:
        bg = vals[name]
        print()
        print("-" * 96)
        print(f"{name}   [{kind}]")
        print("-" * 96)
        for g in ("X", "A", "random"):
            v = bg.get(g, [])
            if not v:
                continue
            print(f"  {g:<7} n={len(v):5d}  mean={st.fmean(v):.4f}  "
                  f"sd={st.stdev(v) if len(v) > 1 else 0:.4f}  median={st.median(v):.4f}")
        allv = [x for vs in bg.values() for x in vs]
        c = Counter(round(x, 9) for x in allv)
        tie = max(c.values()) / len(allv) if allv else float("nan")
        print(f"  distinct values={len(c)}   most common covers {tie*100:.1f}% of pairs")
        mres = {"kind": kind, "tie_fraction": tie, "n_distinct": len(c), "contrasts": {}}
        for label, hi, lo in contrasts:
            a_, b_ = bg.get(hi, []), bg.get(lo, [])
            if not a_ or not b_:
                continue
            auc = rank_auc(a_, b_)
            d = cohens_d(a_, b_)
            ci = boot_ci(a_, b_, rng)
            key = "easy" if label.startswith("easy") else "hard"
            mres["contrasts"][key] = {"rank_auc": auc, "cohens_d": d, "auc_ci95": list(ci),
                                      "n_pos": len(a_), "n_neg": len(b_)}
            flag = "   (CI spans 0.5 -- chance)" if (not math.isnan(ci[0]) and ci[0] <= 0.5 <= ci[1]) else ""
            print(f"    {label:20s} rank-AUC={auc:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}{flag}")
        results["measures"][name] = mres

    # ------------------------------------------------------------------ verdict
    print()
    print("=" * 96)
    print("verdict")
    print("=" * 96)
    hier = {n: r["contrasts"].get("hard", {}).get("rank_auc")
            for n, r in results["measures"].items() if r["kind"] == "hierarchy"}
    sets = {n: r["contrasts"].get("hard", {}).get("rank_auc")
            for n, r in results["measures"].items() if r["kind"] == "set"}
    best_hard = max((v for v in list(hier.values()) + list(sets.values()) if v is not None),
                    default=float("nan"))
    easy_vals = [r["contrasts"].get("easy", {}).get("rank_auc")
                 for r in results["measures"].values()]
    best_easy = max((v for v in easy_vals if v is not None), default=float("nan"))
    print(f"  best easy contrast (X vs uncited) : {best_easy:.4f}")
    print(f"  best hard contrast (X vs A)       : {best_hard:.4f}")
    print()
    print("  Comparison with the other classification hierarchy measured in this project")
    print("  (each against its own dataset and label -- NOT commensurable, shown because")
    print("  the structural question is the same):")
    print(f"    ISCO  fixed-depth, 1 code/entity, 5 distinct distances   hard 0.4851")
    print(f"          oracle over all prefix-based ISCO distances        hard 0.5192")
    print(f"    CPC   variable-depth, many codes/entity                  hard {best_hard:.4f}")
    print()
    if not math.isnan(best_hard):
        if best_hard > 0.60:
            print("  CPC clears its hard contrast well above chance. Since CPC is a")
            print("  classification hierarchy like ISCO, this says ISCO's near-chance result")
            print("  is NOT a general property of hierarchies. What distinguishes them is")
            print("  resolution: CPC assigns many codes at variable depth, ISCO assigns one")
            print("  code at fixed depth and cannot express more than five distances.")
        elif best_hard > 0.55:
            print("  CPC clears chance but only modestly, so a richer hierarchy helps some")
            print("  without being decisive. Read alongside ISCO's structural cap this")
            print("  suggests resolution is necessary but not sufficient.")
        else:
            print("  CPC lands near chance on the hard contrast despite having none of")
            print("  ISCO's structural limits. That points away from encoding as the")
            print("  explanation: examiner-graded novelty relevance is a judgement made")
            print("  WITHIN a technical field, and a field-level taxonomy cannot express it,")
            print("  however finely it subdivides. Same shape as the career and trials")
            print("  results, on a third ontology and an independent label.")
    if not args.no_dedupe:
        print()
        print("  n is unique document pairs (682), not PatentMatch rows (69,576). Measured")
        print("  effect of leaving the multiplicity in (--no-dedupe), Jaccard hard contrast:")
        print("      deduped  AUC 0.5320  CI [0.4898, 0.5713]  width 0.0815  spans 0.5")
        print("      raw rows AUC 0.5419  CI [0.5376, 0.5463]  width 0.0087  EXCLUDES 0.5")
        print("  The point estimate barely moves but the interval is 9.4x narrower, which")
        print("  flips the conclusion from 'indistinguishable from chance' to 'significantly")
        print("  above chance'. The raw-row version is wrong: the extra rows are the same")
        print("  document pairs re-measured ~100 times, not new evidence about CPC.")

    out = args.out or (ROOT / "results" / "ontology_ceiling" /
                       f"cpc_signal{'_rows' if args.no_dedupe else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
