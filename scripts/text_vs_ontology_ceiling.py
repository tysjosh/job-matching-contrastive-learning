#!/usr/bin/env python3
"""Does the ontology carry signal the ENCODER'S TEXT does not already have?

This is the diagnostic the GO/PPI result forced. The story so far:

  * ESCO 0.558, ISCO 0.485, MeSH 0.526-0.607, CPC 0.527 -- near-chance ontology
    ceilings on their hard contrasts, and injection nulls on career.
  * GO 0.8195 -- a strong ontology ceiling, genuine headroom over the model's
    0.8010, and injection STILL contributes -0.0008 (nothing) once the
    negative-diversity confound is controlled for.

So "the ontology has signal about the label" does not predict that injecting it
helps. The obvious remaining explanation is REDUNDANCY: the encoder's input text
may already carry whatever the ontology encodes. GO annotations are curated from
the same literature that STRING's protein descriptions summarise, ESCO skill
lists are extracted from the very resume text the encoder reads, and MeSH terms
are assigned to trials whose eligibility text the encoder also sees. On that
account the ontology is not absent from the model -- it arrives through the text,
and injecting it again adds nothing.

This script tests that directly, per domain, with no training involved:

    AUC_text        frozen-encoder cosine similarity alone
    AUC_onto        ontology set similarity alone
    AUC_combined    both features, 5-fold cross-validated logistic regression
    INCREMENTAL     AUC_combined - AUC_text      <- the quantity that matters
    rho             Spearman correlation between the two similarities

INCREMENTAL is the honest form of the question. A large AUC_onto means nothing on
its own if the text already ranks the pairs the same way; what a model can gain
from the ontology is bounded by what the ontology adds ON TOP OF the text. The
cross-validation matters: fitting two features on the same data the AUC is read
from would manufacture an apparent gain, which is exactly the mistake being
tested for.

Reported on the HARD contrast (good_fit vs potential_fit) and the easy contrast
(good_fit vs no_fit), per this project's convention. Every number is
within-dataset against that dataset's own label; rows are not commensurable
across domains.

Usage
    .venv/bin/python3 scripts/text_vs_ontology_ceiling.py --domain go_ppi
    .venv/bin/python3 scripts/text_vs_ontology_ceiling.py --domain all --max-pairs 4000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

GRADE_OF_LABEL = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}

def load_patents(spec: Dict[str, Any], rng: random.Random, max_pairs: int,
                 rows_per_pair: int = 3):
    """Assemble (texts_a, texts_b, ontology_sim, grades) for the patent domain.

    Patents do not fit the shared JSONL path and need care about the UNIT OF
    ANALYSIS, because the two features live at different levels:

      * CPC codes are a property of the DOCUMENT.
      * PatentMatch text is a property of a CLAIM x PARAGRAPH pair, and one
        document pair recurs ~100 times (69,576 rows over 682 document pairs).

    Scoring rows directly would repeat each CPC comparison ~100 times and shrink
    the CI by ~9x, which on this dataset flips "indistinguishable from chance"
    into "significantly above chance" (measured: CI width 0.0815 -> 0.0087). So
    pairs are deduplicated to the document level.

    That raises the mirror-image problem: if one arbitrary claim/paragraph is
    picked per document pair, the TEXT feature is far noisier than the
    document-level CPC feature, which biases the comparison against text. To keep
    the two features at the same level, text similarity is AVERAGED over up to
    ``rows_per_pair`` rows for that document pair.

    Grades, mapped onto the shared vocabulary:
        X document (cited against novelty/inventive step) -> 2  "good_fit"
        A document (cited as background)                  -> 1  "potential_fit"
        random uncited document pair                      -> 0  "no_fit"
    PatentMatch contains no uncited pairs, so grade 0 is synthesised the same way
    scripts/cpc_signal_diagnosis.py does, to give an easy contrast.
    """
    import numpy as np
    import pandas as pd

    pc = ROOT / "dataset" / "patent_cpc"
    tsv = pc / "patentmatch_mirror" / "test_balanced.tsv"
    cpc_json = pc / "cpc_by_patent.json"
    if not tsv.exists() or not cpc_json.exists():
        print(f"SKIP patents: need {tsv.name} and {cpc_json.name}")
        return None

    from scripts.cpc_signal_diagnosis import ancestor_path, load_scheme

    cpc = {k: v for k, v in json.loads(cpc_json.read_text()).items() if v}
    depth, parent, _n = load_scheme(pc / "ontology" / "CPCSchemeXML202008")

    df = pd.read_csv(tsv, sep="\t", engine="python", on_bad_lines="skip",
                     usecols=["patent_application_id", "cited_document_id",
                              "label", "text", "text_b"])
    df = df.dropna(subset=["patent_application_id", "cited_document_id", "label"])
    df["label"] = df["label"].astype(int)
    n_rows = len(df)

    # dedupe to document pairs, dropping label-ambiguous ones
    labels_by_pair: Dict[tuple, set] = defaultdict(set)
    rows_by_pair: Dict[tuple, list] = defaultdict(list)
    # Access by NAME, not by position. pandas returns usecols in the FILE's column
    # order (index, claim_id, patent_application_id, cited_document_id, text,
    # text_b, label, date), not the order requested, so positional unpacking put
    # the claim text into the label slot and produced 0 usable pairs.
    for r in df.itertuples(index=False):
        a = r.patent_application_id
        b = r.cited_document_id
        if a not in cpc or b not in cpc:
            continue
        labels_by_pair[(a, b)].add(int(r.label))
        if len(rows_by_pair[(a, b)]) < rows_per_pair:
            t, tb = r.text, r.text_b
            if isinstance(t, str) and isinstance(tb, str) and t.strip() and tb.strip():
                rows_by_pair[(a, b)].append((t, tb))
    pairs = [(a, b, next(iter(v))) for (a, b), v in labels_by_pair.items()
             if len(v) == 1 and rows_by_pair[(a, b)]]
    print(f"pairs   : {n_rows} rows -> {len(pairs)} unique document pairs "
          f"with CPC on both sides (deduplicated; see docstring)")
    if max_pairs and len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)

    # IC over the CPC ancestor closure of the documents in play
    anc_cache: Dict[str, list] = {}

    def closure(pid: str) -> set:
        s: set = set()
        for c in cpc[pid]:
            v = anc_cache.get(c)
            if v is None:
                v = ancestor_path(c, parent)
                anc_cache[c] = v
            s.update(v)
        return s

    closures = {pid: closure(pid) for pid in cpc}
    freq: Dict[str, int] = defaultdict(int)
    for s in closures.values():
        for c in s:
            freq[c] += 1
    total = len(closures)
    ic = {c: -math.log(n / total) for c, n in freq.items() if n > 0}

    def simgic(a: str, b: str) -> float:
        ca, cb = closures[a], closures[b]
        den = sum(ic.get(x, 0.0) for x in (ca | cb))
        if den <= 0:
            return 0.0
        return sum(ic.get(x, 0.0) for x in (ca & cb)) / den

    # synthesise grade-0: random application x document with no citation edge
    cited = {(a, b) for a, b, _ in pairs}
    apps = sorted({a for a, _, _ in pairs})
    docs = sorted({b for _, b, _ in pairs})
    n_x = sum(1 for _, _, l in pairs if l == 1)
    randoms = []
    tries = 0
    while len(randoms) < n_x and tries < n_x * 300:
        tries += 1
        a, b = apps[rng.randrange(len(apps))], docs[rng.randrange(len(docs))]
        if (a, b) in cited:
            continue
        # borrow a real claim/paragraph so the text feature stays comparable
        src = rows_by_pair[(a, docs[rng.randrange(len(docs))])] or None
        if not src:
            src = rows_by_pair[pairs[rng.randrange(len(pairs))][:2]]
        if not src:
            continue
        randoms.append((a, b, src[0]))
    print(f"grade-0 : {len(randoms)} synthesised uncited pairs (for the easy contrast)")

    texts_a, texts_b, onto, grades, group = [], [], [], [], []
    for a, b, l in pairs:
        rws = rows_by_pair[(a, b)]
        for (t, tb) in rws:
            texts_a.append(t)
            texts_b.append(tb)
        onto.append(simgic(a, b))
        grades.append(2 if l == 1 else 1)
        group.append(len(rws))
    for a, b, (t, tb) in randoms:
        texts_a.append(t)
        texts_b.append(tb)
        onto.append(simgic(a, b))
        grades.append(0)
        group.append(1)

    return {"texts_a": texts_a, "texts_b": texts_b, "onto": onto,
            "grades": grades, "group": group, "n_rows": n_rows}


def load_career_isco(spec: Dict[str, Any], rng: random.Random, max_pairs: int):
    """Assemble (texts, ISCO similarity, grades) for career's SECOND ontology.

    Career carries two ontologies and they must be screened separately, because
    the screen is a property of the (ontology, label) pairing rather than of the
    domain:
        ESCO  skill graph,  hard-contrast ceiling 0.5518
        ISCO  occupation hierarchy, hard-contrast ceiling 0.4851  (below chance)

    ISCO needs its own loader because its signal is a pair of SCALAR occupation
    codes, not the ``skill_uris`` lists the shared path reads.

    The similarity used here is Lin's information-content measure over shared
    ISCO prefixes, which is the BEST of the five ISCO encodings measured in
    scripts/isco_encoding_diagnosis.py (hard 0.4887 vs 0.4851 for the fixed bands
    training actually uses). Screening on the best available encoding avoids
    strawmanning the ontology: if even the best encoding adds nothing, the
    conclusion is about ISCO and not about how it was coded.

    That script also establishes a hard structural bound worth carrying into any
    discussion of this row: ISCO is fixed-depth (4 digits) and single-position, so
    every prefix-based distance is a monotone transform of "number of shared
    digits" and therefore gives an IDENTICAL rank-AUC. An oracle allowed to order
    the five prefix buckets optimally still caps at 0.5192. No re-encoding of ISCO
    can beat that.
    """
    import csv
    import numpy as np

    split = ROOT / "preprocess/data_splits_v7/train_with_resume_occ.jsonl"
    occ_csv = ROOT / "dataset/esco/occupations_en.csv"
    if not split.exists() or not occ_csv.exists():
        print(f"SKIP career_isco: need {split.name} and {occ_csv.name}")
        return None

    occ_to_isco: Dict[str, str] = {}
    with open(occ_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uri, isco = row.get("conceptUri", ""), row.get("iscoGroup", "")
            if uri and isco:
                occ_to_isco[uri] = isco

    records = []
    with open(split, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    if max_pairs and len(records) > max_pairs:
        records = rng.sample(records, max_pairs)

    # IC over ISCO prefixes, from the empirical code distribution in these pairs
    codes: List[str] = []
    for r in records:
        for side, key in ((r.get("resume"), "occupation_uri"), (r.get("job"), "occupation_uri")):
            if isinstance(side, dict):
                c = occ_to_isco.get(side.get(key, ""), "")
                if c:
                    codes.append(c)
    if not codes:
        print("SKIP career_isco: no ISCO codes resolved")
        return None
    prefix_count: Dict[str, int] = defaultdict(int)
    for c in codes:
        for k in range(1, len(c) + 1):
            prefix_count[c[:k]] += 1
    total = len(codes)
    ic = {p: -math.log(n / total) for p, n in prefix_count.items()}

    def lin_sim(a: str, b: str) -> float:
        shared = 0
        for x, y in zip(a, b):
            if x != y:
                break
            shared += 1
        if shared == 0:
            return 0.0
        num = 2.0 * ic.get(a[:shared], 0.0)
        den = ic.get(a, 0.0) + ic.get(b, 0.0)
        return num / den if den > 0 else 0.0

    from run_phase1_embedding_evaluation import content_to_text

    texts_a, texts_b, onto, grades = [], [], [], []
    missing = 0
    for r in records:
        a, b = r.get("resume"), r.get("job")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        g = GRADE_OF_LABEL.get((r.get("metadata") or {}).get("original_label"))
        if g is None:
            continue
        ca = occ_to_isco.get(a.get("occupation_uri", ""), "")
        cb = occ_to_isco.get(b.get("occupation_uri", ""), "")
        if not ca or not cb:
            missing += 1
            continue
        ta, tb = content_to_text(a, "resume"), content_to_text(b, "job")
        if not ta.strip() or not tb.strip():
            continue
        texts_a.append(ta)
        texts_b.append(tb)
        onto.append(lin_sim(ca, cb))
        grades.append(g)

    print(f"pairs   : {len(grades)} scored, {missing} dropped for a missing "
          f"occupation code on one side")
    return {"texts_a": texts_a, "texts_b": texts_b, "onto": onto,
            "grades": grades, "group": None, "n_rows": len(records)}


DOMAINS: Dict[str, Dict[str, Any]] = {
    "career_isco": {
        "custom": "career_isco",
        "ontology": "isco",
        "note": ("ISCO occupation hierarchy, Lin IC over shared code prefixes "
                 "(the best of five encodings measured); career's SECOND ontology"),
    },
    "patents": {
        "custom": "patents",
        "ontology": "cpc",
        "note": ("CPC scheme, simGIC over the ancestor closure; PatentMatch EPO "
                 "examiner grades (X = cited against novelty, A = background)"),
    },
    "go_ppi": {
        "split": "preprocess/go_ppi_splits/test.jsonl",
        "config": "config/lc_go_ppi_ontneg_stoch.json",
        "ontology": "go",
        "note": "GO biological_process, simGIC; STRING experimental confidence grades",
    },
    "trials": {
        "split": "preprocess/trec_ct_splits/test.jsonl",
        "config": "config/lc_trials_ontneg_only.json",
        "ontology": "mesh",
        "note": "MeSH 2021, best-match-average; TREC eligibility grades",
    },
    "career": {
        "split": "preprocess/data_splits_v7/test.jsonl",
        "config": "config/lc_career_ontneg_only.json",
        "ontology": "esco",
        "note": "ESCO skill graph; good_fit/potential_fit/no_fit",
    },
}


# ----------------------------------------------------------------- statistics
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
    np_, nn = len(pos), len(neg)
    return (rsum - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def boot_ci(pos, neg, rng, n_boot=400, alpha=0.05):
    if not len(pos) or not len(neg):
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        p = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        q = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        vals.append(rank_auc(p, q))
    vals.sort()
    return (vals[int(alpha / 2 * len(vals))],
            vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))])


def cv_auc(X, y, seed: int = 42, folds: int = 5) -> float:
    """Cross-validated AUC of logistic regression on the given features.

    Cross-validated because an in-sample fit on two features would report a gain
    that does not exist out of sample -- the exact artefact this script exists to
    rule out.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    k = min(folds, n_pos, n_neg)
    if k < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000))
        model.fit(X[tr], y[tr])
        scores[te] = model.predict_proba(X[te])[:, 1]
    return rank_auc(scores[y == 1].tolist(), scores[y == 0].tolist())


# ------------------------------------------------------------------- matchers
def build_matcher(kind: str, config):
    if kind == "go":
        from go_ppi_domain.run_config import build_go_matcher
        return build_go_matcher(config)
    if kind == "mesh":
        from trials_domain.run_config import build_mesh_matcher
        return build_mesh_matcher(config)
    if kind == "esco":
        # First arg is positional: esco_graph_path.
        from contrastive_learning.ontology_skill_matcher import OntologySkillMatcher
        kg = getattr(config, "esco_kg_path", None) or getattr(config, "esco_graph_path", None)
        if not kg:
            raise SystemExit("career domain needs esco_kg_path / esco_graph_path in the config")
        return OntologySkillMatcher(kg)
    raise SystemExit(f"unknown ontology kind {kind!r}")


def record_text(slot: Dict[str, Any], which: str) -> str:
    """Encoder-visible text for a slot, matching the training/eval path."""
    from run_phase1_embedding_evaluation import content_to_text
    return content_to_text(slot, which)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="go_ppi",
                    choices=sorted(DOMAINS) + ["all"])
    ap.add_argument("--max-pairs", type=int, default=6000,
                    help="cap on scored pairs per domain (encoding is the cost)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    names = sorted(DOMAINS) if args.domain == "all" else [args.domain]
    rng = random.Random(args.seed)
    all_results: Dict[str, Any] = {}

    for name in names:
        spec = DOMAINS[name]
        print("=" * 96)
        print(f"{name}: does the ontology add anything over the text?")
        print("=" * 96)
        print(f"ontology: {spec['note']}")

        group = None
        if spec.get("custom") in ("patents", "career_isco"):
            loader = (load_patents if spec["custom"] == "patents" else load_career_isco)
            loaded = loader(spec, rng, args.max_pairs)
            if loaded is None:
                continue
            texts_a = loaded["texts_a"]
            texts_b = loaded["texts_b"]
            onto = loaded["onto"]
            grades = loaded["grades"]
            group = loaded["group"]
            model_name_cfg = "sentence-transformers/all-mpnet-base-v2"
        else:
            split = ROOT / spec["split"]
            cfg_path = ROOT / spec["config"]
            if not split.exists():
                print(f"SKIP {name}: missing {split}")
                continue
            print(f"split   : {spec['split']}")

            from contrastive_learning.data_structures import TrainingConfig
            config = TrainingConfig.from_json(str(cfg_path))
            matcher = build_matcher(spec["ontology"], config)
            model_name_cfg = getattr(config, "text_encoder_model", "all-MiniLM-L6-v2")

            records = []
            with open(split, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        records.append(json.loads(line))
            if len(records) > args.max_pairs:
                records = rng.sample(records, args.max_pairs)

            texts_a, texts_b, onto, grades = [], [], [], []
            skipped = 0
            for r in records:
                a, b = r.get("resume"), r.get("job")
                if not isinstance(a, dict) or not isinstance(b, dict):
                    skipped += 1
                    continue
                label = (r.get("metadata") or {}).get("original_label")
                g = GRADE_OF_LABEL.get(label)
                if g is None:
                    skipped += 1
                    continue
                ua = a.get("skill_uris") or []
                ub = b.get("skill_uris") or []
                ta, tb = record_text(a, "resume"), record_text(b, "job")
                if not ta.strip() or not tb.strip():
                    skipped += 1
                    continue
                try:
                    s = float(matcher.ontology_set_similarity(ua, ub)) if (ua and ub) else 0.0
                except Exception:
                    s = 0.0
                texts_a.append(ta)
                texts_b.append(tb)
                onto.append(s)
                grades.append(g)
            print(f"pairs   : {len(grades)} scored, {skipped} skipped")
        by_g = defaultdict(int)
        for g in grades:
            by_g[g] += 1
        print(f"grades  : " + "  ".join(f"{k}={by_g[k]}" for k in sorted(by_g, reverse=True)))
        if min(by_g.get(2, 0), by_g.get(1, 0)) < 30:
            print("  WARNING: too few pairs in one hard-contrast class to read")

        # ---- frozen text similarity
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        model_name = model_name_cfg
        print(f"encoder : {model_name} (frozen, no projection head)")
        enc = SentenceTransformer(model_name)
        uniq = sorted(set(texts_a) | set(texts_b))
        print(f"          encoding {len(uniq)} unique texts ...", flush=True)
        with torch.no_grad():
            embs = enc.encode(uniq, batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        idx = {t: i for i, t in enumerate(uniq)}
        va = embs[[idx[t] for t in texts_a]]
        vb = embs[[idx[t] for t in texts_b]]
        text_sim = np.sum(va * vb, axis=1)

        if group is not None:
            # Patents: several claim x paragraph rows map to one document pair.
            # Average their text similarity so the text feature sits at the same
            # DOCUMENT level as the CPC feature; otherwise text carries per-claim
            # noise the ontology does not and the comparison is unfair to text.
            collapsed = []
            pos = 0
            for k in group:
                collapsed.append(float(np.mean(text_sim[pos:pos + k])))
                pos += k
            text_sim = np.asarray(collapsed, dtype=float)
            print(f"          text similarity averaged over {pos} rows -> "
                  f"{len(text_sim)} document pairs")

        onto_arr = np.asarray(onto, dtype=float)
        g_arr = np.asarray(grades, dtype=int)

        from scipy.stats import spearmanr
        rho = float(spearmanr(text_sim, onto_arr).statistic)

        res: Dict[str, Any] = {
            "split": spec.get("split", "dataset/patent_cpc (PatentMatch test, deduped)"),
            "ontology": spec["note"],
            "n_pairs": len(grades),
            "grades": {str(k): by_g[k] for k in sorted(by_g)},
            "spearman_text_vs_ontology": rho,
            "contrasts": {},
        }

        print()
        print(f"  Spearman(text_sim, ontology_sim) = {rho:+.4f}"
              + ("   <- strongly redundant" if abs(rho) > 0.5 else ""))
        print()
        print(f"  {'contrast':<26} {'AUC_text':>9} {'AUC_onto':>9} {'AUC_both':>9} "
              f"{'INCREMENT':>10}  {'increment 95% CI':<22}")
        print(f"  {'-'*26} {'-'*9} {'-'*9} {'-'*9} {'-'*10}  {'-'*22}")

        for cname, hi, lo in (("hard  good vs potential", 2, 1),
                              ("easy  good vs no_fit", 2, 0)):
            mask = (g_arr == hi) | (g_arr == lo)
            if mask.sum() < 60:
                continue
            y = (g_arr[mask] == hi).astype(int)
            t = text_sim[mask]
            o = onto_arr[mask]
            if y.sum() < 15 or (1 - y).sum() < 15:
                continue

            auc_t = rank_auc(t[y == 1].tolist(), t[y == 0].tolist())
            auc_o = rank_auc(o[y == 1].tolist(), o[y == 0].tolist())
            auc_b = cv_auc(np.column_stack([t, o]), y, seed=args.seed)
            # Text-only CV AUC, so the increment compares like with like: both
            # sides then carry the same cross-validation noise.
            auc_t_cv = cv_auc(np.column_stack([t]), y, seed=args.seed)
            increment = auc_b - auc_t_cv

            # bootstrap CI on the increment
            inc_vals = []
            n = len(y)
            for _ in range(120):
                ii = [rng.randrange(n) for _ in range(n)]
                yy = y[ii]
                if yy.sum() < 15 or (1 - yy).sum() < 15:
                    continue
                b_both = cv_auc(np.column_stack([t[ii], o[ii]]), yy, seed=args.seed)
                b_text = cv_auc(np.column_stack([t[ii]]), yy, seed=args.seed)
                if not (math.isnan(b_both) or math.isnan(b_text)):
                    inc_vals.append(b_both - b_text)
            inc_vals.sort()
            if inc_vals:
                lo_ci = inc_vals[int(0.025 * len(inc_vals))]
                hi_ci = inc_vals[min(len(inc_vals) - 1, int(0.975 * len(inc_vals)))]
            else:
                lo_ci = hi_ci = float("nan")

            flag = ""
            if not math.isnan(lo_ci) and lo_ci <= 0.0 <= hi_ci:
                flag = "  (CI spans 0)"
            print(f"  {cname:<26} {auc_t:>9.4f} {auc_o:>9.4f} {auc_b:>9.4f} "
                  f"{increment:>+10.4f}  [{lo_ci:+.4f},{hi_ci:+.4f}]{flag}")

            res["contrasts"][cname.split()[0]] = {
                "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                "auc_text": auc_t, "auc_ontology": auc_o,
                "auc_text_cv": auc_t_cv, "auc_combined_cv": auc_b,
                "increment": increment, "increment_ci95": [lo_ci, hi_ci],
            }

        all_results[name] = res
        print()

    # ------------------------------------------------------------- conclusion
    if all_results:
        print("=" * 96)
        print("SUMMARY -- what the ontology adds ON TOP OF the text (hard contrast)")
        print("=" * 96)
        print(f"  {'domain':<10} {'AUC_text':>9} {'AUC_onto':>9} {'INCREMENT':>10} "
              f"{'rho':>8}   reading")
        print(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*10} {'-'*8}   {'-'*34}")
        for name, r in all_results.items():
            h = r["contrasts"].get("hard")
            if not h:
                continue
            inc = h["increment"]
            ci = h["increment_ci95"]
            spans0 = not math.isnan(ci[0]) and ci[0] <= 0.0 <= ci[1]
            # Order matters. "text is at chance" has to be checked FIRST: when
            # neither source separates the classes, an increment estimate is a
            # difference between two chance-level numbers and means nothing.
            if h["auc_text"] < 0.55 and h["auc_ontology"] < 0.55:
                reading = "NEITHER source works -- label unlearnable here"
            elif spans0:
                reading = "increment unresolved (CI spans 0)"
            elif inc > 0:
                reading = "ontology ADDS signal beyond text"
            else:
                reading = "ontology actively worse than text alone"
            print(f"  {name:<10} {h['auc_text']:>9.4f} {h['auc_ontology']:>9.4f} "
                  f"{inc:>+10.4f} {r['spearman_text_vs_ontology']:>+8.3f}   {reading}")
        print()
        print("  INCREMENT bounds what ontology injection can achieve. AUC_onto alone")
        print("  does not: an ontology can rank pairs well and still add nothing if the")
        print("  text already ranks them the same way.")
        print()
        print("  Read it together with AUC_text, because there are three distinct")
        print("  situations and they call for different conclusions:")
        print("    AUC_text at chance AND AUC_onto at chance")
        print("        -> the label is not learnable from either source. Nothing about")
        print("           the ontology or the injection method is at fault; the dataset")
        print("           does not contain the distinction. No mechanism can fix this.")
        print("    AUC_text high, increment ~0")
        print("        -> redundancy. The ontology is real but the text already carries")
        print("           it, so injection has nothing to contribute.")
        print("    AUC_text high, increment > 0")
        print("        -> the ontology holds signal the text lacks, and a null training")
        print("           result then indicts the MECHANISM, not the ontology. A model")
        print("           that consumes the ontology as a FEATURE gains the increment;")
        print("           negative selection, which only uses it to pick which examples")
        print("           to show, evidently does not.")

    out = args.out or (ROOT / "results" / "ontology_ceiling" / "text_vs_ontology.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # MERGE rather than overwrite. Each domain takes ~10-25 min (long texts, CPU
    # encoding) so they are normally run one at a time; a plain write would leave
    # only the last domain's result and silently discard the others.
    merged: Dict[str, Any] = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
            if isinstance(existing, dict):
                merged.update(existing)
        except Exception:
            pass
    merged.update(all_results)
    out.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
