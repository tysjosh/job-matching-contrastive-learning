#!/usr/bin/env python3
"""Two-stage retrieve-then-rerank, evaluated as a RANKING task per query.

Why this exists rather than more pairwise AUC
---------------------------------------------
The fusion probe measured a single blended score with one global weight. On trials
that is provably the wrong shape, because the two features win different contrasts:

    trials  soft contrast (eligible vs irrelevant)   text 0.802  MeSH 0.701
            hard contrast (eligible vs ineligible)   text 0.481  MeSH 0.603

Text is the better instrument for deciding whether a trial is about the patient at
all; MeSH is the better instrument for deciding eligibility among trials that are.
One weight has to compromise between the two jobs. A two-stage design does not:

    stage 1  rank the topic's candidate pool by TEXT, keep the top k
    stage 2  re-rank that shortlist with the ONTOLOGY (or a fusion within it)

This also matches how such a system would actually be deployed. The ontology term
never has to be scored against the whole corpus, so ANN indexing on the text
embedding is preserved -- which a single global fused score would break.

Evaluation is per-query ranking, not pairwise AUC, because that is what a
two-stage system changes. Pairwise AUC over a flat pool cannot express "the
shortlist was reordered": it has no notion of a cutoff. Metrics reported per topic
and averaged:

    nDCG@k    graded relevance (eligible=2, ineligible=1, irrelevant=0) with
              2^relevance-1 gains, matching the repository's ordinal evaluator
    P@k       fraction of the top k that are ELIGIBLE (grade 2) -- what a
              clinician screening a shortlist actually experiences
    Recall@k  fraction of all that topic's eligible trials found in the top k

The stage-1 cutoff k1 and the stage-2 weight are selected on VALIDATION and
applied to TEST, so no test statistic informs any choice.

Usage
    .venv/bin/python3 scripts/probe_two_stage_rerank.py --domain trials
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRADE_OF_LABEL = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}

DOMAINS: Dict[str, Dict[str, Any]] = {
    "trials": {
        "config": "config/lc_trials_ontneg_only.json",
        "val": "preprocess/trec_ct_splits/validation.jsonl",
        "test": "preprocess/trec_ct_splits/test.jsonl",
        "ontology": "mesh",
        "group_key": "topic_id",
        "checkpoints": [
            "results/label_budget/full_s42/best_checkpoint.pt",
            "results/label_budget/full_s13/best_checkpoint.pt",
            "results/label_budget/low_ontology_s42/best_checkpoint.pt",
            "results/label_budget/low_ontology_s13/best_checkpoint.pt",
            "results/label_budget/low_random_s42/best_checkpoint.pt",
            "results/label_budget/low_random_s13/best_checkpoint.pt",
        ],
    },
    "go_ppi": {
        "config": "config/lc_go_ppi_ontneg_stoch.json",
        "val": "preprocess/go_ppi_splits/validation.jsonl",
        "test": "preprocess/go_ppi_splits/test.jsonl",
        "ontology": "go",
        "group_key": "protein_id",
        "checkpoints": [
            "results/lc_single_factor/go_ppi_baseline_f100_s42/best_checkpoint.pt",
            "results/lc_single_factor/go_ppi_baseline_f100_s13/best_checkpoint.pt",
            "results/lc_single_factor/go_ppi_baseline_f100_s21/best_checkpoint.pt",
        ],
    },
}


# ------------------------------------------------------------------- metrics
def dcg(relevances: Sequence[float]) -> float:
    """DCG with the same exponential gain convention as run_ordinal_evaluation."""
    return sum((2.0 ** rel - 1.0) / math.log2(i + 2)
               for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_grades: Sequence[int], all_grades: Sequence[int], k: int) -> float:
    ideal = sorted(all_grades, reverse=True)[:k]
    denom = dcg(ideal)
    if denom <= 0:
        return float("nan")
    return dcg(list(ranked_grades)[:k]) / denom


def p_at_k(ranked_grades: Sequence[int], k: int) -> float:
    top = list(ranked_grades)[:k]
    return sum(1 for g in top if g == 2) / len(top) if top else float("nan")


def recall_at_k(ranked_grades: Sequence[int], all_grades: Sequence[int], k: int) -> float:
    total = sum(1 for g in all_grades if g == 2)
    if total == 0:
        return float("nan")
    return sum(1 for g in list(ranked_grades)[:k] if g == 2) / total


# --------------------------------------------------------------------- scoring
def build_matcher(kind: str, config):
    if kind == "go":
        from go_ppi_domain.run_config import build_go_matcher
        return build_go_matcher(config)
    if kind == "mesh":
        from trials_domain.run_config import build_mesh_matcher
        return build_mesh_matcher(config)
    raise SystemExit(f"unknown ontology {kind!r}")


def _record_qid(record: Dict[str, Any], group_key: str) -> str:
    anchor = record.get("resume") or {}
    metadata = record.get("metadata") or {}
    return str(metadata.get("resume_id") or anchor.get(group_key) or "").strip()


def _sample_complete_queries(records: List[Dict[str, Any]], group_key: str,
                             max_records: int, seed: int) -> List[Dict[str, Any]]:
    """Subsample whole query pools for smoke runs; never truncate a query's pool."""
    if not max_records or len(records) <= max_records:
        return records

    import random

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        qid = _record_qid(record, group_key)
        if qid:
            grouped[qid].append(record)

    qids = sorted(grouped)
    random.Random(seed).shuffle(qids)
    chosen: List[Dict[str, Any]] = []
    for qid in qids:
        pool = grouped[qid]
        if chosen and len(chosen) + len(pool) > max_records:
            continue
        chosen.extend(pool)
    return chosen


# Frozen encoder outputs and ontology scores do not depend on the projection-head
# checkpoint. Trials has six checkpoints and long texts, so doing this work once
# per split changes the run from twelve encodes to two without changing any score.
_SPLIT_CACHE: Dict[Any, Any] = {}


def _prepare_split(split_path: Path, config, matcher, group_key: str,
                   max_records: int, seed: int, device_name: str):
    """Encode a complete set of query pools once, independently of checkpoint."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from run_phase1_embedding_evaluation import content_to_text

    key = (str(split_path), group_key, max_records, seed, id(matcher), device_name)
    hit = _SPLIT_CACHE.get(key)
    if hit is not None:
        return hit

    records = []
    with open(split_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    records = _sample_complete_queries(records, group_key, max_records, seed)

    rows = []
    for r in records:
        a, b = r.get("resume"), r.get("job")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        md = r.get("metadata") or {}
        g = GRADE_OF_LABEL.get(md.get("original_label"))
        if g is None:
            continue
        qid = _record_qid(r, group_key)
        if not qid:
            continue
        ta, tb = content_to_text(a, "resume"), content_to_text(b, "job")
        if not ta.strip() or not tb.strip():
            continue
        ua, ub = a.get("skill_uris") or [], b.get("skill_uris") or []
        try:
            s = float(matcher.ontology_set_similarity(ua, ub)) if (ua and ub) else 0.0
        except Exception:
            s = 0.0
        rows.append((qid, ta, tb, s, g))

    device = torch.device(device_name)
    enc = SentenceTransformer(config.text_encoder_model).to(device)
    dim = enc.get_sentence_embedding_dimension()
    uniq = sorted({t for _q, ta, tb, _s, _g in rows for t in (ta, tb)})
    print(f"      encoding {len(uniq)} unique texts for {split_path.name} "
          f"(once, reused across checkpoints) ...", flush=True)
    with torch.no_grad():
        base = enc.encode(uniq, batch_size=64, convert_to_numpy=True,
                          show_progress_bar=False)
    idx = {t: i for i, t in enumerate(uniq)}
    ia = np.asarray([idx[ta] for _q, ta, _tb, _s, _g in rows], dtype=int)
    ib = np.asarray([idx[tb] for _q, _ta, tb, _s, _g in rows], dtype=int)
    onto = np.asarray([s for _q, _ta, _tb, s, _g in rows], dtype=float)
    grades = np.asarray([g for _q, _ta, _tb, _s, g in rows], dtype=int)
    qids = [q for q, _ta, _tb, _s, _g in rows]
    out = (base, ia, ib, onto, grades, qids, dim)
    _SPLIT_CACHE[key] = out
    return out


def score_split(split_path: Path, checkpoint: Path, config, matcher, group_key: str,
                max_records: int = 0, seed: int = 42, device_name: str = "cpu"):
    """Per-query scored candidates: {qid: [(text_sim, onto_sim, grade), ...]}."""
    import numpy as np
    import torch
    from run_phase1_embedding_evaluation import CareerAwareContrastiveModel

    base, ia, ib, onto, grades, qids, dim = _prepare_split(
        split_path, config, matcher, group_key, max_records, seed, device_name)

    device = torch.device(device_name)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    use_sf = any("structured_encoder" in k for k in state.keys())
    model = CareerAwareContrastiveModel(
        input_dim=dim,
        projection_dim=getattr(config, "projection_dim", 128),
        dropout=getattr(config, "projection_dropout", 0.1),
        use_structured_features=use_sf,
        structured_feature_dim=ckpt.get("config", {}).get("structured_feature_dim", 32),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        proj = model(torch.tensor(base, dtype=torch.float32, device=device)).cpu().numpy()
    cosines = np.sum(proj[ia] * proj[ib], axis=1)

    per_query: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    for qid, cos, ont, grade in zip(qids, cosines, onto, grades):
        per_query[qid].append((float(cos), float(ont), int(grade)))
    return per_query


def evaluate(per_query, k1: Optional[int], w2: float, k: int) -> Dict[str, float]:
    """Rank each query, optionally two-stage, and average the metrics.

    ``k1 is None`` means single-stage: rank the whole pool by the fused score with
    weight ``w2`` (w2=0 is text-only). Otherwise stage 1 takes the top ``k1`` by
    text and stage 2 re-ranks only those by the fused score.
    """
    import numpy as np

    nd, pk, rk = [], [], []
    for qid, cands in per_query.items():
        if len(cands) < 2:
            continue
        all_grades = [g for _c, _s, g in cands]
        if not any(g == 2 for g in all_grades):
            continue  # no eligible trial to find; the query cannot be scored

        # normalise within the query, which is what a per-query reranker sees
        cos = np.array([c for c, _s, _g in cands], dtype=float)
        ont = np.array([s for _c, s, _g in cands], dtype=float)
        def nz(x):
            sd = x.std()
            return (x - x.mean()) / sd if sd > 0 else x * 0.0

        if k1 is None:
            score = (1 - w2) * nz(cos) + w2 * nz(ont)
            order = np.argsort(-score)
        else:
            text_order = np.argsort(-cos)
            shortlist = text_order[:k1]
            # Stage 2 may only use the candidates retrieved by stage 1. In
            # particular, ontology scaling cannot inspect the rest of the pool.
            sub_score = ((1 - w2) * nz(cos[shortlist])
                         + w2 * nz(ont[shortlist]))
            reordered = shortlist[np.argsort(-sub_score)]
            shortlist_set = set(shortlist.tolist())
            rest = [i for i in text_order if i not in shortlist_set]
            order = list(reordered) + rest

        ranked = [all_grades[i] for i in order]
        nd.append(ndcg_at_k(ranked, all_grades, k))
        pk.append(p_at_k(ranked, k))
        rk.append(recall_at_k(ranked, all_grades, k))

    def m(v):
        v = [x for x in v if not math.isnan(x)]
        return sum(v) / len(v) if v else float("nan")

    return {"ndcg": m(nd), "p": m(pk), "recall": m(rk), "queries": len(nd)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="trials", choices=sorted(DOMAINS))
    ap.add_argument(
        "--max-records", type=int, default=0,
        help=("optional smoke-test budget; samples complete query pools only. "
              "Default 0 evaluates every record"),
    )
    ap.add_argument("--k", type=int, default=10, help="metric cutoff")
    ap.add_argument("--k1", nargs="+", type=int, default=[20, 50, 100],
                    help="stage-1 shortlist sizes to try")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
        help="embedding/projection device; auto prefers CUDA, then MPS, then CPU",
    )
    args = ap.parse_args(argv)

    import torch
    from contrastive_learning.data_structures import TrainingConfig

    if args.device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    else:
        device_name = args.device

    spec = DOMAINS[args.domain]
    config = TrainingConfig.from_json(str(ROOT / spec["config"]))
    matcher = build_matcher(spec["ontology"], config)
    ckpts = [ROOT / c for c in spec["checkpoints"]]
    ckpts = [c for c in ckpts if c.exists()]
    if not ckpts:
        sys.exit(f"no checkpoints for {args.domain}")

    print("=" * 104)
    print(f"{args.domain}: TWO-STAGE retrieve-then-rerank, per-query ranking metrics")
    print("=" * 104)
    print(f"stage 1: rank the query's pool by TEXT, keep top k1")
    print(f"stage 2: re-rank that shortlist with weight w on the ONTOLOGY")
    print(f"metrics at k={args.k}; k1 and w chosen on VALIDATION, reported on TEST")
    print(f"checkpoints: {len(ckpts)}")
    print(f"device: {device_name}")
    print()

    results = {
        "domain": args.domain,
        "k": args.k,
        "validation_split": spec["val"],
        "test_split": spec["test"],
        "max_records_per_split": args.max_records,
        "sampling_seed": 42,
        "device": device_name,
        "ndcg_gain": "2^relevance-1",
        "selection_metric": f"validation nDCG@{args.k}",
        "per_checkpoint": [],
    }
    agg = defaultdict(list)

    for ck in ckpts:
        tag = ck.parent.name
        print(f"--- {tag}", flush=True)
        pv = score_split(ROOT / spec["val"], ck, config, matcher, spec["group_key"],
                         args.max_records, device_name=device_name)
        pt = score_split(ROOT / spec["test"], ck, config, matcher, spec["group_key"],
                         args.max_records, device_name=device_name)

        # candidate configurations: single-stage fusion, and two-stage at each k1
        configs: List[Tuple[Optional[int], float]] = [(None, w) for w in args.weights]
        for k1 in args.k1:
            configs += [(k1, w) for w in args.weights if w > 0]

        val_scores = {c: evaluate(pv, c[0], c[1], args.k) for c in configs}
        test_scores = {c: evaluate(pt, c[0], c[1], args.k) for c in configs}

        baseline = test_scores[(None, 0.0)]
        # Select single-stage and two-stage configurations independently. Keeping
        # a distinct best_two is essential: otherwise a table labelled "two-stage"
        # silently becomes the single-stage result whenever validation rejects the
        # reranker, making the architectural comparison zero by construction.
        single_configs = [c for c in configs if c[0] is None]
        two_configs = [c for c in configs if c[0] is not None]
        best_single = max(single_configs,
                          key=lambda c: (val_scores[c]["ndcg"]
                                         if not math.isnan(val_scores[c]["ndcg"]) else -1))
        best_two = max(two_configs,
                       key=lambda c: (val_scores[c]["ndcg"]
                                      if not math.isnan(val_scores[c]["ndcg"]) else -1))
        best_overall = max((best_single, best_two),
                           key=lambda c: (val_scores[c]["ndcg"]
                                          if not math.isnan(val_scores[c]["ndcg"]) else -1))
        sel_single = test_scores[best_single]
        sel_two = test_scores[best_two]
        sel_overall = test_scores[best_overall]

        print(f"    queries scored: {baseline['queries']} (test)")
        print(f"    TEXT ONLY                     nDCG@{args.k} {baseline['ndcg']:.4f}  "
              f"P@{args.k} {baseline['p']:.4f}  R@{args.k} {baseline['recall']:.4f}")
        print(f"    best SINGLE-STAGE  w={best_single[1]:.1f}      "
              f"nDCG@{args.k} {sel_single['ndcg']:.4f}  P@{args.k} {sel_single['p']:.4f}  "
              f"R@{args.k} {sel_single['recall']:.4f}")
        print(f"    best TWO-STAGE   k1={best_two[0]} w={best_two[1]:.1f}   "
              f"nDCG@{args.k} {sel_two['ndcg']:.4f}  P@{args.k} {sel_two['p']:.4f}  "
              f"R@{args.k} {sel_two['recall']:.4f}")
        k1txt = "none" if best_overall[0] is None else str(best_overall[0])
        print(f"    validation winner k1={k1txt} w={best_overall[1]:.1f}   "
              f"nDCG@{args.k} {sel_overall['ndcg']:.4f}  "
              f"P@{args.k} {sel_overall['p']:.4f}  "
              f"R@{args.k} {sel_overall['recall']:.4f}")
        for m in ("ndcg", "p", "recall"):
            agg[f"base_{m}"].append(baseline[m])
            agg[f"single_{m}"].append(sel_single[m])
            agg[f"two_{m}"].append(sel_two[m])
        results["per_checkpoint"].append({
            "checkpoint": tag,
            "selected_single": {"w": best_single[1]},
            "selected_two_stage": {"k1": best_two[0], "w": best_two[1]},
            "selected_overall": {"k1": best_overall[0], "w": best_overall[1]},
            "test_baseline": baseline,
            "test_selected_single": sel_single,
            "test_selected_two_stage": sel_two,
            "test_selected_overall": sel_overall,
            "test_grid": {f"{c[0]}|{c[1]}": v for c, v in test_scores.items()},
            "val_grid": {f"{c[0]}|{c[1]}": v for c, v in val_scores.items()},
        })
        print()

    print("=" * 104)
    print(f"POOLED ACROSS {len(ckpts)} CHECKPOINTS (validation-selected, test metrics)")
    print("=" * 104)

    def mean(key):
        v = [x for x in agg[key] if not math.isnan(x)]
        return sum(v) / len(v) if v else float("nan")

    print(f"  {'metric':<10} {'text only':>10} {'1-stage fuse':>13} {'2-stage':>10} "
          f"{'2stage - text':>14} {'2stage - 1stage':>16}")
    print(f"  {'-'*10} {'-'*10} {'-'*13} {'-'*10} {'-'*14} {'-'*16}")
    for m, label in (("ndcg", f"nDCG@{args.k}"), ("p", f"P@{args.k}"),
                     ("recall", f"Recall@{args.k}")):
        b, s, t = mean(f"base_{m}"), mean(f"single_{m}"), mean(f"two_{m}")
        print(f"  {label:<10} {b:>10.4f} {s:>13.4f} {t:>10.4f} "
              f"{t-b:>+14.4f} {t-s:>+16.4f}")
        results.setdefault("pooled", {})[m] = {
            "text_only": b, "single_stage": s, "two_stage": t,
            "two_minus_text": t - b, "two_minus_single": t - s}

    n_two = sum(1 for pc in results["per_checkpoint"]
                if pc["selected_overall"]["k1"] is not None)
    print()
    print(f"  validation preferred a TWO-STAGE configuration on {n_two}/"
          f"{len(results['per_checkpoint'])} checkpoints")
    print()
    print("  How to read this. 'two-stage minus one-stage' is the column that tests the")
    print("  architectural claim: if the two features really are complementary ACROSS")
    print("  contrasts rather than within one, confining the ontology to a text-selected")
    print("  shortlist should beat blending it globally. A value near zero means the")
    print("  extra machinery buys nothing and the simpler single fused score is the")
    print("  right choice.")

    out = args.out or (ROOT / "results" / "ontology_ceiling" /
                       f"two_stage_{args.domain}.json")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
