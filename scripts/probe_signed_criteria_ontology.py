#!/usr/bin/env python3
"""Does the MeSH eligibility null survive pointing the ontology at the CRITERIA?

Motivation
----------
Every ontology result in this project so far scored the trial side using
``condition_browse`` / ``intervention_browse`` MeSH headings (``resolve_trial_uris``
in ``trials_domain/data_converter.py``). Those headings describe what a trial
STUDIES. They are not derived from the eligibility criteria, they carry a median of
2 descriptors per trial, and they have no notion of inclusion vs exclusion.

So the standing conclusion -- "MeSH carries topic signal but not eligibility
signal" -- is confounded. Two explanations fit equally well:

  (a) MeSH lacks decision-relevant structure for this task, or
  (b) MeSH was pointed at the wrong text.

The shuffled-tree control in ``docs/TRIALS_CRITERION_NESTED.md`` rules out tree
GEOMETRY carrying signal over topical headings. It says nothing about hierarchy
over criteria-derived, sign-separated concepts, because that representation has
never been built.

What this script does
---------------------
Extracts MeSH concepts from the eligibility criteria textblock already present in
each trial's ``encoder_view``, splits them at the "Exclusion Criteria" marker, and
builds five ontology features:

    browse        set_sim(patient, browse_headings)          <- the current baseline
    criteria_all  set_sim(patient, inclusion + exclusion)    <- right text, no sign
    criteria_inc  set_sim(patient, inclusion)
    criteria_exc  set_sim(patient, exclusion)
    signed        set_sim(patient, inclusion) - set_sim(patient, exclusion)

Each is fused with the frozen text cosine exactly as ``probe_fusion_mechanism.py``
does -- ``(1-w) * z(cos) + w * z(onto)``, z fitted on validation and applied
unchanged to test, w selected on VALIDATION -- and scored on test.

The primary readout is the WITHIN-TOPIC macro hard AUC (eligible=2 vs
ineligible=1, computed inside each patient then averaged). That is the metric the
signal audit showed sits at chance (0.519/0.520) and that score fusion moved by
-0.0008 to +0.0007. Pooled AUCs are printed alongside only because the existing
published table reports them; pooling across topics lets an ontology take credit
for recognising the disease, which is the confound this project already documented.

Text embeddings are reused from the prepared evaluation cache, so no re-encoding.

Usage
    .venv/bin/python3 scripts/probe_signed_criteria_ontology.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRADE_OF_LABEL = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}

#: MeSH branches admitted from criteria text. Mirrors the converter's topic-side
#: choice: C+F for conditions, D+E for interventions. A/B/G are excluded there
#: because they are weak or actively harmful ("spinal cord conus mass" resolves the
#: Conus snail genus in branch B), and the same reasoning applies here.
CRITERIA_BRANCHES = ("C", "F", "D", "E")

#: Split point between the two criteria sections. 85.4% of trials carry this
#: marker; trials without it contribute an empty exclusion set, which scores 0.0
#: under ``ontology_set_similarity`` -- the same "signal absent" encoding the rest
#: of the pipeline uses.
_EXCLUSION_MARKER = re.compile(r"Exclusion\s+Criteria\s*:?", re.I)
_CRITERIA_BLOCK = re.compile(r"Criteria:(.*)", re.S)

ONTOLOGY_VARIANTS = ("browse", "criteria_all", "criteria_inc", "criteria_exc", "signed",
                     "signed_flip", "signed_repartition")

#: Seed for the two polarity controls. Both are deterministic given this seed and
#: the trial id, so the controls are reproducible without storing them.
CONTROL_SEED = 20260906

CHECKPOINTS = [
    "results/label_budget/full_s42/best_checkpoint.pt",
    "results/label_budget/full_s13/best_checkpoint.pt",
    "results/label_budget/low_ontology_s42/best_checkpoint.pt",
    "results/label_budget/low_ontology_s13/best_checkpoint.pt",
    "results/label_budget/low_random_s42/best_checkpoint.pt",
    "results/label_budget/low_random_s13/best_checkpoint.pt",
]


# --------------------------------------------------------------------- metrics
def rank_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Tie-corrected rank AUC. Same implementation as the existing probe."""
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


def _safe_mean(values: List[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def contrast_metrics(score: np.ndarray, grades: np.ndarray,
                     queries: np.ndarray) -> Dict[str, float]:
    """Global and within-topic macro AUC for the hard and pooled contrasts."""
    out: Dict[str, float] = {}
    hard_mask = (grades == 2) | (grades == 1)
    out["global_hard"] = rank_auc(score[grades == 2].tolist(),
                                  score[grades == 1].tolist())
    out["global_pooled"] = rank_auc(score[grades == 2].tolist(),
                                    score[grades != 2].tolist())

    per_query_hard, per_query_pooled = [], []
    for query in np.unique(queries):
        qm = queries == query
        pos = score[qm & (grades == 2)]
        neg_hard = score[qm & (grades == 1)]
        neg_pool = score[qm & (grades != 2)]
        if len(pos) and len(neg_hard):
            per_query_hard.append(rank_auc(pos.tolist(), neg_hard.tolist()))
        if len(pos) and len(neg_pool):
            per_query_pooled.append(rank_auc(pos.tolist(), neg_pool.tolist()))
    out["within_hard"] = _safe_mean(per_query_hard)
    out["within_pooled"] = _safe_mean(per_query_pooled)
    out["n_queries_hard"] = float(len(per_query_hard))
    return out


# ------------------------------------------------------------------ extraction
def split_criteria(encoder_view: str) -> Tuple[str, str]:
    """Return (inclusion_text, exclusion_text) from a trial's encoder view."""
    block = _CRITERIA_BLOCK.search(encoder_view or "")
    if not block:
        return "", ""
    body = block.group(1)
    parts = _EXCLUSION_MARKER.split(body, maxsplit=1)
    inclusion = parts[0]
    exclusion = parts[1] if len(parts) > 1 else ""
    return inclusion, exclusion


def load_full_criteria(trials_path: Path) -> Dict[str, str]:
    """Untruncated ``eligibility.criteria`` per trial from the converted JSONL.

    ``encoder_view`` truncates criteria at ``CRITERIA_CHAR_BUDGET`` (1200 chars),
    and 40.4% of judged trials exceed that. Because exclusion criteria conventionally
    follow inclusion criteria, truncation preferentially destroys the exclusion
    section -- the single most decision-relevant part of the record. Reading the
    converted JSONL recovers the full text.
    """
    out: Dict[str, str] = {}
    with open(trials_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            criteria = (record.get("eligibility") or {}).get("criteria") or ""
            out[record["nct_id"]] = criteria
    return out


def build_criteria_concepts(records: List[dict], index, cache_path: Path,
                            full_criteria: Optional[Dict[str, str]] = None
                            ) -> Dict[str, Dict[str, List[str]]]:
    """Extract per-section MeSH descriptors for every trial, cached to disk.

    Extraction is a pure function of the criteria text and ``desc2021.gz``, so the
    cache is safe to reuse across runs (same justification the converter uses for
    materialising concepts into the JSONL).
    """
    from trials_domain.concept_extractor import extract_concepts

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        print(f"loaded criteria concepts for {len(cached)} trials from "
              f"{cache_path.relative_to(ROOT)}", flush=True)
        return cached

    views: Dict[str, str] = {}
    for record in records:
        job = record.get("job") or {}
        nct = job.get("nct_id")
        if not nct or nct in views:
            continue
        if full_criteria is not None and nct in full_criteria:
            # Prepend the marker the splitter expects, since the raw criteria
            # field has no "Criteria:" prefix.
            views[nct] = "Criteria:" + full_criteria[nct]
        else:
            views[nct] = job.get("encoder_view") or ""

    out: Dict[str, Dict[str, List[str]]] = {}
    for i, (nct, view) in enumerate(sorted(views.items()), 1):
        inclusion_text, exclusion_text = split_criteria(view)
        out[nct] = {
            "inc": extract_concepts(inclusion_text, index,
                                    restrict_branches=CRITERIA_BRANCHES),
            "exc": extract_concepts(exclusion_text, index,
                                    restrict_branches=CRITERIA_BRANCHES),
        }
        if i % 5000 == 0:
            print(f"  extracted criteria concepts for {i}/{len(views)} trials",
                  flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out))
    print(f"wrote criteria concepts for {len(out)} trials to "
          f"{cache_path.relative_to(ROOT)}", flush=True)
    return out


# ----------------------------------------------------------------- embeddings
def load_prepared_embeddings(cache_root: Path) -> Dict[str, np.ndarray]:
    """Load every cached text embedding, keyed by sha256 of the text.

    The prepared cache from ``evaluate_publication_fusion.py`` stores
    ``chunk_*.npz`` files of (keys, embeddings). Text is unchanged by this
    experiment -- only the ontology feature differs -- so these vectors are
    directly reusable and nothing needs re-encoding.
    """
    by_hash: Dict[str, np.ndarray] = {}
    for chunk in sorted(cache_root.rglob("chunk_*.npz")):
        with np.load(chunk, allow_pickle=False) as data:
            for key, vec in zip(data["keys"], data["embeddings"]):
                by_hash[str(key)] = np.asarray(vec, dtype=np.float32)
    return by_hash


def prepare_split(path: Path, index, matcher, criteria: Dict[str, Dict[str, List[str]]],
                  embed_by_hash: Dict[str, np.ndarray], encoder_model: str
                  ) -> Dict[str, Any]:
    """Assemble text embeddings, all ontology variants, grades and topic ids."""
    from run_phase1_embedding_evaluation import content_to_text

    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    texts_a: List[str] = []
    texts_b: List[str] = []
    grades: List[int] = []
    queries: List[str] = []
    onto: Dict[str, List[float]] = {k: [] for k in ONTOLOGY_VARIANTS}

    def set_sim(a: Sequence[str], b: Sequence[str]) -> float:
        if not a or not b:
            return 0.0
        try:
            return float(matcher.ontology_set_similarity(list(a), list(b)))
        except Exception:
            return 0.0

    missing_embeddings = 0
    for record in records:
        left, right = record.get("resume"), record.get("job")
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        grade = right.get("grade")
        if grade is None:
            grade = GRADE_OF_LABEL.get((record.get("metadata") or {}).get(
                "original_label"))
        if grade not in (0, 1, 2):
            continue
        topic = (record.get("metadata") or {}).get("topic_id")
        if topic is None:
            continue
        text_a = content_to_text(left, "resume")
        text_b = content_to_text(right, "job")
        if not text_a.strip() or not text_b.strip():
            continue

        patient_uris = left.get("skill_uris") or []
        browse_uris = right.get("skill_uris") or []
        nct = right.get("nct_id")
        sections = criteria.get(nct, {"inc": [], "exc": []})
        inc_uris, exc_uris = sections["inc"], sections["exc"]
        all_uris = list(dict.fromkeys(inc_uris + exc_uris))

        s_inc = set_sim(patient_uris, inc_uris)
        s_exc = set_sim(patient_uris, exc_uris)
        onto["browse"].append(set_sim(patient_uris, browse_uris))
        onto["criteria_all"].append(set_sim(patient_uris, all_uris))
        onto["criteria_inc"].append(s_inc)
        onto["criteria_exc"].append(s_exc)
        onto["signed"].append(s_inc - s_exc)

        # --- polarity controls -------------------------------------------------
        # Both preserve the feature's marginal distribution and destroy only the
        # inclusion/exclusion assignment, so a surviving gain would mean the
        # "signed" result comes from something other than criterion polarity.
        # hashlib, not builtin hash(): str hashing is salted per process unless
        # PYTHONHASHSEED is pinned, which would make the controls irreproducible.
        seed = int(hashlib.sha256(
            f"{CONTROL_SEED}:{nct}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        # 1. random per-trial sign flip: same magnitudes, polarity randomised
        flip = 1.0 if rng.random() < 0.5 else -1.0
        onto["signed_flip"].append(flip * (s_inc - s_exc))

        # 2. random repartition: pool both sections and split into pseudo-sections
        #    of the true sizes, so set sizes and concept identities are preserved
        #    but which concepts are "inclusion" is randomised
        pooled = list(all_uris)
        rng.shuffle(pooled)
        n_inc = len(inc_uris)
        pseudo_inc, pseudo_exc = pooled[:n_inc], pooled[n_inc:]
        onto["signed_repartition"].append(
            set_sim(patient_uris, pseudo_inc) - set_sim(patient_uris, pseudo_exc))

        texts_a.append(text_a)
        texts_b.append(text_b)
        grades.append(int(grade))
        queries.append(str(topic))

    unique_texts = list(dict.fromkeys(texts_a + texts_b))
    hashes = {t: hashlib.sha256(t.encode("utf-8")).hexdigest() for t in unique_texts}
    missing = [t for t in unique_texts if hashes[t] not in embed_by_hash]
    if missing:
        from sentence_transformers import SentenceTransformer
        import torch
        print(f"  {len(missing)}/{len(unique_texts)} texts absent from the prepared "
              f"cache; encoding those", flush=True)
        encoder = SentenceTransformer(encoder_model)
        with torch.no_grad():
            vectors = encoder.encode(missing, batch_size=64, convert_to_numpy=True,
                                     show_progress_bar=False)
        for text, vec in zip(missing, vectors):
            embed_by_hash[hashes[text]] = np.asarray(vec, dtype=np.float32)
    else:
        print(f"  all {len(unique_texts)} texts served from the prepared cache",
              flush=True)

    base = np.stack([embed_by_hash[hashes[t]] for t in unique_texts])
    order = {t: i for i, t in enumerate(unique_texts)}
    return {
        "base": base,
        "left_index": np.array([order[t] for t in texts_a]),
        "right_index": np.array([order[t] for t in texts_b]),
        "ontology": {k: np.asarray(v, dtype=float) for k, v in onto.items()},
        "grades": np.asarray(grades, dtype=int),
        "queries": np.asarray(queries),
        "embedding_dim": base.shape[1],
    }


def project(prepared: Dict[str, Any], checkpoint: Path, config) -> np.ndarray:
    """Apply one checkpoint's projection head and return the pair cosine."""
    import torch
    from run_phase1_embedding_evaluation import CareerAwareContrastiveModel

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model = CareerAwareContrastiveModel(
        input_dim=prepared["embedding_dim"],
        projection_dim=getattr(config, "projection_dim", 128),
        dropout=getattr(config, "projection_dropout", 0.1),
        use_structured_features=False,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        projected = model(torch.tensor(prepared["base"], dtype=torch.float32)).numpy()
    left = projected[prepared["left_index"]]
    right = projected[prepared["right_index"]]
    return np.sum(left * right, axis=1)


def z_fit(x: np.ndarray) -> Tuple[float, float]:
    sd = float(np.std(x))
    return float(np.mean(x)), (sd if sd > 0 else 1.0)


def z_apply(x: np.ndarray, stats: Tuple[float, float]) -> np.ndarray:
    return (x - stats[0]) / stats[1]


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--select-on", default="within_pooled",
                    choices=["within_pooled", "within_hard", "global_pooled"],
                    help="validation criterion for the fusion weight")
    ap.add_argument("--full-criteria", action="store_true",
                    help="read untruncated criteria from preprocess/trec_ct/trials.jsonl "
                         "instead of the 1200-char-truncated encoder_view")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    suffix = "_fullcriteria" if args.full_criteria else ""
    if args.out is None:
        args.out = (ROOT / "results" / "ontology_ceiling"
                    / f"signed_criteria_probe{suffix}.json")

    from contrastive_learning.data_structures import TrainingConfig
    from trials_domain.run_config import build_mesh_matcher
    from trials_domain.mesh_ontology import MeshIndex

    config = TrainingConfig.from_json(ROOT / "config" / "lc_trials_ontneg_only.json")
    matcher = build_mesh_matcher(config)
    index = matcher.index

    val_path = ROOT / "preprocess" / "trec_ct_splits" / "validation.jsonl"
    test_path = ROOT / "preprocess" / "trec_ct_splits" / "test.jsonl"

    all_records = []
    for path in (val_path, test_path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_records.append(json.loads(line))

    full_criteria = None
    if args.full_criteria:
        full_criteria = load_full_criteria(
            ROOT / "preprocess" / "trec_ct" / "trials.jsonl")
        print(f"loaded untruncated criteria for {len(full_criteria)} trials")

    criteria = build_criteria_concepts(
        all_records, index,
        ROOT / "embedding_cache" / f"trials_criteria_concepts{suffix}.json",
        full_criteria=full_criteria)

    coverage = {
        "trials": len(criteria),
        "with_inclusion_concepts": sum(1 for v in criteria.values() if v["inc"]),
        "with_exclusion_concepts": sum(1 for v in criteria.values() if v["exc"]),
        "with_neither": sum(1 for v in criteria.values()
                            if not v["inc"] and not v["exc"]),
        "mean_inc": float(np.mean([len(v["inc"]) for v in criteria.values()])),
        "mean_exc": float(np.mean([len(v["exc"]) for v in criteria.values()])),
    }
    print("\ncriteria-concept coverage")
    for k, v in coverage.items():
        print(f"  {k:28} {v:.4g}" if isinstance(v, float) else f"  {k:28} {v}")

    embed_cache = ROOT / "results" / "publication_phase1_fixedval" / \
        "prepared_embeddings" / "trials"
    embed_by_hash = load_prepared_embeddings(embed_cache)
    print(f"\nprepared text embeddings available: {len(embed_by_hash)}", flush=True)

    print("\npreparing validation split ...", flush=True)
    val = prepare_split(val_path, index, matcher, criteria, embed_by_hash,
                        config.text_encoder_model)
    print("preparing test split ...", flush=True)
    test = prepare_split(test_path, index, matcher, criteria, embed_by_hash,
                         config.text_encoder_model)
    print(f"\nvalidation pairs {len(val['grades'])}, test pairs {len(test['grades'])}, "
          f"test topics {len(np.unique(test['queries']))}", flush=True)

    per_checkpoint: List[Dict[str, Any]] = []
    for ckpt_rel in CHECKPOINTS:
        ckpt = ROOT / ckpt_rel
        if not ckpt.exists():
            print(f"SKIP missing checkpoint {ckpt_rel}")
            continue
        tag = Path(ckpt_rel).parent.name
        val_cos = project(val, ckpt, config)
        test_cos = project(test, ckpt, config)
        cos_stats = z_fit(val_cos)
        val_cos_z = z_apply(val_cos, cos_stats)
        test_cos_z = z_apply(test_cos, cos_stats)

        entry: Dict[str, Any] = {"checkpoint": tag, "variants": {}}
        for variant in ONTOLOGY_VARIANTS:
            onto_stats = z_fit(val["ontology"][variant])
            val_onto_z = z_apply(val["ontology"][variant], onto_stats)
            test_onto_z = z_apply(test["ontology"][variant], onto_stats)

            val_curve, test_curve = {}, {}
            for w in args.weights:
                vs = (1 - w) * val_cos_z + w * val_onto_z
                ts = (1 - w) * test_cos_z + w * test_onto_z
                val_curve[f"{w:.1f}"] = contrast_metrics(
                    vs, val["grades"], val["queries"])
                test_curve[f"{w:.1f}"] = contrast_metrics(
                    ts, test["grades"], test["queries"])

            chosen = max(
                args.weights,
                key=lambda w: (val_curve[f"{w:.1f}"][args.select_on]
                               if not math.isnan(val_curve[f"{w:.1f}"][args.select_on])
                               else -1.0))
            entry["variants"][variant] = {
                "selected_w": chosen,
                "test_at_zero": test_curve["0.0"],
                "test_at_selected": test_curve[f"{chosen:.1f}"],
                "test_at_one": test_curve["1.0"],
                "val_curve": val_curve,
                "test_curve": test_curve,
            }
        per_checkpoint.append(entry)
        print(f"  scored {tag}", flush=True)

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 100)
    print("ONTOLOGY-ONLY (w=1.0): can each feature rank eligibility WITHIN a patient?")
    print("=" * 100)
    print(f"  {'variant':16}{'within-topic hard':>20}{'within-topic pooled':>22}"
          f"{'global hard':>14}")
    for variant in ONTOLOGY_VARIANTS:
        wh = _safe_mean([e["variants"][variant]["test_at_one"]["within_hard"]
                         for e in per_checkpoint])
        wp = _safe_mean([e["variants"][variant]["test_at_one"]["within_pooled"]
                         for e in per_checkpoint])
        gh = _safe_mean([e["variants"][variant]["test_at_one"]["global_hard"]
                         for e in per_checkpoint])
        print(f"  {variant:16}{wh:>20.4f}{wp:>22.4f}{gh:>14.4f}")

    text_wh = _safe_mean([e["variants"]["browse"]["test_at_zero"]["within_hard"]
                          for e in per_checkpoint])
    text_wp = _safe_mean([e["variants"]["browse"]["test_at_zero"]["within_pooled"]
                          for e in per_checkpoint])
    print(f"\n  {'text only (w=0)':16}{text_wh:>20.4f}{text_wp:>22.4f}")

    print("\n" + "=" * 100)
    print(f"FUSION, weight selected on validation {args.select_on}, scored on test")
    print("=" * 100)
    print(f"  {'variant':16}{'w':>6}{'within hard delta':>20}"
          f"{'within pooled delta':>22}{'hard up':>10}")
    summary: Dict[str, Any] = {}
    for variant in ONTOLOGY_VARIANTS:
        dh, dp, ws = [], [], []
        for e in per_checkpoint:
            v = e["variants"][variant]
            base_h = v["test_at_zero"]["within_hard"]
            sel_h = v["test_at_selected"]["within_hard"]
            base_p = v["test_at_zero"]["within_pooled"]
            sel_p = v["test_at_selected"]["within_pooled"]
            if not (math.isnan(base_h) or math.isnan(sel_h)):
                dh.append(sel_h - base_h)
            if not (math.isnan(base_p) or math.isnan(sel_p)):
                dp.append(sel_p - base_p)
            ws.append(v["selected_w"])
        mean_w = float(np.mean(ws)) if ws else float("nan")
        ups = sum(1 for x in dh if x > 0)
        sd_h = float(np.std(dh, ddof=1)) if len(dh) > 1 else float("nan")
        sd_p = float(np.std(dp, ddof=1)) if len(dp) > 1 else float("nan")
        print(f"  {variant:16}{mean_w:>6.2f}"
              f"{_safe_mean(dh):>+13.4f} ± {sd_h:.4f}"
              f"{_safe_mean(dp):>+15.4f} ± {sd_p:.4f}"
              f"{ups:>7}/{len(dh)}")
        summary[variant] = {
            "mean_selected_w": mean_w,
            "within_hard_delta_mean": _safe_mean(dh),
            "within_hard_delta_sd": sd_h,
            "within_hard_up": ups,
            "within_pooled_delta_mean": _safe_mean(dp),
            "within_pooled_delta_sd": sd_p,
            "n": len(dh),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "select_on": args.select_on,
        "weights": args.weights,
        "criteria_coverage": coverage,
        "summary": summary,
        "per_checkpoint": per_checkpoint,
    }, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
