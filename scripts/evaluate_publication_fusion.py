#!/usr/bin/env python3
"""Publication-safe Phase-1 evaluation for text-only and ontology score fusion.

This is the domain-integrated companion to ``run_phase1_embedding_evaluation.py``
and ``run_ordinal_evaluation.py``.  It keeps their useful discrimination and
ordinal metrics while fixing the biomedical integration and evaluation issues:

* ``encoder_view`` is serialized by the shared Phase-1 implementation;
* text and ontology features are normalized with VALIDATION statistics only;
* the fusion weight and classification thresholds are selected on VALIDATION;
* test metrics are computed once with every fitted value frozen;
* ranking is macro-averaged over patient topics / anchor proteins, never globally;
* frozen text embeddings and ontology scores are computed once per split and
  reused across all fractions and seeds.

The script evaluates a baseline learning-curve grid.  Text-only and fusion are
paired scoring modes over the SAME checkpoint, so no second training arm is
needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DOMAINS: Dict[str, Dict[str, Any]] = {
    "trials": {
        "config": ROOT / "config" / "lc_trials_baseline.json",
        "validation": ROOT / "preprocess" / "trec_ct_splits" / "validation.jsonl",
        "test": ROOT / "preprocess" / "trec_ct_splits" / "test.jsonl",
        "ontology": "mesh",
        "query_field": "resume_id",
        "run_pattern": "trials_baseline_f{fraction}_s{seed}",
        "grade_names": {2: "eligible", 1: "ineligible", 0: "not_relevant"},
    },
    "go_ppi": {
        "config": ROOT / "config" / "lc_go_ppi_baseline.json",
        "validation": ROOT / "preprocess" / "go_ppi_splits" / "validation.jsonl",
        "test": ROOT / "preprocess" / "go_ppi_splits" / "test.jsonl",
        "ontology": "go",
        "query_field": "resume_id",
        "run_pattern": "go_ppi_baseline_f{fraction}_s{seed}",
        "grade_names": {2: "established", 1: "weak_evidence", 0: "no_interaction"},
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    """Convert numpy scalars and non-finite metrics to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def build_matcher(kind: str, config):
    if kind == "mesh":
        from trials_domain.run_config import build_mesh_matcher
        return build_mesh_matcher(config)
    if kind == "go":
        from go_ppi_domain.run_config import build_go_matcher
        return build_go_matcher(config)
    raise ValueError(f"unknown ontology kind {kind!r}")


def rank_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """AUC from average ranks, including correct handling of tied scores."""
    if not len(pos) or not len(neg):
        return float("nan")
    values = np.concatenate([np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)])
    labels = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos, n_neg = len(pos), len(neg)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def z_stats(values: np.ndarray) -> Tuple[float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values))
    return mean, sd if sd > 0 else 1.0


def z_apply(values: np.ndarray, stats: Tuple[float, float]) -> np.ndarray:
    return (values - stats[0]) / stats[1]


def _load_records(path: Path, max_records: int, sample_seed: int) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if max_records and len(records) > max_records:
        records = random.Random(sample_seed).sample(records, max_records)
    return records


def prepare_split(path: Path, config, matcher, query_field: str,
                  max_records: int = 0, sample_seed: int = 42,
                  frozen_cache: Optional[Dict[str, Any]] = None,
                  prepared_cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Encode split-independent inputs once and retain query/grade provenance."""
    import torch
    from sentence_transformers import SentenceTransformer
    from contrastive_learning.embedding_cache import EmbeddingCache
    from run_phase1_embedding_evaluation import content_to_text

    records = _load_records(path, max_records, sample_seed)
    texts_a: List[str] = []
    texts_b: List[str] = []
    ontology: List[float] = []
    grades: List[int] = []
    queries: List[str] = []
    cached_by_text: Dict[str, np.ndarray] = {}
    cached_by_text_hash: Dict[str, np.ndarray] = {}
    keyer = EmbeddingCache(max_cache_size=0, device=torch.device("cpu"),
                           enable_stats=False) if frozen_cache else None

    split_cache_dir = None
    if prepared_cache_dir is not None and max_records == 0:
        model_slug = str(config.text_encoder_model).replace("/", "_")
        split_cache_dir = (Path(prepared_cache_dir)
                           / f"{path.stem}_{file_sha256(path)[:16]}_{model_slug}")
        split_cache_dir.mkdir(parents=True, exist_ok=True)
        for chunk_path in sorted(split_cache_dir.glob("chunk_*.npz")):
            with np.load(chunk_path, allow_pickle=False) as cached_chunk:
                keys = cached_chunk["keys"]
                vectors = cached_chunk["embeddings"]
                if len(keys) != len(vectors):
                    raise ValueError(f"malformed prepared embedding chunk {chunk_path}")
                for key, vector in zip(keys, vectors):
                    cached_by_text_hash[str(key)] = np.asarray(
                        vector, dtype=np.float32)
        if cached_by_text_hash:
            print(f"loaded {len(cached_by_text_hash)} prepared evaluation "
                  f"embeddings from {split_cache_dir}", flush=True)

    for record in records:
        left, right = record.get("resume"), record.get("job")
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        grade = right.get("grade")
        if grade is None:
            label = (record.get("metadata") or {}).get("original_label")
            grade = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}.get(label)
        if grade not in (0, 1, 2):
            continue
        query = (record.get("metadata") or {}).get(query_field)
        if query is None:
            raise ValueError(f"record in {path} has no metadata.{query_field}")
        text_a = content_to_text(left, "resume")
        text_b = content_to_text(right, "job")
        if not text_a.strip() or not text_b.strip():
            raise ValueError(f"blank encoder text in {path}; biomedical encoder_view not integrated")
        uris_a = left.get("skill_uris") or []
        uris_b = right.get("skill_uris") or []
        try:
            ont = float(matcher.ontology_set_similarity(uris_a, uris_b)) \
                if uris_a and uris_b else 0.0
        except Exception:
            ont = 0.0
        texts_a.append(text_a)
        texts_b.append(text_b)
        if frozen_cache and keyer is not None:
            for content, text_value in ((left, text_a), (right, text_b)):
                cached = frozen_cache.get(keyer.get_content_key(content))
                if cached is not None and text_value not in cached_by_text:
                    if isinstance(cached, torch.Tensor):
                        cached = cached.detach().cpu().numpy()
                    cached_by_text[text_value] = np.asarray(cached, dtype=np.float32)
        ontology.append(ont)
        grades.append(int(grade))
        queries.append(str(query))

    encoder = SentenceTransformer(config.text_encoder_model)
    unique_texts = list(dict.fromkeys(texts_a + texts_b))
    text_hashes = {
        text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in unique_texts
    }
    missing_texts = [
        text for text in unique_texts
        if text not in cached_by_text
        and text_hashes[text] not in cached_by_text_hash
    ]
    print(f"preparing {len(unique_texts)} unique texts for {path.name}: "
          f"{len(unique_texts) - len(missing_texts)} cached, "
          f"{len(missing_texts)} to encode ...", flush=True)
    newly_encoded: Dict[str, np.ndarray] = {}
    if missing_texts:
        encode_chunk_size = 640
        with torch.no_grad():
            for start in range(0, len(missing_texts), encode_chunk_size):
                chunk = missing_texts[start:start + encode_chunk_size]
                encoded = encoder.encode(chunk, batch_size=64,
                                         convert_to_numpy=True,
                                         show_progress_bar=False)
                newly_encoded.update(
                    (text, np.asarray(vector, dtype=np.float32))
                    for text, vector in zip(chunk, encoded))
                if split_cache_dir is not None:
                    keys = np.asarray([text_hashes[text] for text in chunk])
                    vectors = np.asarray(encoded, dtype=np.float32)
                    chunk_id = hashlib.sha256(
                        "|".join(str(key) for key in keys).encode("ascii")
                    ).hexdigest()[:16]
                    np.savez(split_cache_dir / f"chunk_{chunk_id}.npz",
                             keys=keys, embeddings=vectors)
                print(f"  encoded {min(start + len(chunk), len(missing_texts))}/"
                      f"{len(missing_texts)} missing texts", flush=True)

    def resolved_vector(text: str) -> np.ndarray:
        if text in cached_by_text:
            return cached_by_text[text]
        text_hash = text_hashes[text]
        if text_hash in cached_by_text_hash:
            return cached_by_text_hash[text_hash]
        return newly_encoded[text]

    base = np.stack([resolved_vector(text) for text in unique_texts])
    index = {text: i for i, text in enumerate(unique_texts)}
    return {
        "base": np.asarray(base),
        "left_index": np.asarray([index[text] for text in texts_a], dtype=int),
        "right_index": np.asarray([index[text] for text in texts_b], dtype=int),
        "ontology": np.asarray(ontology, dtype=float),
        "grades": np.asarray(grades, dtype=int),
        "queries": np.asarray(queries, dtype=object),
        "embedding_dim": int(encoder.get_sentence_embedding_dimension()),
        "records_scored": len(grades),
        "source_records": len(records),
    }


def load_frozen_embedding_cache(config) -> Dict[str, Any]:
    """Load the reusable frozen-encoder cache written during Phase-1 training."""
    import torch

    cache_path = Path(config.embedding_cache_path)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    if not cache_path.exists():
        print(f"no frozen embedding cache at {cache_path}; encoding all texts",
              flush=True)
        return {}
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = payload.get("cache", {})
    print(f"loaded {len(cache)} frozen text embeddings from {cache_path}",
          flush=True)
    return cache


def text_scores(prepared: Dict[str, Any], checkpoint_path: Path, config) -> np.ndarray:
    """Apply one checkpoint projection head to the shared frozen embeddings."""
    import torch
    from run_phase1_embedding_evaluation import CareerAwareContrastiveModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    use_structured = any("structured_encoder" in key for key in state)
    if use_structured:
        raise ValueError(
            f"{checkpoint_path} uses structured features; the biomedical records "
            "do not define the career structured-feature representation")
    model = CareerAwareContrastiveModel(
        input_dim=prepared["embedding_dim"],
        projection_dim=getattr(config, "projection_dim", 128),
        dropout=getattr(config, "projection_dropout", 0.1),
        use_structured_features=False,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        projected = model(torch.tensor(prepared["base"], dtype=torch.float32)).numpy()
    left = projected[prepared["left_index"]]
    right = projected[prepared["right_index"]]
    return np.sum(left * right, axis=1)


def contrast_metrics(scores: np.ndarray, grades: np.ndarray,
                     queries: np.ndarray) -> Dict[str, Any]:
    definitions = {
        "hard": (2, 1),
        "easy": (2, 0),
        "middle_vs_easy": (1, 0),
    }
    global_auc: Dict[str, float] = {}
    macro_query_auc: Dict[str, Optional[float]] = {}
    query_counts: Dict[str, int] = {}
    for name, (positive_grade, negative_grade) in definitions.items():
        global_auc[name] = rank_auc(
            scores[grades == positive_grade], scores[grades == negative_grade])
        per_query = []
        for query in np.unique(queries):
            mask = queries == query
            pos = scores[mask & (grades == positive_grade)]
            neg = scores[mask & (grades == negative_grade)]
            if len(pos) and len(neg):
                per_query.append(rank_auc(pos, neg))
        macro_query_auc[name] = _safe_mean(per_query)
        query_counts[name] = len(per_query)
    global_auc["pooled"] = rank_auc(scores[grades == 2], scores[grades != 2])
    pooled_per_query = []
    for query in np.unique(queries):
        mask = queries == query
        pos = scores[mask & (grades == 2)]
        neg = scores[mask & (grades != 2)]
        if len(pos) and len(neg):
            pooled_per_query.append(rank_auc(pos, neg))
    macro_query_auc["pooled"] = _safe_mean(pooled_per_query)
    query_counts["pooled"] = len(pooled_per_query)
    return {
        "global_auc": global_auc,
        "macro_query_auc": macro_query_auc,
        "queries_with_both_classes": query_counts,
    }


def ranking_metrics(scores: np.ndarray, grades: np.ndarray, queries: np.ndarray,
                    k: int = 10) -> Dict[str, Any]:
    """Macro per-query retrieval metrics with grade-2 as strict relevance."""
    rows = []
    for query in np.unique(queries):
        mask = queries == query
        query_scores = scores[mask]
        rel = grades[mask]
        if not len(rel):
            continue
        ordered = rel[np.argsort(-query_scores, kind="mergesort")]
        top = ordered[:k]
        gains = np.power(2.0, top) - 1.0
        discounts = np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.sum(gains / discounts))
        ideal = np.sort(rel)[::-1][:k]
        ideal_gains = np.power(2.0, ideal) - 1.0
        idcg = float(np.sum(ideal_gains / np.log2(np.arange(2, len(ideal) + 2))))

        strict = ordered == 2
        total_strict = int(np.sum(rel == 2))
        hit_positions = np.flatnonzero(strict)
        if total_strict:
            precisions = [(i + 1) / (position + 1)
                          for i, position in enumerate(hit_positions)]
            ap = float(np.sum(precisions) / total_strict)
            mrr = float(1.0 / (hit_positions[0] + 1)) if len(hit_positions) else 0.0
            recall = float(np.sum(top == 2) / total_strict)
        else:
            ap = mrr = recall = float("nan")
        rows.append({
            "query": str(query),
            f"ndcg@{k}": dcg / idcg if idcg > 0 else float("nan"),
            f"precision@{k}": float(np.sum(top == 2) / k),
            f"recall@{k}": recall,
            "average_precision": ap,
            "reciprocal_rank": mrr,
        })

    keys = [f"ndcg@{k}", f"precision@{k}", f"recall@{k}",
            "average_precision", "reciprocal_rank"]
    return {
        "n_queries": len(rows),
        "macro": {key: _safe_mean(row[key] for row in rows) for key in keys},
        # Retained for query-level paired bootstrap in the aggregation step.
        "per_query": rows,
    }


def ordinal_metrics(scores: np.ndarray, grades: np.ndarray) -> Dict[str, Any]:
    from scipy import stats

    by_grade = {grade: scores[grades == grade] for grade in (0, 1, 2)}

    def cohens_d(high: np.ndarray, low: np.ndarray) -> Optional[float]:
        if len(high) < 2 or len(low) < 2:
            return None
        pooled_var = (((len(high) - 1) * np.var(high, ddof=1)
                       + (len(low) - 1) * np.var(low, ddof=1))
                      / (len(high) + len(low) - 2))
        return float((np.mean(high) - np.mean(low)) / math.sqrt(pooled_var)) \
            if pooled_var > 0 else 0.0

    tau = stats.kendalltau(scores, grades)
    rho = stats.spearmanr(scores, grades)
    return {
        "tier_stats": {
            str(grade): {
                "n": len(values),
                "mean": float(np.mean(values)) if len(values) else None,
                "sd": float(np.std(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
            }
            for grade, values in by_grade.items()
        },
        "cohens_d": {
            "hard": cohens_d(by_grade[2], by_grade[1]),
            "easy": cohens_d(by_grade[2], by_grade[0]),
            "middle_vs_easy": cohens_d(by_grade[1], by_grade[0]),
        },
        "kendall_tau_b": float(tau.statistic),
        "spearman_rho": float(rho.statistic),
    }


def fit_binary_threshold(scores: np.ndarray, grades: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    labels = (grades == 2).astype(int)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def binary_classification(scores: np.ndarray, grades: np.ndarray,
                          threshold: float) -> Dict[str, Any]:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)

    truth = (grades == 2).astype(int)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {
        "threshold_from_validation": threshold,
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }


def fit_ordinal_thresholds(scores: np.ndarray, grades: np.ndarray,
                           n_quantiles: int = 31) -> Tuple[float, float]:
    """Fit monotone three-class thresholds on validation by macro F1."""
    from sklearn.metrics import f1_score

    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, n_quantiles)))
    best = (-1.0, float(candidates[-1]), float(candidates[0]))
    for low_index, low in enumerate(candidates[:-1]):
        for high in candidates[low_index + 1:]:
            pred = np.where(scores >= high, 2, np.where(scores >= low, 1, 0))
            value = float(f1_score(grades, pred, labels=[0, 1, 2],
                                   average="macro", zero_division=0))
            if value > best[0]:
                best = (value, float(high), float(low))
    return best[1], best[2]


def ordinal_classification(scores: np.ndarray, grades: np.ndarray,
                           thresholds: Tuple[float, float]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    high, low = thresholds
    pred = np.where(scores >= high, 2, np.where(scores >= low, 1, 0))
    return {
        "thresholds_from_validation": {"high": high, "low": low},
        "accuracy": float(accuracy_score(grades, pred)),
        "macro_f1": float(f1_score(grades, pred, labels=[0, 1, 2],
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(grades, pred, labels=[0, 1, 2],
                                      average="weighted", zero_division=0)),
        "ordinal_mae": float(np.mean(np.abs(grades - pred))),
    }


def evaluate_score(scores: np.ndarray, prepared: Dict[str, Any],
                   binary_threshold: Optional[float] = None,
                   ordinal_thresholds: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    grades, queries = prepared["grades"], prepared["queries"]
    out = {
        "contrasts": contrast_metrics(scores, grades, queries),
        "ranking": ranking_metrics(scores, grades, queries),
        "ordinal": ordinal_metrics(scores, grades),
    }
    if binary_threshold is not None:
        out["binary_classification"] = binary_classification(
            scores, grades, binary_threshold)
    if ordinal_thresholds is not None:
        out["ordinal_classification"] = ordinal_classification(
            scores, grades, ordinal_thresholds)
    return out


def selection_value(metrics: Dict[str, Any], metric: str) -> float:
    if metric == "pooled_auc":
        return float(metrics["contrasts"]["global_auc"]["pooled"])
    if metric == "hard_auc":
        return float(metrics["contrasts"]["global_auc"]["hard"])
    if metric == "ndcg@10":
        return float(metrics["ranking"]["macro"]["ndcg@10"])
    raise ValueError(metric)


def validation_selection_value(scores: np.ndarray, prepared: Dict[str, Any],
                               metric: str) -> float:
    """Compute only the validation metric needed for fusion-weight selection."""
    if metric == "pooled_auc":
        grades = prepared["grades"]
        return rank_auc(scores[grades == 2], scores[grades != 2])
    if metric == "hard_auc":
        grades = prepared["grades"]
        return rank_auc(scores[grades == 2], scores[grades == 1])
    if metric == "ndcg@10":
        return float(ranking_metrics(
            scores, prepared["grades"], prepared["queries"])["macro"]["ndcg@10"])
    raise ValueError(metric)


def checkpoint_grid(spec: Dict[str, Any], results_dir: Path,
                    fractions: Sequence[int], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for fraction in fractions:
        for seed in seeds:
            run = results_dir / spec["run_pattern"].format(
                fraction=fraction, seed=seed)
            rows.append({
                "fraction": fraction,
                "seed": seed,
                "run_dir": run,
                "checkpoint": run / "best_checkpoint.pt",
                "config": run / "training_config.json",
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results" / "lc_single_factor")
    parser.add_argument("--fractions", nargs="+", type=int,
                        default=[10, 25, 50, 75, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 21])
    parser.add_argument("--weights", nargs="+", type=float,
                        default=[i / 10 for i in range(11)])
    parser.add_argument("--select-on", choices=["pooled_auc", "hard_auc", "ndcg@10"],
                        default="pooled_auc")
    parser.add_argument("--max-records", type=int, default=0,
                        help="0 evaluates the full split; positive values are smoke tests")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prepared-cache-dir", type=Path, default=None,
                        help="directory for resumable frozen evaluation embeddings")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="list the checkpoint grid without encoding or scoring")
    args = parser.parse_args(argv)

    from contrastive_learning.data_structures import TrainingConfig

    spec = DOMAINS[args.domain]
    grid = checkpoint_grid(spec, args.results_dir, args.fractions, args.seeds)
    missing = [row for row in grid if not row["checkpoint"].exists()]
    if args.list:
        for row in grid:
            state = "ready" if row["checkpoint"].exists() else "missing"
            print(f"{args.domain} f{row['fraction']} s{row['seed']}: {state}  "
                  f"{row['checkpoint']}")
        print(f"ready={len(grid)-len(missing)} missing={len(missing)} total={len(grid)}")
        return 0
    if missing and not args.allow_missing:
        names = "\n".join(str(row["checkpoint"]) for row in missing)
        raise SystemExit(f"{len(missing)} checkpoints missing:\n{names}")
    grid = [row for row in grid if row["checkpoint"].exists()]
    if not grid:
        raise SystemExit("no checkpoints available")

    config = TrainingConfig.from_json(str(spec["config"]))
    matcher = build_matcher(spec["ontology"], config)
    frozen_cache = load_frozen_embedding_cache(config)
    output_path = args.output or (ROOT / "results" / "publication_phase1_fixedval"
                                  / f"{args.domain}_fusion_learning_curve.json")
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    prepared_cache_dir = args.prepared_cache_dir
    if prepared_cache_dir is None:
        prepared_cache_dir = output_path.parent / "prepared_embeddings" / args.domain
    elif not prepared_cache_dir.is_absolute():
        prepared_cache_dir = ROOT / prepared_cache_dir
    validation = prepare_split(spec["validation"], config, matcher,
                               spec["query_field"], args.max_records,
                               frozen_cache=frozen_cache,
                               prepared_cache_dir=prepared_cache_dir)
    test = prepare_split(spec["test"], config, matcher,
                         spec["query_field"], args.max_records,
                         frozen_cache=frozen_cache,
                         prepared_cache_dir=prepared_cache_dir)
    ontology_stats = z_stats(validation["ontology"])
    val_ontology_z = z_apply(validation["ontology"], ontology_stats)
    test_ontology_z = z_apply(test["ontology"], ontology_stats)

    output: Dict[str, Any] = {
        "domain": args.domain,
        "fractions": args.fractions,
        "seeds": args.seeds,
        "selection_metric": args.select_on,
        "weights": args.weights,
        "normalization": "fit mean/sd on validation; apply unchanged to test",
        "thresholds": "fit on validation; apply unchanged to test",
        "ranking": "macro per metadata.resume_id query",
        "max_records_per_split": args.max_records,
        "splits": {
            "validation": {"path": str(spec["validation"]),
                           "sha256": file_sha256(spec["validation"]),
                           "records_scored": validation["records_scored"]},
            "test": {"path": str(spec["test"]),
                     "sha256": file_sha256(spec["test"]),
                     "records_scored": test["records_scored"]},
        },
        "grade_names": {str(key): value for key, value in spec["grade_names"].items()},
        "missing_checkpoints": [str(row["checkpoint"]) for row in missing],
        "prepared_embedding_cache": str(prepared_cache_dir),
        "runs": [],
    }

    for index, row in enumerate(grid, 1):
        print(f"[{index}/{len(grid)}] f{row['fraction']} s{row['seed']}", flush=True)
        val_text = text_scores(validation, row["checkpoint"], config)
        test_text = text_scores(test, row["checkpoint"], config)
        text_stats = z_stats(val_text)
        val_text_z = z_apply(val_text, text_stats)
        test_text_z = z_apply(test_text, text_stats)

        validation_curve: Dict[str, float] = {}
        for weight in args.weights:
            fused = (1 - weight) * val_text_z + weight * val_ontology_z
            validation_curve[str(weight)] = validation_selection_value(
                fused, validation, args.select_on)
        selected_weight = max(args.weights,
                              key=lambda weight: validation_curve[str(weight)])

        val_fused = ((1 - selected_weight) * val_text_z
                     + selected_weight * val_ontology_z)
        test_fused = ((1 - selected_weight) * test_text_z
                      + selected_weight * test_ontology_z)
        text_binary_threshold = fit_binary_threshold(val_text_z, validation["grades"])
        fused_binary_threshold = fit_binary_threshold(val_fused, validation["grades"])
        text_ordinal_thresholds = fit_ordinal_thresholds(
            val_text_z, validation["grades"])
        fused_ordinal_thresholds = fit_ordinal_thresholds(
            val_fused, validation["grades"])

        text_result = evaluate_score(
            test_text_z, test, text_binary_threshold, text_ordinal_thresholds)
        fused_result = evaluate_score(
            test_fused, test, fused_binary_threshold, fused_ordinal_thresholds)
        deltas = {
            "global_auc": {
                name: fused_result["contrasts"]["global_auc"][name]
                      - text_result["contrasts"]["global_auc"][name]
                for name in ("hard", "easy", "middle_vs_easy", "pooled")
            },
            "ranking_macro": {
                name: fused_result["ranking"]["macro"][name]
                      - text_result["ranking"]["macro"][name]
                for name in fused_result["ranking"]["macro"]
            },
        }
        output["runs"].append({
            "fraction": row["fraction"],
            "seed": row["seed"],
            "checkpoint": str(row["checkpoint"]),
            "selected_weight": selected_weight,
            "validation_selection_curve": validation_curve,
            "validation_stats": {
                "text": {"mean": text_stats[0], "sd": text_stats[1]},
                "ontology": {"mean": ontology_stats[0], "sd": ontology_stats[1]},
            },
            "test": {"text": text_result, "fusion": fused_result, "delta": deltas},
        })
        print("  w={:.1f}  pooled AUC {:+.4f}  hard AUC {:+.4f}  nDCG@10 {:+.4f}".format(
            selected_weight,
            deltas["global_auc"]["pooled"],
            deltas["global_auc"]["hard"],
            deltas["ranking_macro"]["ndcg@10"]), flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
