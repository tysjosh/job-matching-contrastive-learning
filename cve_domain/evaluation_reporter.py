"""CVE Evaluation_Reporter — ranking, classification, and separation metrics (Requirement 10).

This module computes CVE-appropriate evaluation metrics on the **test split** and
writes them to ``evaluation_report.json``. It mirrors the career-domain
``contrastive_learning/evaluator.py`` conventions (NDCG / MAP helpers, macro-F1
diagnostics, JSON metric persistence) while implementing the CVE-specific behavior
required by Requirement 10.

What it computes (on a non-empty test split):

* **NDCG and MAP against ``priority_score``** (Req 10.1): the model supplies a
  per-CVE ranking score; NDCG uses the ground-truth ``priority_score`` as graded
  relevance, and MAP uses a binary relevance derived from ``priority_score`` via a
  relevance threshold. Ties are broken by ``cve`` identifier lexicographic ascending
  in *both* the model ranking and the ideal ranking so the metrics are deterministic
  (Req 10.2).
* **Classification metrics (accuracy + macro-averaged F1)** for ``in_kev`` and
  ``priority_band`` (Req 10.3).
* **Embedding-separation diagnostics across ``priority_band`` classes** (Req 10.4),
  analogous to the ordinal three-class separation evaluation in the career domain:
  per-band centroids, intra-class dispersion, inter-centroid distances, and a
  separation ratio.

Skips and reporting:

* If a target label required by a metric is absent from **every** test-split record,
  that metric is skipped and the skip + reason is recorded (Req 10.5).
* If the test split contains no records, all metric computation is skipped and the
  empty-test-split condition is recorded (Req 10.6).
* All computed metrics are written to ``evaluation_report.json`` (Req 10.7).

**No circular evaluation (Req 10.8).** Every reported metric is computed against the
ground-truth supervised labels carried on each ``CVE_View_Record`` — ``priority_score``,
``priority_band``, ``in_kev`` (and ``ransomware`` is a ground-truth label too, though
Req 10.3 only asks for ``in_kev`` and ``priority_band`` classification). These labels
come from NVD / EPSS / CISA, **not** from the ontology. The reporter deliberately does
**not** emit any "did the model retrieve the Positive_Selector's ontology-derived
positive?" metric, because those positives were generated from the ontology rule
(shared CWE / CPE / vendor) used to train Stage 1; scoring the model against them would
grade it on its own training signal. Stage 1's value is judged only indirectly, through
the real-label ranking / classification metrics above.

Design note — inputs. The reporter does **not** run the model. It computes metrics from
model **predictions** (a per-``cve`` mapping of ranking score, ``in_kev`` prediction,
``priority_band`` prediction, and embedding) plus the ground-truth labels read from the
test ``CVE_View_Records``. This keeps the reporter a pure metric function that the
Stage 2 trainer / evaluation driver can call with whatever predictions it produced.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# numpy is only needed for the embedding-separation diagnostics; ranking and
# classification metrics are pure-Python so the reporter still works without it.
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy is normally installed
    NUMPY_AVAILABLE = False

# --- Canonical ground-truth label keys carried on a CVE_View_Record (design "Data Models").
PRIORITY_SCORE_LABEL = "priority_score"
PRIORITY_BAND_LABEL = "priority_band"
IN_KEV_LABEL = "in_kev"
RANSOMWARE_LABEL = "ransomware"

# --- Prediction keys the model driver supplies per cve.
PRED_RANKING_SCORE = "ranking_score"
PRED_IN_KEV = "in_kev"
PRED_PRIORITY_BAND = "priority_band"
PRED_EMBEDDING = "embedding"

# Recognized truthy / falsy tokens for boolean coercion (mirrors data_converter).
_TRUTHY = {"true", "1", "yes", "y", "t"}
_FALSY = {"false", "0", "no", "n", "f", ""}

# Default NDCG cutoffs reported alongside the full-ranking NDCG.
DEFAULT_NDCG_K_VALUES: Tuple[int, ...] = (5, 10)


# --------------------------------------------------------------------------- #
# Report structure
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationReport:
    """Structured evaluation results written to ``evaluation_report.json``.

    Attributes:
        status: ``"ok"`` when metrics were computed, ``"empty_test_split"`` when the
            test split had no records (Req 10.6).
        num_test_records: Number of test-split records considered.
        ranking: NDCG / MAP results, or a skip marker with a reason (Req 10.1/10.2/10.5).
        classification: Accuracy + macro-F1 for ``in_kev`` and ``priority_band``, each
            either a metrics block or a skip marker with a reason (Req 10.3/10.5).
        embedding_separation: Band-separation diagnostics, or a skip marker (Req 10.4/10.5).
        skipped_metrics: Flat map of metric name -> skip reason for every skipped metric.
        notes: Free-form notes, including the explicit no-circular-evaluation statement
            (Req 10.8).
    """

    status: str
    num_test_records: int
    ranking: Dict[str, Any] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)
    embedding_separation: Dict[str, Any] = field(default_factory=dict)
    skipped_metrics: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "num_test_records": self.num_test_records,
            "ranking": self.ranking,
            "classification": self.classification,
            "embedding_separation": self.embedding_separation,
            "skipped_metrics": self.skipped_metrics,
            "notes": self.notes,
        }


class CVEEvaluationReporter:
    """Compute CVE ranking, classification, and embedding-separation metrics (Req 10).

    The reporter is a pure metric function: it takes the test-split
    ``CVE_View_Records`` (which carry the ground-truth ``cve_labels``) and a mapping of
    per-``cve`` model predictions, and returns / writes an :class:`EvaluationReport`.

    Args:
        ndcg_k_values: Cutoffs ``k`` at which to additionally report NDCG@k. The
            full-ranking NDCG is always reported. Defaults to ``(5, 10)``.
        map_relevance_threshold: The ``priority_score`` threshold at or above which a
            CVE counts as *relevant* for MAP's binary relevance. When ``None`` (the
            default), the median ground-truth ``priority_score`` of the test split is
            used, which is deterministic for a fixed test split and always yields at
            least one relevant item. The resolved threshold is recorded in the report.
    """

    def __init__(
        self,
        ndcg_k_values: Optional[Sequence[int]] = None,
        map_relevance_threshold: Optional[float] = None,
    ) -> None:
        self.ndcg_k_values: Tuple[int, ...] = tuple(
            k for k in (ndcg_k_values if ndcg_k_values is not None else DEFAULT_NDCG_K_VALUES) if k > 0
        )
        self.map_relevance_threshold = map_relevance_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        test_records: Sequence[Mapping[str, Any]],
        predictions: Mapping[str, Mapping[str, Any]],
        output_dir: Optional[str] = None,
    ) -> EvaluationReport:
        """Compute all metrics and (optionally) write ``evaluation_report.json``.

        Args:
            test_records: The test-split ``CVE_View_Records``. Each must carry a ``cve``
                identifier and a ``cve_labels`` object with the ground-truth labels.
            predictions: Mapping from ``cve`` identifier to a prediction dict with any of
                the keys ``ranking_score`` (float), ``in_kev`` (bool / probability),
                ``priority_band`` (predicted class string), and ``embedding``
                (sequence of floats).
            output_dir: When given, ``evaluation_report.json`` is written there
                (Req 10.7).

        Returns:
            The :class:`EvaluationReport`.
        """
        records = list(test_records)
        notes = {
            "no_circular_evaluation": (
                "All metrics are computed against ground-truth supervised labels "
                "(priority_score, priority_band, in_kev, ransomware) only. No metric "
                "scores the model against the Positive_Selector's ontology-derived "
                "positive pairs (Req 10.8)."
            )
        }

        # --- Req 10.6: empty test split -> skip everything, record the condition.
        if not records:
            report = EvaluationReport(
                status="empty_test_split",
                num_test_records=0,
                skipped_metrics={
                    "ranking": "empty test split",
                    "classification_in_kev": "empty test split",
                    "classification_priority_band": "empty test split",
                    "embedding_separation": "empty test split",
                },
                notes={**notes, "empty_test_split": "The test split contained no records (Req 10.6)."},
            )
            self._maybe_write(report, output_dir)
            logger.warning("Evaluation skipped: empty test split (Req 10.6).")
            return report

        report = EvaluationReport(status="ok", num_test_records=len(records), notes=notes)

        # --- Req 10.1/10.2: ranking metrics (NDCG + MAP) against priority_score.
        report.ranking = self._evaluate_ranking(records, predictions, report.skipped_metrics)

        # --- Req 10.3: classification metrics for in_kev and priority_band.
        report.classification[IN_KEV_LABEL] = self._evaluate_binary_classification(
            records, predictions, report.skipped_metrics
        )
        report.classification[PRIORITY_BAND_LABEL] = self._evaluate_band_classification(
            records, predictions, report.skipped_metrics
        )
        # priority_band is a deterministic bucketing of priority_score; the softmax
        # band head collapses to the majority class under the ~92% 'watch' skew, so
        # we also report the band derived by bucketing the PREDICTED priority_score
        # at the data's own cut points (an ordinal, non-degenerate band metric).
        report.classification["priority_band_from_score"] = self._evaluate_band_from_score(
            records, predictions, report.skipped_metrics
        )

        # --- Req 10.4: embedding-separation diagnostics across priority_band classes.
        report.embedding_separation = self._evaluate_embedding_separation(
            records, predictions, report.skipped_metrics
        )

        self._maybe_write(report, output_dir)
        logger.info(
            "Evaluation complete on %d test record(s); %d metric(s) skipped.",
            len(records),
            len(report.skipped_metrics),
        )
        return report

    # ------------------------------------------------------------------ #
    # Ranking metrics (Req 10.1, 10.2)
    # ------------------------------------------------------------------ #
    def _evaluate_ranking(
        self,
        records: Sequence[Mapping[str, Any]],
        predictions: Mapping[str, Mapping[str, Any]],
        skipped: Dict[str, str],
    ) -> Dict[str, Any]:
        """NDCG + MAP of the model's ranking against ground-truth priority_score.

        Builds ``(cve, relevance, model_score)`` triples for records that have both a
        ground-truth ``priority_score`` and a model ``ranking_score``. All ordering
        applies a ``cve``-ascending tie-break so the metrics are deterministic (Req 10.2).
        """
        # Req 10.5: priority_score absent from every record -> skip ranking.
        any_score = any(self._get_label(r, PRIORITY_SCORE_LABEL) is not None for r in records)
        if not any_score:
            reason = "priority_score label absent from every test-split record"
            skipped["ranking"] = reason
            return {"skipped": True, "reason": reason}

        triples: List[Tuple[str, float, float]] = []
        missing_prediction = 0
        for record in records:
            cve = str(record.get("cve", ""))
            relevance = self._to_float(self._get_label(record, PRIORITY_SCORE_LABEL))
            if relevance is None:
                continue  # no ground-truth relevance for this record
            pred = predictions.get(cve) or {}
            model_score = self._to_float(pred.get(PRED_RANKING_SCORE))
            if model_score is None:
                missing_prediction += 1
                continue
            triples.append((cve, relevance, model_score))

        if not triples:
            reason = "no model ranking scores available for records with a priority_score label"
            skipped["ranking"] = reason
            return {"skipped": True, "reason": reason}

        # Deterministic model ranking: score descending, cve ascending on ties (Req 10.2).
        model_ranked = sorted(triples, key=lambda t: (-t[2], t[0]))
        # Deterministic ideal ranking: relevance descending, cve ascending on ties (Req 10.2).
        ideal_ranked = sorted(triples, key=lambda t: (-t[1], t[0]))

        model_rels = [rel for _, rel, _ in model_ranked]
        ideal_rels = [rel for _, rel, _ in ideal_ranked]

        result: Dict[str, Any] = {
            "skipped": False,
            "num_ranked": len(triples),
            "num_missing_prediction": missing_prediction,
            "ndcg": self._ndcg(model_rels, ideal_rels, k=None),
            "ndcg_at_k": {
                str(k): self._ndcg(model_rels, ideal_rels, k=k) for k in self.ndcg_k_values
            },
        }

        # MAP with binary relevance derived from priority_score via a threshold.
        threshold = self.map_relevance_threshold
        if threshold is None:
            threshold = self._median([rel for _, rel, _ in triples])
        binary_rel = [1 if rel >= threshold else 0 for _, rel, _ in model_ranked]
        result["map"] = self._average_precision(binary_rel)
        result["map_relevance_threshold"] = threshold
        result["num_relevant"] = sum(binary_rel)
        return result

    @staticmethod
    def _ndcg(model_rels: Sequence[float], ideal_rels: Sequence[float], k: Optional[int]) -> float:
        """Normalized DCG with linear (graded) relevance gains.

        ``model_rels`` / ``ideal_rels`` are the relevance values in the model's ranked
        order and in the ideal (relevance-sorted) order respectively. When ``k`` is
        given, only the top-``k`` positions contribute. Returns 0.0 when the ideal DCG
        is 0 (e.g. all relevances are 0).
        """
        limit = len(model_rels) if k is None else min(k, len(model_rels))

        def dcg(rels: Sequence[float]) -> float:
            total = 0.0
            for i in range(min(limit, len(rels))):
                total += rels[i] / math.log2(i + 2)  # +2 because log2(1) == 0
            return total

        idcg = dcg(ideal_rels)
        if idcg <= 0.0:
            return 0.0
        return dcg(model_rels) / idcg

    @staticmethod
    def _average_precision(binary_rel_in_rank_order: Sequence[int]) -> float:
        """Average precision for a single ranked list (degenerate single-query MAP).

        ``binary_rel_in_rank_order`` is the 0/1 relevance of each item in the model's
        ranked order. Returns 0.0 when there are no relevant items.
        """
        relevant_seen = 0
        precision_sum = 0.0
        for i, rel in enumerate(binary_rel_in_rank_order):
            if rel:
                relevant_seen += 1
                precision_sum += relevant_seen / (i + 1)
        if relevant_seen == 0:
            return 0.0
        return precision_sum / relevant_seen

    # ------------------------------------------------------------------ #
    # Classification metrics (Req 10.3)
    # ------------------------------------------------------------------ #
    def _evaluate_binary_classification(
        self,
        records: Sequence[Mapping[str, Any]],
        predictions: Mapping[str, Mapping[str, Any]],
        skipped: Dict[str, str],
    ) -> Dict[str, Any]:
        """Accuracy + macro-F1 for the binary ``in_kev`` label (Req 10.3)."""
        metric_key = f"classification_{IN_KEV_LABEL}"
        any_label = any(self._get_label(r, IN_KEV_LABEL) is not None for r in records)
        if not any_label:
            reason = "in_kev label absent from every test-split record"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        y_true: List[Any] = []
        y_pred: List[Any] = []
        missing_prediction = 0
        for record in records:
            gt = self._to_bool(self._get_label(record, IN_KEV_LABEL))
            if gt is None:
                continue
            cve = str(record.get("cve", ""))
            pred = predictions.get(cve) or {}
            predicted = self._coerce_binary_prediction(pred.get(PRED_IN_KEV))
            if predicted is None:
                missing_prediction += 1
                continue
            y_true.append(gt)
            y_pred.append(predicted)

        if not y_true:
            reason = "no in_kev predictions available for records with an in_kev label"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        return {
            "skipped": False,
            "num_evaluated": len(y_true),
            "num_missing_prediction": missing_prediction,
            "accuracy": self._accuracy(y_true, y_pred),
            "macro_f1": self._macro_f1(y_true, y_pred),
        }

    def _evaluate_band_classification(
        self,
        records: Sequence[Mapping[str, Any]],
        predictions: Mapping[str, Mapping[str, Any]],
        skipped: Dict[str, str],
    ) -> Dict[str, Any]:
        """Accuracy + macro-F1 for the multiclass ``priority_band`` label (Req 10.3)."""
        metric_key = f"classification_{PRIORITY_BAND_LABEL}"
        any_label = any(self._get_label(r, PRIORITY_BAND_LABEL) is not None for r in records)
        if not any_label:
            reason = "priority_band label absent from every test-split record"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        y_true: List[str] = []
        y_pred: List[str] = []
        missing_prediction = 0
        for record in records:
            gt = self._get_band(record)
            if gt is None:
                continue
            cve = str(record.get("cve", ""))
            pred = predictions.get(cve) or {}
            predicted = pred.get(PRED_PRIORITY_BAND)
            predicted_str = str(predicted).strip() if predicted is not None else ""
            if not predicted_str:
                missing_prediction += 1
                continue
            y_true.append(gt)
            y_pred.append(predicted_str)

        if not y_true:
            reason = "no priority_band predictions available for records with a priority_band label"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        return {
            "skipped": False,
            "num_evaluated": len(y_true),
            "num_missing_prediction": missing_prediction,
            "accuracy": self._accuracy(y_true, y_pred),
            "macro_f1": self._macro_f1(y_true, y_pred),
        }

    def _evaluate_band_from_score(
        self,
        records: Sequence[Mapping[str, Any]],
        predictions: Mapping[str, Mapping[str, Any]],
        skipped: Dict[str, str],
    ) -> Dict[str, Any]:
        """Ordinal ``priority_band`` derived by bucketing the PREDICTED priority_score.

        ``priority_band`` is a deterministic bucketing of ``priority_score`` (the per-
        band score ranges are contiguous and non-overlapping). The trained multiclass
        band head collapses to the majority class ('watch', ~92%) and yields a
        degenerate macro-F1; bucketing the regression head's predicted score at the
        data's own cut points instead produces a non-degenerate, ordinal prediction
        that is also sensitive to Stage 1 quality. Cut points and the severity order
        are derived from the ground-truth (band, score) pairs so the metric adapts to
        whatever bands are present rather than hard-coding thresholds.
        """
        metric_key = "classification_priority_band_from_score"
        # Ground-truth (band, score) pairs define the severity order + cut points.
        pairs = [
            (band, score)
            for r in records
            for band in [self._get_band(r)]
            for score in [self._to_float(self._get_label(r, PRIORITY_SCORE_LABEL))]
            if band is not None and score is not None
        ]
        if not pairs:
            reason = "priority_band and priority_score are not both present on any test record"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        order, thresholds = self._derive_band_cutpoints(pairs)
        if len(order) < 2:
            reason = "fewer than two priority_band classes present; band-from-score is undefined"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        rank = {b: i for i, b in enumerate(order)}

        def bucket(score: float) -> str:
            idx = 0
            for i, t in enumerate(thresholds):
                if score >= t:
                    idx = i + 1
            return order[idx]

        y_true: List[str] = []
        y_pred: List[str] = []
        abs_score_err: List[float] = []
        missing_prediction = 0
        for record in records:
            gt = self._get_band(record)
            if gt is None or gt not in rank:
                continue
            cve = str(record.get("cve", ""))
            pred = predictions.get(cve) or {}
            pscore = self._to_float(pred.get(PRED_RANKING_SCORE))
            if pscore is None:
                missing_prediction += 1
                continue
            y_true.append(gt)
            y_pred.append(bucket(pscore))
            gt_score = self._to_float(self._get_label(record, PRIORITY_SCORE_LABEL))
            if gt_score is not None:
                abs_score_err.append(abs(pscore - gt_score))

        if not y_true:
            reason = "no predicted priority_score available for records with a priority_band label"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        per_class: Dict[str, Dict[str, float]] = {}
        for c in order:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_class[c] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for t in y_true if t == c),
            }

        return {
            "skipped": False,
            "num_evaluated": len(y_true),
            "num_missing_prediction": missing_prediction,
            "accuracy": self._accuracy(y_true, y_pred),
            "macro_f1": self._macro_f1(y_true, y_pred),
            "adjacent_accuracy": self._adjacent_accuracy(y_true, y_pred, rank),
            "mae_priority_score": (sum(abs_score_err) / len(abs_score_err)) if abs_score_err else None,
            "severity_order": list(order),
            "score_cut_points": list(thresholds),
            "per_class": per_class,
        }

    # ------------------------------------------------------------------ #
    # Embedding separation diagnostics (Req 10.4)
    # ------------------------------------------------------------------ #
    def _evaluate_embedding_separation(
        self,
        records: Sequence[Mapping[str, Any]],
        predictions: Mapping[str, Mapping[str, Any]],
        skipped: Dict[str, str],
    ) -> Dict[str, Any]:
        """Per-band centroid / dispersion / separation diagnostics (Req 10.4).

        Analogous to the ordinal three-class separation evaluation in the career
        domain: it groups the provided embeddings by ground-truth ``priority_band``,
        then reports each band's centroid dispersion (mean intra-class distance), the
        pairwise inter-centroid distances, and a separation ratio (mean inter-centroid
        distance / mean intra-class distance).
        """
        metric_key = "embedding_separation"
        any_band = any(self._get_label(r, PRIORITY_BAND_LABEL) is not None for r in records)
        if not any_band:
            reason = "priority_band label absent from every test-split record"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        if not NUMPY_AVAILABLE:
            reason = "numpy is unavailable; embedding-separation diagnostics require numpy"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        # Group embeddings by ground-truth band (bands sorted for deterministic output).
        band_to_vectors: Dict[str, List[List[float]]] = {}
        missing_embedding = 0
        for record in records:
            band = self._get_band(record)
            if band is None:
                continue
            cve = str(record.get("cve", ""))
            pred = predictions.get(cve) or {}
            embedding = pred.get(PRED_EMBEDDING)
            vector = self._to_vector(embedding)
            if vector is None:
                missing_embedding += 1
                continue
            band_to_vectors.setdefault(band, []).append(vector)

        if len(band_to_vectors) < 2:
            reason = (
                "fewer than two priority_band classes have embeddings available; "
                "band separation is undefined"
            )
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        # Validate consistent embedding dimensionality.
        dims = {len(v) for vectors in band_to_vectors.values() for v in vectors}
        if len(dims) != 1:
            reason = f"inconsistent embedding dimensionality across records: {sorted(dims)}"
            skipped[metric_key] = reason
            return {"skipped": True, "reason": reason}

        bands = sorted(band_to_vectors.keys())
        centroids: Dict[str, "np.ndarray"] = {}
        per_band: Dict[str, Dict[str, Any]] = {}
        intra_distances: List[float] = []
        for band in bands:
            matrix = np.asarray(band_to_vectors[band], dtype=float)
            centroid = matrix.mean(axis=0)
            centroids[band] = centroid
            dists = np.linalg.norm(matrix - centroid, axis=1)
            mean_intra = float(dists.mean())
            per_band[band] = {"count": int(matrix.shape[0]), "mean_intra_class_distance": mean_intra}
            intra_distances.extend(dists.tolist())

        # Pairwise inter-centroid distances (deterministic ordering by band name).
        inter_centroid: Dict[str, float] = {}
        inter_values: List[float] = []
        for i in range(len(bands)):
            for j in range(i + 1, len(bands)):
                d = float(np.linalg.norm(centroids[bands[i]] - centroids[bands[j]]))
                inter_centroid[f"{bands[i]}|{bands[j]}"] = d
                inter_values.append(d)

        mean_intra = float(np.mean(intra_distances)) if intra_distances else 0.0
        mean_inter = float(np.mean(inter_values)) if inter_values else 0.0
        separation_ratio = (mean_inter / mean_intra) if mean_intra > 0 else None

        return {
            "skipped": False,
            "num_bands": len(bands),
            "num_missing_embedding": missing_embedding,
            "per_band": per_band,
            "inter_centroid_distances": inter_centroid,
            "mean_intra_class_distance": mean_intra,
            "mean_inter_centroid_distance": mean_inter,
            "separation_ratio": separation_ratio,
        }

    # ------------------------------------------------------------------ #
    # Classification helpers (pure Python, explicit zero-division handling)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        """Fraction of exact matches. ``y_true`` / ``y_pred`` must be equal length."""
        if not y_true:
            return 0.0
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    @staticmethod
    def _macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        """Macro-averaged F1 over the union of classes present in truth or prediction.

        F1 for a class is defined as 0.0 when precision + recall == 0 (matching the
        common ``zero_division=0`` convention), so the metric is always well-defined.
        """
        classes = sorted({str(c) for c in y_true} | {str(c) for c in y_pred})
        if not classes:
            return 0.0
        f1_scores: List[float] = []
        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if str(t) == cls and str(p) == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if str(t) != cls and str(p) == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if str(t) == cls and str(p) != cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        return sum(f1_scores) / len(f1_scores)

    @staticmethod
    def _derive_band_cutpoints(
        pairs: Sequence["tuple[str, float]"],
    ) -> "tuple[List[str], List[float]]":
        """Derive the band severity order + score cut points from (band, score) pairs.

        Bands are ordered by their minimum observed ground-truth score (ascending
        severity); the cut point between two adjacent bands is the minimum score of
        the higher band (the per-band score ranges are contiguous/non-overlapping, so
        this recovers the definitional threshold, e.g. 45 / 70 / 85).
        """
        scores_by_band: Dict[str, List[float]] = {}
        for band, score in pairs:
            scores_by_band.setdefault(band, []).append(score)
        order = sorted(scores_by_band, key=lambda b: min(scores_by_band[b]))
        thresholds = [min(scores_by_band[order[i + 1]]) for i in range(len(order) - 1)]
        return order, thresholds

    @staticmethod
    def _adjacent_accuracy(
        y_true: Sequence[str], y_pred: Sequence[str], rank: Mapping[str, int]
    ) -> float:
        """Fraction of predictions within one severity rank of the truth."""
        if not y_true:
            return 0.0
        ok = sum(
            1 for t, p in zip(y_true, y_pred)
            if abs(rank.get(t, 0) - rank.get(p, rank.get(t, 0))) <= 1
        )
        return ok / len(y_true)

    # ------------------------------------------------------------------ #
    # Field accessors / coercion helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_label(record: Mapping[str, Any], key: str) -> Any:
        """Return a supervised label from ``cve_labels`` (falling back to top-level).

        Returns ``None`` when the label key is absent. Reads the canonical
        ``cve_labels`` object first (design "Data Models"), then a top-level key for
        robustness.
        """
        labels = record.get("cve_labels")
        if isinstance(labels, Mapping) and key in labels:
            return labels[key]
        return record.get(key)

    @classmethod
    def _get_band(cls, record: Mapping[str, Any]) -> Optional[str]:
        """Return the trimmed ground-truth ``priority_band`` or ``None`` when missing/empty."""
        value = cls._get_label(record, PRIORITY_BAND_LABEL)
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Coerce ``value`` to float, returning ``None`` when missing / non-numeric."""
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(result) or math.isinf(result):
            return None
        return result

    @staticmethod
    def _to_bool(value: Any) -> Optional[bool]:
        """Coerce a ground-truth label to bool via recognized truthy/falsy tokens.

        Returns ``None`` when the value is missing or unrecognized (so the caller can
        treat it as an absent label rather than silently defaulting).
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        token = str(value).strip().lower()
        if token in _TRUTHY:
            return True
        if token in _FALSY:
            return False
        return None

    @staticmethod
    def _coerce_binary_prediction(value: Any) -> Optional[bool]:
        """Coerce a binary prediction (bool, 0/1, or probability) to bool.

        A float / int in a probability-like range is thresholded at 0.5; explicit
        booleans and recognized truthy/falsy tokens pass through. Returns ``None`` when
        the prediction is missing / unusable.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) >= 0.5
        token = str(value).strip().lower()
        if token in _TRUTHY:
            return True
        if token in _FALSY and token != "":
            return False
        return None

    @staticmethod
    def _to_vector(value: Any) -> Optional[List[float]]:
        """Coerce an embedding to a list of floats, or ``None`` when unusable."""
        if value is None:
            return None
        try:
            vector = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
        return vector if vector else None

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        """Deterministic median of ``values`` (returns 0.0 for an empty sequence)."""
        if not values:
            return 0.0
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        if n % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    # ------------------------------------------------------------------ #
    # IO
    # ------------------------------------------------------------------ #
    @staticmethod
    def _maybe_write(report: EvaluationReport, output_dir: Optional[str]) -> None:
        """Write ``evaluation_report.json`` under ``output_dir`` when provided (Req 10.7)."""
        if output_dir is None:
            return
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        logger.info("Evaluation report written to %s", report_path)
