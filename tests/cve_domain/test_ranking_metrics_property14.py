"""Property-based test for CVE ranking metrics (Property 14).

Component under test: ``CVEEvaluationReporter`` in
``cve_domain/evaluation_reporter.py`` — specifically its ``evaluate()`` ranking
block, which computes NDCG and MAP of the model's ranking against the
ground-truth ``priority_score``, breaking ties by ``cve`` identifier
lexicographic ascending order so the metrics are deterministic (Req 10.1, 10.2).

Property 14 (design.md):

    For any set of test records with ``priority_score`` targets, a ranking in
    perfect ``priority_score`` order yields NDCG = 1.0, MAP lies in ``[0, 1]``, and
    permuting records that share a ``priority_score`` leaves the computed metrics
    unchanged because ties are broken by ``cve`` lexicographic ascending order.

This test exercises three facets of Property 14:

1. **Correctness** — the reporter's NDCG and MAP match an *independent* reference
   computation written here (a genuine oracle, not a mirror of the production
   code), and MAP always lies in ``[0, 1]`` while NDCG lies in ``[0, 1]``.
2. **Perfect ordering** — when the model ranking score equals the ground-truth
   ``priority_score`` for every record, NDCG == 1.0 exactly (even in the presence
   of ties, because both the model and ideal rankings apply the same
   ``cve``-ascending tie-break).
3. **Determinism under ties** — shuffling the input record order (records share
   ``priority_score`` and/or model ``ranking_score`` values by construction)
   leaves the computed ranking metrics byte-for-byte identical.

Records are generated from small shared ``priority_score`` / ``ranking_score``
pools so that ties are deliberately frequent, which is what makes the tie-break
determinism (Req 10.2) meaningfully tested.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_ranking_metrics_property14.py
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cve_domain.evaluation_reporter import CVEEvaluationReporter


# ---------------------------------------------------------------------------
# Generation helpers
#
# Unique cve ids drawn from a fixed universe; priority_score and ranking_score
# each drawn from small pools so that ties (equal priority_score and/or equal
# model ranking_score) occur frequently. Frequent ties are exactly what stress
# the cve-ascending tie-break that Property 14 / Req 10.2 require.
# ---------------------------------------------------------------------------

_ID_UNIVERSE = [f"CVE-2024-{n:04d}" for n in range(40)]
_SCORE_POOL = [0.0, 25.0, 50.0, 75.0, 100.0]
_RANK_POOL = [1.0, 2.0, 3.0, 4.0, 5.0]


@st.composite
def ranking_inputs(draw) -> dict:
    """Draw test_records (cve + priority_score) and predictions (cve -> ranking_score).

    Ties are forced by sampling both the ground-truth priority_score and the
    model ranking_score from small pools.
    """
    cve_ids: List[str] = draw(
        st.lists(st.sampled_from(_ID_UNIVERSE), min_size=2, max_size=16, unique=True)
    )

    records: List[dict] = []
    predictions: Dict[str, dict] = {}
    for cve in cve_ids:
        score = draw(st.sampled_from(_SCORE_POOL))
        rank = draw(st.sampled_from(_RANK_POOL))
        records.append({"cve": cve, "cve_labels": {"priority_score": score}})
        predictions[cve] = {"ranking_score": rank}

    shuffle_seed = draw(st.integers(min_value=0, max_value=1_000_000))
    return {"records": records, "predictions": predictions, "shuffle_seed": shuffle_seed}


# ---------------------------------------------------------------------------
# Independent reference (oracle) — recomputes NDCG / MAP from first principles.
#
# This is deliberately written separately from the production code so the test
# is a genuine oracle. It replicates the *definitions* the reporter documents:
#   * model ranking:  score descending, cve ascending on ties
#   * ideal ranking:  relevance descending, cve ascending on ties
#   * NDCG:           linear gains, log2(rank+1) discount, normalized by ideal
#   * MAP:            binary relevance at the median priority_score threshold,
#                     single-query average precision over the model ranking
# ---------------------------------------------------------------------------

def _reference_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _reference_dcg(rels: Sequence[float], limit: int) -> float:
    total = 0.0
    for i in range(min(limit, len(rels))):
        total += rels[i] / math.log2(i + 2)
    return total


def _reference_ndcg(
    model_rels: Sequence[float], ideal_rels: Sequence[float], k: Optional[int]
) -> float:
    limit = len(model_rels) if k is None else min(k, len(model_rels))
    idcg = _reference_dcg(ideal_rels, limit)
    if idcg <= 0.0:
        return 0.0
    return _reference_dcg(model_rels, limit) / idcg


def _reference_average_precision(binary_rel: Sequence[int]) -> float:
    relevant_seen = 0
    precision_sum = 0.0
    for i, rel in enumerate(binary_rel):
        if rel:
            relevant_seen += 1
            precision_sum += relevant_seen / (i + 1)
    if relevant_seen == 0:
        return 0.0
    return precision_sum / relevant_seen


def _reference_ranking(
    records: Sequence[dict],
    predictions: Dict[str, dict],
    k_values: Sequence[int],
) -> dict:
    """Independently compute the ranking metrics for comparison with the reporter."""
    triples: List[Tuple[str, float, float]] = []
    for rec in records:
        cve = rec["cve"]
        rel = float(rec["cve_labels"]["priority_score"])
        model_score = float(predictions[cve]["ranking_score"])
        triples.append((cve, rel, model_score))

    # Deterministic model ranking (score desc, cve asc) and ideal ranking
    # (relevance desc, cve asc) — the tie-break that makes metrics deterministic.
    model_ranked = sorted(triples, key=lambda t: (-t[2], t[0]))
    ideal_ranked = sorted(triples, key=lambda t: (-t[1], t[0]))
    model_rels = [rel for _, rel, _ in model_ranked]
    ideal_rels = [rel for _, rel, _ in ideal_ranked]

    ndcg_full = _reference_ndcg(model_rels, ideal_rels, k=None)
    ndcg_at_k = {str(k): _reference_ndcg(model_rels, ideal_rels, k=k) for k in k_values}

    threshold = _reference_median([rel for _, rel, _ in triples])
    binary_rel = [1 if rel >= threshold else 0 for _, rel, _ in model_ranked]
    ap = _reference_average_precision(binary_rel)

    return {
        "ndcg": ndcg_full,
        "ndcg_at_k": ndcg_at_k,
        "map": ap,
        "map_relevance_threshold": threshold,
        "num_relevant": sum(binary_rel),
        "num_ranked": len(triples),
    }


# ---------------------------------------------------------------------------
# Property 14
# ---------------------------------------------------------------------------

# Feature: cve-vulnerability-ranking, Property 14: Ranking metrics are correct and deterministic under ties
@settings(max_examples=200, deadline=None)
@given(inputs=ranking_inputs())
def test_ranking_metrics_match_independent_reference(inputs):
    """Property 14 — correctness (Validates: Requirements 10.1, 10.2).

    The reporter's NDCG / MAP match an independent reference computation, NDCG lies
    in [0, 1], and MAP lies in [0, 1].
    """
    records = inputs["records"]
    predictions = inputs["predictions"]

    reporter = CVEEvaluationReporter()
    report = reporter.evaluate(records, predictions)
    ranking = report.ranking

    assert ranking.get("skipped") is False
    expected = _reference_ranking(records, predictions, reporter.ndcg_k_values)

    # (a) NDCG matches the independent oracle and is bounded in [0, 1].
    assert ranking["ndcg"] == pytest.approx(expected["ndcg"], rel=1e-9, abs=1e-12)
    assert 0.0 <= ranking["ndcg"] <= 1.0 + 1e-9
    for k, value in expected["ndcg_at_k"].items():
        assert ranking["ndcg_at_k"][k] == pytest.approx(value, rel=1e-9, abs=1e-12)
        assert 0.0 <= ranking["ndcg_at_k"][k] <= 1.0 + 1e-9

    # (b) MAP matches the independent oracle and lies in [0, 1].
    assert ranking["map"] == pytest.approx(expected["map"], rel=1e-9, abs=1e-12)
    assert 0.0 <= ranking["map"] <= 1.0 + 1e-9

    # Threshold and counts also agree with the reference.
    assert ranking["map_relevance_threshold"] == pytest.approx(
        expected["map_relevance_threshold"], rel=1e-9, abs=1e-12
    )
    assert ranking["num_relevant"] == expected["num_relevant"]
    assert ranking["num_ranked"] == expected["num_ranked"]


# Feature: cve-vulnerability-ranking, Property 14: Ranking metrics are correct and deterministic under ties
@settings(max_examples=200, deadline=None)
@given(inputs=ranking_inputs())
def test_perfect_priority_order_yields_ndcg_one(inputs):
    """Property 14 — perfect ordering (Validates: Requirements 10.1, 10.2).

    When the model ranking score equals the ground-truth priority_score for every
    record, the model ranking coincides with the ideal ranking (both apply the same
    cve-ascending tie-break), so NDCG == 1.0 exactly — even with tied scores.
    """
    records = inputs["records"]
    # Model ranks perfectly: ranking_score == priority_score for every cve.
    predictions = {
        rec["cve"]: {"ranking_score": rec["cve_labels"]["priority_score"]}
        for rec in records
    }

    report = CVEEvaluationReporter().evaluate(records, predictions)
    ranking = report.ranking

    assert ranking.get("skipped") is False

    # NDCG normalizes by the ideal DCG. When every priority_score is 0 the ideal
    # DCG is 0 and NDCG is 0.0 by the reporter's documented convention; perfect
    # ordering yields exactly 1.0 only when at least one relevance is positive.
    all_zero = all(rec["cve_labels"]["priority_score"] <= 0.0 for rec in records)
    expected_ndcg = 0.0 if all_zero else 1.0

    assert ranking["ndcg"] == pytest.approx(expected_ndcg, abs=1e-12)
    for value in ranking["ndcg_at_k"].values():
        assert value == pytest.approx(expected_ndcg, abs=1e-12)
    # MAP is still a valid probability.
    assert 0.0 <= ranking["map"] <= 1.0 + 1e-9


# Feature: cve-vulnerability-ranking, Property 14: Ranking metrics are correct and deterministic under ties
@settings(max_examples=200, deadline=None)
@given(inputs=ranking_inputs())
def test_ranking_metrics_deterministic_under_input_permutation(inputs):
    """Property 14 — determinism under ties (Validates: Requirements 10.1, 10.2).

    Permuting the input record order (records deliberately share priority_score
    and/or ranking_score values) leaves the computed ranking metrics identical,
    because ties are broken by cve lexicographic ascending order rather than by
    input position.
    """
    records = inputs["records"]
    predictions = inputs["predictions"]

    reporter = CVEEvaluationReporter()
    baseline = reporter.evaluate(records, predictions).ranking

    shuffled = list(records)
    random.Random(inputs["shuffle_seed"]).shuffle(shuffled)
    # A different order that still contains exactly the same records.
    permuted = reporter.evaluate(shuffled, predictions).ranking

    assert permuted == baseline, (
        "ranking metrics changed under input permutation; ties are not being "
        "broken deterministically by cve"
    )
