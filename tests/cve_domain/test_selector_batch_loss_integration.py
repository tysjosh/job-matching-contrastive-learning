"""Integration test: CVENegativeSelector -> BatchProcessor -> ContrastiveLossEngine (task 7.9).

Feature: cve-vulnerability-ranking
Requirements: 5.3

This test wires the three real components together, exactly as Stage 1 does, and
confirms the two claims of Requirement 5.3:

1. **Selected negatives reach the negative slots that feed the denominator.**
   The tiered negatives chosen by ``CVENegativeSelector`` (Req 5) are routed
   through the additive ``BatchProcessor`` seam (task 7.8,
   ``_select_cve_negatives``) into ``ContrastiveTriplet.negatives`` — the exact
   list the loss engine iterates to build the InfoNCE denominator. We assert the
   negative ``cve`` ids on the produced triplet equal the selector's output for
   that anchor.

2. **The loss math is unchanged.** The negatives flow into the *unmodified*
   ``ContrastiveLossEngine._infonce_loss`` denominator. We prove this two ways:

   * A direct call to ``_infonce_loss`` on tiny tensors reproduces the standard
     InfoNCE value ``-log(exp(pos) / (exp(pos) + sum(exp(neg_i))))`` computed
     independently — every selected negative appears in the denominator, and the
     result matches the reference formula bit-for-bit.
   * A full ``compute_loss`` forward through the real triplet path yields a
     finite, differentiable scalar, and an instrumented (non-mutating) wrapper
     around ``_infonce_loss`` confirms it is called with exactly one negative per
     selected id.

The pipeline is exercised on tiny hand-built tensors so no SentenceTransformer
encoder is required. The loss engine's own ``_infonce_loss`` is never modified —
the wrapper only records how many negatives were passed and delegates to the
original implementation.

Run from the repo root with::

    .venv/bin/python -m pytest tests/cve_domain/test_selector_batch_loss_integration.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import pytest

torch = pytest.importorskip("torch")

from contrastive_learning.batch_processor import BatchProcessor
from contrastive_learning.data_structures import (
    ContrastiveTriplet,
    TrainingConfig,
    TrainingSample,
)
from contrastive_learning.loss_engine import ContrastiveLossEngine
from cve_domain.negative_selector import CVENegativeSelector
from cve_domain.ontology_adapter import CVEOntologyAdapter
from cve_domain.record_adapter import CVERecordAdapter


# ---------------------------------------------------------------------------
# Fixtures: a small CVE world (view records + tiered denominator pools)
# ---------------------------------------------------------------------------

#: Six CVEs form the split. Each anchor's pools reference the *other* five, so
#: negatives are always drawn from genuine, present sibling CVEs (no random
#: fallback and no dummy negative needed).
_CVE_IDS: List[str] = [f"CVE-2024-{n:04d}" for n in range(1, 7)]

_MAX_NEGATIVES = 4
_SEED = 7
_TEMPERATURE = 0.1
_TIER_RATIOS = {"hard": 0.5, "medium": 0.25, "easy": 0.25}


def _view_record(cve: str) -> Dict[str, object]:
    """A minimal CVE_View_Record carrying the encoder view + labels the adapter reads."""
    return {
        "cve": cve,
        "encoder_view": f"{cve}. Synthetic vulnerability profile for {cve}.",
        "nvd_published": "2024-01-01T00:00:00.000",
        "cve_labels": {"in_kev": False, "ransomware": False},
        "ontology": {"cwes": ["CWE-79"], "cpes": [], "vendors": ["acme"]},
    }


def _write_pools(path: Path) -> None:
    """Write a ``cve_denominator_pools.jsonl`` giving every anchor tiered siblings.

    For anchor i, the other five CVEs are split across the three tiers so the
    selector has enough pooled candidates to satisfy ``max_negatives_per_anchor``
    without ever needing the random-sample fallback.
    """
    lines = []
    for idx, anchor in enumerate(_CVE_IDS):
        others = [c for c in _CVE_IDS if c != anchor]
        # Deterministic, non-trivial spread across tiers.
        hard = others[:2]
        medium = others[2:4]
        easy = others[4:]
        lines.append(
            {
                "cve": anchor,
                "hard_negatives": hard,
                "medium_negatives": medium,
                "easy_negatives": easy,
            }
        )
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )


@pytest.fixture
def view_lookup() -> Dict[str, Dict[str, object]]:
    """cve id -> CVE_View_Record, preserving a stable (insertion) order."""
    return {cve: _view_record(cve) for cve in _CVE_IDS}


@pytest.fixture
def selector(tmp_path: Path) -> CVENegativeSelector:
    """A CVENegativeSelector over a temp denominator-pools file (Req 5)."""
    pools_path = tmp_path / "cve_denominator_pools.jsonl"
    _write_pools(pools_path)
    adapter = CVEOntologyAdapter(str(pools_path))
    return CVENegativeSelector(
        ontology_adapter=adapter,
        max_negatives_per_anchor=_MAX_NEGATIVES,
        tier_ratios=_TIER_RATIOS,
        seed=_SEED,
    )


@pytest.fixture
def config() -> TrainingConfig:
    """A CVE-domain contrastive config.

    ``use_pathway_negatives=False`` keeps the career ESCO-graph requirement out
    of the picture; the CVE selector is injected separately. ``loss_type`` stays
    at the default ``"infonce"`` so the standard denominator is exercised.
    """
    return TrainingConfig(
        domain_adapter="cve",
        use_pathway_negatives=False,
        pathway_weight=0.0,
        loss_type="infonce",
        temperature=_TEMPERATURE,
        max_negatives_per_anchor=_MAX_NEGATIVES,
        num_epochs=1,
        batch_size=8,
    )


@pytest.fixture
def batch_processor(
    config: TrainingConfig,
    selector: CVENegativeSelector,
    view_lookup: Dict[str, Dict[str, object]],
) -> BatchProcessor:
    """A BatchProcessor with the CVE tiered negative selector injected (task 7.8)."""
    bp = BatchProcessor(config)
    bp.set_cve_negative_selector(selector, view_lookup, present_ids=set(_CVE_IDS))
    return bp


def _build_batch(config: TrainingConfig, view_lookup: Dict[str, Dict[str, object]]) -> List[TrainingSample]:
    """Build anchor+positive CVE TrainingSamples via the real CVERecordAdapter.

    Each anchor is paired with the *next* CVE as its ontology-related positive
    (the CVEPositiveSelector's role is stubbed here by attaching a ``positive``),
    then mapped to a TrainingSample through the seam adapter.
    """
    adapter = CVERecordAdapter(config)
    samples: List[TrainingSample] = []
    for idx, cve in enumerate(_CVE_IDS):
        positive_cve = _CVE_IDS[(idx + 1) % len(_CVE_IDS)]
        record = dict(view_lookup[cve])
        record["positive"] = dict(view_lookup[positive_cve])
        sample = adapter.build_sample(record, line_number=idx, config=config)
        assert sample is not None, "adapter should produce a sample for a paired anchor"
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# 1. Selected negatives reach the triplet negative slots (Req 5.3, claim 1)
# ---------------------------------------------------------------------------


def test_selected_negatives_reach_triplet_negative_slots(
    batch_processor: BatchProcessor,
    selector: CVENegativeSelector,
    config: TrainingConfig,
    view_lookup: Dict[str, Dict[str, object]],
) -> None:
    """The triplet negatives are exactly the selector's tiered output per anchor."""
    batch = _build_batch(config, view_lookup)
    triplets = batch_processor.process_batch(batch)

    # One triplet per positive anchor sample.
    assert len(triplets) == len(batch)

    split_ids = list(view_lookup.keys())
    present_ids = set(_CVE_IDS)

    # A fresh, identically-seeded selector reproduces the expected selection
    # (deterministic per-anchor RNG), independent of the batch processor's own
    # selector instance / accumulated report state.
    reference_adapter = selector.ontology_adapter
    reference_selector = CVENegativeSelector(
        ontology_adapter=reference_adapter,
        max_negatives_per_anchor=_MAX_NEGATIVES,
        tier_ratios=_TIER_RATIOS,
        seed=_SEED,
    )

    for triplet in triplets:
        anchor_cve = triplet.view_metadata["resume_id"]
        expected = reference_selector.select_negatives(
            anchor_cve, split_ids, present_ids=present_ids
        )

        got = [neg["cve"] for neg in triplet.negatives]

        # No dummy negative should have been needed (pools are well populated).
        assert "dummy_negative" not in got, "unexpected dummy negative fallback"
        # The selected tiered negatives reach the negative slots verbatim.
        assert got == expected, (
            f"anchor {anchor_cve}: triplet negatives {got} != selector output {expected}"
        )
        # Bounded and anchor-excluded invariants carried through the seam.
        assert len(got) <= _MAX_NEGATIVES
        assert anchor_cve not in got
        assert len(got) == len(set(got))
        # Each negative carries a real encoder_view materialized from the lookup.
        for neg in triplet.negatives:
            assert neg["encoder_view"] == view_lookup[neg["cve"]]["encoder_view"]


# ---------------------------------------------------------------------------
# 2a. Negatives populate the unchanged InfoNCE denominator (Req 5.3, claim 2)
# ---------------------------------------------------------------------------


def _unit(*values: float) -> "torch.Tensor":
    """L2-normalized 1-D tensor (embeddings are normalized before the loss)."""
    t = torch.tensor(values, dtype=torch.float32)
    return t / t.norm()


def test_negatives_enter_infonce_denominator_math_unchanged(
    batch_processor: BatchProcessor,
    config: TrainingConfig,
    view_lookup: Dict[str, Dict[str, object]],
) -> None:
    """Feeding the selected negatives into the real ``_infonce_loss`` matches the
    standard InfoNCE formula exactly — the loss math is unchanged and every
    negative is present in the denominator."""
    batch = _build_batch(config, view_lookup)
    triplets = batch_processor.process_batch(batch)
    triplet = triplets[0]
    n_neg = len(triplet.negatives)
    assert n_neg >= 1

    engine = ContrastiveLossEngine(config)

    # Distinct normalized embeddings: one anchor, one positive, one per negative.
    anchor = _unit(1.0, 0.0, 0.0)
    positive = _unit(0.9, 0.3, 0.0)
    negatives = [
        _unit(math.cos(a), math.sin(a), 0.2)
        for a in [0.5 + 0.3 * i for i in range(n_neg)]
    ]

    loss = engine._infonce_loss(anchor, positive, negatives)

    # --- Finite scalar. ---
    assert torch.isfinite(loss).all()

    # --- Independent reference: standard InfoNCE over pos + all N negatives. ---
    temp = config.temperature
    pos_sim = torch.dot(anchor, positive) / temp
    neg_sims = torch.stack([torch.dot(anchor, neg) / temp for neg in negatives])
    denom = torch.exp(pos_sim) + torch.sum(torch.exp(neg_sims))
    expected = -torch.log(torch.exp(pos_sim) / denom)

    # The denominator built from exactly the N selected negatives reproduces the
    # engine's value bit-for-bit -> math unchanged, all negatives included.
    assert torch.allclose(loss, expected, atol=1e-6), (
        f"engine loss {loss.item()} != reference InfoNCE {expected.item()}"
    )


# ---------------------------------------------------------------------------
# 2b. Full compute_loss forward through the real triplet path (Req 5.3, claim 2)
# ---------------------------------------------------------------------------


def test_compute_loss_forward_consumes_all_selected_negatives(
    batch_processor: BatchProcessor,
    config: TrainingConfig,
    view_lookup: Dict[str, Dict[str, object]],
) -> None:
    """A full ``compute_loss`` forward over the produced triplet is finite and
    differentiable, and the unchanged ``_infonce_loss`` is invoked with exactly
    one negative embedding per selected id (the negatives reach the denominator
    through the real routing)."""
    batch = _build_batch(config, view_lookup)
    triplet = batch_processor.process_batch(batch)[0]
    n_neg = len(triplet.negatives)

    engine = ContrastiveLossEngine(config)

    # Build an embeddings dict keyed exactly as the loss engine looks them up.
    embeddings: Dict[str, "torch.Tensor"] = {}
    dim = 8
    torch.manual_seed(0)

    def _put(content: Dict[str, object]) -> None:
        key = engine._get_content_key(content)
        if key not in embeddings:
            vec = torch.randn(dim, requires_grad=True)
            embeddings[key] = vec / vec.norm()

    _put(triplet.anchor)
    _put(triplet.positive)
    for neg in triplet.negatives:
        _put(neg)

    # Instrument (do NOT modify) the denominator function to record how many
    # negatives it receives, then delegate to the real implementation.
    recorded: Dict[str, int] = {}
    original_infonce = engine._infonce_loss

    def _spy(anchor, positive, negs):
        recorded["negative_count"] = len(negs)
        return original_infonce(anchor, positive, negs)

    engine._infonce_loss = _spy  # type: ignore[assignment]

    loss = engine.compute_loss([triplet], embeddings)

    # --- Finite, differentiable scalar loss. ---
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    assert loss.requires_grad

    # --- Every selected negative reached the InfoNCE denominator. ---
    assert recorded.get("negative_count") == n_neg, (
        f"denominator saw {recorded.get('negative_count')} negatives, "
        f"expected {n_neg} selected negatives"
    )
