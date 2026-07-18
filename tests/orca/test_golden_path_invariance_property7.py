#!/usr/bin/env python3
"""Property-based golden-path invariance test for the ORCA loss-engine factory.

Feature: orca, Task 1.4 (optional property test).

Property 7: Golden-path invariance
-----------------------------------
With ``orca_enabled=False`` (the default), the ``make_loss_engine`` factory MUST
return an instance of the pre-ORCA ``ContrastiveLossEngine`` and the loss it
computes on a fixed batch/seed MUST be *bit-for-bit* identical to a directly
constructed ``ContrastiveLossEngine`` on the same batch. This guards ORCA's
hard additivity constraint: the career InfoNCE path stays byte-identical while
ORCA is off (Requirement 7.1).

The property uses Hypothesis to vary the batch contents (number of triplets,
number of negatives per triplet, and every embedding value) under the
configured ``training_seed``. For every generated batch the factory-produced
engine and a directly-instantiated baseline engine must agree bit-for-bit.

Validates: Requirements 7.1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from hypothesis import given, settings, strategies as st

# Make the repo root importable so ``contrastive_learning`` / ``orca`` resolve
# regardless of the directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contrastive_learning.data_structures import ContrastiveTriplet, TrainingConfig  # noqa: E402
from contrastive_learning.loss_engine import ContrastiveLossEngine  # noqa: E402
from orca.factory import make_loss_engine  # noqa: E402

# Small embedding width keeps the generated batches cheap while still
# exercising the full dot-product / exp / log reduction path.
_EMBED_DIM = 8


def _make_config() -> TrainingConfig:
    """A default career config with ORCA off and a fixed reproducibility seed."""
    return TrainingConfig(
        batch_size=32,
        loss_type="infonce",
        temperature=0.1,
        shuffle_data=False,
        training_phase="supervised",
        # orca_enabled defaults to False; be explicit to document intent.
        orca_enabled=False,
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies: a whole batch (triplets + aligned embeddings).
# ---------------------------------------------------------------------------

# Bounded, finite floats keep the exp/log path numerically well-behaved while
# still varying every embedding coordinate Hypothesis draws.
_coord = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def _batches(draw):
    """Draw a list of ``ContrastiveTriplet`` and a matching raw-embedding map.

    Each item (anchor / positive / negative) is given a unique content dict so
    its embedding key is distinct; the raw vectors are returned keyed by a
    stable ``item_id`` that the test later resolves to content keys via the
    engine's own ``_get_content_key`` (identical for both engines).
    """
    n_triplets = draw(st.integers(min_value=1, max_value=4))

    triplets: List[dict] = []
    raw_vectors: Dict[str, List[float]] = {}
    item_counter = 0

    def _new_item() -> Tuple[dict, str]:
        nonlocal item_counter
        item_id = f"item_{item_counter}"
        item_counter += 1
        content = {"text": item_id}
        vec = draw(st.lists(_coord, min_size=_EMBED_DIM, max_size=_EMBED_DIM))
        raw_vectors[item_id] = vec
        return content, item_id

    for _ in range(n_triplets):
        n_neg = draw(st.integers(min_value=1, max_value=5))
        anchor, _ = _new_item()
        positive, _ = _new_item()
        negatives = []
        for _ in range(n_neg):
            neg, _ = _new_item()
            negatives.append(neg)
        triplets.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negatives": negatives,
                "career_distances": [1.0] * n_neg,
            }
        )

    return triplets, raw_vectors


def _build_triplets(spec: List[dict]) -> List[ContrastiveTriplet]:
    return [
        ContrastiveTriplet(
            anchor=t["anchor"],
            positive=t["positive"],
            negatives=t["negatives"],
            career_distances=t["career_distances"],
            view_metadata={},
        )
        for t in spec
    ]


def _build_embeddings(
    engine: ContrastiveLossEngine, spec: List[dict], raw_vectors: Dict[str, List[float]]
) -> Dict[str, torch.Tensor]:
    """Map each item's content key (via the engine) to its L2-normalized vector."""
    embeddings: Dict[str, torch.Tensor] = {}

    def _register(content: dict) -> None:
        item_id = content["text"]
        vec = torch.tensor(raw_vectors[item_id], dtype=torch.float32)
        # L2-normalize to mirror real projected embeddings; guard the zero vector.
        norm = torch.linalg.vector_norm(vec)
        if float(norm) > 0.0:
            vec = vec / norm
        key = engine._get_content_key(content)
        embeddings[key] = vec

    for t in spec:
        _register(t["anchor"])
        _register(t["positive"])
        for neg in t["negatives"]:
            _register(neg)

    return embeddings


# ---------------------------------------------------------------------------
# The property.
# ---------------------------------------------------------------------------


# Feature: orca, Property 7: Golden-path invariance
@settings(max_examples=200, deadline=None)
@given(batch=_batches())
def test_property_7_golden_path_invariance(batch):
    """Factory (ORCA off) returns ContrastiveLossEngine and is bit-for-bit
    identical to a directly-constructed baseline on the same fixed batch/seed."""
    spec, raw_vectors = batch
    config = _make_config()

    # Seed under the configured training_seed for reproducibility of the fixed
    # batch (the InfoNCE reduction is deterministic, but this pins any global RNG
    # either engine might touch during construction).
    torch.manual_seed(config.training_seed)
    factory_engine = make_loss_engine(config)

    # (1) Golden-path contract: the factory returns the UNCHANGED engine class,
    #     never an ORCA engine, while orca_enabled is False.
    assert isinstance(factory_engine, ContrastiveLossEngine)
    assert type(factory_engine) is ContrastiveLossEngine

    torch.manual_seed(config.training_seed)
    baseline_engine = ContrastiveLossEngine(config)

    # Build the identical fixed batch for both engines. Content keys come from
    # each engine's own hashing but are identical because the class is identical.
    factory_triplets = _build_triplets(spec)
    baseline_triplets = _build_triplets(spec)
    factory_embeddings = _build_embeddings(factory_engine, spec, raw_vectors)
    baseline_embeddings = _build_embeddings(baseline_engine, spec, raw_vectors)

    factory_loss = factory_engine.compute_loss(factory_triplets, factory_embeddings)
    baseline_loss = baseline_engine.compute_loss(baseline_triplets, baseline_embeddings)

    # (2) Bit-for-bit equality of the loss on the fixed batch.
    assert torch.equal(factory_loss, baseline_loss), (
        "Golden-path loss diverged from the pre-ORCA baseline while "
        f"orca_enabled=False\nfactory={factory_loss!r}\nbaseline={baseline_loss!r}"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
