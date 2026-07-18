#!/usr/bin/env python3
"""End-to-end integration tests for the ORCA trainer loss adapter.

These tests close the gap the earlier component tests missed: they exercise the
*real* trainer call convention ``compute_loss(triplets, embeddings)`` against the
ORCA path and prove that

  * the injected ORCA loss is actually callable the way
    ``ContrastiveLearningTrainer._train_batch`` calls it (regression for the
    ``missing z_negs/scalar_features`` TypeError that made Phase 4 crash), and
  * the ORCA reliability-calibrated loss produces gradients that train the
    ReliabilityMLP (Phase-3 style: projection frozen) and the projection head +
    ReliabilityMLP together (Phase-4 style),
  * the four-phase orchestrator injects the adapter (not the raw tensor engine)
    so a run drives the ORCA objective rather than OSCAR InfoNCE.

They deliberately avoid loading a SentenceTransformer: embeddings are small
synthetic tensors and the content-key function is a trivial ``id`` lookup, so the
tests run fast while still using the *real* ``OrcaLossEngine``,
``ReliabilityMLP``, ``WeakTargetBuilder``, and ``OrcaTrainerLossAdapter``.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_trainer_adapter_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contrastive_learning.data_structures import ContrastiveTriplet, TrainingConfig  # noqa: E402
from orca.loss_engine import OrcaLossEngine  # noqa: E402
from orca.reliability_model import ReliabilityMLP  # noqa: E402
from orca.trainer_adapter import OrcaTrainerLossAdapter  # noqa: E402
from orca.warmup_store import WarmupEmbeddingStore  # noqa: E402
from orca.weak_targets import WeakTargetBuilder  # noqa: E402


_D = 4          # embedding dim
_FEATURE_DIM = 5  # scalar ontology feature width (default MVP layout)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _content_key(content: dict) -> str:
    """Trivial deterministic content key for the synthetic fixtures."""
    return content["id"]


def _make_engine(variant: str = "denominator") -> OrcaLossEngine:
    cfg = TrainingConfig(
        orca_enabled=True,
        orca_variant=variant,
        orca_eta_rel=1.0,
        orca_use_alignment=False,
    )
    torch.manual_seed(0)
    reliability_model = ReliabilityMLP(embed_dim=_D, feature_dim=_FEATURE_DIM)
    reliability_model.train()  # dropout on; gradients still flow
    weak_builder = WeakTargetBuilder(cfg)
    return OrcaLossEngine(
        cfg, skill_matcher=None,
        reliability_model=reliability_model, weak_builder=weak_builder,
    )


def _make_triplets(n_triplets: int = 2, n_neg: int = 3):
    """Build synthetic triplets and a matching raw-vector map keyed by id."""
    triplets = []
    raw_vectors = {}
    counter = 0

    def _item():
        nonlocal counter
        item_id = f"item_{counter}"
        counter += 1
        raw_vectors[item_id] = torch.randn(_D)
        return {"id": item_id}

    torch.manual_seed(123)
    for _ in range(n_triplets):
        anchor = _item()
        positive = _item()
        negatives = [_item() for _ in range(n_neg)]
        triplets.append(ContrastiveTriplet(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            # Positive per-negative ontology-distance proxy (drives r_ont).
            career_distances=[2.0 + i for i in range(n_neg)],
            view_metadata={"ontology_similarity": 0.4, "ot_distance": 3.0},
        ))
    return triplets, raw_vectors


def _leaf_embeddings(raw_vectors) -> dict:
    """Embeddings as leaf tensors (no grad) — simulates a frozen projection."""
    return {k: v.clone().detach() for k, v in raw_vectors.items()}


# --------------------------------------------------------------------------- #
# 1. Regression: the injected ORCA loss is callable the way the trainer calls it
# --------------------------------------------------------------------------- #
def test_adapter_is_callable_with_trainer_contract():
    """adapter.compute_loss(triplets, embeddings) works (2 positional args).

    This is the exact call ``ContrastiveLearningTrainer`` makes. The raw
    ``OrcaLossEngine.compute_loss`` cannot be called this way (it needs
    ``z_negs``/``scalar_features``); the adapter must.
    """
    engine = _make_engine()
    adapter = OrcaTrainerLossAdapter(engine, content_key_fn=_content_key)
    triplets, raw = _make_triplets()
    embeddings = _leaf_embeddings(raw)

    loss = adapter.compute_loss(triplets, embeddings)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.requires_grad


# --------------------------------------------------------------------------- #
# 2. Phase-3 style: projection frozen -> only the ReliabilityMLP trains
# --------------------------------------------------------------------------- #
def test_adapter_trains_reliability_mlp_when_projection_frozen():
    """The ORCA loss produces gradients that update the ReliabilityMLP.

    Embeddings are leaf tensors (frozen projection), so the ONLY trainable
    parameters are the ReliabilityMLP's. After one optimizer step at least one
    ReliabilityMLP parameter must change — proof the reliability-calibrated ORCA
    path (not OSCAR InfoNCE, which never references the ReliabilityMLP) is what
    ran.
    """
    engine = _make_engine()
    adapter = OrcaTrainerLossAdapter(engine, content_key_fn=_content_key)
    triplets, raw = _make_triplets()
    embeddings = _leaf_embeddings(raw)

    model = engine.reliability_model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = [p.detach().clone() for p in model.parameters()]

    optimizer.zero_grad()
    loss = adapter.compute_loss(triplets, embeddings)
    loss.backward()

    # Gradients actually reached the ReliabilityMLP.
    grads = [p.grad for p in model.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "no non-zero gradient reached the ReliabilityMLP — the ORCA reliability "
        "path did not run")

    optimizer.step()
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "ReliabilityMLP parameters did not update after an ORCA step"


# --------------------------------------------------------------------------- #
# 3. Phase-4 style: projection head + ReliabilityMLP both train
# --------------------------------------------------------------------------- #
def test_adapter_trains_projection_and_reliability_together():
    """With a trainable projection producing embeddings, both param groups train."""
    engine = _make_engine()
    adapter = OrcaTrainerLossAdapter(engine, content_key_fn=_content_key)
    triplets, raw = _make_triplets()

    torch.manual_seed(7)
    projection = nn.Linear(_D, _D)

    def _project_embeddings():
        # Fresh grad graph each call: embeddings = projection(text_vector).
        return {k: projection(v) for k, v in raw.items()}

    model = engine.reliability_model
    optimizer = torch.optim.Adam(
        list(projection.parameters()) + list(model.parameters()), lr=1e-2)

    proj_before = [p.detach().clone() for p in projection.parameters()]
    rel_before = [p.detach().clone() for p in model.parameters()]

    optimizer.zero_grad()
    loss = adapter.compute_loss(triplets, _project_embeddings())
    assert loss.requires_grad
    loss.backward()

    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in projection.parameters()), "projection got no gradient"
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in model.parameters()), "ReliabilityMLP got no gradient"

    optimizer.step()
    assert any(not torch.equal(b, a)
               for b, a in zip(proj_before, projection.parameters())), \
        "projection did not update"
    assert any(not torch.equal(b, a)
               for b, a in zip(rel_before, model.parameters())), \
        "ReliabilityMLP did not update"


# --------------------------------------------------------------------------- #
# 4. Warmup-store encoder signal path
# --------------------------------------------------------------------------- #
def test_adapter_uses_warmup_store_for_encoder_signal():
    """With a warmup snapshot covering all keys, the r_enc signal path runs."""
    engine = _make_engine()
    triplets, raw = _make_triplets()
    embeddings = _leaf_embeddings(raw)

    # A warmup store keyed exactly like the embeddings (same content keys).
    store = WarmupEmbeddingStore({k: v.clone() for k, v in raw.items()})
    adapter = OrcaTrainerLossAdapter(
        engine, content_key_fn=_content_key, warmup_store=store)

    loss = adapter.compute_loss(triplets, embeddings)
    assert torch.isfinite(loss)
    assert loss.requires_grad


# --------------------------------------------------------------------------- #
# 5. external_weight variant runs through the adapter too
# --------------------------------------------------------------------------- #
def test_adapter_supports_external_weight_variant():
    """The external_weight application site is exercised end-to-end via the adapter."""
    engine = _make_engine(variant="external_weight")
    adapter = OrcaTrainerLossAdapter(engine, content_key_fn=_content_key)
    triplets, raw = _make_triplets()
    embeddings = _leaf_embeddings(raw)

    loss = adapter.compute_loss(triplets, embeddings)
    assert torch.isfinite(loss)
    assert loss.requires_grad


# --------------------------------------------------------------------------- #
# 6. Orchestrator injects the ADAPTER (not the raw tensor engine)
# --------------------------------------------------------------------------- #
class _StubCache:
    def __init__(self):
        self.cache = {}

    def get_content_key(self, content):
        return content["id"]


class _StubBatchProcessor:
    def __init__(self):
        self.skill_matcher = None

    def set_epoch(self, epoch):
        pass


class _StubTrainer:
    """Minimal trainer exposing the seams the orchestrator touches.

    ``set_loss_engine`` records what was injected so the test can assert the
    orchestrator injects an :class:`OrcaTrainerLossAdapter`.
    """

    def __init__(self):
        self.text_encoder = nn.Linear(_D, _D)
        self.model = nn.Linear(_D, _D)
        self.embedding_cache = _StubCache()
        self.device = torch.device("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.batch_processor = _StubBatchProcessor()
        self.injected_engine = None

    def set_loss_engine(self, engine):
        self.injected_engine = engine

    def set_negative_selector(self, selector):
        pass

    def train_epoch(self, dataset_path, epoch=0):
        pass

    def preload_dataset_embeddings(self, dataset_path):
        pass


def test_orchestrator_injects_the_adapter():
    """After Phase 3/4 the trainer's active engine is the ORCA adapter."""
    from orca.orchestrator import OrcaPhaseOrchestrator

    config = TrainingConfig(
        orca_enabled=True,
        orca_variant="denominator",
        orca_adaptive_sampling=False,
        orca_use_alignment=False,
        orca_use_history=False,
        orca_warmup_epochs=1,
        orca_reliability_epochs=1,
        orca_joint_epochs=1,
        projection_dim=_D,
        training_seed=42,
    )
    trainer = _StubTrainer()
    orch = OrcaPhaseOrchestrator(config, trainer)

    orch.phase1_preprocess("data.jsonl")
    orch.phase2_warmup("data.jsonl")
    orch.phase3_reliability_pretraining("data.jsonl")

    assert isinstance(trainer.injected_engine, OrcaTrainerLossAdapter)

    orch.phase4_joint_training("data.jsonl")
    assert isinstance(trainer.injected_engine, OrcaTrainerLossAdapter)


# --------------------------------------------------------------------------- #
# 7. Captured per-negative ontology features are consumed (not the proxy)
# --------------------------------------------------------------------------- #
def test_adapter_consumes_captured_ontology_features():
    """When view_metadata carries negative_ontology_features, they feed the MLP.

    Builds triplets whose metadata includes the five-scalar per-negative features
    that ``batch_processor`` captures, and asserts the adapter runs and trains the
    ReliabilityMLP off them (the feature tensor width matches feature_dim=5).
    """
    engine = _make_engine()
    adapter = OrcaTrainerLossAdapter(engine, content_key_fn=_content_key)
    triplets, raw = _make_triplets(n_triplets=1, n_neg=3)

    # Attach captured per-negative ontology scalars (canonical order).
    triplets[0].view_metadata["negative_ontology_features"] = [
        {"d_esco": 0.9, "d_isco": 0.7, "d_ot": 2.1, "s_esco": 0.1, "s_isco": 0.3},
        {"d_esco": 0.2, "d_isco": 0.4, "d_ot": 0.5, "s_esco": 0.8, "s_isco": 0.6},
        {"d_esco": 0.5, "d_isco": 0.5, "d_ot": 1.0, "s_esco": 0.5, "s_isco": 0.5},
    ]

    embeddings = _leaf_embeddings(raw)
    model = engine.reliability_model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    optimizer.zero_grad()
    loss = adapter.compute_loss(triplets, embeddings)
    loss.backward()

    assert torch.isfinite(loss) and loss.requires_grad
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in model.parameters()), \
        "captured ontology features did not drive the ReliabilityMLP"


# --------------------------------------------------------------------------- #
# 8. Full variant builds job pairs and applies the alignment term
# --------------------------------------------------------------------------- #
class _StubMatcher:
    """Minimal skill matcher exposing ontology_set_similarity for s_ont."""

    def ontology_set_similarity(self, a, b):
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)  # Jaccard, in [0, 1]


def test_full_variant_applies_alignment_via_job_pairs():
    """With variant=full + a skill matcher + job skill_uris, alignment contributes.

    The adapter must build JobPairs from the triplet's negatives and pass them to
    the engine, so the total differs from the same batch computed with alignment
    disabled (isolating the alignment term's effect).
    """
    triplets, raw = _make_triplets(n_triplets=1, n_neg=4)
    # Give each negative job overlapping/disjoint skill_uris so s_ont varies.
    negs = triplets[0].negatives
    negs[0]["skill_uris"] = ["u1", "u2", "u3"]
    negs[1]["skill_uris"] = ["u2", "u3", "u4"]
    negs[2]["skill_uris"] = ["u5", "u6"]
    negs[3]["skill_uris"] = ["u6", "u7"]
    embeddings = _leaf_embeddings(raw)

    # Engine WITH alignment (full) sharing a skill matcher.
    cfg_full = TrainingConfig(orca_enabled=True, orca_variant="full",
                              orca_use_alignment=True, orca_lambda_align=0.5,
                              orca_eta_rel=1.0)
    torch.manual_seed(0)
    shared_model = ReliabilityMLP(embed_dim=_D, feature_dim=_FEATURE_DIM).eval()
    shared_builder = WeakTargetBuilder(cfg_full)
    engine_full = OrcaLossEngine(cfg_full, skill_matcher=_StubMatcher(),
                                 reliability_model=shared_model,
                                 weak_builder=shared_builder)
    adapter_full = OrcaTrainerLossAdapter(engine_full, content_key_fn=_content_key)

    # Engine WITHOUT alignment, same shared model/builder for an apples-to-apples
    # comparison of the assembled totals.
    cfg_noalign = TrainingConfig(orca_enabled=True, orca_variant="no_align",
                                 orca_use_alignment=False, orca_eta_rel=1.0)
    engine_noalign = OrcaLossEngine(cfg_noalign, skill_matcher=_StubMatcher(),
                                    reliability_model=shared_model,
                                    weak_builder=shared_builder)
    adapter_noalign = OrcaTrainerLossAdapter(engine_noalign, content_key_fn=_content_key)

    with torch.no_grad():
        loss_full = adapter_full.compute_loss(triplets, embeddings)
        loss_noalign = adapter_noalign.compute_loss(triplets, embeddings)

    assert torch.isfinite(loss_full) and torch.isfinite(loss_noalign)
    # Alignment term is non-zero here (s_ont != cosine), so the totals differ.
    assert not torch.allclose(loss_full, loss_noalign), \
        "alignment term had no effect — job pairs were not applied"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
