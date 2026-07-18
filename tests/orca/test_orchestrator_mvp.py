#!/usr/bin/env python3
"""Unit tests for MVP orchestration and seeding (task 2.11).

Component under test: ``orca/orchestrator.py`` — :class:`OrcaPhaseOrchestrator`
(the four-phase ORCA schedule) plus its interaction with
:class:`orca.warmup_store.WarmupEmbeddingStore`.

These tests exercise the MVP orchestration contract without loading a real
``ContrastiveLearningTrainer`` (which would require a SentenceTransformer). A
lightweight duck-typed stub trainer stands in, exposing exactly the seams the
orchestrator reaches for: ``text_encoder`` / ``model`` (nn.Module parameter
groups), ``embedding_cache`` (with a ``cache`` dict), ``device``, ``optimizer``,
``set_loss_engine``, ``train_epoch``, ``batch_processor`` (``set_epoch`` +
``skill_matcher``), and ``preload_dataset_embeddings``.

Coverage:
  * Strict phase ordering (Requirement 8.1): out-of-order phases raise
    ``OrcaPhaseError``; a full ``run()`` executes 1 → 2 → 3 → 4.
  * Warmup-snapshot immutability (Requirement 8.3): mutating the trainer's
    embedding cache after Phase 2 does not alter the frozen store.
  * Parameter-freeze correctness per phase (Requirements 8.4, 8.5): Phase 3
    trains only the ReliabilityMLP; Phase 4 trains projection + ReliabilityMLP
    with the encoder frozen.
  * Configuration errors: a missing warmup snapshot in Phase 3/4 without the
    live-fallback flag raises (Requirement 8.6); an invalid/absent
    ``training_seed`` raises ``OrcaConfigError`` (Requirement 10.3);
    ``orca_enabled=False`` raises.

Run from the repo root with::

    .venv/bin/python -m pytest tests/orca/test_orchestrator_mvp.py -v

Requirements: 8.1, 8.6, 10.3, 11.3
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

from contrastive_learning.data_structures import TrainingConfig  # noqa: E402
from orca.config import OrcaConfigError  # noqa: E402
from orca.orchestrator import OrcaPhaseError, OrcaPhaseOrchestrator, Phase  # noqa: E402
from orca.trainer_adapter import OrcaTrainerLossAdapter  # noqa: E402
from orca.warmup_store import WarmupEmbeddingStore  # noqa: E402


# --------------------------------------------------------------------------- #
# Lightweight duck-typed stub trainer (no SentenceTransformer required)
# --------------------------------------------------------------------------- #
class _StubCache:
    """Minimal embedding cache: exposes a ``cache`` dict of key -> tensor.

    Mirrors the real ``EmbeddingCache`` closely enough for the orchestrator,
    which now resolves a content-key function from the cache when it injects the
    ORCA loss adapter (Phase 3/4).
    """

    def __init__(self, entries=None):
        self.cache = dict(entries or {})

    def get_content_key(self, content):
        """Deterministic content key (mirrors EmbeddingCache.get_content_key)."""
        if isinstance(content, dict):
            return content.get("id") or str(sorted(content.items()))
        return str(content)


class _StubBatchProcessor:
    """Exposes the two seams the orchestrator touches: ``set_epoch`` + matcher."""

    def __init__(self):
        self.skill_matcher = None
        self.epochs_seen = []

    def set_epoch(self, epoch):
        self.epochs_seen.append(epoch)


class _StubTrainer:
    """Duck-typed stand-in for ``ContrastiveLearningTrainer``.

    Only the attributes/methods the orchestrator reaches for are implemented.
    ``text_encoder`` and ``model`` are real (tiny) nn.Modules so parameter
    freeze/unfreeze assertions exercise genuine ``requires_grad`` flags.
    """

    def __init__(self, embed_dim=8, cache_entries=None):
        self.text_encoder = nn.Linear(embed_dim, embed_dim)   # frozen encoder stand-in
        self.model = nn.Linear(embed_dim, embed_dim)          # projection head stand-in
        self.embedding_cache = _StubCache(cache_entries)
        self.device = torch.device("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.batch_processor = _StubBatchProcessor()

        # Call recorders for assertions.
        self.train_epoch_calls = []
        self.loss_engine = None
        self.preload_calls = []

    def set_loss_engine(self, engine):
        self.loss_engine = engine

    def train_epoch(self, dataset_path, epoch=0):
        self.train_epoch_calls.append((str(dataset_path), epoch))

    def preload_dataset_embeddings(self, dataset_path):
        self.preload_calls.append(str(dataset_path))


# --------------------------------------------------------------------------- #
# Config / orchestrator factories
# --------------------------------------------------------------------------- #
def _mvp_config(**overrides):
    """Build a valid MVP ``TrainingConfig`` (ORCA-Denominator, seeded).

    Adaptive sampling / alignment / history are OFF (the MVP surface) and epoch
    counts are 1 each to keep the phase loop fast. Overrides let individual
    tests flip a single field (e.g. an invalid ``training_seed``).
    """
    params = dict(
        orca_enabled=True,
        orca_variant="denominator",
        orca_adaptive_sampling=False,
        orca_use_alignment=False,
        orca_use_history=False,
        orca_warmup_epochs=1,
        orca_reliability_epochs=1,
        orca_joint_epochs=1,
        projection_dim=8,
        training_seed=123,
    )
    params.update(overrides)
    return TrainingConfig(**params)


def _make_orchestrator(config=None, trainer=None):
    config = config if config is not None else _mvp_config()
    trainer = trainer if trainer is not None else _StubTrainer(embed_dim=8)
    return OrcaPhaseOrchestrator(config, trainer), config, trainer


# ======================================================= strict phase ordering
def test_phase3_before_phase2_raises():
    """Calling Phase 3 before Phase 2 completes raises (Requirement 8.1)."""
    orch, _, _ = _make_orchestrator()
    with pytest.raises(OrcaPhaseError):
        orch.phase3_reliability_pretraining("data.jsonl")


def test_phase4_before_phase3_raises():
    """Calling Phase 4 before Phase 3 completes raises (Requirement 8.1)."""
    orch, _, _ = _make_orchestrator()
    orch.phase1_preprocess("data.jsonl")
    orch.phase2_warmup("data.jsonl")
    with pytest.raises(OrcaPhaseError):
        orch.phase4_joint_training("data.jsonl")


def test_phase2_before_phase1_raises():
    """Warmup cannot begin before preprocessing completes (Requirement 8.1)."""
    orch, _, _ = _make_orchestrator()
    with pytest.raises(OrcaPhaseError):
        orch.phase2_warmup("data.jsonl")


def test_full_run_executes_phases_in_order():
    """``run()`` drives 1 → 2 → 3 → 4 and lands in the JOINT phase (Req 8.1)."""
    cache = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    trainer = _StubTrainer(embed_dim=8, cache_entries=cache)
    orch, _, _ = _make_orchestrator(trainer=trainer)

    store = orch.run("data.jsonl")

    assert orch.current_phase == Phase.JOINT
    # 1 warmup + 1 reliability + 1 joint epoch each drive train_epoch once.
    assert len(trainer.train_epoch_calls) == 3
    # The ORCA loss is injected as the trainer-compatible adapter, which wraps
    # the tensor-level OrcaLossEngine (the trainer calls compute_loss(triplets,
    # embeddings), a contract the raw engine cannot satisfy).
    assert isinstance(trainer.loss_engine, OrcaTrainerLossAdapter)
    assert trainer.loss_engine.engine is orch.loss_engine
    # run() returns the captured warmup snapshot.
    assert isinstance(store, WarmupEmbeddingStore)
    assert len(store) == 2


# =================================================== warmup-snapshot immutability
def test_warmup_snapshot_immutable_after_cache_mutation():
    """Mutating the cache after Phase 2 leaves the frozen snapshot unchanged.

    Requirement 8.3: the warmup snapshot must stay bit-for-bit unchanged for the
    remainder of the run even as the underlying cache is mutated.
    """
    cache = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    trainer = _StubTrainer(embed_dim=8, cache_entries=cache)
    orch, _, _ = _make_orchestrator(trainer=trainer)

    orch.phase1_preprocess("data.jsonl")
    store = orch.phase2_warmup("data.jsonl")

    original_a = store["a"].clone()
    original_b = store["b"].clone()
    original_keys = store.content_keys()

    # Mutate the source cache every way a live trainer might: in-place edit,
    # reassignment, new key, and key deletion.
    trainer.embedding_cache.cache["a"].add_(100.0)
    trainer.embedding_cache.cache["b"] = torch.tensor([-9.0, -9.0])
    trainer.embedding_cache.cache["c"] = torch.tensor([7.0, 8.0])
    del trainer.embedding_cache.cache["a"]

    # The snapshot is untouched by any of the above.
    assert store.content_keys() == original_keys
    assert torch.equal(store["a"], original_a)
    assert torch.equal(store["b"], original_b)
    assert "c" not in store


# =============================================== parameter-freeze correctness
def _all_requires_grad(module, flag):
    return all(p.requires_grad == flag for p in module.parameters())


def test_phase3_freezes_all_but_reliability_mlp():
    """Phase 3 trains only the ReliabilityMLP; encoder+projection frozen (Req 8.4)."""
    trainer = _StubTrainer(embed_dim=8)
    orch, _, _ = _make_orchestrator(trainer=trainer)

    orch.phase1_preprocess("data.jsonl")
    orch.phase2_warmup("data.jsonl")
    orch.phase3_reliability_pretraining("data.jsonl")

    assert _all_requires_grad(trainer.text_encoder, False)
    assert _all_requires_grad(trainer.model, False)
    assert orch.reliability_model is not None
    assert _all_requires_grad(orch.reliability_model, True)


def test_phase4_trains_projection_and_reliability_encoder_frozen():
    """Phase 4 trains projection + ReliabilityMLP, encoder frozen (Req 8.5)."""
    trainer = _StubTrainer(embed_dim=8)
    orch, _, _ = _make_orchestrator(trainer=trainer)

    orch.phase1_preprocess("data.jsonl")
    orch.phase2_warmup("data.jsonl")
    orch.phase3_reliability_pretraining("data.jsonl")
    orch.phase4_joint_training("data.jsonl")

    assert _all_requires_grad(trainer.text_encoder, False)     # encoder frozen
    assert _all_requires_grad(trainer.model, True)             # projection trainable
    assert _all_requires_grad(orch.reliability_model, True)    # reliability trainable


# ===================================================== configuration errors
def test_missing_warmup_snapshot_in_phase3_raises():
    """Phase 3 without a warmup snapshot (and no fallback) raises (Req 8.6)."""
    trainer = _StubTrainer(embed_dim=8)
    orch, _, _ = _make_orchestrator(trainer=trainer)

    # Force the ordering guard to pass while leaving the snapshot uncaptured, so
    # the warmup-availability guard is what fails.
    orch._completed = Phase.WARMUP
    assert orch.warmup_store is None
    with pytest.raises(OrcaPhaseError):
        orch.phase3_reliability_pretraining("data.jsonl")


def test_missing_warmup_snapshot_in_phase4_raises():
    """Phase 4 without a warmup snapshot (and no fallback) raises (Req 8.6)."""
    trainer = _StubTrainer(embed_dim=8)
    orch, _, _ = _make_orchestrator(trainer=trainer)

    orch._completed = Phase.RELIABILITY
    assert orch.warmup_store is None
    with pytest.raises(OrcaPhaseError):
        orch.phase4_joint_training("data.jsonl")


def test_live_warmup_fallback_allows_missing_snapshot():
    """With the fallback flag set, a missing snapshot does not raise (Req 8.6)."""
    config = _mvp_config(orca_allow_live_warmup_fallback=True)
    trainer = _StubTrainer(embed_dim=8)
    orch, _, _ = _make_orchestrator(config=config, trainer=trainer)

    orch._completed = Phase.WARMUP
    assert orch.warmup_store is None
    # Must not raise: the live-fallback flag permits proceeding.
    orch.phase3_reliability_pretraining("data.jsonl")
    assert orch.current_phase == Phase.RELIABILITY


@pytest.mark.parametrize("bad_seed", [None, "42", 42.0, True])
def test_invalid_or_absent_training_seed_raises(bad_seed):
    """A non-integer / absent ``training_seed`` raises OrcaConfigError (Req 10.3)."""
    config = _mvp_config()
    # Bypass TrainingConfig type expectations by setting the attribute directly.
    config.training_seed = bad_seed
    trainer = _StubTrainer(embed_dim=8)
    with pytest.raises(OrcaConfigError):
        OrcaPhaseOrchestrator(config, trainer)


def test_orca_disabled_raises():
    """Constructing the orchestrator with ORCA off raises (Req 11.3)."""
    config = _mvp_config()
    config.orca_enabled = False
    trainer = _StubTrainer(embed_dim=8)
    with pytest.raises(OrcaConfigError):
        OrcaPhaseOrchestrator(config, trainer)


def test_seed_everything_is_deterministic_and_returns_seed():
    """``seed_everything`` seeds from ``training_seed`` and is reproducible (Req 10.3)."""
    orch, config, _ = _make_orchestrator()

    seed = orch.seed_everything()
    assert seed == config.training_seed
    first = torch.rand(4)

    returned = orch.seed_everything()
    assert returned == config.training_seed
    second = torch.rand(4)

    # Re-seeding from the same training_seed reproduces the RNG stream.
    assert torch.equal(first, second)
