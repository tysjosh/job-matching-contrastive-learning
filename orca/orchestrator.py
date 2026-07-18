"""OrcaPhaseOrchestrator: drives the four-phase ORCA training schedule.

Implemented in Milestone 1 (task 2.10). See design section B.7.

The orchestrator wraps an existing OSCAR ``ContrastiveLearningTrainer`` and runs
the fixed four-phase ORCA schedule *without forking the trainer*:

  * **Phase 1 — Ontology preprocessing.** Reuse OSCAR's ESCO/ISCO linking and
    ontology distances (``d_esco``, ``d_isco``, ``d_ot``, coverage). No gradient
    updates are applied (Requirement 8.2).
  * **Phase 2 — Encoder warmup.** Train normal InfoNCE for ``orca_warmup_epochs``
    using the *unchanged* OSCAR loss engine, then freeze a bit-for-bit snapshot
    of the warmup embeddings into a :class:`~orca.warmup_store.WarmupEmbeddingStore`
    (Requirement 8.3).
  * **Phase 3 — Reliability pretraining.** Train ONLY the ``ReliabilityMLP`` on
    the frozen warmup embeddings; the text encoder and projection head stay
    bit-for-bit identical to their post-Phase-2 values (Requirement 8.4).
  * **Phase 4 — Joint ORCA training.** Train the projection head + ``ReliabilityMLP``
    with the text encoder frozen, driving the trainer with the injected
    :class:`~orca.loss_engine.OrcaLossEngine`. At the MVP milestone adaptive
    sampling is OFF, so OSCAR's fixed ontology-bucket negative selection is kept
    (Requirements 8.5, 11.2).

Integration is dependency-inverted: this module imports from
``contrastive_learning`` (allowed), but ``contrastive_learning`` never imports
``orca`` (Requirement 7.6). The orchestrator attaches to the trainer only
through the additive ``set_loss_engine`` / ``set_negative_selector`` seams.

Strict ordering (Requirement 8.1) is enforced by an internal phase counter: a
phase cannot run until its predecessor has completed. Phase 3/4 raise a clear
configuration error if the warmup snapshot is unavailable and
``orca_allow_live_warmup_fallback`` is not set (Requirement 8.6). All randomness
is seeded solely from ``config.training_seed`` (Requirements 10.1, 10.3).
"""

from __future__ import annotations

import logging
import random
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from orca.adaptive_sampler import AdaptiveNegativeSampler
from orca.config import OrcaConfig, OrcaConfigError
from orca.factory import make_loss_engine
from orca.trainer_adapter import OrcaTrainerLossAdapter
from orca.warmup_store import WarmupEmbeddingStore

logger = logging.getLogger(__name__)

# Variants whose design-table row enables adaptive negative sampling (design
# B.9). ``denominator`` and ``external_weight`` explicitly disable it, so even
# with ``orca_adaptive_sampling`` true they retain OSCAR bucket selection
# (Requirement 5.7). The adaptive sampler is only ever injected during Phase 4
# when the active variant is in this set AND ``orca_adaptive_sampling`` is true.
ADAPTIVE_SAMPLING_VARIANTS = frozenset(
    {"no_ontology", "no_future", "no_align", "full"}
)


class Phase(IntEnum):
    """The four ORCA phases, in their fixed execution order (Requirement 8.1)."""

    NONE = 0
    PREPROCESS = 1
    WARMUP = 2
    RELIABILITY = 3
    JOINT = 4


class OrcaPhaseError(OrcaConfigError):
    """Raised when the four-phase schedule is driven out of order or a required
    precondition (e.g. a warmup snapshot) is missing.

    Subclasses :class:`~orca.config.OrcaConfigError` so callers can treat every
    ORCA-configuration/ordering problem uniformly.
    """


class OrcaPhaseOrchestrator:
    """Runs the 4-phase ORCA schedule on top of a ``ContrastiveLearningTrainer``.

    Args:
        config: A ``TrainingConfig`` with ``orca_enabled`` true and the
            ``orca_*`` fields populated. ``training_seed`` must be an integer
            (Requirement 10.3).
        trainer: A constructed OSCAR ``ContrastiveLearningTrainer``. The
            orchestrator reuses its encoder (``text_encoder``), projection head
            (``model``), optimizer, embedding cache, and per-epoch training loop.
        skill_matcher: Optional OSCAR ``OntologySkillMatcher`` forwarded to the
            loss-engine factory. Defaults to the trainer's batch-processor skill
            matcher when available.

    Raises:
        OrcaConfigError: If ORCA is not enabled, if ``training_seed`` is missing
            or non-integer, or if the MVP-ordering guard in the factory rejects
            the configuration.
    """

    def __init__(self, config, trainer, skill_matcher=None):
        if not getattr(config, "orca_enabled", False):
            raise OrcaConfigError(
                "OrcaPhaseOrchestrator requires orca_enabled=True; the four-phase "
                "schedule must not run on the byte-identical career path."
            )

        self.config = config
        self.orca_cfg = OrcaConfig.from_training_config(config)
        self.trainer = trainer

        # Validate the seed up front so an invalid seed fails before any phase
        # runs and before any RNG is touched (Requirement 10.3).
        self._validate_seed()

        # Resolve the skill matcher (prefer the caller's, else the trainer's).
        if skill_matcher is None:
            batch_processor = getattr(trainer, "batch_processor", None)
            skill_matcher = getattr(batch_processor, "skill_matcher", None)

        # Build the ORCA loss engine via the factory. The factory enforces the
        # MVP-ordering guard (Requirement 11.4) and constructs the ReliabilityMLP
        # + WeakTargetBuilder, so the orchestrator never touches those internals
        # directly. Raises OrcaConfigError for a non-MVP configuration.
        self.loss_engine = make_loss_engine(config, skill_matcher)
        self.reliability_model = getattr(self.loss_engine, "reliability_model", None)

        # Parameter groups (duck-typed; any may be absent in a stubbed trainer).
        self.encoder = getattr(trainer, "text_encoder", None)
        self.projection = getattr(trainer, "model", None)

        # Warmup snapshot lifecycle: captured at the end of Phase 2, then frozen.
        self.warmup_store: Optional[WarmupEmbeddingStore] = None

        # Trainer-compatible loss adapter: bridges the trainer's
        # ``compute_loss(triplets, embeddings)`` call convention to the
        # tensor-level OrcaLossEngine. Built lazily on first injection (Phase 3),
        # reused for Phase 4. None on the byte-identical career path.
        self.loss_adapter: Optional[OrcaTrainerLossAdapter] = None

        # Adaptive negative sampler: constructed and injected only during Phase 4
        # when the active variant enables adaptive sampling (task 4.3). Stays
        # None on the ORCA-Denominator MVP path so OSCAR bucket selection is
        # retained (Requirements 5.7, 11.2).
        self.adaptive_sampler: Optional[AdaptiveNegativeSampler] = None

        # Strict-ordering state: highest phase that has completed so far.
        self._completed: Phase = Phase.NONE

        logger.info(
            "OrcaPhaseOrchestrator initialized (variant=%s, warmup=%d, "
            "reliability=%d, joint=%d, allow_live_warmup_fallback=%s)",
            self.orca_cfg.variant, self.orca_cfg.warmup_epochs,
            self.orca_cfg.reliability_epochs, self.orca_cfg.joint_epochs,
            self.orca_cfg.allow_live_warmup_fallback,
        )

    # ============================================================ public entry
    def run(self, dataset_path: Union[str, Path]):
        """Execute Phases 1 → 2 → 3 → 4 in strict order (Requirement 8.1).

        Seeds every RNG from ``training_seed`` first (Requirements 10.1, 10.2),
        then runs each phase; a phase never begins before its predecessor has
        completed. Returns the captured :class:`WarmupEmbeddingStore` so callers
        can inspect the frozen warmup embeddings.
        """
        self.seed_everything()

        self.phase1_preprocess(dataset_path)
        self.phase2_warmup(dataset_path)
        self.phase3_reliability_pretraining(dataset_path)
        self.phase4_joint_training(dataset_path)

        logger.info("ORCA four-phase schedule complete.")
        return self.warmup_store

    @property
    def current_phase(self) -> Phase:
        """The most recently completed phase (``Phase.NONE`` before any run)."""
        return self._completed

    # =============================================================== seeding
    def _validate_seed(self) -> int:
        """Return the integer ``training_seed`` or raise (Requirement 10.3).

        ``bool`` is rejected explicitly: although ``bool`` is a subclass of
        ``int`` in Python, a boolean seed is almost certainly a misconfiguration.
        """
        seed = getattr(self.config, "training_seed", None)
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise OrcaConfigError(
                "ORCA reproducibility requires an integer config.training_seed; "
                f"got {seed!r}. Set a valid training_seed before training."
            )
        return seed

    def seed_everything(self) -> int:
        """Seed Python, NumPy, and torch RNGs solely from ``training_seed``.

        Draws from no unseeded / time-based source (Requirement 10.1) so two runs
        with identical config, seed, and inputs are bit-for-bit reproducible
        (Requirement 10.2). Returns the seed used.
        """
        seed = self._validate_seed()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("ORCA seeded all RNG from training_seed=%d", seed)
        return seed

    # ============================================================ phase 1
    def phase1_preprocess(self, dataset_path: Union[str, Path]) -> None:
        """Phase 1: reuse OSCAR ontology preprocessing; apply NO gradient updates.

        OSCAR's skill matcher / career graph already produce ESCO links, ISCO
        codes, and the ontology distances lazily as batches are processed; ORCA
        reuses them unchanged (Requirement 9.1). This phase therefore performs no
        training and mutates no model parameters (Requirement 8.2); it only marks
        the phase complete so the strict-ordering guard admits Phase 2.
        """
        logger.info("ORCA Phase 1: reusing OSCAR ontology preprocessing "
                    "(no gradient updates).")
        self._completed = Phase.PREPROCESS

    # ============================================================ phase 2
    def phase2_warmup(self, dataset_path: Union[str, Path]) -> WarmupEmbeddingStore:
        """Phase 2: encoder warmup then capture the frozen warmup snapshot.

        Runs ``orca_warmup_epochs`` epochs of the *unchanged* OSCAR InfoNCE path
        (the ORCA loss engine is NOT injected during warmup), then populates the
        embedding cache over every training-set item and snapshots it into a
        read-only :class:`WarmupEmbeddingStore` that stays bit-for-bit unchanged
        for the rest of the run (Requirement 8.3).
        """
        self._require_completed(Phase.PREPROCESS, Phase.WARMUP)

        logger.info("ORCA Phase 2: encoder warmup for %d epoch(s) "
                    "(normal InfoNCE).", self.orca_cfg.warmup_epochs)
        self._execute_epochs(dataset_path, self.orca_cfg.warmup_epochs, "warmup")

        # Ensure the cache covers every training-set item before snapshotting.
        # ``_train_epoch`` may clear the cache between epochs, so repopulate it
        # (a no-op read from disk cache when present) right before the snapshot.
        self._ensure_embeddings_cached(dataset_path)
        self.warmup_store = WarmupEmbeddingStore.snapshot(self.trainer.embedding_cache)
        logger.info("ORCA Phase 2: captured warmup snapshot of %d embeddings.",
                    len(self.warmup_store))

        self._completed = Phase.WARMUP
        return self.warmup_store

    # ============================================================ phase 3
    def phase3_reliability_pretraining(self, dataset_path: Union[str, Path]) -> None:
        """Phase 3: train ONLY the ReliabilityMLP on frozen warmup embeddings.

        Freezes the text encoder and projection head so their parameters stay
        bit-for-bit identical to their post-Phase-2 values (Requirement 8.4) and
        unfreezes only the ReliabilityMLP. Raises if the warmup snapshot is
        unavailable and ``orca_allow_live_warmup_fallback`` is not set
        (Requirement 8.6).
        """
        self._require_completed(Phase.WARMUP, Phase.RELIABILITY)
        self._require_warmup_available(Phase.RELIABILITY)

        # Encoder + projection frozen; reliability model is the only trainable set.
        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.projection, False)
        self._set_requires_grad(self.reliability_model, True)
        self._move_reliability_to_device()
        self._rebuild_optimizer(self._reliability_parameters())

        # Inject the ORCA loss adapter so the batches actually drive the
        # ReliabilityMLP through the reliability-calibrated loss (the projection
        # is frozen, so only the ReliabilityMLP receives gradients). Without this
        # the trainer would run OSCAR's InfoNCE, which never references the
        # ReliabilityMLP and would train nothing this phase.
        self._inject_orca_loss()

        logger.info("ORCA Phase 3: reliability pretraining for %d epoch(s) "
                    "(only ReliabilityMLP trainable, frozen warmup embeddings, "
                    "OrcaLossEngine active).",
                    self.orca_cfg.reliability_epochs)
        self._execute_epochs(dataset_path, self.orca_cfg.reliability_epochs,
                             "reliability")

        self._completed = Phase.RELIABILITY

    # ============================================================ phase 4
    def phase4_joint_training(self, dataset_path: Union[str, Path]) -> None:
        """Phase 4: joint ORCA training of the projection head + ReliabilityMLP.

        Keeps the text encoder frozen (Requirement 8.5), unfreezes the projection
        head and ReliabilityMLP, injects the :class:`OrcaLossEngine` through the
        trainer's ``set_loss_engine`` seam, and trains for ``orca_joint_epochs``.
        At the MVP milestone adaptive sampling is OFF, so no negative selector is
        injected and OSCAR's fixed ontology-bucket selection is retained
        (Requirements 8.5, 11.2). Raises if the warmup snapshot is unavailable and
        the live-fallback flag is not set (Requirement 8.6).
        """
        self._require_completed(Phase.RELIABILITY, Phase.JOINT)
        self._require_warmup_available(Phase.JOINT)

        # Encoder frozen; projection head + reliability model trainable.
        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.projection, True)
        self._set_requires_grad(self.reliability_model, True)
        self._move_reliability_to_device()
        self._rebuild_optimizer(
            self._projection_parameters() + self._reliability_parameters()
        )

        # Inject the ORCA loss adapter via the additive seam. The adapter wraps
        # the tensor-level OrcaLossEngine so the trainer's
        # ``compute_loss(triplets, embeddings)`` call drives the
        # reliability-calibrated denominator (projection head + ReliabilityMLP
        # both trainable this phase).
        self._inject_orca_loss()

        # Wire the negative selector for Phase 4 (task 4.3). When the active
        # variant enables adaptive sampling, inject the deterministic
        # AdaptiveNegativeSampler through the trainer's ``set_negative_selector``
        # seam; the per-epoch curriculum (``gamma_s`` / ``random_mix``) is then
        # applied in :meth:`_set_epoch` before each epoch. Otherwise (the MVP
        # ORCA-Denominator path, and ORCA-ExternalWeight) no selector is
        # injected and OSCAR's fixed ontology-bucket selection is retained
        # (Requirements 5.7, 8.5, 11.2).
        self._configure_phase4_negative_selector()

        logger.info("ORCA Phase 4: joint training for %d epoch(s) "
                    "(projection + ReliabilityMLP trainable, encoder frozen, "
                    "OrcaLossEngine active, %s).",
                    self.orca_cfg.joint_epochs,
                    "adaptive negative sampling"
                    if self.adaptive_sampler is not None
                    else "OSCAR bucket sampling")
        self._execute_epochs(dataset_path, self.orca_cfg.joint_epochs, "joint")

        self._completed = Phase.JOINT

    # ================================================ phase-4 negative selector
    def _adaptive_sampling_active(self) -> bool:
        """Whether Phase 4 should drive the AdaptiveNegativeSampler.

        True only when the master switch ``orca_adaptive_sampling`` is set AND
        the active variant's design-table row enables adaptive sampling AND a
        ReliabilityMLP is available to score candidates. ``denominator`` and
        ``external_weight`` are excluded even with the switch on (Requirement
        5.7), so those variants keep OSCAR bucket selection.
        """
        return (
            self.orca_cfg.adaptive_sampling
            and self.orca_cfg.variant in ADAPTIVE_SAMPLING_VARIANTS
            and self.reliability_model is not None
        )

    def _configure_phase4_negative_selector(self) -> None:
        """Inject or clear the Phase-4 negative selector (task 4.3).

        Constructs and injects an :class:`AdaptiveNegativeSampler` when adaptive
        sampling is active; otherwise clears any selector so OSCAR bucket
        selection is used (Requirements 5.7, 8.5). Injection is dependency
        inverted through the trainer's ``set_negative_selector`` seam, so this
        module remains the only side that references ``orca`` (Requirement 7.6).
        """
        set_selector = getattr(self.trainer, "set_negative_selector", None)

        if self._adaptive_sampling_active():
            self.adaptive_sampler = AdaptiveNegativeSampler(
                self.config, self.reliability_model)
            if not callable(set_selector):
                raise OrcaPhaseError(
                    "ORCA joint: adaptive sampling is enabled but the trainer "
                    "does not expose the set_negative_selector seam; cannot "
                    "inject the AdaptiveNegativeSampler for Phase 4."
                )
            set_selector(self.adaptive_sampler)
            logger.info(
                "ORCA Phase 4: injected AdaptiveNegativeSampler (variant=%s, "
                "gamma_s=%.3f, random_mix=%.3f).",
                self.orca_cfg.variant, self.orca_cfg.gamma_s,
                self.orca_cfg.random_mix)
        else:
            # Fall back to OSCAR bucket selection: ensure no stale selector is
            # left injected (a no-op when none was ever set).
            self.adaptive_sampler = None
            if callable(set_selector):
                set_selector(None)
            logger.info(
                "ORCA Phase 4: adaptive sampling off for variant=%s; retaining "
                "OSCAR bucket selection.", self.orca_cfg.variant)

    def _curriculum_schedule(self, epoch: int) -> tuple:
        """Return the ``(gamma_s, random_mix)`` curriculum for a joint epoch.

        Design B.5: early epochs explore broadly (``gamma_s=0.5`` with a high
        ``random_mix``) and later epochs focus on uncertain negatives (higher
        ``gamma_s`` with a lower ``random_mix``). The base ``random_mix`` comes
        from the config; it is scaled down as training progresses. The schedule
        is a deterministic function of ``epoch`` and the configured
        ``joint_epochs``, so it stays reproducible under a fixed seed.
        """
        total = max(1, int(self.orca_cfg.joint_epochs))
        frac = epoch / total
        base_mix = self.orca_cfg.random_mix
        if frac < 1.0 / 3.0:
            gamma_s, random_mix = 0.5, base_mix
        elif frac < 2.0 / 3.0:
            gamma_s, random_mix = 1.0, base_mix * 0.5
        else:
            gamma_s, random_mix = 2.0, base_mix * 0.2
        random_mix = min(1.0, max(0.0, random_mix))
        return gamma_s, random_mix

    def _apply_curriculum(self, epoch: int) -> None:
        """Update the adaptive sampler's ``gamma_s`` / ``random_mix`` for ``epoch``.

        A no-op unless an :class:`AdaptiveNegativeSampler` is active (i.e. only
        during Phase 4 with adaptive sampling on). Applied before the epoch is
        propagated to the batch processor so the sampler's curriculum state is
        current for that epoch's selection (design B.5 curriculum).
        """
        if self.adaptive_sampler is None:
            return
        gamma_s, random_mix = self._curriculum_schedule(epoch)
        self.adaptive_sampler.gamma_s = gamma_s
        self.adaptive_sampler.random_mix = random_mix
        logger.debug(
            "ORCA curriculum @ epoch %d: gamma_s=%.3f, random_mix=%.3f",
            epoch, gamma_s, random_mix)

    # ================================================= loss-engine injection
    def _inject_orca_loss(self) -> None:
        """Build (once) and inject the trainer-compatible ORCA loss adapter.

        The trainer calls ``active_loss_engine.compute_loss(triplets,
        embeddings)``; the tensor-level :class:`OrcaLossEngine` cannot be called
        that way. :class:`~orca.trainer_adapter.OrcaTrainerLossAdapter` bridges the
        two — assembling per-triplet ORCA tensors, ontology scalar features, and
        warmup-embedding weak-target inputs — then delegates to the tested engine.

        Built lazily on first use and reused across Phase 3/4; the current warmup
        snapshot is (re)attached each time so Phase 3 and Phase 4 both derive the
        encoder-similarity weak-target signal from the frozen embeddings.
        """
        inject = getattr(self.trainer, "set_loss_engine", None)
        if not callable(inject):
            raise OrcaPhaseError(
                "ORCA: trainer does not expose the set_loss_engine seam; cannot "
                "activate the OrcaLossEngine."
            )

        if self.loss_adapter is None:
            content_key_fn = self._resolve_content_key_fn()
            self.loss_adapter = OrcaTrainerLossAdapter(
                self.loss_engine,
                content_key_fn=content_key_fn,
                warmup_store=self.warmup_store,
                ot_distance_scale=getattr(self.config, "ot_distance_scale", 10.0),
            )
        else:
            # Keep the adapter's warmup snapshot current (it is captured in
            # Phase 2, i.e. after the adapter might first be referenced).
            self.loss_adapter.set_warmup_store(self.warmup_store)

        inject(self.loss_adapter)

    def _resolve_content_key_fn(self):
        """Return the trainer's content-key function (matches the embeddings dict).

        Prefers the embedding cache's ``get_content_key`` (what the trainer uses
        to key the embeddings it passes to the loss engine); falls back to the
        trainer's ``_get_content_key`` shim.
        """
        cache = getattr(self.trainer, "embedding_cache", None)
        fn = getattr(cache, "get_content_key", None)
        if callable(fn):
            return fn
        fn = getattr(self.trainer, "_get_content_key", None)
        if callable(fn):
            return fn
        raise OrcaPhaseError(
            "ORCA: cannot resolve a content-key function from the trainer; "
            "expected trainer.embedding_cache.get_content_key or "
            "trainer._get_content_key."
        )

    # ====================================================== ordering / guards
    def _require_completed(self, predecessor: Phase, target: Phase) -> None:
        """Enforce strict phase ordering (Requirement 8.1).

        Raises :class:`OrcaPhaseError` unless ``predecessor`` (and thus every
        earlier phase) has completed before ``target`` begins.
        """
        if self._completed < predecessor:
            raise OrcaPhaseError(
                f"Cannot begin {target.name} (Phase {int(target)}): its "
                f"predecessor {predecessor.name} (Phase {int(predecessor)}) has "
                f"not completed. Highest completed phase is "
                f"{self._completed.name} (Phase {int(self._completed)}). Phases "
                f"must run strictly 1 -> 2 -> 3 -> 4."
            )

    def _require_warmup_available(self, phase: Phase) -> None:
        """Guard the warmup snapshot for Phase 3/4 (Requirement 8.6).

        If no snapshot has been captured, raise a configuration error unless
        ``orca_allow_live_warmup_fallback`` permits using live cached embeddings;
        in that case log the fallback and continue. Halts without applying any
        gradient updates when it raises.
        """
        if self.warmup_store is not None:
            return
        if self.orca_cfg.allow_live_warmup_fallback:
            logger.warning(
                "ORCA %s: warmup snapshot unavailable; proceeding with live "
                "cached embeddings because orca_allow_live_warmup_fallback=True.",
                phase.name,
            )
            return
        raise OrcaPhaseError(
            f"ORCA {phase.name} (Phase {int(phase)}) requires warmup embeddings, "
            "but none were captured: Phase 2 must precede Phase 3. Run Phase 2 "
            "first, or set orca_allow_live_warmup_fallback=True to fall back to "
            "live cached embeddings. Halting without gradient updates."
        )

    # ====================================================== parameter helpers
    @staticmethod
    def _set_requires_grad(module, flag: bool) -> None:
        """Set ``requires_grad`` on every parameter of ``module`` (if present)."""
        if module is None:
            return
        params = getattr(module, "parameters", None)
        if not callable(params):
            return
        for p in module.parameters():
            p.requires_grad = flag

    def _reliability_parameters(self) -> list:
        """Trainable parameter list for the ReliabilityMLP (empty if absent)."""
        if self.reliability_model is None:
            return []
        return [p for p in self.reliability_model.parameters()]

    def _projection_parameters(self) -> list:
        """Trainable parameter list for the projection head (empty if absent)."""
        if self.projection is None:
            return []
        return [p for p in self.projection.parameters()]

    def _move_reliability_to_device(self) -> None:
        """Move the ReliabilityMLP to the trainer's device, if both exist."""
        device = getattr(self.trainer, "device", None)
        if self.reliability_model is not None and device is not None:
            self.reliability_model.to(device)

    def _rebuild_optimizer(self, params: list) -> None:
        """Rebuild the trainer's optimizer over exactly ``params``.

        Each phase trains a different parameter group (Requirement 8.4/8.5); the
        trainer's default optimizer only covers the projection head, so the
        orchestrator rebuilds it to include the ReliabilityMLP where required.
        Skips silently when there are no trainable params (e.g. a stubbed
        trainer) so phase setup stays inspectable in tests.
        """
        trainable = [p for p in params if getattr(p, "requires_grad", False)]
        if not trainable:
            logger.debug("No trainable parameters to rebuild optimizer over; "
                         "leaving trainer.optimizer unchanged.")
            return
        lr = getattr(self.config, "learning_rate", 1e-3)
        weight_decay = getattr(self.config, "weight_decay", 0.0)
        self.trainer.optimizer = torch.optim.Adam(
            trainable, lr=lr, weight_decay=weight_decay
        )

    # ====================================================== epoch execution
    def _execute_epochs(self, dataset_path: Union[str, Path], num_epochs: int,
                        label: str) -> None:
        """Run ``num_epochs`` training epochs through the trainer's public loop.

        Delegates to ``trainer.train_epoch`` so ORCA reuses OSCAR's batch
        processing, embedding, and optimization machinery unchanged. Guarded so a
        non-positive epoch count is a well-defined no-op.
        """
        if num_epochs is None or num_epochs <= 0:
            logger.info("ORCA %s: 0 epochs requested; nothing to train.", label)
            return
        train_epoch = getattr(self.trainer, "train_epoch", None)
        if not callable(train_epoch):
            raise OrcaPhaseError(
                f"ORCA {label}: trainer does not expose a callable train_epoch; "
                "cannot execute the phase."
            )
        for epoch in range(num_epochs):
            self._set_epoch(epoch)
            train_epoch(dataset_path, epoch=epoch)

    def _set_epoch(self, epoch: int) -> None:
        """Propagate the phase-local epoch to curriculum-aware components.

        Applies the adaptive-sampler curriculum (``gamma_s`` / ``random_mix``)
        for this epoch first (a no-op unless adaptive sampling is active), then
        forwards the epoch to the batch processor via ``set_epoch`` so its
        curriculum-aware negative selection sees the current epoch (design B.5).
        """
        self._apply_curriculum(epoch)
        batch_processor = getattr(self.trainer, "batch_processor", None)
        setter = getattr(batch_processor, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def _ensure_embeddings_cached(self, dataset_path: Union[str, Path]) -> None:
        """Populate the embedding cache over every training-set item.

        The trainer may clear its cache between epochs, so before snapshotting we
        repopulate it via ``preload_dataset_embeddings`` (which itself no-ops when
        a disk cache is present). This guarantees the warmup snapshot covers every
        training-set item (Requirement 8.3).
        """
        preload = getattr(self.trainer, "preload_dataset_embeddings", None)
        if callable(preload):
            preload(dataset_path)
