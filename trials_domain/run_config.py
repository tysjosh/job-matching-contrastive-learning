"""Wire the trials domain into a constructed trainer.

One function, :func:`attach_trials_domain`, performs every domain-specific
injection through the additive seams:

  * ``BatchProcessor.set_ontology_matcher`` — the MeSH matcher plus the coarse
    ``branch_distance`` callable, replacing the ESCO matcher and the ISCO
    occupation-code lookup. This is what lets ORCA capture a real five-scalar
    ``d_esco``/``d_isco``/``d_ot``/``s_esco``/``s_isco`` decomposition on this
    domain instead of falling back to the single blended ``career_distances``
    proxy.
  * ``BatchProcessor.set_negative_selector`` — the grade-aware
    :class:`TrialsNegativeSelector`, so each topic's negatives come from its own
    graded qrels.

Nothing here is imported by ``contrastive_learning`` or ``orca``; the dependency
points one way only.

Ordering note
-------------
Call this **after** the trainer is constructed and **before** training starts. On
an ORCA run that means before ``OrcaPhaseOrchestrator.run``, because Phase 2's
warmup embeddings are captured from the batches the selector produces — attaching
later would train the warmup encoder on default negatives and then switch the
distribution underneath the frozen snapshot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from trials_domain.mesh_ontology import MeshIndex, MeshMatcher
from trials_domain.negative_selector import TrialsNegativeSelector
import trials_domain.record_adapter  # noqa: F401  registers the "trials" adapter

logger = logging.getLogger(__name__)


def build_mesh_matcher(config) -> MeshMatcher:
    """Build the MeSH matcher from the ``mesh_*`` config fields."""
    descriptor_path = getattr(
        config, "mesh_descriptor_path",
        "trec-clinical-trials/raw/ontology/desc2021.gz",
    )
    cache_path = getattr(
        config, "mesh_index_cache", "embedding_cache/mesh2021_index.pkl"
    )
    index = MeshIndex.build_or_load(descriptor_path, cache_path)
    return MeshMatcher(
        index,
        alpha=getattr(config, "mesh_alpha", 0.5),
        max_hops=getattr(config, "mesh_max_hops", 12),
        similarity_mode=getattr(config, "mesh_similarity_mode", "hierarchy"),
    )


def attach_trials_domain(
    trainer,
    config,
    split: str = "train",
    matcher: Optional[MeshMatcher] = None,
) -> dict:
    """Inject the MeSH matcher and grade-aware negative selector into ``trainer``.

    Args:
        trainer: A constructed ``ContrastiveLearningTrainer``.
        config: The ``TrainingConfig`` (needs the ``trials_*`` and ``mesh_*`` fields).
        split: Which split's graded pools to load ("train", "validation", "test").
        matcher: Optional prebuilt matcher, to share one index across splits.

    Returns:
        A summary dict describing what was attached, for run-log provenance.

    Raises:
        RuntimeError: If the trainer does not expose the required seams, or if
            ORCA is enabled but the ontology-matcher seam is absent. Failing loudly
            matters here: without the matcher seam the run would silently proceed
            on the ``career_distances`` proxy and the ontology ablations would be
            measuring nothing.
    """
    batch_processor = getattr(trainer, "batch_processor", None)
    if batch_processor is None:
        raise RuntimeError(
            "attach_trials_domain requires a trainer exposing .batch_processor"
        )

    set_matcher = getattr(batch_processor, "set_ontology_matcher", None)
    if not callable(set_matcher):
        raise RuntimeError(
            "BatchProcessor does not expose set_ontology_matcher; the trials "
            "domain cannot supply MeSH features. Without it ORCA would fall back "
            "to the blended career_distances proxy and the d_esco/d_isco "
            "decomposition would be lost."
        )

    matcher = matcher or build_mesh_matcher(config)
    # branch_distance is the d_isco analogue: coarse hierarchy agreement over the
    # two sides' condition descriptors. Passed as the coarse_distance_fn, which
    # takes URI lists rather than ISCO's two scalar occupation codes.
    set_matcher(matcher, coarse_distance_fn=matcher.branch_distance)

    split_dir = Path(getattr(config, "trials_split_dir", "preprocess/trec_ct_splits"))
    converted_dir = Path(
        getattr(config, "trials_converted_dir", "preprocess/trec_ct")
    )
    total_epochs = int(
        getattr(config, "orca_joint_epochs", None)
        or getattr(config, "num_epochs", 10)
    )
    selector = TrialsNegativeSelector.from_split_dir(
        split_dir,
        converted_dir,
        split,
        matcher=matcher,
        total_epochs=total_epochs,
        training_seed=int(getattr(config, "training_seed", 42)),
        start_hard_ratio=float(getattr(config, "trials_start_hard_ratio", 0.2)),
        end_hard_ratio=float(getattr(config, "trials_end_hard_ratio", 0.6)),
    )

    # Attach through the DOMAIN selector slot, not the ORCA seam.
    #
    # The ORCA seam (``set_negative_selector``) is owned by the phase
    # orchestrator: for every non-adaptive variant — including the
    # ORCA-Denominator MVP — ``_configure_phase4_negative_selector`` calls
    # ``set_selector(None)`` at the start of joint training. Attaching there meant
    # the graded pool was silently dropped exactly when it mattered, and Phase 4
    # trained on random in-batch negatives (other topics' eligible trials) with
    # the grade-1 judgments never reaching the loss. The domain slot has its own
    # lifetime and survives that clear.
    set_selector = getattr(batch_processor, "set_domain_negative_selector", None)
    if not callable(set_selector):
        raise RuntimeError(
            "BatchProcessor does not expose set_domain_negative_selector; the "
            "trials domain cannot supply its graded negatives. Without it the "
            "ORCA orchestrator clears any selector installed through the ORCA "
            "seam at the start of Phase 4."
        )
    set_selector(selector)

    summary = {
        "split": split,
        "mesh_descriptors": len(matcher.index),
        "mesh_alpha": matcher.alpha,
        "mesh_max_hops": matcher.max_hops,
        "selector_attached_via": "domain_slot",
        "curriculum": {
            "start_hard_ratio": selector.start_hard_ratio,
            "end_hard_ratio": selector.end_hard_ratio,
            "total_epochs": selector.total_epochs,
        },
        "pools": selector.pool_summary(),
    }
    logger.info("Trials domain attached: %s", summary)
    return summary
