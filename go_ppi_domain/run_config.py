"""Inject the GO matcher and grade-aware negative selector into a trainer.

One function, :func:`attach_go_ppi_domain`, performs every domain-specific
injection through the existing additive seams:

  * ``BatchProcessor.set_ontology_matcher`` -- the :class:`GoMatcher` plus its
    coarse ``branch_distance``, replacing the ESCO matcher and the ISCO scalar
    lookup. This is what lets ORCA capture a real ``d_esco``/``d_isco``/``s_esco``
    decomposition on this domain instead of degrading to the blended
    ``career_distances`` proxy.
  * ``BatchProcessor.set_domain_negative_selector`` -- the grade-aware
    :class:`GoPpiNegativeSelector`, so each anchor's negatives come from its own
    graded pool.

Nothing here is imported by ``contrastive_learning`` or ``orca``; the dependency
points one way only.

Ordering
--------
Call **after** the trainer is constructed and **before** training starts. On an
ORCA run that means before ``OrcaPhaseOrchestrator.run``, because Phase 2's warmup
embeddings are captured from the batches the selector produces -- attaching later
would train the warmup encoder on default negatives and then swap the distribution
out from under the frozen snapshot.

Why the DOMAIN selector slot and not the ORCA one
-------------------------------------------------
``set_negative_selector`` is owned by the phase orchestrator, which calls it with
``None`` at the start of Phase 4 for every non-adaptive variant. A selector
attached there is silently dropped exactly when it matters, and Phase 4 then
trains on random in-batch negatives with the graded grade-1 pairs never reaching
the loss. ``set_domain_negative_selector`` has its own lifetime and survives that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from go_ppi_domain.go_ontology import GoIndex, GoMatcher, GoMultiAspectMatcher
from go_ppi_domain.negative_selector import GoPpiNegativeSelector
import go_ppi_domain.record_adapter  # noqa: F401  registers the "go_ppi" adapter

logger = logging.getLogger(__name__)


def build_go_matcher(config) -> GoMatcher | GoMultiAspectMatcher:
    """Build the GO matcher from the ``go_*`` config fields."""
    obo_path = getattr(config, "go_obo_path", "dataset/go_ppi/bulk/go-basic.obo")
    gaf_path = getattr(config, "go_annotation_path",
                       "dataset/go_ppi/bulk/goa_human.gaf.gz")
    aspect = getattr(config, "go_aspect", "P")
    if aspect not in {"P", "F", "C", "A"}:
        raise ValueError(f"unknown GO aspect: {aspect}")
    mode = getattr(config, "go_similarity_mode", "simgic")
    base_cache = Path(getattr(config, "go_index_cache", "preprocess/go_ppi/go_index_P.pkl"))

    def one(branch: str) -> GoMatcher:
        # Existing configurations point at go_index_P.pkl. Use sibling caches
        # for F and C so a branch run cannot overwrite the BP index.
        cache = base_cache.with_name(f"go_index_{branch}.pkl") if base_cache.name == "go_index_P.pkl" else base_cache
        if aspect == "A" and base_cache.name != "go_index_P.pkl":
            raise ValueError("hybrid GO runs require go_index_P.pkl as the cache base")
        index = GoIndex.build_or_load(obo_path, gaf_path, cache, aspect=branch)
        return GoMatcher(index, alpha=float(getattr(config, "go_alpha", 0.5)),
                         max_hops=int(getattr(config, "go_max_hops", 12)),
                         similarity_mode=mode)

    if aspect == "A":
        return GoMultiAspectMatcher(
            {branch: one(branch) for branch in ("P", "F", "C")},
            weights=getattr(config, "go_aspect_weights", None))
    return one(aspect)


def attach_go_ppi_domain(
    trainer,
    config,
    split: str = "train",
    matcher: Optional[GoMatcher] = None,
) -> dict:
    """Inject the GO matcher and graded negative selector into ``trainer``.

    Args:
        trainer: A constructed ``ContrastiveLearningTrainer``.
        config: The ``TrainingConfig`` (needs the ``go_*`` / ``go_ppi_*`` fields).
        split: Which split's graded pools to load.
        matcher: Optional prebuilt matcher, to share one index across splits.

    Returns:
        A summary dict describing what was attached, for run-log provenance.

    Raises:
        RuntimeError: If the trainer does not expose the required seams. Failing
            loudly matters: without the matcher seam the run proceeds happily on
            the ``career_distances`` proxy and the ontology arm silently becomes
            the baseline while still reporting as an ontology arm.
    """
    batch_processor = getattr(trainer, "batch_processor", None)
    if batch_processor is None:
        raise RuntimeError(
            "attach_go_ppi_domain requires a trainer exposing .batch_processor")

    set_matcher = getattr(batch_processor, "set_ontology_matcher", None)
    if not callable(set_matcher):
        raise RuntimeError(
            "BatchProcessor does not expose set_ontology_matcher; the go_ppi "
            "domain cannot supply GO features. Without it ORCA would fall back to "
            "the blended career_distances proxy and the d_esco/d_isco "
            "decomposition would be lost.")

    matcher = matcher or build_go_matcher(config)
    # branch_distance is the d_isco analogue: coarse agreement over the two sides'
    # shallow GO ancestors. Takes URI lists rather than ISCO's two scalar codes.
    set_matcher(matcher, coarse_distance_fn=matcher.branch_distance)

    split_dir = Path(getattr(config, "go_ppi_split_dir", "preprocess/go_ppi_splits"))
    converted_dir = Path(getattr(config, "go_ppi_converted_dir", "preprocess/go_ppi"))
    total_epochs = int(
        getattr(config, "orca_joint_epochs", None) or getattr(config, "num_epochs", 10))

    selector = GoPpiNegativeSelector.from_split_dir(
        split_dir,
        converted_dir,
        split,
        matcher=matcher,
        total_epochs=total_epochs,
        training_seed=int(getattr(config, "training_seed", 42)),
        start_hard_ratio=float(getattr(config, "go_ppi_start_hard_ratio", 0.2)),
        end_hard_ratio=float(getattr(config, "go_ppi_end_hard_ratio", 0.6)),
        go_tiered=bool(getattr(config, "go_ppi_go_tiered_negatives", False)),
        go_score_cap=int(getattr(config, "go_ppi_go_score_cap", 400)),
        tier_sampling=str(getattr(config, "go_ppi_tier_sampling", "deterministic")),
        tier_window_frac=float(getattr(config, "go_ppi_tier_window_frac", 0.34)),
    )

    set_selector = getattr(batch_processor, "set_domain_negative_selector", None)
    if not callable(set_selector):
        raise RuntimeError(
            "BatchProcessor does not expose set_domain_negative_selector; the "
            "go_ppi domain cannot supply its graded negatives. Without it the ORCA "
            "orchestrator clears any selector installed through the ORCA seam at "
            "the start of Phase 4.")
    set_selector(selector)

    # An arm flagged as ontology-tiered but holding no matcher would train exactly
    # as the baseline while reporting as the ontology arm — the silent-null failure
    # this project has already been bitten by twice.
    if bool(getattr(config, "go_ppi_go_tiered_negatives", False)) and selector.matcher is None:
        raise RuntimeError(
            "go_ppi_go_tiered_negatives=True but the selector received no matcher; "
            "this arm would be identical to the baseline.")

    summary = {
        "split": split,
        "go_terms": len(matcher.index),
        "go_aspect": matcher.index.meta.get("aspect"),
        "go_similarity_mode": matcher.similarity_mode,
        "go_genes_annotated": matcher.index.meta.get("genes_annotated"),
        "go_alpha": matcher.alpha,
        "go_max_hops": matcher.max_hops,
        "go_tiered": bool(selector.go_tiered),
        "tier_sampling": selector.tier_sampling,
        "tier_window_frac": selector.tier_window_frac,
        "selector_attached_via": "domain_slot",
        "curriculum": {
            "start_hard_ratio": selector.start_hard_ratio,
            "end_hard_ratio": selector.end_hard_ratio,
            "total_epochs": selector.total_epochs,
        },
        "pools": selector.pool_summary(),
    }
    logger.info("GO/PPI domain attached: %s", summary)
    return summary
