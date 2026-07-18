"""make_loss_engine: the loss-engine factory that keeps the career path
byte-identical when ``orca_enabled`` is false.

Implemented in Milestone 0 (task 1.3). See design section B.8.

This module is the single integration seam that decides *which* loss engine the
shared trainer uses. Its most important contract is the default branch: when
``orca_enabled`` is false (the default), it returns the unchanged pre-ORCA
``ContrastiveLossEngine`` so the career InfoNCE/ordinal path stays bit-for-bit
identical (Requirement 7.1, Property 7).

Only when ORCA is explicitly enabled does the factory construct the ORCA
components (``ReliabilityMLP`` + ``WeakTargetBuilder``) and return an
``OrcaLossEngine`` (Requirement 7.3). Before doing so it enforces MVP milestone
ordering (Requirement 11.4): enabling a non-MVP variant, adaptive sampling, or
the ontology-alignment loss before those milestones exist raises a clear
configuration error rather than silently running an unimplemented path.

The ORCA-only imports are performed lazily inside the enabled branch on purpose:
the sibling ORCA modules are filled in by parallel milestone tasks and may still
be stubs, so keeping their imports out of module scope guarantees the
byte-identical default branch works regardless of their state.
"""

from contrastive_learning.loss_engine import ContrastiveLossEngine

from orca.config import OrcaConfig


def make_loss_engine(config, skill_matcher=None):
    """Return the loss engine appropriate for ``config``.

    Default path (``orca_enabled`` false) returns the UNCHANGED pre-ORCA
    ``ContrastiveLossEngine`` so the career/ordinal path is byte-identical
    (Requirement 7.1). The ORCA engine is constructed only when ORCA is
    explicitly enabled (Requirement 7.3).

    Args:
        config: A ``TrainingConfig`` (or any object exposing the ``orca_*``
            attributes plus the shared ``projection_dim`` / ``temperature``).
        skill_matcher: Optional OSCAR ``OntologySkillMatcher`` forwarded to the
            chosen engine (used for on-the-fly ontology features / ordinal phi).

    Returns:
        A ``ContrastiveLossEngine`` when ORCA is disabled, otherwise an
        ``OrcaLossEngine``.

    Raises:
        OrcaConfigError: If ``orca_variant`` is unrecognized, or if a non-MVP
            variant, adaptive sampling, or the alignment loss is enabled before
            those milestones exist (MVP-ordering guard, Requirement 11.4).
    """
    # ── Golden path: ORCA off → return the unchanged engine, untouched. ──
    if not getattr(config, "orca_enabled", False):
        return ContrastiveLossEngine(config, skill_matcher)

    # ── ORCA enabled: validate config and construct the appropriate engine /
    #    model configuration for the active variant. ``from_training_config``
    #    rejects an unrecognized ``orca_variant`` (Requirement 6.8); ``validate``
    #    then checks per-variant consistency (the variant's implied flags). All
    #    six recognized variants are now valid — Milestones 2 (adaptive sampling)
    #    and 3 (alignment) are complete — so the blanket MVP gate is no longer
    #    applied here; behavior is resolved per variant from the design B.9
    #    variant table on OrcaConfig. ──
    orca_cfg = OrcaConfig.from_training_config(config)
    orca_cfg.validate()

    # Lazy imports keep the byte-identical default branch working even while
    # these sibling modules are still stubs from parallel milestone tasks.
    from orca.reliability_model import ReliabilityMLP
    from orca.weak_targets import WeakTargetBuilder
    from orca.loss_engine import OrcaLossEngine

    embed_dim = getattr(config, "projection_dim", 128)

    # Feature layout is variant-driven (design B.2 hook): ORCA-NoOntology
    # constructs the ReliabilityMLP with the ontology scalars excluded, so its
    # ``feature_dim`` is the coverage-only (or zero) width (Requirement 6.4).
    # The ``scalar_features`` supplied at call time must match this width.
    feature_dim = orca_cfg.effective_feature_dim()
    reliability = ReliabilityMLP(embed_dim, feature_dim)

    # Weak targets: ORCA-NoFuture forces the history signal off so its weight is
    # redistributed across the remaining present signals via renormalization
    # (Requirement 6.5). Other variants keep the config's ``orca_use_history``.
    weak = WeakTargetBuilder(config)
    if orca_cfg.forces_history_off():
        weak.use_history = False

    return OrcaLossEngine(config, skill_matcher, reliability, weak)
