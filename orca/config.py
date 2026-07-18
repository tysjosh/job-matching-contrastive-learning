"""OrcaConfig: accessor/validation helper over the additive ``orca_*`` fields on
``TrainingConfig``.

Implemented in Milestone 0 (task 1.2). See design section B.9.

This module reads the ``orca_*`` fields that live on ``TrainingConfig`` (kept
there so the config stays a single source of truth) and exposes them behind a
small, validated accessor. Validation is intentionally minimal at this
milestone: it checks that ``orca_variant`` is one of the six recognized
switches and raises :class:`OrcaConfigError` otherwise. Later milestones layer
MVP-ordering guards on top of this helper.
"""

from dataclasses import dataclass

# The six recognized ORCA variant switches (design B.9 variant table). Order
# mirrors the design's table for readability.
RECOGNIZED_VARIANTS = (
    "no_ontology",
    "external_weight",
    "denominator",
    "no_future",
    "no_align",
    "full",
)

# The single ORCA variant available at the MVP milestone (ORCA-Denominator,
# Requirement 11.1/11.2). Every other variant depends on adaptive sampling,
# alignment, or history signals that arrive in later milestones, so enabling
# them before those milestones land is a configuration error (Requirement 11.4).
MVP_VARIANT = "denominator"

# Number of scalar ontology features (``d_esco``, ``d_isco``, ``d_ot``,
# ``s_esco``, ``s_isco``) that lead the reliability feature vector. The
# ORCA-NoOntology variant excludes exactly these five, leaving the coverage-only
# (or zero-width) layout (Requirement 6.4, design B.2 variant hook).
NUM_ONTOLOGY_SCALARS = 5

# Site at which reliability is applied for each variant (design B.9 variant
# table). ``external_weight`` multiplies the FINAL Standard_InfoNCE loss
# (Requirement 6.1); every other variant folds reliability into the
# Reliability_Calibrated_Denominator (Requirements 6.2, 6.4-6.7).
RELIABILITY_SITE_EXTERNAL = "external_weight"
RELIABILITY_SITE_DENOMINATOR = "denominator"

# Per-variant implied behavior (the design B.9 variant table encoded as data).
# The engine, factory, and orchestrator all resolve their behavior from this one
# table so a variant's implied flags stay consistent across every seam
# (Requirements 6.1, 6.2, 6.4-6.7).
#
# Fields:
#   reliability_site    — where reliability is applied (external vs denominator)
#   adaptive_sampling   — whether the variant's row enables adaptive sampling
#   alignment           — whether the OntologyAlignmentLoss is included
#   use_ontology        — whether ESCO/ISCO/OT scalar features are included
#   history_off         — whether the history weak-target signal is forced off
VARIANT_TABLE = {
    "no_ontology": {
        "reliability_site": RELIABILITY_SITE_DENOMINATOR,
        "adaptive_sampling": True,
        "alignment": False,
        "use_ontology": False,
        "history_off": False,
    },
    "external_weight": {
        "reliability_site": RELIABILITY_SITE_EXTERNAL,
        "adaptive_sampling": False,
        "alignment": False,
        "use_ontology": True,
        "history_off": False,
    },
    "denominator": {
        "reliability_site": RELIABILITY_SITE_DENOMINATOR,
        "adaptive_sampling": False,
        "alignment": False,
        "use_ontology": True,
        "history_off": False,
    },
    "no_future": {
        "reliability_site": RELIABILITY_SITE_DENOMINATOR,
        "adaptive_sampling": True,
        "alignment": False,
        "use_ontology": True,
        "history_off": True,
    },
    "no_align": {
        "reliability_site": RELIABILITY_SITE_DENOMINATOR,
        "adaptive_sampling": True,
        "alignment": False,
        "use_ontology": True,
        "history_off": False,
    },
    "full": {
        "reliability_site": RELIABILITY_SITE_DENOMINATOR,
        "adaptive_sampling": True,
        "alignment": True,
        "use_ontology": True,
        "history_off": False,
    },
}


class OrcaConfigError(ValueError):
    """Raised when the ``orca_*`` configuration is invalid or inconsistent."""


@dataclass(frozen=True)
class OrcaConfig:
    """Validated, read-only view of the ``orca_*`` fields on ``TrainingConfig``.

    Construct with :meth:`from_training_config`. The accessor copies each field
    value so downstream ORCA components depend on this small surface rather than
    reaching into the larger ``TrainingConfig``.
    """

    enabled: bool
    variant: str
    r_min: float
    omega: float
    beta: float
    gamma_enc: float
    lambda_ont: float
    lambda_enc: float
    lambda_hist: float
    use_history: bool
    eta_rel: float
    use_alignment: bool
    lambda_align: float
    feature_dim: int
    use_ontology_features: bool
    adaptive_sampling: bool
    sampling_epsilon: float
    gamma_s: float
    random_mix: float
    warmup_epochs: int
    reliability_epochs: int
    joint_epochs: int
    allow_live_warmup_fallback: bool

    @classmethod
    def from_training_config(cls, config) -> "OrcaConfig":
        """Read the ``orca_*`` fields from ``config`` and validate them.

        Args:
            config: A ``TrainingConfig`` (or any object exposing the ``orca_*``
                attributes). Missing attributes fall back to the OFF /
                OSCAR-equivalent defaults via ``getattr``.

        Returns:
            A validated :class:`OrcaConfig`.

        Raises:
            OrcaConfigError: If ``orca_variant`` is not one of the six
                recognized values.
        """
        variant = getattr(config, "orca_variant", "denominator")
        if variant not in RECOGNIZED_VARIANTS:
            raise OrcaConfigError(
                f"orca_variant must be one of {RECOGNIZED_VARIANTS}, got: {variant!r}"
            )

        return cls(
            enabled=getattr(config, "orca_enabled", False),
            variant=variant,
            r_min=getattr(config, "orca_r_min", 0.05),
            omega=getattr(config, "orca_omega", 0.5),
            beta=getattr(config, "orca_beta", 1.0),
            gamma_enc=getattr(config, "orca_gamma_enc", 5.0),
            lambda_ont=getattr(config, "orca_lambda_ont", 0.5),
            lambda_enc=getattr(config, "orca_lambda_enc", 0.5),
            lambda_hist=getattr(config, "orca_lambda_hist", 0.0),
            use_history=getattr(config, "orca_use_history", False),
            eta_rel=getattr(config, "orca_eta_rel", 1.0),
            use_alignment=getattr(config, "orca_use_alignment", False),
            lambda_align=getattr(config, "orca_lambda_align", 0.1),
            feature_dim=getattr(config, "orca_feature_dim", 5),
            use_ontology_features=getattr(config, "orca_use_ontology_features", True),
            adaptive_sampling=getattr(config, "orca_adaptive_sampling", True),
            sampling_epsilon=getattr(config, "orca_sampling_epsilon", 0.1),
            gamma_s=getattr(config, "orca_gamma_s", 0.5),
            random_mix=getattr(config, "orca_random_mix", 0.5),
            warmup_epochs=getattr(config, "orca_warmup_epochs", 3),
            reliability_epochs=getattr(config, "orca_reliability_epochs", 3),
            joint_epochs=getattr(config, "orca_joint_epochs", 10),
            allow_live_warmup_fallback=getattr(
                config, "orca_allow_live_warmup_fallback", False
            ),
        )

    def validate_mvp(self) -> None:
        """Assert this config matches the MVP milestone (ORCA-Denominator).

        The MVP is delivered first (Requirement 11.1): ``orca_variant`` must be
        ``"denominator"`` with adaptive sampling, ontology alignment, and the
        history weak-target signal all disabled (Requirement 11.2). Any other
        combination requires a later milestone that does not exist yet, so
        enabling it now raises a clear :class:`OrcaConfigError` naming every
        offending condition (Requirement 11.4).

        This is the single source of truth for the MVP-ordering guard so the
        loss-engine factory and the phase orchestrator enforce identical rules
        rather than duplicating the logic. Callers invoke it only when ORCA is
        enabled; with ``orca_enabled`` false the guard never runs and the
        byte-identical career path is untouched (Requirement 7.1).

        Raises:
            OrcaConfigError: If a non-MVP variant, adaptive sampling, the
                ontology-alignment loss, or the history signal is enabled.
        """
        offending = []

        if self.variant != MVP_VARIANT:
            offending.append(
                f"orca_variant={self.variant!r} (only {MVP_VARIANT!r} is "
                f"available at the MVP milestone)"
            )
        if self.adaptive_sampling:
            offending.append(
                "orca_adaptive_sampling=True (adaptive negative sampling lands "
                "in a later milestone)"
            )
        if self.use_alignment:
            offending.append(
                "orca_use_alignment=True (ontology alignment lands in a later "
                "milestone)"
            )
        if self.use_history:
            offending.append(
                "orca_use_history=True (history weak-target signal lands in a "
                "later milestone)"
            )

        if offending:
            raise OrcaConfigError(
                "ORCA is enabled with features that are not part of the MVP "
                "milestone (ORCA-Denominator). Disable the following before "
                "training: " + "; ".join(offending) + "."
            )

    # ------------------------------------------------------------ variant spec
    def _variant_row(self) -> dict:
        """Return this variant's row from :data:`VARIANT_TABLE`.

        ``from_training_config`` already guarantees ``variant`` is one of the six
        recognized values (Requirement 6.8), so the lookup never falls back.
        """
        return VARIANT_TABLE[self.variant]

    def uses_ontology_features(self) -> bool:
        """Whether the reliability feature layout includes the ontology scalars.

        The ``no_ontology`` variant excludes the ESCO/ISCO/OT scalar features
        (Requirement 6.4); the ``orca_use_ontology_features`` flag can also turn
        them off independently. Either one excludes them.
        """
        return self._variant_row()["use_ontology"] and self.use_ontology_features

    def effective_feature_dim(self) -> int:
        """Reliability feature width for the active variant (design B.2 hook).

        For variants that keep the ontology scalars this is the configured
        ``orca_feature_dim``. For ``no_ontology`` (or when
        ``orca_use_ontology_features`` is false) the five ontology scalars are
        removed, leaving the coverage-only (or zero) width (Requirement 6.4),
        floored at ``0``.
        """
        if self.uses_ontology_features():
            return self.feature_dim
        return max(0, self.feature_dim - NUM_ONTOLOGY_SCALARS)

    def reliability_site(self) -> str:
        """Where reliability is applied: ``"external_weight"`` or ``"denominator"``.

        ``external_weight`` multiplies the final Standard_InfoNCE loss
        (Requirement 6.1); all other variants fold reliability into the
        Reliability_Calibrated_Denominator (Requirements 6.2, 6.4-6.7).
        """
        return self._variant_row()["reliability_site"]

    def uses_alignment(self) -> bool:
        """Whether the OntologyAlignmentLoss is part of the total loss.

        Only ``full`` includes it (Requirement 6.7); every other variant excludes
        it (Requirements 6.6 and the design B.9 table). The explicit
        ``orca_use_alignment`` flag can additionally request it (Requirement 4.3).
        """
        return self._variant_row()["alignment"] or self.use_alignment

    def uses_adaptive_sampling(self) -> bool:
        """Whether Phase 4 drives the adaptive negative sampler for this variant.

        True only when the variant's row enables adaptive sampling AND the master
        ``orca_adaptive_sampling`` switch is set. ``external_weight`` and
        ``denominator`` keep it off regardless (Requirements 6.1, 6.2, 5.7).
        """
        return self._variant_row()["adaptive_sampling"] and self.adaptive_sampling

    def forces_history_off(self) -> bool:
        """Whether the variant forces the history weak-target signal off.

        ``no_future`` excludes the history signal and redistributes its weight
        via renormalization (Requirement 6.5).
        """
        return self._variant_row()["history_off"]

    def validate(self) -> None:
        """Validate variant-specific consistency for an ORCA-enabled config.

        This is the per-variant replacement for the blanket MVP gate: all six
        recognized variants are valid (Milestones 2 and 3 are complete), and only
        a truly unrecognized ``orca_variant`` raises — which
        :meth:`from_training_config` already enforces (Requirement 6.8). Here we
        additionally reject configurations whose explicit flags directly
        contradict the active variant's implied semantics, so a variant's
        behavior is unambiguous:

          * enabling ``orca_use_alignment`` on a non-``full`` variant contradicts
            the design B.9 table (only ORCA-Full carries alignment; Req 6.6/6.7);
          * a reduced feature layout must not go negative.

        Raises:
            OrcaConfigError: If an explicit flag is inconsistent with the variant.
        """
        offending = []

        if self.use_alignment and not self._variant_row()["alignment"]:
            offending.append(
                f"orca_use_alignment=True is only valid for the 'full' variant; "
                f"variant {self.variant!r} excludes the ontology-alignment loss"
            )

        if self.effective_feature_dim() < 0:
            offending.append(
                f"orca_feature_dim={self.feature_dim} is too small for the "
                f"{self.variant!r} feature layout"
            )

        if offending:
            raise OrcaConfigError(
                "Inconsistent ORCA variant configuration: "
                + "; ".join(offending) + "."
            )
