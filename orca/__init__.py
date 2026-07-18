"""
ORCA (Ontology-Regularized Contrastive Alignment with Uncertainty).

An additive, config-gated training mode layered on top of the existing OSCAR
contrastive-learning pipeline. All ORCA code lives in this package; OSCAR
modules never import from here (Isolation constraint). ORCA activates only when
``config.orca_enabled`` is true (default false).

Public exports are kept minimal on purpose. Later milestones fill in the
reliability model, weak-target builder, loss engine, adaptive sampler,
alignment loss, orchestrator, warmup store, factory, and config helpers.
"""

from orca.comparison import (
    ComparisonBatch,
    ComparisonResult,
    run_application_site_comparison,
)
from orca.config import OrcaConfig, OrcaConfigError
from orca.trainer_adapter import OrcaTrainerLossAdapter
from orca.types import OntologyFeatures, ReliabilityBatch, WeakTargets
from orca.weak_targets import WeakTargetBuilder

__all__ = [
    "OntologyFeatures",
    "ReliabilityBatch",
    "WeakTargets",
    "OrcaConfig",
    "OrcaConfigError",
    "WeakTargetBuilder",
    "OrcaTrainerLossAdapter",
    "ComparisonBatch",
    "ComparisonResult",
    "run_application_site_comparison",
]
