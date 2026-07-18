"""
Shared data models for the ORCA training subsystem.

These types are consumed across the ORCA package (reliability model, weak-target
builder, loss engine, adaptive sampler). They intentionally reuse the scalar
ontology features that OSCAR already computes (``d_esco``, ``d_isco``, ``d_ot``,
``s_esco``, ``s_isco``, ``coverage``) rather than recomputing them.

See design section B.1 for the authoritative definitions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

# Neutral default substituted for a missing scalar ontology feature. A value of
# 0.0 keeps a missing feature from biasing the ReliabilityMLP input and mirrors
# the "signal-absent" handling described in the design's error-handling section.
NEUTRAL_SCALAR_DEFAULT: float = 0.0


@dataclass
class OntologyFeatures:
    """Scalar ontology features per (resume, negative_job) pair, reused from OSCAR.

    Every scalar field is optional: a value of ``None`` marks the feature as
    missing, in which case :meth:`as_vector` substitutes a neutral default so
    downstream consumers never crash on absent ontology signals.
    """

    d_esco: Optional[float] = None   # ESCO skill-graph distance
    d_isco: Optional[float] = None   # ISCO occupation-hierarchy distance
    d_ot: Optional[float] = None     # optimal-transport (ot_distance) between skill sets
    s_esco: Optional[float] = None   # ESCO set-similarity
    s_isco: Optional[float] = None   # ISCO proximity (0..1)
    coverage: Optional[torch.Tensor] = None  # coverage_features vector (length = coverage_dim)

    def as_vector(self) -> torch.Tensor:
        """Flatten to ``[d_esco, d_isco, d_ot, s_esco, s_isco, *coverage]``.

        Missing scalars are replaced with :data:`NEUTRAL_SCALAR_DEFAULT`. A
        missing coverage vector contributes no coverage elements.

        Returns:
            A 1-D float tensor of length ``5 + coverage_dim``.
        """
        scalars = [
            self.d_esco,
            self.d_isco,
            self.d_ot,
            self.s_esco,
            self.s_isco,
        ]
        values = [
            NEUTRAL_SCALAR_DEFAULT if v is None else float(v)
            for v in scalars
        ]
        vector = torch.tensor(values, dtype=torch.float32)

        if self.coverage is not None:
            coverage = self.coverage.reshape(-1).to(dtype=torch.float32)
            vector = torch.cat([vector, coverage], dim=-1)

        return vector


@dataclass
class ReliabilityBatch:
    """Aligned tensors for one batch of anchors x K negatives."""

    z_r: torch.Tensor              # (B, K, embed_dim) anchor (resume) projected embedding
    z_neg: torch.Tensor            # (B, K, embed_dim) negative-job projected embedding
    scalar_features: torch.Tensor  # (B, K, feature_dim) ontology feature vectors
    valid_mask: torch.Tensor       # (B, K) bool; False where a negative slot is padding/missing


@dataclass
class WeakTargets:
    """Per-negative weak reliability targets in [0,1] and which signals were present."""

    r_tilde: torch.Tensor                 # (B, K) blended target
    signals_present: Dict[str, bool] = field(default_factory=dict)  # {'ont':.., 'enc':.., 'hist':..}


@dataclass
class JobPairs:
    """A batch of sampled job pairs carrying ontology metadata for the
    OntologyAlignmentLoss (ORCA-Full only).

    Each pair ``(job_a, job_b)`` supplies the two projected job embeddings and a
    scalar ontology set-similarity target ``s_ont`` in ``[0, 1]``. The alignment
    loss (design B.6) pulls the pair's cosine similarity toward ``s_ont``.

    Tensor contract (``P`` = number of pairs in the batch):
      * ``z_a``   : ``(P, embed_dim)`` first job's projected embedding
      * ``z_b``   : ``(P, embed_dim)`` second job's projected embedding
      * ``s_ont`` : ``(P,)`` ontology set-similarity target per pair
    """

    z_a: torch.Tensor      # (P, embed_dim) first job's projected embedding
    z_b: torch.Tensor      # (P, embed_dim) second job's projected embedding
    s_ont: torch.Tensor    # (P,) ontology set-similarity target in [0, 1]
