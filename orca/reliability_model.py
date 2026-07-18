"""ReliabilityMLP: r_psi = P(true_negative | resume, negative_job).

Implemented in Milestone 1 (task 2.1). See design section B.2.

The ReliabilityMLP consumes a pair of projected embeddings (anchor/resume and
negative-job) plus a scalar ontology-feature vector, and emits a per-negative
reliability in ``[0, 1]`` — the probability that a sampled negative is a *true*
negative rather than an ambiguous / possibly-false one.

Input layout (concatenated along the trailing dimension)::

    [z_r, z_j, z_r * z_j, abs(z_r - z_j), scalar_features]

giving ``input_dim = 4 * embed_dim + feature_dim``:
  - ``4 * embed_dim``: the two embeddings, their elementwise product, and their
    elementwise absolute difference.
  - ``feature_dim``: ``[d_esco, d_isco, d_ot, s_esco, s_isco, *coverage_features]``.
"""

import torch
import torch.nn as nn


class ReliabilityMLP(nn.Module):
    """r_psi = P(true_negative | resume, negative_job).

    Args:
        embed_dim: Width of each projected embedding (``z_r`` / ``z_j``).
        feature_dim: Width of the scalar ontology-feature vector. For the
            ORCA-NoOntology variant this is reduced to the coverage-only (or
            zero) width; the module simply reads whatever ``feature_dim`` the
            active ``OrcaConfig`` supplies, so no branch is needed in
            :meth:`forward`.
        hidden1: Width of the first hidden layer (design default 256).
        hidden2: Width of the second hidden layer (design default 128).
        dropout: Dropout probability applied after each ReLU (design default 0.3).
    """

    def __init__(self, embed_dim: int, feature_dim: int,
                 hidden1: int = 256, hidden2: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        input_dim = 4 * embed_dim + feature_dim
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1), nn.Sigmoid(),
        )

    def forward(self, z_r: torch.Tensor, z_j: torch.Tensor,
                scalar_features: torch.Tensor) -> torch.Tensor:
        """Compute the per-negative reliability.

        Args:
            z_r: Anchor (resume) embedding of shape ``(..., embed_dim)``.
            z_j: Negative-job embedding of shape ``(..., embed_dim)`` sharing the
                leading dimensions of ``z_r``.
            scalar_features: Ontology-feature vector of shape
                ``(..., feature_dim)`` sharing the same leading dimensions.

        Returns:
            A reliability tensor of shape ``z_r.shape[:-1]`` with every element
            finite and in the closed interval ``[0, 1]``.

        Raises:
            ValueError: If the leading dimensions of ``z_r``, ``z_j``, and
                ``scalar_features`` do not match (Requirement 1.5).

        Notes:
            The input tensors are never mutated in place (Requirement 1.4); all
            operations (``cat``, ``*``, ``abs``, ``-``) allocate new tensors.
        """
        # Requirement 1.5: leading dims (everything except the trailing feature
        # axis) of all three inputs must match, else raise without returning.
        leading_r = z_r.shape[:-1]
        leading_j = z_j.shape[:-1]
        leading_f = scalar_features.shape[:-1]
        if not (leading_r == leading_j == leading_f):
            raise ValueError(
                "ReliabilityMLP requires matching leading dimensions for z_r, "
                f"z_j, and scalar_features; got z_r={tuple(z_r.shape)}, "
                f"z_j={tuple(z_j.shape)}, "
                f"scalar_features={tuple(scalar_features.shape)}"
            )

        # Requirement 1.3: concatenate [z_r, z_j, z_r*z_j, |z_r - z_j|, feats].
        # All ops are out-of-place, so inputs stay unmodified (Requirement 1.4).
        x = torch.cat(
            [z_r, z_j, z_r * z_j, torch.abs(z_r - z_j), scalar_features],
            dim=-1,
        )
        # Requirement 1.1: squeeze the trailing singleton so the output shape
        # equals z_r.shape[:-1]. Requirement 1.2: Sigmoid keeps values in [0,1].
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def score(self, anchor: torch.Tensor, candidates: torch.Tensor,
              scalar_features: torch.Tensor) -> torch.Tensor:
        """Detached reliability scores for the adaptive sampler.

        Broadcasts a single ``anchor`` embedding across a set of ``candidates``
        and returns per-candidate reliabilities with gradients detached, so the
        sampler can consume them without touching the autograd graph.

        Args:
            anchor: Anchor (resume) embedding of shape ``(..., embed_dim)``. It
                is broadcast against ``candidates`` along the candidate axis; a
                shape of ``(embed_dim,)`` broadcasts to every candidate.
            candidates: Candidate negative-job embeddings of shape
                ``(N, embed_dim)`` (or any shape broadcastable with ``anchor``).
            scalar_features: Ontology-feature vectors aligned with
                ``candidates`` of shape ``(N, feature_dim)``.

        Returns:
            A detached reliability tensor whose shape matches the broadcast
            leading dimensions of the candidate set.
        """
        anchor_b = torch.broadcast_to(anchor, candidates.shape)
        reliability = self.forward(anchor_b, candidates, scalar_features)
        return reliability.detach()
