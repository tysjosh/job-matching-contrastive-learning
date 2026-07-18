"""OntologyAlignmentLoss: aligns job-pair cosine similarity to ontology
set-similarity (ORCA-Full only).

Implemented in Milestone 3 (task 5.1). See design section B.6.

Core idea (design B.6): for sampled job pairs ``(job_a, job_b)`` that carry
ontology metadata, pull the pair's embedding cosine similarity toward the
ontology set-similarity target ``s_ont``:

    loss_align = mean( (cosine(z_a, z_b) - s_ont) ** 2 )

The loss is a plain callable (no learnable parameters of its own); it is
constructed by :class:`~orca.loss_engine.OrcaLossEngine` as
``OntologyAlignmentLoss(config)`` and only enabled for ORCA-Full
(``orca_use_alignment=True``). The scalar weight ``orca_lambda_align`` is applied
by the loss engine, not here (Requirement 4.3).
"""

import torch
import torch.nn.functional as F


class OntologyAlignmentLoss:
    """Mean-squared error between job-pair cosine similarity and ontology
    set-similarity.

    ``loss_align = mean( (cosine(z_a, z_b) - s_ont) ** 2 )`` over the sampled job
    pairs (design B.6). Constructed as ``OntologyAlignmentLoss(config)`` to match
    how :class:`~orca.loss_engine.OrcaLossEngine` builds it; the config is
    retained for parity/extensibility even though the MSE form needs no
    hyperparameters of its own.
    """

    def __init__(self, config=None):
        self.config = config

    def __call__(self, job_pairs) -> torch.Tensor:
        """Compute the ontology-alignment loss for a batch of job pairs.

        Args:
            job_pairs: An object exposing ``z_a`` ``(P, D)``, ``z_b`` ``(P, D)``
                job embeddings and ``s_ont`` ``(P,)`` ontology set-similarity
                targets (e.g. :class:`orca.types.JobPairs`).

        Returns:
            A scalar loss tensor. An empty batch (no pairs) yields a zero scalar
            so the term contributes nothing and never produces a NaN mean.
        """
        z_a = job_pairs.z_a
        z_b = job_pairs.z_b
        s_ont = job_pairs.s_ont

        # Empty pairs → return a zero scalar (mean of an empty tensor is NaN).
        if z_a.shape[0] == 0:
            return torch.zeros((), dtype=z_a.dtype, device=z_a.device)

        cos = F.cosine_similarity(z_a, z_b, dim=-1)
        s_ont = s_ont.to(cos.dtype)
        return ((cos - s_ont) ** 2).mean()
