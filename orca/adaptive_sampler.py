"""AdaptiveNegativeSampler: deterministic, seeded sampler that draws uncertain
negatives more often.

Implemented in Milestone 2 (task 4.1). See design section B.5.

The sampler scores every candidate negative by *how uncertain* it is — a low
reliability (an ambiguous / possibly-false negative) yields a high sampling
score, so uncertain negatives are drawn more often ("sample uncertain
negatives more, punish them less"). Selection mixes a uniform-random portion
(exploration) with a score-proportional portion (exploitation) and is fully
reproducible under a fixed ``(training_seed, epoch)`` (Requirements 5.1, 10.1,
10.2): every source of randomness is seeded solely from
``training_seed + epoch`` via a CPU :class:`torch.Generator`, so runs are
byte-identical across processes and machines.
"""

import torch

from .config import OrcaConfigError


class AdaptiveNegativeSampler:
    """Deterministic, seeded adaptive negative selector (design B.5).

    Injected as the Phase-4 negative selector when the active variant enables
    adaptive sampling. Uncertain negatives (reliability near ``0``) receive a
    higher sampling score and are therefore selected more frequently, while the
    output remains reproducible for a fixed ``(seed, epoch, candidates)``.

    Args:
        cfg: A ``TrainingConfig`` (or any object exposing the ``orca_*`` and
            ``training_seed`` attributes). The sampler reads
            ``orca_sampling_epsilon`` (``epsilon`` > 0), ``orca_gamma_s``
            (``gamma_s`` >= 0), ``orca_random_mix`` (in ``[0, 1]``), and
            ``training_seed``.
        reliability_model: A :class:`~orca.reliability_model.ReliabilityMLP`
            exposing the detached ``score(anchor, candidates, scalar_features)``
            helper.
    """

    def __init__(self, cfg, reliability_model):
        self.epsilon = getattr(cfg, "orca_sampling_epsilon", 0.1)
        self.gamma_s = getattr(cfg, "orca_gamma_s", 0.5)
        self.random_mix = getattr(cfg, "orca_random_mix", 0.5)
        self.seed = getattr(cfg, "training_seed", None)
        self.reliability_model = reliability_model

    def sampling_scores(self, reliability: torch.Tensor) -> torch.Tensor:
        """Per-candidate sampling score ``epsilon + (1 - reliability)**gamma_s``.

        Requirement 5.2: with ``epsilon > 0`` and ``gamma_s >= 0`` this is a
        non-increasing function of reliability, so a lower-reliability (more
        uncertain) candidate always receives a sampling score greater than or
        equal to that of any higher-reliability candidate. The ``1 - reliability``
        base is clamped to ``[0, 1]`` so a fractional ``gamma_s`` never produces
        NaN even if a reliability marginally exceeds its ``[0, 1]`` contract.
        """
        uncertainty = (1.0 - reliability).clamp(min=0.0)
        return self.epsilon + uncertainty ** self.gamma_s

    def select(self, anchor: torch.Tensor, candidates: torch.Tensor,
               scalar_features: torch.Tensor, k: int, epoch: int) -> torch.Tensor:
        """Select ``min(k, n)`` distinct negatives from ``candidates``.

        The selection is the concatenation of a uniform-random portion followed
        by a score-proportional portion, all drawn without replacement:

          1. Draw ``n_random = min(round(random_mix * k), m)`` candidates
             uniformly at random (Requirement 5.4).
          2. Draw the remaining ``m - n_random`` from the not-yet-selected
             candidates with probability proportional to their sampling scores
             (Requirement 5.4).

        where ``m = min(k, n)`` and ``n`` is the candidate count.

        Args:
            anchor: Anchor (resume) embedding of shape ``(..., embed_dim)``,
                broadcast across the candidate set by the reliability model.
            candidates: Candidate negative-job embeddings of shape
                ``(n, embed_dim)``.
            scalar_features: Ontology-feature vectors aligned 1:1 with
                ``candidates`` of shape ``(n, feature_dim)``.
            k: Requested number of negatives.
            epoch: Current training epoch; combined with ``training_seed`` to
                seed the generator so different epochs draw different (but
                reproducible) negatives.

        Returns:
            A 1-D ``LongTensor`` of exactly ``min(k, n)`` distinct candidate
            indices (Requirement 5.3), ordered ``[random..., scored...]``. An
            empty ``LongTensor`` when ``candidates`` is empty or ``k <= 0``
            (Requirement 5.5).

        Raises:
            OrcaConfigError: If ``training_seed`` is missing or not an integer,
                since a reproducible seed is required for deterministic
                selection (Requirements 10.1, 10.2).
        """
        n = int(candidates.shape[0]) if candidates.ndim >= 1 else 0

        # Requirement 5.5: empty candidate set or non-positive request yields an
        # empty selection without raising.
        if n == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long)

        # Requirements 10.1/10.2: determinism requires a valid integer seed. A
        # bool is an int subclass but is not a meaningful seed, so reject it.
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise OrcaConfigError(
                "AdaptiveNegativeSampler requires an integer training_seed for "
                f"deterministic selection; got: {self.seed!r}"
            )

        m = min(k, n)

        # Requirement 5.6 / 10.1: seed all randomness solely from
        # (training_seed + epoch) on a CPU generator so draws are byte-identical
        # across processes and machines. All index math runs on CPU for the
        # same reason.
        gen = torch.Generator().manual_seed(self.seed + int(epoch))

        # Requirement 5.2: detached per-candidate reliabilities -> sampling
        # scores. Kept on CPU (float64) so multinomial matches the generator's
        # device and is numerically stable.
        reliability = self.reliability_model.score(anchor, candidates, scalar_features)
        reliability = reliability.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        scores = self.sampling_scores(reliability)

        # Requirement 5.4: uniform-random portion first.
        n_random = min(int(round(self.random_mix * k)), m)
        n_scored = m - n_random

        selected_parts = []

        if n_random > 0:
            uniform_weights = torch.ones(n, dtype=torch.float64)
            random_idx = torch.multinomial(
                uniform_weights, n_random, replacement=False, generator=gen
            )
            selected_parts.append(random_idx)
        else:
            random_idx = torch.empty(0, dtype=torch.long)

        if n_scored > 0:
            # Requirement 5.3: draw the remainder from the *not-yet-selected*
            # candidates by zeroing the already-picked positions, guaranteeing
            # distinct, without-replacement selection.
            scored_weights = scores.clone()
            scored_weights[random_idx] = 0.0
            scored_idx = torch.multinomial(
                scored_weights, n_scored, replacement=False, generator=gen
            )
            selected_parts.append(scored_idx)

        selected = torch.cat(selected_parts) if selected_parts else torch.empty(0, dtype=torch.long)
        return selected.to(torch.long)
