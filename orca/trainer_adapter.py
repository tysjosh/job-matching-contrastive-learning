"""OrcaTrainerLossAdapter: bridge the OSCAR trainer's loss-engine call convention
to the tensor-level :class:`~orca.loss_engine.OrcaLossEngine`.

Why this exists
---------------
``ContrastiveLearningTrainer`` computes a batch of ``ContrastiveTriplet`` objects
and a ``content_key -> embedding`` dict, then calls its active loss engine as::

    loss = active_loss_engine.compute_loss(triplets, embeddings)

OSCAR's ``ContrastiveLossEngine`` accepts exactly that. ``OrcaLossEngine``,
however, is a *tensor* API (``compute_loss(z_r, z_pos, z_negs, scalar_features,
...)``) that was unit/property-tested in isolation. Injecting the raw ORCA engine
into the trainer therefore fails at call time (missing ``z_negs`` /
``scalar_features``).

This adapter is the missing seam. It exposes the trainer-compatible
``compute_loss(triplets, embeddings)`` signature and, per triplet, assembles the
tensors the ORCA engine needs:

  * ``z_r`` / ``z_pos`` / ``z_negs`` from the embeddings dict (via the same
    content-key function the trainer uses),
  * per-negative ``scalar_features`` from the ontology signals OSCAR already
    attached to the triplet (per-negative ``career_distances`` plus sample-level
    ``ontology_similarity`` / ``ot_distance``), padded/truncated to the
    ReliabilityMLP's ``feature_dim``,
  * weak-target inputs: an ontology-distance proxy for ``r_ont`` and, when a
    :class:`~orca.warmup_store.WarmupEmbeddingStore` is available, the frozen
    warmup embeddings for the encoder-similarity signal ``r_enc``.

It then delegates to the *unchanged, tested* ``OrcaLossEngine.compute_loss`` per
triplet (batch dim ``B = 1``) and averages, mirroring OSCAR's per-triplet
averaging. All variant branching, the reliability BCE term, the reliability
floor, and the non-finite guards therefore come straight from the tested engine.

Isolation note: this module imports only ``torch`` (and ORCA types). It does NOT
import ``contrastive_learning`` — the content-key function and warmup store are
injected by the orchestrator, so the OSCAR package still contains no reference to
``orca`` (Requirement 7.6).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class OrcaTrainerLossAdapter:
    """Trainer-compatible wrapper around a tensor-level :class:`OrcaLossEngine`.

    Args:
        engine: The constructed :class:`~orca.loss_engine.OrcaLossEngine` (holds
            the ReliabilityMLP, WeakTargetBuilder, variant, temperature, floor).
        content_key_fn: The function mapping a content dict to its embedding key
            — pass the trainer's ``embedding_cache.get_content_key`` so keys
            match the ``embeddings`` dict the trainer supplies.
        warmup_store: Optional read-only warmup-embedding snapshot used to derive
            the encoder-similarity weak-target signal ``r_enc``. When ``None``
            (e.g. before Phase 2 completes) the encoder signal is simply omitted
            and the weak-target builder renormalizes over the remaining signals.
        ot_distance_scale: Scale used to normalize the sample-level
            ``ot_distance`` feature into ``[0, 1]`` (mirrors OSCAR's
            ``ot_distance_scale``).
    """

    def __init__(
        self,
        engine,
        *,
        content_key_fn: Callable[[dict], str],
        warmup_store=None,
        ot_distance_scale: float = 10.0,
    ):
        self.engine = engine
        self.content_key_fn = content_key_fn
        self.warmup_store = warmup_store
        self.ot_distance_scale = float(ot_distance_scale) or 10.0

        model = getattr(engine, "reliability_model", None)
        # The ReliabilityMLP records the feature width it was built with (which
        # already reflects the no_ontology reduction). Fall back to 5 scalars.
        self.feature_dim = int(getattr(model, "feature_dim", 5))
        self._warned_alignment = False

    # ------------------------------------------------------------------ wiring
    def set_warmup_store(self, warmup_store) -> None:
        """Attach (or replace) the frozen warmup-embedding snapshot."""
        self.warmup_store = warmup_store

    def set_epoch(self, epoch: int) -> None:
        """No-op epoch hook for trainer compatibility.

        The trainer advances curriculum via the batch processor and the OSCAR
        engine; ORCA's per-epoch behavior (adaptive-sampler curriculum) is driven
        by the orchestrator, so nothing is needed here. Present so the trainer can
        call ``set_epoch`` uniformly on whichever engine is active.
        """
        return None

    # --------------------------------------------------------------- main entry
    def compute_loss(self, triplets: List, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Trainer-facing loss: assemble ORCA tensors per triplet and average.

        Mirrors ``ContrastiveLossEngine.compute_loss(triplets, embeddings)`` so it
        can be injected as the trainer's active loss engine. Each triplet is
        turned into a ``B = 1`` batch and passed through the tested
        ``OrcaLossEngine.compute_loss``; the per-triplet scalars are averaged.

        Returns:
            A scalar loss tensor. If no triplet could be assembled (e.g. missing
            embeddings), returns a zero scalar that carries gradient, matching the
            OSCAR engine's empty-batch behavior.
        """
        if not triplets:
            raise ValueError("Cannot compute loss for empty triplets list")
        if not embeddings:
            raise ValueError("Embeddings dictionary cannot be empty")

        device = self._infer_device(embeddings)
        per_triplet_losses = []

        for triplet in triplets:
            try:
                contribution = self._triplet_loss(triplet, embeddings, device)
            except Exception as exc:  # defensive: skip a malformed triplet
                logger.warning("ORCA adapter skipped a triplet: %s", exc)
                continue
            if contribution is not None:
                per_triplet_losses.append(contribution)

        if not per_triplet_losses:
            logger.warning(
                "ORCA adapter produced no per-triplet losses for this batch; "
                "returning a zero-gradient loss.")
            return torch.zeros((), device=device, requires_grad=True)

        return torch.stack(per_triplet_losses).mean()

    # ------------------------------------------------------------- per triplet
    def _triplet_loss(self, triplet, embeddings, device) -> Optional[torch.Tensor]:
        """Assemble one triplet's ORCA tensors and delegate to the engine."""
        anchor_key = self.content_key_fn(triplet.anchor)
        positive_key = self.content_key_fn(triplet.positive)
        if anchor_key not in embeddings or positive_key not in embeddings:
            return None

        z_r = embeddings[anchor_key]        # (D,)
        z_pos = embeddings[positive_key]    # (D,)

        neg_keys = []
        neg_embs = []
        neg_positions = []  # index into triplet.negatives for aligned metadata
        for i, negative in enumerate(triplet.negatives):
            nk = self.content_key_fn(negative)
            if nk in embeddings:
                neg_keys.append(nk)
                neg_embs.append(embeddings[nk])
                neg_positions.append(i)

        if not neg_embs:
            return None

        z_negs = torch.stack(neg_embs, dim=0)          # (K, D)
        k = z_negs.shape[0]

        scalar_features = self._build_scalar_features(
            triplet, neg_positions, k, z_negs.dtype, device)   # (K, F)

        weak_targets = self._build_weak_targets(
            triplet, neg_positions, anchor_key, neg_keys, z_negs, device)  # (K,) or None

        # Add the B = 1 batch dimension the engine's tensor API expects.
        z_r_b = z_r.unsqueeze(0)                    # (1, D)
        z_pos_b = z_pos.unsqueeze(0)               # (1, D)
        z_negs_b = z_negs.unsqueeze(0)             # (1, K, D)
        feats_b = scalar_features.unsqueeze(0)     # (1, K, F)
        weak_b = None if weak_targets is None else weak_targets.unsqueeze(0)  # (1, K)

        # ``full`` variant: build in-batch job pairs from this triplet's
        # negatives so the OntologyAlignmentLoss receives real
        # (z_a, z_b, s_ont) triples. Skipped (job_pairs=None) when alignment is
        # off or the ontology inputs to compute s_ont are unavailable.
        job_pairs = self._build_job_pairs(triplet, neg_keys, z_negs, device=z_negs.device)

        return self.engine.compute_loss(
            z_r_b, z_pos_b, z_negs_b, feats_b, weak_targets=weak_b, job_pairs=job_pairs,
        )

    # -------------------------------------------------------------- job pairs
    def _build_job_pairs(self, triplet, neg_keys, z_negs, device):
        """Assemble ``JobPairs`` from this triplet's negatives for the alignment loss.

        Only builds pairs when the alignment term is active (``full`` variant),
        a skill matcher is available to score ``s_ont``, and every negative
        carries ``skill_uris``. Pairs consecutive negatives ``(0,1), (2,3), ...``
        to bound cost; ``s_ont`` is the skill-set similarity between the two
        jobs' ESCO URIs. Returns ``None`` when alignment is off or inputs are
        missing, so the engine simply omits the alignment term.
        """
        if getattr(self.engine, "alignment", None) is None:
            return None

        matcher = getattr(self.engine, "skill_matcher", None)
        if matcher is None or not hasattr(matcher, "ontology_set_similarity"):
            if not self._warned_alignment:
                logger.warning(
                    "ORCA alignment is enabled but no skill matcher is available "
                    "to compute s_ont; the alignment term is omitted.")
                self._warned_alignment = True
            return None

        negatives = getattr(triplet, "negatives", [])
        # Map each present negative (aligned with z_negs rows) to its job dict.
        key_to_row = {k: i for i, k in enumerate(neg_keys)}
        jobs = []
        for job in negatives:
            key = self.content_key_fn(job)
            if key in key_to_row:
                jobs.append((key_to_row[key], job))
        if len(jobs) < 2:
            return None

        from orca.types import JobPairs

        a_rows, b_rows, s_onts = [], [], []
        for i in range(0, len(jobs) - 1, 2):
            row_a, job_a = jobs[i]
            row_b, job_b = jobs[i + 1]
            uris_a = job_a.get("skill_uris", []) if isinstance(job_a, dict) else []
            uris_b = job_b.get("skill_uris", []) if isinstance(job_b, dict) else []
            if not uris_a or not uris_b:
                continue
            try:
                s_ont = float(matcher.ontology_set_similarity(uris_a, uris_b))
            except Exception:
                continue
            a_rows.append(row_a)
            b_rows.append(row_b)
            s_onts.append(s_ont)

        if not a_rows:
            return None

        z_a = z_negs[a_rows]                                   # (P, D)
        z_b = z_negs[b_rows]                                   # (P, D)
        s_ont_t = torch.tensor(s_onts, dtype=z_negs.dtype, device=device)  # (P,)
        return JobPairs(z_a=z_a, z_b=z_b, s_ont=s_ont_t)

    # --------------------------------------------------------- feature assembly
    # Canonical ordering of the five ontology scalars (design B.1 / B.2).
    _ONTOLOGY_SCALAR_ORDER = ("d_esco", "d_isco", "d_ot", "s_esco", "s_isco")

    def _build_scalar_features(self, triplet, neg_positions, k, dtype, device) -> torch.Tensor:
        """Per-negative ontology feature vectors of width ``self.feature_dim``.

        Preferred source (Req 9.1): the per-negative
        ``view_metadata['negative_ontology_features']`` that OSCAR's batch
        processor captured from the skill matcher — the true
        ``[d_esco, d_isco, d_ot, s_esco, s_isco]`` scalars, ordered canonically.

        Fallback (older data / random negatives without captured features): the
        per-negative ``career_distances`` blended distance plus the sample-level
        ``ontology_similarity`` and normalized ``ot_distance``.

        Either way the row is sliced or zero-padded to exactly ``feature_dim`` so
        the layout matches whatever the ReliabilityMLP was built with (including
        the ``no_ontology`` reduced/zero width).
        """
        if self.feature_dim == 0:
            # no_ontology with a zero-width feature layout: (K, 0) tensor.
            return torch.zeros((k, 0), dtype=dtype, device=device)

        vm = getattr(triplet, "view_metadata", {}) or {}
        neg_feats = vm.get("negative_ontology_features")

        rows = []
        if neg_feats:
            for pos in neg_positions:
                feat = neg_feats[pos] if pos < len(neg_feats) else {}
                row = [float(feat.get(name, 0.0)) for name in self._ONTOLOGY_SCALAR_ORDER]
                rows.append(self._fit_width(row))
        else:
            career_distances = getattr(triplet, "career_distances", None) or []
            ont_sim = vm.get("ontology_similarity")
            ont_sim = float(ont_sim) if ont_sim is not None else 0.0
            ot_dist = vm.get("ot_distance")
            norm_ot = (
                max(0.0, 1.0 - float(ot_dist) / self.ot_distance_scale)
                if ot_dist is not None else 0.0
            )
            for pos in neg_positions:
                cdist = float(career_distances[pos]) if pos < len(career_distances) else 0.0
                rows.append(self._fit_width([cdist, ont_sim, norm_ot]))

        return torch.tensor(rows, dtype=dtype, device=device)

    def _fit_width(self, row: list) -> list:
        """Slice or zero-pad ``row`` to exactly ``self.feature_dim`` entries."""
        if len(row) < self.feature_dim:
            return row + [0.0] * (self.feature_dim - len(row))
        return row[: self.feature_dim]

    # ----------------------------------------------------------- weak targets
    def _build_weak_targets(
        self, triplet, neg_positions, anchor_key, neg_keys, z_negs, device
    ) -> Optional[torch.Tensor]:
        """Per-negative weak targets ``r_tilde`` from ontology + encoder signals.

        * Ontology signal: uses the per-negative ``career_distances`` as an
          ontology-distance proxy ``d_ont`` (passed as equal ``d_esco``/``d_isco``
          so the blend equals that value for any ``omega``). This is the signal
          OSCAR's negative selection already produced.
        * Encoder signal: when a warmup snapshot is available and covers the
          anchor and the negatives, uses the frozen warmup cosine similarity for
          ``r_enc``; otherwise the encoder signal is omitted and the builder
          renormalizes over the ontology signal alone.

        Returns ``None`` only if the weak-target builder is unavailable, in which
        case the engine simply skips the reliability-BCE term.
        """
        builder = getattr(self.engine, "weak_builder", None)
        if builder is None:
            return None

        vm = getattr(triplet, "view_metadata", {}) or {}
        neg_feats = vm.get("negative_ontology_features")

        # Prefer the captured true scalars: pass the real d_esco / d_isco so the
        # weak-target builder blends r_ont from the ontology decomposition rather
        # than the single blended career-distance proxy.
        if neg_feats:
            d_esco_vals, d_isco_vals = [], []
            for p in neg_positions:
                feat = neg_feats[p] if p < len(neg_feats) else {}
                d_esco_vals.append(float(feat.get("d_esco", 0.0)))
                d_isco_vals.append(float(feat.get("d_isco", 0.0)))
            d_esco = torch.tensor(d_esco_vals, dtype=z_negs.dtype, device=device)
            d_isco = torch.tensor(d_isco_vals, dtype=z_negs.dtype, device=device)
        else:
            career_distances = getattr(triplet, "career_distances", None) or []
            d_vals = [
                float(career_distances[p]) if p < len(career_distances) else 0.0
                for p in neg_positions
            ]
            # Proxy: use the blended distance for both ESCO and ISCO inputs so
            # d_ont equals it for any omega.
            d_esco = torch.tensor(d_vals, dtype=z_negs.dtype, device=device)
            d_isco = d_esco

        # Encoder signal from frozen warmup embeddings, if the snapshot covers
        # every content key involved.
        z_r_warm = None
        z_neg_warm = None
        store = self.warmup_store
        if store is not None and anchor_key in store and all(nk in store for nk in neg_keys):
            anchor_warm = store[anchor_key].to(device)               # (D,)
            z_r_warm = anchor_warm.unsqueeze(0).expand(len(neg_keys), -1)  # (K, D)
            z_neg_warm = torch.stack(
                [store[nk].to(device) for nk in neg_keys], dim=0)    # (K, D)

        r_tilde = builder.build(
            d_esco=d_esco,
            d_isco=d_isco,
            z_r_warmup=z_r_warm,
            z_neg_warmup=z_neg_warm,
        )
        # ``build`` returns a scalar neutral default only when NO signal is
        # present; here the ontology signal is always present, so r_tilde is (K,).
        return r_tilde.to(device)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _infer_device(embeddings: Dict[str, torch.Tensor]) -> torch.device:
        for value in embeddings.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")
