"""OrcaLossEngine: mirror of ``ContrastiveLossEngine`` with a reliability-calibrated
denominator.

Implemented for the MVP milestone (task 2.6) and extended for ORCA-Full (task
5.2): when ``orca_use_alignment`` is true the assembled total loss gains the
``orca_lambda_align * loss_align`` term (Requirements 4.3-4.5). The
``external_weight`` branch is exercised by a later milestone (task 6.1). The
existing ``contrastive_learning/loss_engine.py`` is never modified. See design
sections B.4 and B.6.

Core idea (design B.4): OSCAR's InfoNCE denominator
``pos + sum_k(exp(sim_k / tau))`` becomes a *reliability-calibrated* denominator
``pos + sum_k(reliability_k * exp(sim_k / tau))``, where each per-negative
reliability is first clamped to ``[r_min, 1.0]`` so ambiguous negatives push less
strongly while no negative is ever fully dropped. When every reliability equals
``1.0`` the calibrated denominator collapses back to the standard InfoNCE
denominator, so ORCA-Denominator reduces exactly to Standard_InfoNCE
(Requirement 3.1).

The numerical-stability recipe (``eps``, ``max_exp``, the ``[eps, 1e10]``
denominator clamp, the ``[eps, 1.0]`` ratio clamp, and the zero-gradient
fallback on non-finite values) mirrors ``ContrastiveLossEngine._infonce_loss``
exactly so the two engines agree bit-for-bit in the reliability==1 case.

Tensor contract for the core math (shape-general via trailing dims):
  * ``z_r``    : ``(..., D)``   anchor (resume) projected embedding
  * ``z_pos``  : ``(..., D)``   positive-job projected embedding
  * ``z_negs`` : ``(..., K, D)``negative-job projected embeddings
  * ``reliability`` : ``(..., K)`` per-negative reliabilities in ``[0, 1]``
  * returns a per-anchor loss of shape ``(...)``
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class OrcaLossEngine:
    """Mirror of ``ContrastiveLossEngine``'s InfoNCE with a reliability-calibrated
    denominator.

    Constructed by :func:`orca.factory.make_loss_engine` ONLY when
    ``config.orca_enabled`` is true. The existing ``ContrastiveLossEngine`` is
    never modified.

    Args:
        config: A ``TrainingConfig`` exposing ``temperature`` and the ``orca_*``
            fields (see design B.9).
        skill_matcher: Optional OSCAR ``OntologySkillMatcher`` (forwarded for
            parity with the OSCAR engine; ontology scalars themselves are sourced
            through the weak-target builder / feature vectors).
        reliability_model: The :class:`~orca.reliability_model.ReliabilityMLP`
            producing per-negative reliabilities.
        weak_builder: The :class:`~orca.weak_targets.WeakTargetBuilder` producing
            the weak targets ``r_tilde`` used to supervise the reliability model.
    """

    def __init__(self, config, skill_matcher, reliability_model, weak_builder):
        self.config = config
        self.skill_matcher = skill_matcher
        self.reliability_model = reliability_model
        self.weak_builder = weak_builder

        self.tau = config.temperature
        self.r_min = getattr(config, "orca_r_min", 0.05)          # reliability floor
        self.eta_rel = getattr(config, "orca_eta_rel", 1.0)       # weight on reliability BCE
        self.lambda_align = getattr(config, "orca_lambda_align", 0.1)
        self.variant = getattr(config, "orca_variant", "denominator")

        # Numerical-stability constants — identical to ContrastiveLossEngine so
        # the reliability==1 case reduces to Standard_InfoNCE bit-for-bit.
        self.eps = 1e-8
        self.max_exp = 50.0        # prevent overflow in exp()
        self.max_denom = 1e10      # denominator upper clamp

        # Alignment is included only for ORCA-Full (Requirement 6.7) or when the
        # explicit ``orca_use_alignment`` flag requests it (Requirement 4.3);
        # every other variant excludes it, which is exactly equal to
        # ``orca_lambda_align = 0`` (Requirements 4.4, 6.6). Resolving from the
        # variant keeps the design B.9 table the single source of truth. The
        # import is lazy so the alignment module need not exist for MVP imports.
        self.alignment = None
        alignment_active = (
            self.variant == "full" or getattr(config, "orca_use_alignment", False)
        )
        if alignment_active:
            from orca.alignment import OntologyAlignmentLoss  # noqa: WPS433 (lazy)
            self.alignment = OntologyAlignmentLoss(config)

        logger.info(
            "Initialized OrcaLossEngine(variant=%s, temperature=%s, r_min=%s, "
            "eta_rel=%s, use_alignment=%s)",
            self.variant, self.tau, self.r_min, self.eta_rel,
            self.alignment is not None,
        )

    # ------------------------------------------------------------------ helpers
    def clamp_reliability(self, reliability: torch.Tensor) -> torch.Tensor:
        """Clamp every per-negative reliability to ``[r_min, 1.0]``.

        Requirement 3.2 / Property 2: every value that enters the denominator is
        ``>= r_min`` and ``<= 1.0``, so no negative is ever fully dropped.
        """
        return torch.clamp(reliability, self.r_min, 1.0)

    def _neg_term(self, z_r: torch.Tensor, z_negs: Optional[torch.Tensor],
                  reliability: Optional[torch.Tensor]) -> torch.Tensor:
        """Reliability-weighted negative mass ``sum_k(reliability_k*exp(sim_k/tau))``.

        Returns a per-anchor tensor of shape ``z_r.shape[:-1]``. An empty negative
        set (``z_negs is None`` or ``K == 0``) contributes zero mass so the
        denominator reduces to the positive term alone (Requirement 3.4).
        """
        leading = z_r.shape[:-1]
        if z_negs is None or z_negs.shape[-2] == 0:
            return torch.zeros(leading, dtype=z_r.dtype, device=z_r.device)

        # sim(z_r, neg_k) / tau, clamped to avoid exp() overflow.
        neg_sim = (z_r.unsqueeze(-2) * z_negs).sum(-1) / self.tau
        neg_sim = torch.clamp(neg_sim, max=self.max_exp)
        neg_exp = torch.exp(neg_sim)

        if reliability is not None:
            reliability = self.clamp_reliability(reliability)
            neg_exp = reliability * neg_exp
        return neg_exp.sum(-1)

    def _pos_term(self, z_r: torch.Tensor, z_pos: torch.Tensor) -> torch.Tensor:
        """Positive term ``exp(sim(z_r, z_pos)/tau)`` (per-anchor)."""
        pos_sim = (z_r * z_pos).sum(-1) / self.tau
        pos_sim = torch.clamp(pos_sim, max=self.max_exp)
        return torch.exp(pos_sim)

    def _zero_grad_loss(self, reference: torch.Tensor) -> torch.Tensor:
        """A zero-valued loss that carries zero gradient to all parameters.

        Returned on non-finite values (Requirements 3.5, 4.5). It is a fresh leaf
        disconnected from the autograd graph, so backprop leaves every model
        parameter unchanged for that step while preserving the reference shape.
        """
        return torch.zeros(
            reference.shape, dtype=reference.dtype, device=reference.device,
            requires_grad=True,
        )

    # --------------------------------------------------------------- core losses
    def orca_denominator(self, z_r: torch.Tensor, z_pos: torch.Tensor,
                         z_negs: Optional[torch.Tensor],
                         reliability: Optional[torch.Tensor]):
        """Reliability-calibrated denominator (design B.4, Requirements 3.2-3.4).

        ``denom = pos + sum_k(clamp(reliability_k, r_min, 1.0) * exp(sim_k/tau))``.
        Each reliability is clamped to ``[r_min, 1.0]`` before it enters the sum
        (Requirement 3.2); an empty negative set yields ``denom = pos``
        (Requirement 3.4).

        Returns:
            A ``(pos, denom)`` pair of per-anchor tensors.
        """
        pos = self._pos_term(z_r, z_pos)
        neg = self._neg_term(z_r, z_negs, reliability)
        return pos, pos + neg

    def orca_loss(self, z_r: torch.Tensor, z_pos: torch.Tensor,
                  z_negs: Optional[torch.Tensor],
                  reliability: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-anchor ORCA InfoNCE loss with the reliability-calibrated denominator.

        With every ``reliability_k = 1.0`` this equals :meth:`standard_infonce`
        bit-for-bit (Requirement 3.1), because ``clamp(1.0, r_min, 1.0) == 1.0``
        and every clamp/eps/max_exp constant matches OSCAR's engine.

        Requirement 3.5: if the denominator (or the resulting loss) is non-finite,
        the similarities and denominator are already clamped to their configured
        bounds; a residual non-finite value triggers a warning and a zero-gradient
        loss so model parameters stay unchanged for the step.
        """
        pos, denom = self.orca_denominator(z_r, z_pos, z_negs, reliability)
        denom = torch.clamp(denom, self.eps, self.max_denom)
        ratio = torch.clamp(pos / denom, self.eps, 1.0)
        loss = -torch.log(ratio)

        if not torch.isfinite(loss).all() or not torch.isfinite(denom).all():
            logger.warning(
                "Non-finite ORCA denominator/loss detected after clamping; "
                "returning a zero-gradient loss for this step."
            )
            return self._zero_grad_loss(loss)
        return loss

    def standard_infonce(self, z_r: torch.Tensor, z_pos: torch.Tensor,
                         z_negs: Optional[torch.Tensor]) -> torch.Tensor:
        """Standard (uncalibrated) InfoNCE — the ``reliability == 1`` reference.

        Faithfully mirrors ``ContrastiveLossEngine._infonce_loss`` (same clamps,
        ``eps``, ``max_exp``, and zero-gradient fallback) so ORCA-Denominator
        provably reduces to it (Requirement 3.1) and the ORCA-ExternalWeight
        variant can reuse it as its uncalibrated base (Requirement 6.1).
        """
        pos = self._pos_term(z_r, z_pos)
        neg = self._neg_term(z_r, z_negs, reliability=None)  # reliability == 1
        denom = torch.clamp(pos + neg, self.eps, self.max_denom)
        ratio = torch.clamp(pos / denom, self.eps, 1.0)
        loss = -torch.log(ratio)

        if not torch.isfinite(loss).all():
            logger.warning(
                "Non-finite InfoNCE loss detected; returning a zero-gradient loss."
            )
            return self._zero_grad_loss(loss)
        return loss

    def reliability_loss(self, reliability: torch.Tensor,
                         r_tilde: torch.Tensor) -> torch.Tensor:
        """Mean BCE between predicted reliabilities and weak targets ``r_tilde``.

        Requirement 4.1: both operands lie in ``[0, 1]`` and the result is a
        single non-negative scalar reduced by the mean over all
        (resume, negative_job) pairs. Requirement 4.5: a non-finite result yields
        a zero-gradient scalar with a warning.
        """
        r_tilde = torch.broadcast_to(r_tilde.to(reliability.dtype), reliability.shape)
        # Defensively clamp into the open-ended [0,1] domain BCE requires.
        reliability = torch.clamp(reliability, 0.0, 1.0)
        r_tilde = torch.clamp(r_tilde, 0.0, 1.0)
        loss = F.binary_cross_entropy(reliability, r_tilde, reduction="mean")

        if not torch.isfinite(loss).all():
            logger.warning(
                "Non-finite reliability (BCE) loss detected; returning a "
                "zero-gradient loss."
            )
            return self._zero_grad_loss(loss)
        return loss

    # ------------------------------------------------------------ full assembly
    def compute_loss(
        self,
        z_r: torch.Tensor,
        z_pos: torch.Tensor,
        z_negs: torch.Tensor,
        scalar_features: torch.Tensor,
        *,
        weak_targets: Optional[torch.Tensor] = None,
        weak_target_inputs: Optional[dict] = None,
        job_pairs=None,
    ) -> torch.Tensor:
        """Assemble the total ORCA loss for one batch (design B.4).

        Total loss = ``loss_main + orca_eta_rel * loss_rel`` (Requirement 4.2),
        plus ``orca_lambda_align * loss_align`` only when alignment is enabled
        (Requirement 4.3); with alignment off the result equals
        ``lambda_align = 0`` (Requirement 4.4). At the MVP milestone
        (``orca_variant="denominator"``, adaptive sampling off, alignment off)
        only the first two terms contribute.

        Args:
            z_r: Anchor embeddings ``(B, D)``.
            z_pos: Positive-job embeddings ``(B, D)``.
            z_negs: Negative-job embeddings ``(B, K, D)``.
            scalar_features: Ontology feature vectors ``(B, K, feature_dim)`` fed
                to the reliability model.
            weak_targets: Precomputed ``r_tilde`` ``(B, K)``. If omitted and
                ``weak_target_inputs`` is provided, the weak-target builder is
                invoked to produce them.
            weak_target_inputs: Kwargs forwarded to
                :meth:`WeakTargetBuilder.build` when ``weak_targets`` is not
                supplied.
            job_pairs: Optional job-pair batch consumed by the alignment loss
                (ORCA-Full only; ignored at the MVP milestone).

        Returns:
            A scalar total-loss tensor. Non-finite intermediate or assembled
            values yield a zero-gradient scalar (Requirement 4.5).
        """
        reliability = self._predict_reliability(z_r, z_negs, scalar_features)

        # ── Main contrastive term: reliability applied at the variant's site. ──
        if self.variant == "external_weight":
            # ORCA-ExternalWeight (Requirement 6.1): reliability is applied as a
            # single scalar multiplier on the FINAL Standard_InfoNCE loss; the
            # InfoNCE denominator is NOT calibrated with reliability. This is the
            # headline contrast against ORCA-Denominator (Requirement 6.3): the
            # two variants share identical reliability tensors and differ only in
            # where reliability enters. ``mean(reliability)`` is detached so the
            # main loss backprops through the InfoNCE path only, while the
            # reliability model is still supervised by the BCE term below.
            base = self.standard_infonce(z_r, z_pos, z_negs)
            loss_main = reliability.detach().mean() * base.mean()
        else:
            # ORCA-Denominator / -NoOntology / -NoFuture / -NoAlign / -Full
            # (Requirements 6.2, 6.4-6.7): reliability inside the denominator, no
            # final-loss multiplier.
            loss_main = self.orca_loss(z_r, z_pos, z_negs, reliability).mean()

        total = loss_main

        # ── Reliability supervision: + eta_rel * mean BCE(reliability, r_tilde). ──
        if weak_targets is None and weak_target_inputs is not None:
            weak_targets = self.weak_builder.build(**weak_target_inputs)
        if weak_targets is not None:
            loss_rel = self.reliability_loss(reliability, weak_targets)
            total = total + self.eta_rel * loss_rel

        # ── ORCA-Full objective (task 5.2): + orca_lambda_align * loss_align. ──
        # Added only when alignment is enabled (``orca_use_alignment=True`` ->
        # ``self.alignment is not None``, Requirement 4.3) and a ``JobPairs``
        # batch (orca.types.JobPairs) is supplied. When alignment is disabled the
        # term is absent, exactly equal to ``orca_lambda_align = 0``
        # (Requirement 4.4), preserving the MVP denominator path unchanged.
        if self.alignment is not None and job_pairs is not None:
            total = total + self.lambda_align * self.alignment(job_pairs)

        # Extend the non-finite guard to the assembled total (Requirement 4.5).
        if not torch.isfinite(total).all():
            logger.warning(
                "Non-finite assembled ORCA total loss detected; returning a "
                "zero-gradient loss for this step."
            )
            return self._zero_grad_loss(total)
        return total

    # ------------------------------------------------------------------ internal
    def _predict_reliability(self, z_r: torch.Tensor, z_negs: torch.Tensor,
                             scalar_features: torch.Tensor) -> torch.Tensor:
        """Run the reliability model per negative, broadcasting the anchor.

        ``z_r`` ``(B, D)`` is broadcast to ``(B, K, D)`` so each negative is
        scored against its anchor; returns reliabilities of shape ``(B, K)``.
        """
        z_r_b = z_r.unsqueeze(-2).expand_as(z_negs)
        return self.reliability_model(z_r_b, z_negs, scalar_features)
