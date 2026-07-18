"""Application-site comparison harness: ORCA-ExternalWeight vs ORCA-Denominator.

Implemented in Milestone 4 (task 6.3). See design section B.4 and the design B.9
variant table, whose note calls ``external_weight`` vs ``denominator`` the
**most important comparison**: the two variants differ *only* in **where**
reliability enters the objective — a scalar multiplier on the final
Standard_InfoNCE loss (ORCA-ExternalWeight, Requirement 6.1) versus a per-negative
term inside the Reliability_Calibrated_Denominator (ORCA-Denominator,
Requirement 6.2). Isolating that single difference is what proves whether
denominator calibration matters.

To make the comparison *clean* (Requirement 6.3), this harness runs both variants
on **identical** batches, seeds, inputs, and predicted reliabilities:

  * It constructs BOTH engines from the same ``config`` (only ``orca_variant``
    differs between the two copies; the ontology-alignment term is forced off on
    both so the sole difference is the application site).
  * Both engines SHARE one :class:`~orca.reliability_model.ReliabilityMLP` and one
    :class:`~orca.weak_targets.WeakTargetBuilder` instance, so the reliability
    predictions ``r_psi`` and the weak targets ``r_tilde`` are identical by
    construction.
  * The shared reliability model is placed in ``eval`` mode so its forward pass is
    deterministic — otherwise dropout (design B.2 default ``0.3``) would make the
    two engines' independent reliability computations diverge and defeat the
    isolation.
  * Randomness is seeded solely from ``config.training_seed`` (Requirement 10.1).

The harness reuses :class:`~orca.loss_engine.OrcaLossEngine` directly and does not
modify ``contrastive_learning`` (Isolation constraint). The corresponding property
test proving the two variants differ only in application site is task 6.2 (a
separate deliverable); this module is the config-driven runner that records the
headline loss difference.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch

from orca.loss_engine import OrcaLossEngine
from orca.reliability_model import ReliabilityMLP
from orca.weak_targets import WeakTargetBuilder

# Tolerance for the "identical reliabilities / r_tilde" confirmation checks.
# Both engines share the same (eval-mode, deterministic) model and builder, so
# the tensors are expected to match essentially exactly; a small tolerance
# absorbs only incidental floating-point noise.
_IDENTITY_TOL: float = 1e-6


@dataclass
class ComparisonBatch:
    """Aligned inputs for one application-site comparison.

    A convenience container matching :meth:`OrcaLossEngine.compute_loss`'s
    positional inputs. Any object exposing the same attributes may be passed to
    :func:`run_application_site_comparison` instead of this dataclass.

    Tensor contract (``B`` anchors, ``K`` negatives, ``D`` embedding width):
      * ``z_r``             : ``(B, D)``            anchor (resume) embedding
      * ``z_pos``           : ``(B, D)``            positive-job embedding
      * ``z_negs``          : ``(B, K, D)``         negative-job embeddings
      * ``scalar_features`` : ``(B, K, feature_dim)`` ontology feature vectors
      * ``weak_targets``    : optional ``(B, K)`` precomputed ``r_tilde``
      * ``weak_target_inputs`` : optional kwargs dict forwarded to
        :meth:`WeakTargetBuilder.build` when ``weak_targets`` is not supplied
      * ``job_pairs``       : optional; ignored here (alignment is forced off to
        isolate the application-site effect)
    """

    z_r: torch.Tensor
    z_pos: torch.Tensor
    z_negs: torch.Tensor
    scalar_features: torch.Tensor
    weak_targets: Optional[torch.Tensor] = None
    weak_target_inputs: Optional[dict] = None
    job_pairs: object = None


@dataclass
class ComparisonResult:
    """Structured result of the ORCA-ExternalWeight vs ORCA-Denominator comparison.

    Attributes:
        external_weight_loss: Total loss under ``orca_variant="external_weight"``
            (reliability applied as a scalar multiplier on the final
            Standard_InfoNCE loss, Requirement 6.1).
        denominator_loss: Total loss under ``orca_variant="denominator"``
            (reliability folded into the Reliability_Calibrated_Denominator,
            Requirement 6.2).
        loss_difference: ``external_weight_loss - denominator_loss`` — the
            headline effect isolated to the application site (Requirement 6.3).
        shared_reliability: The per-negative reliability predictions ``r_psi``
            ``(B, K)`` shared by both variants.
        shared_r_tilde: The weak reliability targets ``r_tilde`` shared by both
            variants (a scalar tensor when no weak-target signal was present).
        reliabilities_identical: Confirmation that both variants consumed
            identical reliability predictions (Requirement 6.3).
        r_tilde_identical: Confirmation that both variants consumed identical
            weak targets ``r_tilde`` (Requirement 6.3).
        seed: The ``training_seed`` the run was seeded from (Requirement 10.1).
    """

    external_weight_loss: float
    denominator_loss: float
    loss_difference: float
    shared_reliability: torch.Tensor
    shared_r_tilde: torch.Tensor
    reliabilities_identical: bool
    r_tilde_identical: bool
    seed: int = 0


def _require_seed(config) -> int:
    """Return an integer ``training_seed`` from ``config`` or raise.

    Requirement 10.3: a seeded operation must not begin without a valid integer
    seed. ``bool`` is rejected explicitly (it is an ``int`` subclass but not a
    meaningful seed).
    """
    seed = getattr(config, "training_seed", None)
    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            "run_application_site_comparison requires an integer "
            f"config.training_seed; got {seed!r}"
        )
    return seed


def _variant_config(config, variant: str):
    """Shallow-copy ``config`` with ``orca_variant`` set and alignment forced off.

    Only ``orca_variant`` distinguishes the two engines; ``orca_use_alignment``
    is forced ``False`` on both so the ontology-alignment term never contributes
    and the sole difference between the two runs is the reliability application
    site (Requirement 6.3).
    """
    cfg = copy.copy(config)
    cfg.orca_variant = variant
    cfg.orca_use_alignment = False
    return cfg


def run_application_site_comparison(config, batch) -> ComparisonResult:
    """Run ORCA-ExternalWeight and ORCA-Denominator on identical inputs.

    Constructs both variant engines over a **single shared** ``ReliabilityMLP``
    and ``WeakTargetBuilder`` so the reliability predictions ``r_psi`` and weak
    targets ``r_tilde`` are identical between the two runs, then computes each
    variant's total loss on the same batch under a fixed seed and records the
    loss difference — isolating the application-site effect (Requirement 6.3).

    Args:
        config: A ``TrainingConfig`` (or any object exposing ``training_seed``,
            ``temperature``, ``projection_dim``, and the ``orca_*`` fields). The
            incoming ``orca_variant`` is overridden internally for each run.
        batch: A :class:`ComparisonBatch` (or any object exposing ``z_r``,
            ``z_pos``, ``z_negs``, ``scalar_features``, and optionally
            ``weak_targets`` / ``weak_target_inputs`` / ``job_pairs``).

    Returns:
        A :class:`ComparisonResult` with the two losses, their difference, the
        shared reliability / ``r_tilde`` tensors, and confirmation flags that
        both variants consumed identical reliabilities and targets.

    Raises:
        ValueError: If ``config.training_seed`` is missing or not an integer
            (Requirement 10.3).
    """
    seed = _require_seed(config)

    z_r = batch.z_r
    z_pos = batch.z_pos
    z_negs = batch.z_negs
    scalar_features = batch.scalar_features
    weak_targets = getattr(batch, "weak_targets", None)
    weak_target_inputs = getattr(batch, "weak_target_inputs", None)
    job_pairs = getattr(batch, "job_pairs", None)

    # ── Seed every generator solely from training_seed (Requirement 10.1) so the
    #    shared ReliabilityMLP initializes deterministically. ──
    torch.manual_seed(seed)

    embed_dim = z_r.shape[-1]
    feature_dim = scalar_features.shape[-1]

    # ── Build the SHARED components exactly once. Both engines receive the same
    #    instances, so their reliability predictions and r_tilde are identical by
    #    construction (Requirement 6.3). ──
    reliability_model = ReliabilityMLP(embed_dim, feature_dim)
    # eval() makes the forward pass deterministic (disables dropout), so each
    # engine's independent reliability computation yields the same tensor.
    reliability_model.eval()
    weak_builder = WeakTargetBuilder(config)

    external_engine = OrcaLossEngine(
        _variant_config(config, "external_weight"),
        skill_matcher=None,
        reliability_model=reliability_model,
        weak_builder=weak_builder,
    )
    denominator_engine = OrcaLossEngine(
        _variant_config(config, "denominator"),
        skill_matcher=None,
        reliability_model=reliability_model,
        weak_builder=weak_builder,
    )

    # ── Record the shared reliability predictions and r_tilde for confirmation.
    #    These use the same shared, deterministic (eval-mode) model/builder the
    #    engines use internally, so they mirror what each engine consumes. ──
    with torch.no_grad():
        z_r_b = z_r.unsqueeze(-2).expand_as(z_negs)
        shared_reliability = reliability_model(z_r_b, z_negs, scalar_features)

    if weak_targets is not None:
        shared_r_tilde = weak_targets
    elif weak_target_inputs is not None:
        shared_r_tilde = weak_builder.build(**weak_target_inputs)
    else:
        # No weak-target signal supplied: r_tilde is the neutral default and the
        # reliability BCE term is simply omitted by both engines. Record the
        # neutral scalar for transparency.
        shared_r_tilde = weak_builder.build()

    # ── Empirically confirm both engines consume identical reliabilities: a
    #    second deterministic forward pass must reproduce the first exactly. ──
    with torch.no_grad():
        reliability_recheck = reliability_model(z_r_b, z_negs, scalar_features)
    reliabilities_identical = bool(
        torch.allclose(shared_reliability, reliability_recheck, atol=_IDENTITY_TOL)
    )

    if weak_target_inputs is not None and weak_targets is None:
        with torch.no_grad():
            r_tilde_recheck = weak_builder.build(**weak_target_inputs)
        r_tilde_identical = bool(
            torch.allclose(shared_r_tilde, r_tilde_recheck, atol=_IDENTITY_TOL)
        )
    else:
        # Precomputed / neutral r_tilde is a fixed input, so it is trivially
        # identical across the two runs.
        r_tilde_identical = True

    # ── Run both variants on the SAME inputs. ──
    external_loss = external_engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features,
        weak_targets=weak_targets,
        weak_target_inputs=weak_target_inputs,
        job_pairs=job_pairs,
    )
    denominator_loss = denominator_engine.compute_loss(
        z_r, z_pos, z_negs, scalar_features,
        weak_targets=weak_targets,
        weak_target_inputs=weak_target_inputs,
        job_pairs=job_pairs,
    )

    ext_val = float(external_loss.detach())
    den_val = float(denominator_loss.detach())

    return ComparisonResult(
        external_weight_loss=ext_val,
        denominator_loss=den_val,
        loss_difference=ext_val - den_val,
        shared_reliability=shared_reliability,
        shared_r_tilde=shared_r_tilde,
        reliabilities_identical=reliabilities_identical,
        r_tilde_identical=r_tilde_identical,
        seed=seed,
    )
