"""WeakTargetBuilder: blends ontology/encoder/history signals into ``r_tilde``.

Implemented in Milestone 1 (task 2.3) and extended in Milestone 5 (task 7.1).
See design section B.3.

``r_tilde`` is a per-negative weak reliability target in ``[0, 1]`` where
``1.0`` means "likely true negative" and ``0.0`` means
"ambiguous / possibly-false negative". It is built by blending the reliability
signals that are *present* for a batch — ontology distance, warmup-encoder
cosine similarity, and (optionally) interaction history — and renormalizing the
active blend weights so they sum to ``1.0``.

Signal semantics (design B.3):
  * ``r_ont`` grows with the blended ontology distance ``d_ont`` — a job that is
    far in ontology space is more likely a genuine negative.
  * ``r_enc`` shrinks with warmup-embedding cosine similarity — a job that looks
    semantically close is a suspicious (possibly-false) negative.

The ontology scalars (``d_esco``, ``d_isco``) are reused from OSCAR's
skill-matcher / ``career_graph.py`` (they are never recomputed here). A missing
scalar is expected to arrive pre-substituted with its neutral default (see
``orca.types.OntologyFeatures.as_vector``); a fully missing signal is dropped
from the blend and its weight redistributed across the remaining signals.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Neutral reliability used when no signal is present for a pair (design B.3 /
# Requirement 2.5). 0.5 encodes maximum uncertainty about the negative.
NEUTRAL_R_TILDE: float = 0.5

# Tolerance within which the active blend weights must sum to 1.0
# (Requirement 2.1 / 2.3).
WEIGHT_SUM_TOL: float = 1e-6

# Recognized interaction-history status labels and the reliability they map to
# (design B.3):
#   * ``later_positive``      -> 0.0  a negative that later became a positive is
#                                     ambiguous / a likely false negative.
#   * ``repeatedly_negative`` -> 1.0  a negative that stayed negative across
#                                     repeated interactions is a reliable true
#                                     negative.
# Any other / unrecognized status is treated as *missing* (see
# ``WeakTargetBuilder.history_reliability``). Labels are matched
# case-insensitively.
HISTORY_STATUS_RELIABILITY: Dict[str, float] = {
    "later_positive": 0.0,
    "repeatedly_negative": 1.0,
}


def _history_status_value(token) -> Optional[float]:
    """Normalize a single interaction token to a reliability, or ``None``.

    Accepts the flexible per-pair representations documented on
    :meth:`WeakTargetBuilder.history_reliability`: a status string, a mapping
    carrying a ``type``/``status`` key, or an enum-like object exposing a
    ``value``/``name``/``type``/``status`` attribute. Returns ``0.0`` or ``1.0``
    for a recognized status and ``None`` for anything unrecognized (missing).
    """
    if token is None:
        return None

    label: Optional[str] = None
    if isinstance(token, str):
        label = token
    elif isinstance(token, dict):
        raw = token.get("type", token.get("status"))
        label = raw if isinstance(raw, str) else None
    else:
        # Enum-like / object with a status-bearing attribute.
        for attr in ("type", "status", "value", "name"):
            raw = getattr(token, attr, None)
            if isinstance(raw, str):
                label = raw
                break

    if label is None:
        return None
    return HISTORY_STATUS_RELIABILITY.get(label.strip().lower())


class WeakTargetBuilder:
    """Builds ``r_tilde`` in ``[0, 1]`` by blending present reliability signals.

    Args:
        cfg: A ``TrainingConfig`` (or any object exposing the ``orca_*``
            attributes). Fields are read with defaults so the builder never
            crashes on a partially-populated config.
    """

    def __init__(self, cfg):
        self.omega = getattr(cfg, "orca_omega", 0.5)            # ISCO vs ESCO mix in d_ont
        self.beta = getattr(cfg, "orca_beta", 1.0)              # ontology distance sharpness
        self.gamma = getattr(cfg, "orca_gamma_enc", 5.0)        # encoder similarity sharpness
        self.lambda_ont = float(getattr(cfg, "orca_lambda_ont", 0.5))
        self.lambda_enc = float(getattr(cfg, "orca_lambda_enc", 0.5))
        self.lambda_hist = float(getattr(cfg, "orca_lambda_hist", 0.0))
        self.use_history = getattr(cfg, "orca_use_history", False)  # False for static datasets

    # ------------------------------------------------------------------ signals
    def ontology_reliability(self, d_esco, d_isco) -> torch.Tensor:
        """Ontology-derived reliability ``r_ont`` from ESCO/ISCO distances.

        ``d_ont = (1 - omega) * d_esco + omega * d_isco`` and
        ``r_ont = 1 - exp(-beta * d_ont)``. With ``beta >= 0`` and ``d_ont >= 0``
        this is non-decreasing in ``d_ont`` (Requirement 2.2 / Property 8): a
        larger ontology distance yields a reliability greater than or equal to
        that at any smaller distance.

        Accepts python scalars or tensors; the result is always a tensor.
        """
        d_esco = torch.as_tensor(d_esco, dtype=torch.float32)
        d_isco = torch.as_tensor(d_isco, dtype=torch.float32)
        d_ont = (1.0 - self.omega) * d_esco + self.omega * d_isco
        return 1.0 - torch.exp(-self.beta * d_ont)

    def encoder_reliability(self, z_r_warmup, z_neg_warmup) -> torch.Tensor:
        """Encoder-derived reliability ``r_enc`` from warmup cosine similarity.

        ``r_enc = 1 - sigmoid(gamma * cos)`` on the frozen warmup embeddings.
        With ``gamma >= 0`` this is non-increasing in the cosine similarity
        (Requirement 2.2 / Property 8): a higher similarity yields a reliability
        less than or equal to that at any lower similarity.
        """
        cos = F.cosine_similarity(z_r_warmup, z_neg_warmup, dim=-1)
        p_amb_enc = torch.sigmoid(self.gamma * cos)
        return 1.0 - p_amb_enc

    def history_reliability(self, interaction) -> Optional[torch.Tensor]:
        """History-derived reliability ``r_hist`` (temporal datasets only).

        Interprets an interaction *signal* into a per-negative reliability in
        ``[0, 1]`` (design B.3). Two statuses are recognized:

          * ``later_positive``      -> ``0.0`` — a negative that was later shown
            to be positive is ambiguous / a likely false negative.
          * ``repeatedly_negative`` -> ``1.0`` — a negative that stayed negative
            across repeated interactions is a reliable true negative.

        Input contract for ``interaction``:
          * A single status: a string (``"later_positive"`` /
            ``"repeatedly_negative"``, case-insensitive), a mapping with a
            ``"type"`` or ``"status"`` key, or an enum-like object exposing a
            ``value``/``name``/``type``/``status`` attribute. Returns a scalar
            ``r_hist`` tensor.
          * A batch: a sequence (``list``/``tuple``) of such tokens, aligned
            1:1 with the ``(resume, negative_job)`` pairs. Returns a 1-D
            ``r_hist`` tensor broadcastable against the ontology/encoder signals.
            Recognized tokens map to ``0.0``/``1.0``; individual unrecognized
            tokens within an otherwise-present batch are filled with the neutral
            :data:`NEUTRAL_R_TILDE` (``0.5``) so ``r_hist`` stays in ``[0, 1]``.

        Returns ``None`` (signal *missing*) when ``interaction`` is ``None``, an
        empty sequence, or contains no recognized status at all — so ``build``
        drops history from the blend and redistributes its weight across the
        remaining present signals (Requirements 2.3, 6.5).
        """
        if interaction is None:
            return None

        # Batch of per-pair tokens.
        if isinstance(interaction, (list, tuple)):
            if len(interaction) == 0:
                return None
            values = [_history_status_value(tok) for tok in interaction]
            if all(v is None for v in values):
                # No pair carries a recognized status -> signal missing.
                return None
            filled = [NEUTRAL_R_TILDE if v is None else v for v in values]
            return torch.tensor(filled, dtype=torch.float32)

        # Single status shared across the batch.
        value = _history_status_value(interaction)
        if value is None:
            return None
        return torch.tensor(value, dtype=torch.float32)

    # -------------------------------------------------------------------- blend
    def build(
        self,
        d_esco=None,
        d_isco=None,
        z_r_warmup=None,
        z_neg_warmup=None,
        interaction=None,
    ) -> torch.Tensor:
        """Blend the present reliability signals into ``r_tilde``.

        A signal is *present* when its inputs are available (non-``None``):
          * ontology: both ``d_esco`` and ``d_isco`` provided;
          * encoder: both ``z_r_warmup`` and ``z_neg_warmup`` provided;
          * history: ``orca_use_history`` is true, ``interaction`` provided, and
            ``history_reliability`` returns a non-``None`` value.

        The active blend weights (``lambda_*`` of the present signals) are
        renormalized to sum to ``1.0`` within :data:`WEIGHT_SUM_TOL`
        (Requirements 2.1, 2.3). When no signal is present, the neutral default
        :data:`NEUTRAL_R_TILDE` (``0.5``) is returned without renormalization and
        without raising (Requirement 2.5). When ``orca_use_history`` is false the
        history term is excluded, which is equivalent to setting
        ``orca_lambda_hist = 0`` and renormalizing over the ontology and encoder
        signals (Requirement 2.4).

        Returns:
            A tensor of ``r_tilde`` values in ``[0, 1]`` broadcast to the common
            shape of the present signals, or a scalar ``0.5`` tensor when no
            signal is present.
        """
        r_tilde, _ = self.build_targets(
            d_esco=d_esco,
            d_isco=d_isco,
            z_r_warmup=z_r_warmup,
            z_neg_warmup=z_neg_warmup,
            interaction=interaction,
        )
        return r_tilde

    def build_targets(
        self,
        d_esco=None,
        d_isco=None,
        z_r_warmup=None,
        z_neg_warmup=None,
        interaction=None,
    ) -> Tuple[torch.Tensor, Dict[str, bool]]:
        """Like :meth:`build` but also returns which signals were present.

        Returns:
            A ``(r_tilde, signals_present)`` pair where ``signals_present`` is a
            dict ``{'ont': bool, 'enc': bool, 'hist': bool}``.
        """
        terms = []
        weights = []
        signals_present = {"ont": False, "enc": False, "hist": False}

        if d_esco is not None and d_isco is not None:
            terms.append(self.ontology_reliability(d_esco, d_isco))
            weights.append(self.lambda_ont)
            signals_present["ont"] = True

        if z_r_warmup is not None and z_neg_warmup is not None:
            terms.append(self.encoder_reliability(z_r_warmup, z_neg_warmup))
            weights.append(self.lambda_enc)
            signals_present["enc"] = True

        if self.use_history and interaction is not None:
            r_hist = self.history_reliability(interaction)
            if r_hist is not None:
                terms.append(r_hist)
                weights.append(self.lambda_hist)
                signals_present["hist"] = True

        # No signal present -> neutral default, no renormalization (Req 2.5).
        if not terms:
            return torch.tensor(NEUTRAL_R_TILDE, dtype=torch.float32), signals_present

        # Renormalize active weights to sum to 1.0 (Req 2.1, 2.3). If the present
        # signals carry zero total weight, fall back to an equal split so the
        # blend stays well-defined and the weights still sum to 1.0.
        w = torch.tensor(weights, dtype=torch.float32, device=terms[0].device)
        total = w.sum()
        if total <= 0.0:
            w = torch.ones_like(w) / w.numel()
        else:
            w = w / total

        r_tilde = sum(wi * ti for wi, ti in zip(w, terms))
        # Convex combination of values already in [0, 1]; clamp defensively to
        # absorb any floating-point overshoot (Req 2.1).
        r_tilde = torch.clamp(r_tilde, 0.0, 1.0)
        return r_tilde, signals_present
