"""Stage 2 supervised fine-tuning driver for the CVE domain (task 10.2).

Stage 2 is the second half of the two-stage CVE pipeline (Req 8). It loads the
**Stage 1 contrastive checkpoint** (the frozen encoder + the projection head that
maps encoder embeddings to the 128-dim contrastive space) and attaches the four
:class:`~cve_domain.supervised_heads.CVESupervisedHeads` over those **frozen**
embeddings (Req 8.4, 9.1). Only the supervised heads are trained; the encoder and
projection head stay frozen by default to avoid the catastrophic forgetting the
prior CDCL work observed when unfreezing the sentence-transformer (Req 9.1).

This module is a thin, **additive** wiring layer. It reuses ``FineTuningTrainer``
conventions (frozen ``SentenceTransformer`` encoder, JSONL dataloader batching,
best-checkpoint-by-validation-loss) but operates on ``CVE_View_Record`` lines
(reading the pre-serialized ``encoder_view`` text verbatim, the same way the
Stage 1 trainer does — see ``contrastive_learning/trainer.py``) and on the four
CVE supervised targets rather than the career resume/job binary classifier. It
does **not** modify ``FineTuningTrainer``, ``trainer.py``, or the loss math.

Two guards required by the spec:

* **Missing Stage 1 checkpoint (Req 8.6).** If the referenced checkpoint path
  does not exist, the trainer stops *before* training and reports the missing
  path. This check runs at construction time and needs neither ``torch`` nor the
  encoder, so it fails fast.
* **Absent supervised label (Req 8.7).** A head is only built and trained when
  its label is present on **at least one** training record. If a required label
  is absent from *every* training record, that head is skipped and the skip plus
  its reason is recorded in the Stage 2 report. Per-sample masking additionally
  excludes individual records that lack a given (otherwise-enabled) label from
  that head's loss.

Losses (Req 8.5): ``MSELoss`` for the ``priority_score`` regression head,
``CrossEntropyLoss`` for the ``priority_band`` multiclass head, and
``BCEWithLogitsLoss`` for the ``in_kev`` / ``ransomware`` binary heads.

Requirements: 8.4, 8.6, 8.7
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from contrastive_learning.data_structures import TrainingConfig

from .stage1 import load_view_records
from .supervised_heads import (
    IN_KEV_HEAD,
    PRIORITY_BAND_HEAD,
    PRIORITY_SCORE_HEAD,
    RANSOMWARE_HEAD,
)

try:  # torch is a hard runtime dependency for the actual training path.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sentence_transformers import SentenceTransformer

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

#: The four supervised heads, in a fixed order for deterministic reporting.
ALL_HEADS = (PRIORITY_SCORE_HEAD, PRIORITY_BAND_HEAD, IN_KEV_HEAD, RANSOMWARE_HEAD)

#: Batch size used for the one-time frozen-encoder embedding pass. This is pure
#: inference (no gradients), so it can be far larger than the training batch size
#: for a big speedup on GPU/MPS; the actual batch is max(training batch_size, this).
EMBED_INFERENCE_BATCH_SIZE = 256
#: Log embedding progress every N inference batches (visibility on large splits).
EMBED_LOG_EVERY_N_BATCHES = 20

#: priority_score is a 0–100 label. The regression head is trained on the
#: normalized [0, 1] target so its MSE lives on the same scale as the
#: cross-entropy / BCE classification heads (otherwise the raw 0–100 MSE — squared
#: — dominates the summed loss and starves the band / in_kev / ransomware heads).
#: Predictions are denormalized back to 0–100 in :meth:`predict_records`.
PRIORITY_SCORE_SCALE = 100.0

#: Best-by-validation Stage 2 checkpoint filename.
STAGE2_BEST_CHECKPOINT = "stage2_best_checkpoint.pt"
#: Stage 2 report filename (enabled/skipped heads + reasons + metrics).
STAGE2_REPORT_FILENAME = "stage2_report.json"

#: Recognized truthy / falsy tokens for the boolean labels (mirrors the converter).
_TRUTHY = {"true", "t", "yes", "y", "1", "1.0"}
_FALSY = {"false", "f", "no", "n", "0", "0.0", ""}


# --------------------------------------------------------------------------- #
# Label reading helpers (pure Python; no torch needed)
# --------------------------------------------------------------------------- #
def _record_labels(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``cve_labels`` mapping carried on a ``CVE_View_Record``."""
    labels = record.get("cve_labels")
    return dict(labels) if isinstance(labels, Mapping) else {}


def _numeric_priority_score(value: Any) -> Optional[float]:
    """Return ``value`` as a float in ``[0, 100]`` or ``None`` when invalid."""
    if isinstance(value, bool):  # bools are ints in Python; reject explicitly.
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            score = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if 0.0 <= score <= 100.0:
        return score
    return None


def _band_value(value: Any) -> Optional[str]:
    """Return a non-empty ``priority_band`` string or ``None`` when absent."""
    if value is None:
        return None
    band = str(value).strip()
    return band or None


def _bool_label(value: Any) -> Optional[bool]:
    """Parse a truthy/falsy label; ``None`` when missing/unparseable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in _TRUTHY:
        return True
    if token in _FALSY:
        return False
    return None


# --------------------------------------------------------------------------- #
# Head configuration (which heads to build, with skip reasons)
# --------------------------------------------------------------------------- #
@dataclass
class HeadConfiguration:
    """Which supervised heads to build for a Stage 2 run, and why others skipped.

    Attributes:
        enabled: Head name -> whether it will be built and trained.
        skip_reasons: Head name -> human-readable reason it was skipped (Req 8.7),
            present only for skipped heads.
        band_vocabulary: Sorted list of distinct ``priority_band`` values seen in
            the training records (defines the multiclass head's class order).
    """

    enabled: Dict[str, bool] = field(default_factory=dict)
    skip_reasons: Dict[str, str] = field(default_factory=dict)
    band_vocabulary: List[str] = field(default_factory=list)

    @property
    def band_to_index(self) -> Dict[str, int]:
        """Map each ``priority_band`` value to its class index."""
        return {band: idx for idx, band in enumerate(self.band_vocabulary)}

    def any_enabled(self) -> bool:
        return any(self.enabled.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled_heads": [h for h in ALL_HEADS if self.enabled.get(h)],
            "skipped_heads": self.skip_reasons,
            "band_vocabulary": self.band_vocabulary,
        }


def detect_head_configuration(
    records: Sequence[Mapping[str, Any]]
) -> HeadConfiguration:
    """Decide which heads to build from the labels present in ``records``.

    A head is enabled iff its label is present on at least one training record;
    otherwise the head is skipped with a recorded reason (Req 8.7). For the
    ``priority_band`` head the distinct band values also define the multiclass
    output size — a single distinct band is treated as too few classes to train a
    classifier, so the band head is skipped in that case.

    This is pure Python (no torch), so the absent-label skip logic is testable
    without loading the encoder.
    """
    has_score = False
    has_in_kev = False
    has_ransomware = False
    bands: set[str] = set()

    for record in records:
        labels = _record_labels(record)
        if not has_score and _numeric_priority_score(labels.get(PRIORITY_SCORE_HEAD)) is not None:
            has_score = True
        band = _band_value(labels.get(PRIORITY_BAND_HEAD))
        if band is not None:
            bands.add(band)
        if not has_in_kev and _bool_label(labels.get(IN_KEV_HEAD)) is not None:
            has_in_kev = True
        if not has_ransomware and _bool_label(labels.get(RANSOMWARE_HEAD)) is not None:
            has_ransomware = True

    config = HeadConfiguration()
    config.band_vocabulary = sorted(bands)

    # priority_score (regression).
    config.enabled[PRIORITY_SCORE_HEAD] = has_score
    if not has_score:
        config.skip_reasons[PRIORITY_SCORE_HEAD] = (
            "priority_score is absent from every training record "
            "(no numeric value within [0, 100])"
        )

    # priority_band (multiclass) — needs at least two distinct classes.
    if len(config.band_vocabulary) >= 2:
        config.enabled[PRIORITY_BAND_HEAD] = True
    else:
        config.enabled[PRIORITY_BAND_HEAD] = False
        if not config.band_vocabulary:
            config.skip_reasons[PRIORITY_BAND_HEAD] = (
                "priority_band is absent from every training record"
            )
        else:
            config.skip_reasons[PRIORITY_BAND_HEAD] = (
                "priority_band has only one distinct class "
                f"({config.band_vocabulary[0]!r}); need >= 2 classes to train"
            )

    # in_kev (binary).
    config.enabled[IN_KEV_HEAD] = has_in_kev
    if not has_in_kev:
        config.skip_reasons[IN_KEV_HEAD] = (
            "in_kev is absent from every training record"
        )

    # ransomware (binary).
    config.enabled[RANSOMWARE_HEAD] = has_ransomware
    if not has_ransomware:
        config.skip_reasons[RANSOMWARE_HEAD] = (
            "ransomware is absent from every training record"
        )

    return config


# --------------------------------------------------------------------------- #
# Frozen projection head reconstructed from the Stage 1 checkpoint
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:

    class _FrozenProjectionHead(nn.Module):
        """Reconstructs the Stage 1 contrastive projection head for inference.

        Mirrors the ``CareerAwareContrastiveModel`` projection architecture used by
        the Stage 1 trainer (``contrastive_learning/trainer.py``):
        ``Linear(input_dim, hidden) -> ReLU -> Dropout -> Linear(hidden,
        projection_dim)`` followed by L2 normalization. The submodule is named
        ``projection_head`` with the same layer indices (0 and 3) so the Stage 1
        checkpoint's ``model_state_dict`` loads directly.

        The layer dimensions are inferred from the checkpoint weights, so this
        works regardless of the encoder (384-dim MiniLM, 768-dim mpnet, ...) and
        the configured ``projection_dim``.
        """

        def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int) -> None:
            super().__init__()
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),  # inference-time; dropout is a no-op in eval()
                nn.Linear(hidden_dim, projection_dim),
            )
            self.projection_dim = projection_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            projected = self.projection_head(x)
            return nn.functional.normalize(projected, p=2, dim=-1)

    def _build_projection_from_state_dict(
        state_dict: Mapping[str, "torch.Tensor"]
    ) -> "_FrozenProjectionHead":
        """Build and load the frozen projection head from a checkpoint state dict.

        Infers ``input_dim`` / ``hidden_dim`` / ``projection_dim`` from the two
        Linear layers' shapes so no encoder load is needed to size the module.
        """
        first_w = state_dict.get("projection_head.0.weight")
        last_w = state_dict.get("projection_head.3.weight")
        if first_w is None or last_w is None:
            raise ValueError(
                "Stage 1 checkpoint model_state_dict does not contain the expected "
                "'projection_head.0.weight' / 'projection_head.3.weight' keys; got "
                f"keys: {sorted(state_dict.keys())}"
            )
        hidden_dim, input_dim = int(first_w.shape[0]), int(first_w.shape[1])
        projection_dim = int(last_w.shape[0])
        module = _FrozenProjectionHead(input_dim, hidden_dim, projection_dim)
        # Load only the projection_head.* tensors (ignore any extra keys).
        proj_state = {
            k: v for k, v in state_dict.items() if k.startswith("projection_head.")
        }
        module.load_state_dict(proj_state, strict=True)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module


# --------------------------------------------------------------------------- #
# Stage 2 trainer
# --------------------------------------------------------------------------- #
@dataclass
class Stage2Result:
    """Outcome of a Stage 2 run (or a prepared-but-not-trained dry run)."""

    head_configuration: HeadConfiguration
    stage1_checkpoint_path: str
    best_checkpoint_path: Optional[str] = None
    best_val_loss: Optional[float] = None
    epochs_trained: int = 0
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage1_checkpoint_path": self.stage1_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epochs_trained,
            **self.head_configuration.to_dict(),
        }


class CVEStage2Trainer:
    """Stage 2 supervised fine-tuning over frozen Stage 1 embeddings.

    Args:
        config: The CVE Run_Config (a :class:`TrainingConfig`). ``num_epochs``,
            ``batch_size``, ``learning_rate``, ``weight_decay``,
            ``text_encoder_model`` and ``freeze_text_encoder`` are read from it.
        output_dir: Directory for the Stage 2 checkpoint and report.
        stage1_checkpoint_path: Path to the Stage 1 ``best_checkpoint.pt`` (defaults
            to ``config.pretrained_model_path``).

    Raises:
        FileNotFoundError: If the Stage 1 checkpoint path does not exist. This is
            the Req 8.6 guard and fires at construction — before any encoder load
            or training — so a misconfigured run stops immediately.
        ValueError: If no Stage 1 checkpoint path is configured at all.
    """

    def __init__(
        self,
        config: TrainingConfig,
        output_dir: str | Path,
        stage1_checkpoint_path: Optional[str] = None,
        base_embeddings: bool = False,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Base-embeddings mode (the "no Stage 1" honest baseline): attach the
        # supervised heads directly over the frozen sentence-transformer output
        # with NO Stage 1 projection (identity). No checkpoint is required, so the
        # Req 8.6 guard below is intentionally skipped for this explicit mode.
        self.base_embeddings = bool(base_embeddings)

        if self.base_embeddings:
            self.stage1_checkpoint_path = None
        else:
            checkpoint = stage1_checkpoint_path or getattr(
                config, "pretrained_model_path", None
            )
            if not checkpoint or not str(checkpoint).strip():
                raise ValueError(
                    "A Stage 1 checkpoint path is required for Stage 2 (set "
                    "config.pretrained_model_path or pass stage1_checkpoint_path), "
                    "or set base_embeddings=True for the no-Stage-1 baseline."
                )
            self.stage1_checkpoint_path = str(checkpoint)

            # --- Req 8.6: stop before training when the checkpoint does not exist.
            if not Path(self.stage1_checkpoint_path).exists():
                raise FileNotFoundError(
                    "Stage 1 checkpoint referenced by the Stage 2 Run_Config does not "
                    f"exist; cannot start Stage 2 training. Missing path: "
                    f"{self.stage1_checkpoint_path}"
                )

        # Torch-heavy state, built lazily in prepare()/train().
        self.device = None
        self.text_encoder = None
        self.projection = None  # stays None in base-embeddings (identity) mode
        self._base_embedding_dim: Optional[int] = None
        self.heads = None
        self.head_config: Optional[HeadConfiguration] = None
        # Per-head class-imbalance weights (band CE weight vector + binary
        # pos_weights), populated in train() when cve_class_balanced_heads is on.
        self._class_weights: Dict[str, "torch.Tensor"] = {}

    # ------------------------------------------------------------------ #
    # Encoder + projection loading (torch-heavy; deferred)
    # ------------------------------------------------------------------ #
    def _ensure_torch(self) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and sentence-transformers are required to run CVE Stage 2 "
                "training. Install with: pip install torch sentence-transformers"
            )

    def _load_encoder_and_projection(self) -> None:
        """Load the frozen encoder and (unless base mode) the Stage 1 projection."""
        self._ensure_torch()
        if self.text_encoder is not None and (
            self.projection is not None or self.base_embeddings
        ):
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base-embeddings baseline: frozen encoder only, identity projection.
        if self.base_embeddings:
            self.text_encoder = SentenceTransformer(self.config.text_encoder_model)
            self.text_encoder.max_seq_length = 512
            self.text_encoder.to(self.device)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self._base_embedding_dim = int(
                self.text_encoder.get_sentence_embedding_dimension()
            )
            self.projection = None
            logger.info(
                "Stage 2 (base-embeddings baseline): loaded frozen encoder '%s' "
                "(dim=%d), no Stage 1 projection.",
                self.config.text_encoder_model,
                self._base_embedding_dim,
            )
            return

        # Frozen projection head reconstructed from the Stage 1 checkpoint.
        checkpoint = torch.load(
            self.stage1_checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = (
            checkpoint.get("model_state_dict")
            if isinstance(checkpoint, Mapping)
            else None
        )
        if state_dict is None:
            raise ValueError(
                "Stage 1 checkpoint does not contain 'model_state_dict'; cannot "
                f"reconstruct the projection head from {self.stage1_checkpoint_path}"
            )
        self.projection = _build_projection_from_state_dict(state_dict).to(self.device)

        # Frozen sentence-transformer encoder (same model used in Stage 1).
        self.text_encoder = SentenceTransformer(self.config.text_encoder_model)
        self.text_encoder.max_seq_length = 512
        self.text_encoder.to(self.device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        logger.info(
            "Stage 2: loaded frozen encoder '%s' + projection head "
            "(input=%d, projection_dim=%d) from %s",
            self.config.text_encoder_model,
            self.projection.projection_head[0].in_features,
            self.projection.projection_dim,
            self.stage1_checkpoint_path,
        )

    @property
    def embedding_dim(self) -> int:
        """The frozen embedding dim the heads operate over.

        The Stage 1 projection output dim in the normal path; the raw encoder
        dimension in the base-embeddings baseline (identity projection).
        """
        if self.text_encoder is None:
            self._load_encoder_and_projection()
        if self.base_embeddings:
            return int(self._base_embedding_dim)
        return int(self.projection.projection_dim)

    def _embed_views(self, views: Sequence[str]) -> "torch.Tensor":
        """Encode ``encoder_view`` texts to the frozen embeddings the heads use.

        Applies the Stage 1 projection in the normal path; returns the raw frozen
        encoder embedding unchanged in the base-embeddings baseline.
        """
        with torch.no_grad():
            text_emb = self.text_encoder.encode(
                list(views),
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
            )
            if text_emb.dim() == 1:
                text_emb = text_emb.unsqueeze(0)
            if self.base_embeddings:
                # SentenceTransformer.encode can return an inference-mode tensor
                # (torch.inference_mode internally); clone to a normal tensor so
                # the trainable heads can save it for backward. In the projection
                # path below the projection op already yields a normal tensor.
                return text_emb.detach().clone()
            projected = self.projection(text_emb)
        return projected

    def _embed_records_once(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Tuple[List[Mapping[str, Any]], Optional["torch.Tensor"]]:
        """Embed each valid record's ``encoder_view`` exactly once (frozen encoder).

        Returns the records that carry a non-empty ``encoder_view`` (the same
        filter :meth:`_iter_batches` applies) together with a **CPU** tensor of
        their embeddings, aligned by index. Because the encoder and the Stage 1
        projection are frozen, these embeddings are identical on every epoch, so
        computing them once and reusing the slices avoids re-encoding the whole
        split each epoch — a large speedup with byte-identical results. The cache
        is kept on CPU and moved to the device per batch to keep peak device
        memory bounded on large splits.
        """
        valid = [
            r for r in records
            if isinstance(r.get("encoder_view"), str) and r["encoder_view"].strip()
        ]
        if not valid:
            return [], None
        # Embedding is pure frozen-encoder inference (no gradients), so it can use a
        # much larger batch than the training ``batch_size`` (which is sized for the
        # head optimizer step). A bigger inference batch is markedly faster on
        # GPU/MPS with identical results.
        embed_batch = max(int(self.config.batch_size), EMBED_INFERENCE_BATCH_SIZE)
        total = len(valid)
        chunks: List["torch.Tensor"] = []
        logger.info("Embedding %d records once (frozen encoder, batch=%d)...", total, embed_batch)
        for start in range(0, total, embed_batch):
            views = [r["encoder_view"] for r in valid[start:start + embed_batch]]
            chunks.append(self._embed_views(views).detach().to("cpu"))
            done = min(start + embed_batch, total)
            if done == total or (start // embed_batch) % EMBED_LOG_EVERY_N_BATCHES == 0:
                logger.info("  embedded %d/%d records (%.0f%%)", done, total, 100.0 * done / total)
        return valid, torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------ #
    # Head construction
    # ------------------------------------------------------------------ #
    def build_heads(self, head_config: HeadConfiguration) -> Any:
        """Build :class:`CVESupervisedHeads` for the enabled heads (Req 8.4)."""
        self._load_encoder_and_projection()
        # Imported here so the module import stays torch-optional.
        from .supervised_heads import CVESupervisedHeads

        num_bands = (
            len(head_config.band_vocabulary)
            if head_config.enabled.get(PRIORITY_BAND_HEAD)
            else None
        )
        heads = CVESupervisedHeads(
            embedding_dim=self.embedding_dim,
            num_priority_bands=num_bands,
            enable_priority_score=head_config.enabled.get(PRIORITY_SCORE_HEAD, False),
            enable_priority_band=head_config.enabled.get(PRIORITY_BAND_HEAD, False),
            enable_in_kev=head_config.enabled.get(IN_KEV_HEAD, False),
            enable_ransomware=head_config.enabled.get(RANSOMWARE_HEAD, False),
        ).to(self.device)
        self.heads = heads
        self.head_config = head_config
        return heads

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def _iter_batches(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Iterator[List[Mapping[str, Any]]]:
        """Yield fixed-size batches of view records with a non-empty encoder view."""
        batch: List[Mapping[str, Any]] = []
        batch_size = max(1, int(self.config.batch_size))
        for record in records:
            view = record.get("encoder_view")
            if not isinstance(view, str) or not view.strip():
                continue
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _head_targets(
        self, batch: Sequence[Mapping[str, Any]], head_config: HeadConfiguration
    ) -> Dict[str, Tuple["torch.Tensor", "torch.Tensor"]]:
        """Build per-head (mask, target) tensors for a batch.

        The mask selects the batch rows that carry a valid label for the head, so
        records missing an otherwise-enabled label are excluded from that head's
        loss (per-sample masking beyond the all-absent skip of Req 8.7).
        """
        n = len(batch)
        targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        band_to_index = head_config.band_to_index

        if head_config.enabled.get(PRIORITY_SCORE_HEAD):
            mask, vals = [], []
            for rec in batch:
                score = _numeric_priority_score(_record_labels(rec).get(PRIORITY_SCORE_HEAD))
                mask.append(score is not None)
                # Train on the normalized [0, 1] target (see PRIORITY_SCORE_SCALE).
                vals.append(score / PRIORITY_SCORE_SCALE if score is not None else 0.0)
            targets[PRIORITY_SCORE_HEAD] = (
                torch.tensor(mask, dtype=torch.bool, device=self.device),
                torch.tensor(vals, dtype=torch.float32, device=self.device),
            )

        if head_config.enabled.get(PRIORITY_BAND_HEAD):
            mask, vals = [], []
            for rec in batch:
                band = _band_value(_record_labels(rec).get(PRIORITY_BAND_HEAD))
                idx = band_to_index.get(band) if band is not None else None
                mask.append(idx is not None)
                vals.append(idx if idx is not None else 0)
            targets[PRIORITY_BAND_HEAD] = (
                torch.tensor(mask, dtype=torch.bool, device=self.device),
                torch.tensor(vals, dtype=torch.long, device=self.device),
            )

        for head in (IN_KEV_HEAD, RANSOMWARE_HEAD):
            if head_config.enabled.get(head):
                mask, vals = [], []
                for rec in batch:
                    parsed = _bool_label(_record_labels(rec).get(head))
                    mask.append(parsed is not None)
                    vals.append(1.0 if parsed else 0.0)
                targets[head] = (
                    torch.tensor(mask, dtype=torch.bool, device=self.device),
                    torch.tensor(vals, dtype=torch.float32, device=self.device),
                )
        assert n == len(batch)
        return targets

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    def _compute_class_weights(
        self, records: Sequence[Mapping[str, Any]], head_config: HeadConfiguration
    ) -> Dict[str, "torch.Tensor"]:
        """Derive class-imbalance weights from the training label distribution.

        The classification heads otherwise minimize their loss by predicting the
        majority class (observed collapse: constant band / constant in_kev). We
        counter this with:

        * ``priority_band`` (CrossEntropy): inverse-frequency class weights,
          normalized so the mean weight is ~1, aligned to ``band_to_index``.
        * ``in_kev`` / ``ransomware`` (BCEWithLogits): ``pos_weight = n_neg/n_pos``
          so the positive (minority) class is up-weighted.

        Weights are computed on the training split only and live on ``self.device``.
        """
        weights: Dict[str, "torch.Tensor"] = {}
        # Cap for the binary pos_weight. Raw neg/pos can reach ~1000x on the full
        # data (e.g. ransomware: 254 pos / 277k neg), which makes the minority
        # loss dominate and destabilizes training (observed val_loss blow-up).
        # Capping keeps the up-weighting useful without wrecking optimization.
        pos_weight_cap = float(getattr(self.config, "cve_pos_weight_cap", 10.0))

        if head_config.enabled.get(PRIORITY_BAND_HEAD):
            idx_map = head_config.band_to_index
            counts = [0] * len(idx_map)
            for rec in records:
                band = _band_value(_record_labels(rec).get(PRIORITY_BAND_HEAD))
                idx = idx_map.get(band) if band is not None else None
                if idx is not None:
                    counts[idx] += 1
            total = sum(counts)
            k = len(counts)
            if total > 0 and k > 0:
                # SQRT of inverse-frequency (softer than raw inverse-freq), then
                # normalize to mean 1. This breaks majority-collapse without the
                # extreme ~180x weights raw inverse-freq produces on skewed bands.
                raw = [math.sqrt(total / (k * c)) if c > 0 else 1.0 for c in counts]
                mean_w = sum(raw) / len(raw)
                w = [x / mean_w for x in raw]
                weights["band"] = torch.tensor(w, dtype=torch.float32, device=self.device)
                logger.info("Stage 2 band class weights (sqrt-inv-freq, mean-norm): %s (counts=%s)",
                            [round(x, 3) for x in w], counts)

        for head in (IN_KEV_HEAD, RANSOMWARE_HEAD):
            if head_config.enabled.get(head):
                pos = neg = 0
                for rec in records:
                    val = _bool_label(_record_labels(rec).get(head))
                    if val is True:
                        pos += 1
                    elif val is False:
                        neg += 1
                if pos > 0 and neg > 0:
                    raw_pw = neg / pos
                    pw = min(raw_pw, pos_weight_cap)
                    weights[head] = torch.tensor(pw, dtype=torch.float32, device=self.device)
                    logger.info("Stage 2 %s pos_weight=%.3f (capped from raw %.1f; pos=%d, neg=%d)",
                                head, pw, raw_pw, pos, neg)
        return weights

    def _compute_loss(
        self,
        outputs: Mapping[str, "torch.Tensor"],
        targets: Mapping[str, Tuple["torch.Tensor", "torch.Tensor"]],
    ) -> "torch.Tensor":
        """Sum the per-head masked losses (Req 8.5).

        MSE for priority_score, CrossEntropy for priority_band, BCEWithLogits for
        the binary heads. Each head's loss is averaged over only the masked
        (label-present) rows; heads with no valid row in the batch contribute 0.
        When class-balanced heads are enabled, the band CE uses inverse-frequency
        class weights and the binary heads use a positive-class ``pos_weight``
        (see :meth:`_compute_class_weights`).
        """
        mse = nn.MSELoss(reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean", weight=self._class_weights.get("band"))
        bce = nn.BCEWithLogitsLoss(reduction="mean")

        total = torch.zeros((), dtype=torch.float32, device=self.device)

        if PRIORITY_SCORE_HEAD in outputs:
            mask, tgt = targets[PRIORITY_SCORE_HEAD]
            if mask.any():
                pred = outputs[PRIORITY_SCORE_HEAD][mask]
                total = total + mse(pred, tgt[mask])

        if PRIORITY_BAND_HEAD in outputs:
            mask, tgt = targets[PRIORITY_BAND_HEAD]
            if mask.any():
                logits = outputs[PRIORITY_BAND_HEAD][mask]
                total = total + ce(logits, tgt[mask])

        for head in (IN_KEV_HEAD, RANSOMWARE_HEAD):
            if head in outputs:
                mask, tgt = targets[head]
                if mask.any():
                    head_bce = (
                        nn.BCEWithLogitsLoss(reduction="mean", pos_weight=self._class_weights[head])
                        if head in self._class_weights
                        else bce
                    )
                    total = total + head_bce(outputs[head][mask], tgt[mask])

        return total

    # ------------------------------------------------------------------ #
    # Prepare (dry) + train
    # ------------------------------------------------------------------ #
    def prepare(self, train_records: Sequence[Mapping[str, Any]]) -> HeadConfiguration:
        """Detect which heads to build from the training labels (Req 8.7).

        Returns the :class:`HeadConfiguration` (enabled heads + skip reasons +
        band vocabulary) without loading the encoder, so a run's head plan can be
        inspected cheaply before committing to training.
        """
        head_config = detect_head_configuration(train_records)
        for head in ALL_HEADS:
            if not head_config.enabled.get(head):
                logger.warning(
                    "Stage 2: skipping '%s' head — %s",
                    head,
                    head_config.skip_reasons.get(head, "label absent"),
                )
        self.head_config = head_config
        return head_config

    def train(
        self,
        train_path: str | Path,
        validation_path: Optional[str | Path] = None,
    ) -> Stage2Result:
        """Run Stage 2: attach heads over frozen embeddings and train (Req 8.4-8.7).

        Args:
            train_path: Path to the training ``CVE_View_Records`` JSONL.
            validation_path: Optional validation split JSONL, used to select the
                best checkpoint by validation loss.

        Returns:
            A :class:`Stage2Result` describing the enabled/skipped heads, the best
            checkpoint path (when trained), and the written report path.
        """
        self._ensure_torch()

        train_records = load_view_records(train_path)
        head_config = self.prepare(train_records)

        result = Stage2Result(
            head_configuration=head_config,
            stage1_checkpoint_path=self.stage1_checkpoint_path,
        )

        if not head_config.any_enabled():
            # Every label absent -> nothing to train. Record and stop (Req 8.7).
            logger.warning(
                "Stage 2: no supervised label present on any training record; "
                "all heads skipped, nothing to train."
            )
            result.report_path = self._write_report(result)
            return result

        validation_records = (
            load_view_records(validation_path) if validation_path else []
        )

        self._load_encoder_and_projection()
        heads = self.build_heads(head_config)

        # Class-imbalance weighting for the classification heads (counters the
        # majority-class collapse). Gated by config; on by default.
        if getattr(self.config, "cve_class_balanced_heads", True):
            self._class_weights = self._compute_class_weights(train_records, head_config)
        else:
            self._class_weights = {}

        weight_decay = getattr(self.config, "weight_decay", 0.0)
        optimizer = optim.Adam(
            heads.trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=weight_decay,
        )

        # The encoder (and Stage 1 projection) are frozen, so a record's embedding
        # is identical on every epoch. Embed each split once up front and reuse the
        # cached slices — this avoids re-encoding the whole training/validation set
        # on every epoch (byte-identical results, far less compute on large splits).
        train_valid, train_emb = self._embed_records_once(train_records)
        val_valid, val_emb = self._embed_records_once(validation_records)

        best_val_loss = float("inf")
        epochs = max(1, int(self.config.num_epochs))
        batch_size = max(1, int(self.config.batch_size))
        n_train = len(train_valid)
        for epoch in range(epochs):
            heads.train()
            epoch_losses: List[float] = []
            for start in range(0, n_train, batch_size):
                batch = train_valid[start:start + batch_size]
                embeddings = train_emb[start:start + batch_size].to(self.device)
                targets = self._head_targets(batch, head_config)
                outputs = heads(embeddings)
                loss = self._compute_loss(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads.trainable_parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")

            val_loss = None
            if val_valid:
                val_loss = self._evaluate_cached(heads, val_valid, val_emb, head_config)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    result.best_checkpoint_path = self._save_checkpoint(
                        heads, head_config, epoch, val_loss
                    )
            logger.info(
                "Stage 2 epoch %d/%d: train_loss=%.4f%s",
                epoch + 1,
                epochs,
                train_loss,
                f", val_loss={val_loss:.4f}" if val_loss is not None else "",
            )
            result.epochs_trained = epoch + 1

        # With no usable validation split, persist the final heads as the checkpoint.
        if not val_valid:
            result.best_checkpoint_path = self._save_checkpoint(
                heads, head_config, result.epochs_trained - 1, best_val_loss
            )
        else:
            result.best_val_loss = best_val_loss if best_val_loss != float("inf") else None

        result.report_path = self._write_report(result)
        return result

    def _evaluate_cached(
        self,
        heads: Any,
        records: Sequence[Mapping[str, Any]],
        embeddings: "torch.Tensor",
        head_config: HeadConfiguration,
    ) -> float:
        """Average combined head loss over pre-embedded validation records.

        Same as :meth:`_evaluate` but consumes the one-time embedding cache
        (:meth:`_embed_records_once`) instead of re-encoding each epoch.
        """
        heads.eval()
        losses: List[float] = []
        batch_size = max(1, int(self.config.batch_size))
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                emb = embeddings[start:start + batch_size].to(self.device)
                targets = self._head_targets(batch, head_config)
                outputs = heads(emb)
                losses.append(float(self._compute_loss(outputs, targets).item()))
        return sum(losses) / len(losses) if losses else float("inf")

    def _evaluate(
        self,
        heads: Any,
        records: Sequence[Mapping[str, Any]],
        head_config: HeadConfiguration,
    ) -> float:
        """Average combined head loss over the validation records (no grad)."""
        heads.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in self._iter_batches(records):
                embeddings = self._embed_views([r["encoder_view"] for r in batch])
                targets = self._head_targets(batch, head_config)
                outputs = heads(embeddings)
                losses.append(float(self._compute_loss(outputs, targets).item()))
        return sum(losses) / len(losses) if losses else float("inf")

    # ------------------------------------------------------------------ #
    # Inference (predictions for the Evaluation_Reporter)
    # ------------------------------------------------------------------ #
    def load_heads_from_checkpoint(self, checkpoint_path: str | Path) -> "HeadConfiguration":
        """Rebuild the trained heads from a Stage 2 checkpoint for inference.

        Lets a runner resume straight to prediction/evaluation without retraining
        when ``stage2_best_checkpoint.pt`` already exists. Reconstructs the head
        configuration (enabled heads + band vocabulary) and loads the weights over
        the frozen encoder/projection.
        """
        self._load_encoder_and_projection()
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        enabled = set(checkpoint.get("enabled_heads", []))
        band_vocab = list(checkpoint.get("band_vocabulary", []))
        head_config = HeadConfiguration(
            enabled={h: (h in enabled) for h in ALL_HEADS},
            band_vocabulary=band_vocab,
        )
        heads = self.build_heads(head_config)
        heads.load_state_dict(checkpoint["heads_state_dict"])
        heads.eval()
        self.heads = heads
        self.head_config = head_config
        return head_config

    def predict_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Predict per-CVE outputs for the trained heads, keyed by ``cve``.

        Produces the prediction dict the :class:`CVEEvaluationReporter` consumes:

        - ``ranking_score``: the ``priority_score`` regression head output (the
          model's ranking signal, scored against ground-truth ``priority_score``).
        - ``in_kev``: probability from the ``in_kev`` binary head (sigmoid).
        - ``priority_band``: the argmax class from the multiclass head, mapped back
          through the training band vocabulary.
        - ``embedding``: the frozen embedding the heads operate over (for the
          band-separation diagnostics).

        Only keys for enabled heads are emitted; the reporter skips metrics whose
        prediction/label is absent. Must be called after :meth:`train` (or after
        heads have been built), so ``self.heads`` / ``self.head_config`` are set.
        """
        self._ensure_torch()
        if self.heads is None or self.head_config is None:
            raise RuntimeError(
                "predict_records requires trained heads; call train() first."
            )

        band_vocab = self.head_config.band_vocabulary
        predictions: Dict[str, Dict[str, Any]] = {}

        self.heads.eval()
        with torch.no_grad():
            for batch in self._iter_batches(records):
                embeddings = self._embed_views([r["encoder_view"] for r in batch])
                outputs = self.heads(embeddings)
                for i, rec in enumerate(batch):
                    cve = str(rec.get("cve", "")).strip()
                    if not cve:
                        continue
                    pred: Dict[str, Any] = {"embedding": embeddings[i].detach().cpu().tolist()}
                    if PRIORITY_SCORE_HEAD in outputs:
                        # Head is trained on the normalized [0, 1] target; map the
                        # prediction back to the human-meaningful 0–100 scale so it
                        # is comparable to the ground-truth priority_score.
                        pred["ranking_score"] = (
                            float(outputs[PRIORITY_SCORE_HEAD][i].item()) * PRIORITY_SCORE_SCALE
                        )
                    if IN_KEV_HEAD in outputs:
                        pred["in_kev"] = float(torch.sigmoid(outputs[IN_KEV_HEAD][i]).item())
                    if PRIORITY_BAND_HEAD in outputs and band_vocab:
                        idx = int(torch.argmax(outputs[PRIORITY_BAND_HEAD][i]).item())
                        if 0 <= idx < len(band_vocab):
                            pred["priority_band"] = band_vocab[idx]
                    predictions[cve] = pred
        return predictions

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save_checkpoint(
        self,
        heads: Any,
        head_config: HeadConfiguration,
        epoch: int,
        val_loss: float,
    ) -> str:
        path = self.output_dir / STAGE2_BEST_CHECKPOINT
        torch.save(
            {
                "epoch": epoch,
                "heads_state_dict": heads.state_dict(),
                "enabled_heads": [h for h in ALL_HEADS if head_config.enabled.get(h)],
                "band_vocabulary": head_config.band_vocabulary,
                "embedding_dim": self.embedding_dim,
                "stage1_checkpoint_path": self.stage1_checkpoint_path,
                "val_loss": None if val_loss == float("inf") else val_loss,
            },
            path,
        )
        return str(path)

    def _write_report(self, result: Stage2Result) -> str:
        path = self.output_dir / STAGE2_REPORT_FILENAME
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
        return str(path)
