"""Supervised heads over frozen pretrained embeddings (task 10.1).

Stage 2 of the two-stage CVE pipeline attaches supervised heads on top of the
**frozen** pretrained embeddings produced by the Stage 1 contrastive encoder
(the 128-dim projection-head output). The encoder is frozen by default to avoid
the catastrophic forgetting the prior CDCL work observed when unfreezing the
sentence-transformer (Req 9.1), so these heads are the only trainable component
in the default Stage 2 configuration.

Four heads are provided, one per supervised target carried on each
``CVE_View_Record`` (Req 8.5):

- ``priority_score`` — a **regression** head (single raw output) predicting the
  float priority target in ``[0, 100]``.
- ``priority_band`` — a **multiclass** head emitting one logit per priority band.
- ``in_kev`` — a **binary** head (single logit) for CISA KEV membership.
- ``ransomware`` — a **binary** head (single logit) for known ransomware use.

The heads emit raw scores / logits (no final activation) so the Stage 2 trainer
can pair them with the appropriate loss (``MSELoss`` for regression,
``CrossEntropyLoss`` for the multiclass band, ``BCEWithLogitsLoss`` for the two
binary heads). Keeping activation out of the module also makes the outputs
directly usable for the ranking / classification metrics in the evaluation
reporter.

Each head can be **individually enabled or skipped**. Stage 2 (task 10.2) skips
any head whose supervised label is absent from every training record (Req 8.7),
so :class:`CVESupervisedHeads` only builds and runs the heads it is told to
enable, and its ``forward`` returns a dict containing only the enabled heads.

Requirements: 8.5, 9.1
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard runtime dependency
    TORCH_AVAILABLE = False


#: Canonical head names, matching the supervised label keys on a CVE_View_Record.
PRIORITY_SCORE_HEAD = "priority_score"
PRIORITY_BAND_HEAD = "priority_band"
IN_KEV_HEAD = "in_kev"
RANSOMWARE_HEAD = "ransomware"

#: Default embedding dimension: the Stage 1 projection-head output (128-dim).
DEFAULT_EMBEDDING_DIM = 128
#: Default hidden width for each head's MLP.
DEFAULT_HIDDEN_DIM = 64
#: Default dropout applied inside each head for regularization.
DEFAULT_DROPOUT = 0.1


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for CVE supervised heads. "
            "Install with: pip install torch"
        )


if TORCH_AVAILABLE:

    def _init_linear_weights(module: nn.Module) -> None:
        """Xavier-initialize the linear layers of ``module`` (bias -> 0)."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    class SupervisedHead(nn.Module):
        """A small MLP head over a frozen embedding.

        Architecture: ``Linear(embedding_dim, hidden_dim) -> ReLU -> Dropout ->
        Linear(hidden_dim, output_dim)``. The head emits raw scores / logits with
        no final activation so the trainer can apply the appropriate loss.

        Args:
            embedding_dim: Dimension of the frozen input embedding.
            output_dim: Number of outputs (1 for regression/binary, ``num_classes``
                for multiclass).
            hidden_dim: Width of the hidden layer.
            dropout: Dropout probability applied after the hidden activation.
            squeeze_output: When ``True`` (regression / binary heads with a single
                output), the trailing size-1 dimension is squeezed so the head
                returns shape ``(batch,)`` instead of ``(batch, 1)``.
        """

        def __init__(
            self,
            embedding_dim: int,
            output_dim: int,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            dropout: float = DEFAULT_DROPOUT,
            squeeze_output: bool = False,
        ) -> None:
            super().__init__()
            if embedding_dim <= 0:
                raise ValueError(f"embedding_dim must be positive, got: {embedding_dim}")
            if output_dim <= 0:
                raise ValueError(f"output_dim must be positive, got: {output_dim}")
            if hidden_dim <= 0:
                raise ValueError(f"hidden_dim must be positive, got: {hidden_dim}")
            if not 0.0 <= dropout <= 1.0:
                raise ValueError(f"dropout must be between 0.0 and 1.0, got: {dropout}")

            self.output_dim = output_dim
            self.squeeze_output = squeeze_output
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            _init_linear_weights(self.mlp)

        def forward(self, embeddings: "torch.Tensor") -> "torch.Tensor":
            out = self.mlp(embeddings)
            if self.squeeze_output and out.shape[-1] == 1:
                out = out.squeeze(-1)
            return out

    class CVESupervisedHeads(nn.Module):
        """Groups the four supervised heads over frozen pretrained embeddings.

        Any subset of heads can be enabled. ``forward`` returns a dict keyed by
        head name containing only the enabled heads' outputs, so Stage 2 can skip
        a head whose label is absent from every training record (Req 8.7).

        Output shapes (per enabled head):
          - ``priority_score`` -> ``(batch,)`` raw regression score
          - ``priority_band``  -> ``(batch, num_priority_bands)`` class logits
          - ``in_kev``         -> ``(batch,)`` binary logit
          - ``ransomware``     -> ``(batch,)`` binary logit

        Args:
            embedding_dim: Dimension of the frozen input embedding (Stage 1
                projection-head output; default 128).
            num_priority_bands: Number of ``priority_band`` classes for the
                multiclass head. Required only when the band head is enabled.
            hidden_dim: Hidden width shared by every head's MLP.
            dropout: Dropout probability shared by every head.
            enable_priority_score / enable_priority_band / enable_in_kev /
            enable_ransomware: Whether to build and run each head. Disabled heads
                are not created and are omitted from ``forward``'s output.
        """

        def __init__(
            self,
            embedding_dim: int = DEFAULT_EMBEDDING_DIM,
            num_priority_bands: Optional[int] = None,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            dropout: float = DEFAULT_DROPOUT,
            enable_priority_score: bool = True,
            enable_priority_band: bool = True,
            enable_in_kev: bool = True,
            enable_ransomware: bool = True,
        ) -> None:
            _require_torch()
            super().__init__()

            if not any(
                [
                    enable_priority_score,
                    enable_priority_band,
                    enable_in_kev,
                    enable_ransomware,
                ]
            ):
                raise ValueError("At least one supervised head must be enabled")

            self.embedding_dim = embedding_dim
            self.num_priority_bands = num_priority_bands

            # Regression head: single raw output for priority_score in [0, 100].
            self.priority_score_head = (
                SupervisedHead(
                    embedding_dim, output_dim=1, hidden_dim=hidden_dim,
                    dropout=dropout, squeeze_output=True,
                )
                if enable_priority_score
                else None
            )

            # Multiclass head: one logit per priority band.
            self.priority_band_head = None
            if enable_priority_band:
                if num_priority_bands is None or num_priority_bands < 2:
                    raise ValueError(
                        "num_priority_bands must be >= 2 when the priority_band head "
                        f"is enabled, got: {num_priority_bands}"
                    )
                self.priority_band_head = SupervisedHead(
                    embedding_dim, output_dim=num_priority_bands,
                    hidden_dim=hidden_dim, dropout=dropout,
                )

            # Binary heads: a single logit each.
            self.in_kev_head = (
                SupervisedHead(
                    embedding_dim, output_dim=1, hidden_dim=hidden_dim,
                    dropout=dropout, squeeze_output=True,
                )
                if enable_in_kev
                else None
            )
            self.ransomware_head = (
                SupervisedHead(
                    embedding_dim, output_dim=1, hidden_dim=hidden_dim,
                    dropout=dropout, squeeze_output=True,
                )
                if enable_ransomware
                else None
            )

        @property
        def enabled_heads(self) -> Dict[str, bool]:
            """Map of head name -> whether it is enabled (built)."""
            return {
                PRIORITY_SCORE_HEAD: self.priority_score_head is not None,
                PRIORITY_BAND_HEAD: self.priority_band_head is not None,
                IN_KEV_HEAD: self.in_kev_head is not None,
                RANSOMWARE_HEAD: self.ransomware_head is not None,
            }

        def forward(self, embeddings: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            """Run every enabled head over the frozen ``embeddings``.

            Args:
                embeddings: Frozen pretrained embeddings, shape
                    ``(batch, embedding_dim)``.

            Returns:
                A dict keyed by head name containing only the enabled heads'
                outputs. Disabled heads are omitted.
            """
            outputs: Dict[str, "torch.Tensor"] = {}
            if self.priority_score_head is not None:
                outputs[PRIORITY_SCORE_HEAD] = self.priority_score_head(embeddings)
            if self.priority_band_head is not None:
                outputs[PRIORITY_BAND_HEAD] = self.priority_band_head(embeddings)
            if self.in_kev_head is not None:
                outputs[IN_KEV_HEAD] = self.in_kev_head(embeddings)
            if self.ransomware_head is not None:
                outputs[RANSOMWARE_HEAD] = self.ransomware_head(embeddings)
            return outputs

        def trainable_parameters(self):
            """Return the head parameters that require gradients.

            Convenience for the Stage 2 optimizer setup: over frozen embeddings
            these heads are the only trainable component (Req 9.1).
            """
            return [p for p in self.parameters() if p.requires_grad]
