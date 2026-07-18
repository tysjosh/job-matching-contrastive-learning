"""WarmupEmbeddingStore: read-only snapshot of warmup embeddings captured at the
end of Phase 2.

The store is a frozen ``content_key -> embedding`` map used by ORCA to derive
reliability targets (encoder-similarity signal) and to pretrain the
``ReliabilityMLP`` on stable embeddings. Capturing a snapshot lets ORCA keep the
reused warmup embeddings *bit-for-bit identical* for the remainder of a run even
as the projection head continues to train in Phase 4 (Requirements 8.3, 9.2).

Design reference: the "Data Models" note describing
``a read-only map content_key -> frozen_embedding captured at the end of Phase 2``.

Implementation notes:
  - ``snapshot`` detaches and clones every cached embedding so later encoder /
    cache mutations cannot alter the frozen snapshot (bit-for-bit constraint).
  - The source :class:`EmbeddingCache` is never mutated: we only read its
    ``cache`` mapping and copy out of it.
  - The store is read-only after capture: there is no public API to add,
    replace, or remove entries, and lookups return defensive detached clones.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, Mapping

import torch

logger = logging.getLogger(__name__)


class WarmupEmbeddingStore:
    """Immutable ``content_key -> frozen_embedding`` snapshot.

    Construct via :meth:`snapshot`, which captures every item currently held in
    an OSCAR :class:`~contrastive_learning.embedding_cache.EmbeddingCache`
    without mutating it. Once captured, the stored embeddings never change.
    """

    def __init__(self, embeddings: Mapping[str, torch.Tensor]):
        """Store a frozen copy of ``embeddings``.

        Prefer :meth:`snapshot` for capturing from an OSCAR embedding cache. The
        constructor is kept general so the store can also be built from a plain
        mapping (e.g. in tests). Every embedding is detached and cloned so the
        store owns private tensors that no external mutation can reach.
        """
        frozen: Dict[str, torch.Tensor] = {}
        for content_key, embedding in embeddings.items():
            frozen[content_key] = self._freeze(embedding)
        # Store privately; there is no setter, so the mapping is effectively
        # read-only for the store's lifetime.
        self._embeddings: Dict[str, torch.Tensor] = frozen
        logger.info(
            "WarmupEmbeddingStore captured %d frozen warmup embeddings",
            len(self._embeddings),
        )

    @staticmethod
    def _freeze(embedding: torch.Tensor) -> torch.Tensor:
        """Return a detached, gradient-free clone of ``embedding``.

        Cloning decouples the stored tensor from the source cache's storage, so
        subsequent in-place edits or reassignments in the cache leave the
        snapshot bit-for-bit unchanged. ``detach`` drops any autograd history.
        """
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"WarmupEmbeddingStore expects torch.Tensor embeddings, "
                f"got {type(embedding)!r}"
            )
        return embedding.detach().clone()

    @classmethod
    def snapshot(cls, embedding_cache) -> "WarmupEmbeddingStore":
        """Capture a read-only snapshot of every item in ``embedding_cache``.

        Args:
            embedding_cache: An OSCAR
                :class:`~contrastive_learning.embedding_cache.EmbeddingCache`
                (or any object exposing a ``cache`` mapping of
                ``content_key -> torch.Tensor``). The cache is only read from and
                is never modified.

        Returns:
            A :class:`WarmupEmbeddingStore` holding a frozen copy of every
            cached ``content_key -> embedding`` pair (i.e. every training-set
            item that has been embedded into the cache).
        """
        source = getattr(embedding_cache, "cache", None)
        if source is None:
            raise AttributeError(
                "embedding_cache must expose a 'cache' mapping of "
                "content_key -> torch.Tensor to snapshot"
            )
        # Iterate over a shallow copy of the items view so we never rely on the
        # cache remaining unchanged mid-iteration, and never touch it otherwise.
        return cls(dict(source))

    def get(self, content_key: str) -> torch.Tensor:
        """Return a defensive clone of the frozen embedding for ``content_key``.

        Raises:
            KeyError: If ``content_key`` was not captured in the snapshot.
        """
        if content_key not in self._embeddings:
            raise KeyError(content_key)
        # Hand out a clone so callers cannot mutate the store's private tensor
        # in place and violate the bit-for-bit guarantee.
        return self._embeddings[content_key].detach().clone()

    def __getitem__(self, content_key: str) -> torch.Tensor:
        return self.get(content_key)

    def __contains__(self, content_key: object) -> bool:
        return content_key in self._embeddings

    def __len__(self) -> int:
        return len(self._embeddings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._embeddings)

    def content_keys(self) -> frozenset:
        """Return the immutable set of captured content keys."""
        return frozenset(self._embeddings.keys())

    def covers(self, content_keys) -> bool:
        """Return True if every key in ``content_keys`` is present in the store.

        Use this to verify the snapshot covers every training-set item before
        Phase 3/4 (Requirement 8.3: the snapshot must cover every training-set
        item).
        """
        return all(key in self._embeddings for key in content_keys)

    def missing_keys(self, content_keys) -> frozenset:
        """Return the subset of ``content_keys`` not present in the snapshot.

        A convenience for diagnosing incomplete coverage; an empty result means
        full coverage of the requested keys.
        """
        return frozenset(key for key in content_keys if key not in self._embeddings)
