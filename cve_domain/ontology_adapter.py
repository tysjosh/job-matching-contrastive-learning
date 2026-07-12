"""CVE Ontology_Adapter — denominator-pool index + Cyber_KG access (Req 5.1, 5.7).

This module implements :class:`CVEOntologyAdapter`, the CVE-domain analogue of the
career domain's ESCO-graph usage. It maps the CVE ontology artifacts onto the data
the CDCL negative-selection path needs:

* It loads ``cve_denominator_pools.jsonl`` and indexes each ``Denominator_Pool`` by
  its ``cve`` identifier, retaining exactly one pool per unique ``cve`` (first wins)
  — Requirement 5.1.
* It provides access to the ``Cyber_KG`` (``cyber_kg.gexf``), the domain analogue of
  the ESCO knowledge graph, loaded through NetworkX like ``CareerGraph`` does.
* If the pools file cannot be read, or one of its lines cannot be parsed as a
  ``Denominator_Pool``, it stops **before training** by raising
  :class:`CVEOntologyLoadError`, which names the offending file or line
  — Requirement 5.7.

The ``CVENegativeSelector`` (task 7.4) consumes this adapter's index to draw tiered
negatives; the ``CVEPositiveSelector`` (task 7.10) may consult the Cyber_KG. Only the
loading/indexing/access surface (task 7.1) lives here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

try:  # NetworkX is only required when the Cyber_KG is actually accessed.
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in NetworkX-less envs
    NETWORKX_AVAILABLE = False


class CVEOntologyLoadError(Exception):
    """Raised when the ontology pools cannot be loaded (Req 5.7).

    The error stops the pipeline before training and reports the offending file
    and, when the failure is a per-line parse error, the 1-based line number and
    the underlying reason.

    Attributes:
        file_path: The ontology file whose read/parse failed.
        line_number: The 1-based line number that failed to parse, or ``None``
            when the failure is a whole-file read error.
        reason: A human-readable description of what went wrong.
    """

    def __init__(
        self,
        file_path: str,
        reason: str,
        line_number: Optional[int] = None,
    ) -> None:
        self.file_path = file_path
        self.line_number = line_number
        self.reason = reason
        if line_number is not None:
            message = (
                f"Failed to parse denominator pool at {file_path}:{line_number}: "
                f"{reason}"
            )
        else:
            message = f"Failed to read ontology file {file_path}: {reason}"
        super().__init__(message)


@dataclass(frozen=True)
class DenominatorPool:
    """One CVE's tiered negative pools (one line of the pools JSONL).

    Attributes:
        cve: The anchor CVE identifier this pool belongs to.
        hard_negatives: Ontology hard-negative CVE identifiers (may be empty).
        medium_negatives: Ontology medium-negative CVE identifiers (may be empty).
        easy_negatives: Ontology easy-negative CVE identifiers (may be empty).
    """

    cve: str
    hard_negatives: List[str] = field(default_factory=list)
    medium_negatives: List[str] = field(default_factory=list)
    easy_negatives: List[str] = field(default_factory=list)


# The three tiered negative list fields carried by a Denominator_Pool line.
_TIER_KEYS = ("hard_negatives", "medium_negatives", "easy_negatives")


def _coerce_negative_list(value: Any, key: str) -> List[str]:
    """Validate and normalize one tier's negative list.

    A missing key defaults to an empty list (the data legitimately carries empty
    ``hard``/``medium`` tiers). A present value that is not a list of strings is a
    parse failure — the caller turns it into a :class:`CVEOntologyLoadError`.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be a list, got {type(value).__name__}")
    result: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"'{key}' must contain only strings, got {type(item).__name__}"
            )
        result.append(item)
    return result


def _parse_pool(obj: Any) -> DenominatorPool:
    """Turn a decoded JSON value into a :class:`DenominatorPool`.

    Raises ``ValueError`` when the object is not a valid Denominator_Pool (not a
    JSON object, or missing/empty ``cve``); the caller wraps it with file/line
    context for Req 5.7.
    """
    if not isinstance(obj, dict):
        raise ValueError("line is not a JSON object")
    cve = obj.get("cve")
    if not isinstance(cve, str) or not cve.strip():
        raise ValueError("missing or empty 'cve' identifier")
    cve = cve.strip()
    return DenominatorPool(
        cve=cve,
        hard_negatives=_coerce_negative_list(obj.get("hard_negatives"), "hard_negatives"),
        medium_negatives=_coerce_negative_list(
            obj.get("medium_negatives"), "medium_negatives"
        ),
        easy_negatives=_coerce_negative_list(obj.get("easy_negatives"), "easy_negatives"),
    )


class CVEOntologyAdapter:
    """Loads and indexes CVE ontology pools and provides Cyber_KG access.

    On construction the adapter eagerly reads and indexes
    ``cve_denominator_pools.jsonl`` so that any read/parse failure surfaces
    **before training** (Req 5.7). The Cyber_KG graph is loaded lazily on first
    access to avoid the NetworkX cost when only the pools are needed.

    Args:
        denominator_pools_path: Path to ``cve_denominator_pools.jsonl``.
        cyber_kg_path: Optional path to ``cyber_kg.gexf``. When provided, the graph
            is available via :attr:`cyber_kg`.

    Raises:
        CVEOntologyLoadError: If the pools file cannot be read or one of its lines
            cannot be parsed as a Denominator_Pool.
    """

    def __init__(
        self,
        denominator_pools_path: str,
        cyber_kg_path: Optional[str] = None,
    ) -> None:
        self.denominator_pools_path = denominator_pools_path
        self.cyber_kg_path = cyber_kg_path

        # Index of pools by unique cve id (first-wins per Req 5.1).
        self._pools: Dict[str, DenominatorPool] = {}
        # Count of later duplicate-cve lines discarded (kept for reporting).
        self.duplicate_pool_count = 0

        # Lazily-loaded Cyber_KG.
        self._cyber_kg: Optional[Any] = None

        self._load_pools()

    # ---- Denominator pool loading / access (Req 5.1, 5.7) -------------------

    def _load_pools(self) -> None:
        """Read and index the denominator pools, stopping on any failure (Req 5.7)."""
        try:
            handle = open(self.denominator_pools_path, "r", encoding="utf-8")
        except OSError as exc:
            raise CVEOntologyLoadError(
                self.denominator_pools_path, reason=str(exc)
            ) from exc

        try:
            with handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        # Tolerate blank separator lines in JSONL.
                        continue
                    try:
                        obj = json.loads(stripped)
                        pool = _parse_pool(obj)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise CVEOntologyLoadError(
                            self.denominator_pools_path,
                            reason=str(exc),
                            line_number=line_number,
                        ) from exc

                    # First-wins per unique cve id (Req 5.1): keep the first pool
                    # seen and discard later duplicates, counting them.
                    if pool.cve in self._pools:
                        self.duplicate_pool_count += 1
                        continue
                    self._pools[pool.cve] = pool
        except OSError as exc:  # read error partway through the file
            raise CVEOntologyLoadError(
                self.denominator_pools_path, reason=str(exc)
            ) from exc

    def get_pool(self, cve: str) -> Optional[DenominatorPool]:
        """Return the indexed Denominator_Pool for ``cve``, or ``None`` if absent."""
        return self._pools.get(cve)

    def has_pool(self, cve: str) -> bool:
        """Return True when a Denominator_Pool is indexed for ``cve``."""
        return cve in self._pools

    def pool_count(self) -> int:
        """Return the number of unique indexed Denominator_Pools."""
        return len(self._pools)

    def cve_ids(self) -> Iterator[str]:
        """Iterate the indexed CVE identifiers in first-seen (insertion) order."""
        return iter(self._pools.keys())

    def __len__(self) -> int:
        return len(self._pools)

    def __contains__(self, cve: object) -> bool:
        return cve in self._pools

    # ---- Cyber_KG access ----------------------------------------------------

    @property
    def cyber_kg(self) -> Any:
        """Return the Cyber_KG graph, loading ``cyber_kg.gexf`` on first access.

        The graph is the CVE-domain analogue of the ESCO knowledge graph and is
        loaded with NetworkX like ``CareerGraph`` does. Loading is lazy so callers
        that only need the denominator pools never pay the graph-read cost.

        Raises:
            CVEOntologyLoadError: If no Cyber_KG path was configured, NetworkX is
                unavailable, or the graph file cannot be read.
        """
        if self._cyber_kg is None:
            self._cyber_kg = self._load_cyber_kg()
        return self._cyber_kg

    def _load_cyber_kg(self) -> Any:
        """Load the Cyber_KG GEXF graph, reporting failures via CVEOntologyLoadError."""
        if not self.cyber_kg_path:
            raise CVEOntologyLoadError(
                "<cyber_kg>", reason="no Cyber_KG path was configured"
            )
        if not NETWORKX_AVAILABLE:
            raise CVEOntologyLoadError(
                self.cyber_kg_path,
                reason=(
                    "NetworkX is required for Cyber_KG access; install it via "
                    "'pip install networkx'"
                ),
            )
        if not os.path.exists(self.cyber_kg_path):
            raise CVEOntologyLoadError(
                self.cyber_kg_path, reason="Cyber_KG graph file not found"
            )
        try:
            return nx.read_gexf(self.cyber_kg_path)
        except Exception as exc:  # networkx raises a variety of parse errors
            raise CVEOntologyLoadError(
                self.cyber_kg_path, reason=str(exc)
            ) from exc

    def has_cyber_kg_node(self, node: str) -> bool:
        """Return True when ``node`` is present in the Cyber_KG."""
        return node in self.cyber_kg
