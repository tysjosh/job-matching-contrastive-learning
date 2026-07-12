"""CVEPositiveSelector — ontology-related positive selection for Stage 1 (Req 13).

This module implements :class:`CVEPositiveSelector`, the CVE-domain
``Positive_Selector``. For a CVE anchor it selects an *ontology-related* CVE from
the anchor's **same split** to serve as the contrastive positive — the anchor and
its selected positive are the two views of a training pair (the resume/job
analogue in the career domain), so **no data augmentation** and **no resume/job
mirror shim** are used.

Ontology relatedness is read from the ``ontology`` object carried on each
``CVE_View_Record`` (``{cwes, cpes, vendors}``), produced by the
``CVEDataConverter``. (The ``cyber_kg.gexf`` graph is intentionally *not*
consulted here: it is currently malformed on disk, and the per-record ontology
fields already carry the CWE / CPE / vendor structure the cascade needs.)

Selection uses a **priority cascade** (Req 13.1), in order:

1. a candidate sharing at least one CWE **and** at least one CPE or vendor with
   the anchor;
2. else a candidate sharing at least one CWE;
3. else a candidate sharing at least one vendor or product.

Selection rules:

* **Same split only** (Req 13.2): the inverted index is built per split, so every
  candidate is drawn only from the anchor's own split.
* **Seeded / reproducible** (Req 13.3): the choice among equally-eligible
  candidates at a cascade level uses a per-anchor RNG derived from the Run_Config
  ``split_seed``, so identical positive assignments are produced across repeated
  runs with the same seed. Candidate sets are sorted before the seeded draw so the
  result never depends on set iteration order.
* **Exclusions** (Req 13.4): the anchor's own ``cve`` identifier and any ``cve``
  already selected as a *negative* for that anchor are removed from the candidate
  set at every cascade level.
* **No fabrication** (Req 13.5): an anchor with no in-split ontology candidate
  across all three cascade levels is **excluded from Stage 1** and counted; no
  positive is fabricated.
* **Reporting** (Req 13.6): on split completion the count of anchors resolved at
  each cascade level and the excluded-anchor count are written to
  ``positive_selection_report.json``.

**Efficiency over ~347K records** (design 7a): a naive pairwise sibling search is
quadratic. Instead the selector builds, **per split**, an inverted index
(CWE → cves, CPE → cves, vendor → cves) once, then resolves each anchor's
candidates by set unions / intersections against that index. This keeps positive
selection near-linear in the number of records rather than scanning all pairs.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# The three cascade levels, in priority order (Req 13.1). The keys are the report
# field names used in positive_selection_report.json.
CASCADE_CWE_AND_CPE_OR_VENDOR = "cwe_and_cpe_or_vendor"
CASCADE_CWE = "cwe"
CASCADE_VENDOR_OR_PRODUCT = "vendor_or_product"
CASCADE_LEVELS = (
    CASCADE_CWE_AND_CPE_OR_VENDOR,
    CASCADE_CWE,
    CASCADE_VENDOR_OR_PRODUCT,
)

# Report artifact filename written on split completion (Req 13.6).
REPORT_FILENAME = "positive_selection_report.json"


@dataclass
class PositiveSelectionReport:
    """Structured accounting of positive selection over a split (Req 13.6).

    Attributes:
        resolved_by_cascade_level: Count of anchors resolved at each cascade
            level — ``cwe_and_cpe_or_vendor`` (shares a CWE and a CPE/vendor),
            ``cwe`` (shares a CWE only), and ``vendor_or_product`` (shares a
            vendor/product only) (Req 13.1, 13.6).
        excluded_anchor_count: Count of anchors excluded from Stage 1 because they
            had no in-split ontology candidate across all cascade levels; no
            positive was fabricated for them (Req 13.5).
        anchors_processed: Number of anchors selection was run for (context).
    """

    resolved_by_cascade_level: Dict[str, int] = field(
        default_factory=lambda: {level: 0 for level in CASCADE_LEVELS}
    )
    excluded_anchor_count: int = 0
    anchors_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolved_by_cascade_level": dict(self.resolved_by_cascade_level),
            "excluded_anchor_count": self.excluded_anchor_count,
            "anchors_processed": self.anchors_processed,
        }


def _clean_str(value: Any) -> str:
    """Return ``value`` as a whitespace-trimmed string (``""`` when falsy)."""
    if value is None:
        return ""
    return str(value).strip()


def _clean_token_list(value: Any) -> List[str]:
    """Normalize an ontology token list to trimmed, non-empty strings.

    The ``ontology`` object's ``cwes`` / ``cpes`` / ``vendors`` are already
    list-valued on a ``CVE_View_Record``, but this tolerates a stray scalar and
    drops blank tokens so the inverted index only holds meaningful keys.
    """
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, Iterable):
        result: List[str] = []
        for item in value:
            token = _clean_str(item)
            if token:
                result.append(token)
        return result
    return []


class CVEPositiveSelector:
    """Select an ontology-related positive CVE per anchor within a split.

    The selector is built once per split from that split's ``CVE_View_Records``:
    it constructs the inverted index and per-anchor ontology token sets up front
    so each anchor's positive is resolved by cheap set operations (Req 13, design
    7a efficiency note).

    Args:
        seed: The Run_Config ``split_seed``, used to derive per-anchor
            deterministic RNGs so selection is reproducible (Req 13.3).
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

        # Inverted indexes over the current split: token -> set of cve ids.
        self._cwe_index: Dict[str, Set[str]] = {}
        self._cpe_index: Dict[str, Set[str]] = {}
        self._vendor_index: Dict[str, Set[str]] = {}

        # Per-anchor ontology token sets: cve -> (cwes, cpes, vendors).
        self._anchor_ontology: Dict[str, Tuple[Set[str], Set[str], Set[str]]] = {}

        # Ordered list of anchor ids in the current split (index build order).
        self._anchor_ids: List[str] = []

        # Report accumulated across select_positive calls; reset per split run.
        self.report = PositiveSelectionReport()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Any) -> "CVEPositiveSelector":
        """Build a selector from a ``TrainingConfig``-like Run_Config object.

        Reads ``split_seed`` from ``config`` (see ``TrainingConfig``).
        """
        return cls(seed=getattr(config, "split_seed", 42))

    # ------------------------------------------------------------------ #
    # Inverted-index build (per split, once — design 7a)
    # ------------------------------------------------------------------ #
    def build_index(self, split_records: Sequence[Mapping[str, Any]]) -> None:
        """Build the per-split inverted index and per-anchor ontology sets.

        Called once per split before resolving positives. A record without a
        present ``cve`` is ignored; duplicate ``cve`` ids keep the first
        occurrence's ontology (first-wins, matching the converter's dedup).

        Args:
            split_records: The ``CVE_View_Records`` of a single split. Each should
                carry a ``cve`` identifier and an ``ontology`` object
                (``{cwes, cpes, vendors}``).
        """
        self._cwe_index = {}
        self._cpe_index = {}
        self._vendor_index = {}
        self._anchor_ontology = {}
        self._anchor_ids = []

        for record in split_records:
            if not isinstance(record, Mapping):
                continue
            cve = _clean_str(record.get("cve"))
            if not cve or cve in self._anchor_ontology:
                continue

            ontology = record.get("ontology")
            ontology = ontology if isinstance(ontology, Mapping) else {}
            cwes = set(_clean_token_list(ontology.get("cwes")))
            cpes = set(_clean_token_list(ontology.get("cpes")))
            vendors = set(_clean_token_list(ontology.get("vendors")))

            self._anchor_ontology[cve] = (cwes, cpes, vendors)
            self._anchor_ids.append(cve)

            for token in cwes:
                self._cwe_index.setdefault(token, set()).add(cve)
            for token in cpes:
                self._cpe_index.setdefault(token, set()).add(cve)
            for token in vendors:
                self._vendor_index.setdefault(token, set()).add(cve)

    # ------------------------------------------------------------------ #
    # Candidate lookup helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _union(index: Mapping[str, Set[str]], tokens: Iterable[str]) -> Set[str]:
        """Union of the cve-id sets indexed under each of ``tokens``."""
        result: Set[str] = set()
        for token in tokens:
            bucket = index.get(token)
            if bucket:
                result |= bucket
        return result

    def _pick(self, anchor_cve: str, candidates: Set[str]) -> str:
        """Deterministically pick one candidate using the seeded per-anchor RNG.

        Candidates are sorted before the draw so the result is independent of set
        iteration order and reproducible across runs with the same seed (Req 13.3).
        """
        rng = random.Random(f"{self.seed}:{anchor_cve}")
        return rng.choice(sorted(candidates))

    # ------------------------------------------------------------------ #
    # Per-anchor selection (Req 13.1–13.5)
    # ------------------------------------------------------------------ #
    def select_positive(
        self,
        anchor_cve: str,
        excluded_negatives: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Select the ontology-related positive for one anchor, updating the report.

        Args:
            anchor_cve: The anchor CVE identifier. Must be present in the built
                index; an unknown anchor has no ontology and is excluded.
            excluded_negatives: CVE ids already selected as *negatives* for this
                anchor, excluded from the positive-candidate set (Req 13.4).

        Returns:
            The selected positive CVE id, or ``None`` when the anchor has no
            in-split ontology candidate across all cascade levels (the anchor is
            then excluded from Stage 1 and counted — Req 13.5). No positive is
            fabricated.
        """
        anchor_cve = _clean_str(anchor_cve)
        self.report.anchors_processed += 1

        anchor_ontology = self._anchor_ontology.get(anchor_cve)
        if anchor_ontology is None:
            # Unknown anchor -> no ontology -> excluded, no fabrication (Req 13.5).
            self.report.excluded_anchor_count += 1
            return None

        anchor_cwes, anchor_cpes, anchor_vendors = anchor_ontology

        # Build the exclusion set: the anchor's own id + already-selected
        # negatives for this anchor (Req 13.4).
        excluded: Set[str] = {anchor_cve}
        if excluded_negatives is not None:
            for neg in excluded_negatives:
                cid = _clean_str(neg)
                if cid:
                    excluded.add(cid)

        # Candidate universes per shared-signal, with exclusions removed.
        cwe_matches = self._union(self._cwe_index, anchor_cwes) - excluded
        cpe_matches = self._union(self._cpe_index, anchor_cpes) - excluded
        vendor_matches = self._union(self._vendor_index, anchor_vendors) - excluded

        # --- Priority cascade (Req 13.1). ---
        # Level 1: shares a CWE AND (a CPE OR a vendor).
        level1 = cwe_matches & (cpe_matches | vendor_matches)
        if level1:
            self.report.resolved_by_cascade_level[CASCADE_CWE_AND_CPE_OR_VENDOR] += 1
            return self._pick(anchor_cve, level1)

        # Level 2: shares a CWE.
        if cwe_matches:
            self.report.resolved_by_cascade_level[CASCADE_CWE] += 1
            return self._pick(anchor_cve, cwe_matches)

        # Level 3: shares a vendor / product.
        if vendor_matches:
            self.report.resolved_by_cascade_level[CASCADE_VENDOR_OR_PRODUCT] += 1
            return self._pick(anchor_cve, vendor_matches)

        # No candidate at any level -> exclude the anchor, do not fabricate (Req 13.5).
        self.report.excluded_anchor_count += 1
        return None

    # ------------------------------------------------------------------ #
    # Split-level driver + reporting (Req 13.6)
    # ------------------------------------------------------------------ #
    def select_for_split(
        self,
        split_records: Sequence[Mapping[str, Any]],
        negatives_by_anchor: Optional[Mapping[str, Iterable[str]]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build the index, select a positive for every anchor, write the report.

        Args:
            split_records: The ``CVE_View_Records`` of the anchor's split. Each
                must carry a ``cve`` identifier and an ``ontology`` object.
            negatives_by_anchor: Optional mapping of anchor ``cve`` to the ids
                already selected as negatives for that anchor, excluded from the
                positive candidates (Req 13.4).
            output_dir: When provided, ``positive_selection_report.json`` is
                written there on completion (Req 13.6).

        Returns:
            A mapping of each *resolved* anchor ``cve`` to its selected positive
            ``cve``. Excluded anchors (no in-split ontology sibling) are omitted
            from the mapping (Req 13.5).
        """
        self.reset_report()
        self.build_index(split_records)

        selections: Dict[str, str] = {}
        for anchor in self._anchor_ids:
            excluded_negatives = None
            if negatives_by_anchor is not None:
                excluded_negatives = negatives_by_anchor.get(anchor)
            positive = self.select_positive(anchor, excluded_negatives)
            if positive is not None:
                selections[anchor] = positive

        if output_dir is not None:
            self.write_report(output_dir)

        return selections

    def reset_report(self) -> None:
        """Reset the accumulated report (called at the start of a split run)."""
        self.report = PositiveSelectionReport()

    def write_report(self, output_dir: str) -> Path:
        """Write ``positive_selection_report.json`` under ``output_dir`` (Req 13.6)."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / REPORT_FILENAME
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(self.report.to_dict(), handle, indent=2)
        logger.info("Wrote positive-selection report to %s", report_path)
        return report_path
