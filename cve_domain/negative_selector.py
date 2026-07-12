"""CVENegativeSelector — tiered ontology negatives + fallback cascade (Req 5, 6).

This module implements :class:`CVENegativeSelector`, which supplies the tiered
(hard / medium / easy) ontology-constrained negatives for a CVE anchor into the
InfoNCE denominator. It is the CVE-domain analogue of the career pathway/ontology
negative selection, and it consumes the :class:`~cve_domain.ontology_adapter.CVEOntologyAdapter`
index (``get_pool(cve) -> DenominatorPool``). The loss math in
``contrastive_learning/loss_engine.py`` is **not** touched — this selector only
chooses *which* CVE identifiers become negatives.

Selection for a single anchor CVE (:meth:`select_negatives`):

1. Read the anchor's ``hard`` / ``medium`` / ``easy`` pools from the adapter's
   index (an anchor with no indexed pool is treated as all-tiers-empty).
2. Compute per-tier target counts: normalize the configured per-tier ratios to
   sum to 1, multiply by ``max_negatives_per_anchor``, and round to the nearest
   integer (Req 5.5).
3. Draw from each tier by seeded random selection up to that tier's target,
   deduplicating by ``cve`` id and never exceeding the configured maximum
   (Req 5.2, 5.4).
4. Fallback cascade for any shortfall (Req 6): an empty/short ``hard`` tier is
   filled from ``medium`` then ``easy``; an empty/short ``medium`` tier is filled
   from ``easy``; all remaining pooled negatives are supplied before any random
   draw (Req 6.1–6.3). If pooled negatives are still insufficient (fewer pooled
   than the maximum, or all tiers empty), the shortfall is filled by a seeded
   random sample of *other* ``CVE_View_Records`` in the **same split**, excluding
   the anchor's own id and any already-selected id (Req 6.3, 6.4).
5. The anchor's own id is always excluded (Req 6.8) and the returned negatives
   contain no duplicate ids (Req 5.2, 6.6).
6. A referenced negative id that is not present among the converted
   ``CVE_View_Records`` is skipped and counted (Req 5.6).
7. If the unique candidates available (pooled + same-split) are fewer than the
   configured maximum, all available candidates are supplied and the per-anchor
   deficit is recorded without terminating selection (Req 6.5).

On split completion (:meth:`select_for_split`), the per-tier fallback usage
counts and the count of anchors that used the random-sample fallback are written
to ``negative_selection_report.json`` (Req 6.7).

Because ~95% of CVEs have populated hard tiers, the random-sample path is a
defensive minority safety net; the report makes its usage observable.

Requirements: 5.2, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .ontology_adapter import CVEOntologyAdapter, DenominatorPool

logger = logging.getLogger(__name__)

# The three ontology negative tiers, in hardest-first priority order. This order
# governs both the per-tier target draw and the pooled fallback fill (Req 6.1/6.2:
# an empty/short higher tier is filled from the tiers below it, hardest first).
TIER_NAMES = ("hard", "medium", "easy")

# Default per-tier ratios, matching TrainingConfig.negative_tier_ratios so the
# selector and the Run_Config agree when no explicit ratios are supplied.
DEFAULT_TIER_RATIOS: Dict[str, float] = {"hard": 0.34, "medium": 0.33, "easy": 0.33}

# Report artifact filename written on split completion (Req 6.7).
REPORT_FILENAME = "negative_selection_report.json"


@dataclass
class NegativeSelectionReport:
    """Structured accounting of negative selection over a split (design "NegativeSelectionReport").

    Attributes:
        per_tier_selected_counts: Total negatives supplied from each ontology
            tier across all anchors (random-sample negatives are not attributed
            to a tier).
        per_tier_fallback_usage: Count of anchors for which the ``hard`` /
            ``medium`` tier could not meet its target and therefore required
            fallback filling from lower tiers (Req 6.1, 6.2).
        random_fallback_anchor_count: Count of anchors that used the same-split
            random-sample fallback (Req 6.3, 6.4).
        skipped_missing_id_count: Count of referenced pool ids skipped because
            they are absent from the converted ``CVE_View_Records`` (Req 5.6).
        deficit_anchor_count: Count of anchors supplied with fewer than the
            configured maximum because unique candidates ran out (Req 6.5).
        anchors_processed: Number of anchors selection was run for (context).
    """

    per_tier_selected_counts: Dict[str, int] = field(
        default_factory=lambda: {"hard": 0, "medium": 0, "easy": 0}
    )
    per_tier_fallback_usage: Dict[str, int] = field(
        default_factory=lambda: {"hard": 0, "medium": 0}
    )
    random_fallback_anchor_count: int = 0
    skipped_missing_id_count: int = 0
    deficit_anchor_count: int = 0
    anchors_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_tier_selected_counts": dict(self.per_tier_selected_counts),
            "per_tier_fallback_usage": dict(self.per_tier_fallback_usage),
            "random_fallback_anchor_count": self.random_fallback_anchor_count,
            "skipped_missing_id_count": self.skipped_missing_id_count,
            "deficit_anchor_count": self.deficit_anchor_count,
            "anchors_processed": self.anchors_processed,
        }


def _normalize_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    """Normalize per-tier ratios to sum to 1 over the three tiers (Req 5.5).

    Missing tier keys default to 0. Negative ratios are rejected. If the three
    ratios sum to 0 (or none are supplied), the tiers fall back to equal shares
    so target computation stays well defined.
    """
    values = {tier: float(ratios.get(tier, 0.0)) for tier in TIER_NAMES}
    for tier, value in values.items():
        if value < 0:
            raise ValueError(f"Negative tier ratio for {tier!r} is negative: {value}")
    total = sum(values.values())
    if total <= 0:
        return {tier: 1.0 / len(TIER_NAMES) for tier in TIER_NAMES}
    return {tier: value / total for tier, value in values.items()}


class CVENegativeSelector:
    """Select tiered ontology negatives for CVE anchors with a fallback cascade.

    Args:
        ontology_adapter: The loaded :class:`CVEOntologyAdapter` providing
            ``get_pool(cve) -> DenominatorPool`` per anchor (Req 5.1).
        max_negatives_per_anchor: The maximum negatives per anchor; must be a
            positive integer of at least 1 (Req 5.4).
        tier_ratios: Optional per-tier ratios (``hard``/``medium``/``easy``);
            normalized to sum to 1 (Req 5.5). Defaults to
            :data:`DEFAULT_TIER_RATIOS`.
        seed: The split seed from the Run_Config, used to derive per-anchor
            deterministic RNGs so selection is reproducible (Req 11.3, 6.3).
    """

    def __init__(
        self,
        ontology_adapter: CVEOntologyAdapter,
        max_negatives_per_anchor: int,
        tier_ratios: Optional[Mapping[str, float]] = None,
        seed: int = 42,
    ) -> None:
        if int(max_negatives_per_anchor) < 1:
            raise ValueError(
                "max_negatives_per_anchor must be a positive integer of at least 1, "
                f"got {max_negatives_per_anchor!r}"
            )
        self.ontology_adapter = ontology_adapter
        self.max_negatives_per_anchor = int(max_negatives_per_anchor)
        self.tier_ratios = _normalize_ratios(tier_ratios or DEFAULT_TIER_RATIOS)
        self.seed = int(seed)

        # Report accumulated across select_negatives calls; reset per split run.
        self.report = NegativeSelectionReport()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls, ontology_adapter: CVEOntologyAdapter, config: Any
    ) -> "CVENegativeSelector":
        """Build a selector from a ``TrainingConfig``-like Run_Config object.

        Reads ``max_negatives_per_anchor``, ``negative_tier_ratios`` and
        ``split_seed`` from ``config`` (see ``TrainingConfig``).
        """
        return cls(
            ontology_adapter=ontology_adapter,
            max_negatives_per_anchor=getattr(config, "max_negatives_per_anchor", 20),
            tier_ratios=getattr(config, "negative_tier_ratios", None),
            seed=getattr(config, "split_seed", 42),
        )

    # ------------------------------------------------------------------ #
    # Target computation (Req 5.5)
    # ------------------------------------------------------------------ #
    def _tier_targets(self) -> Dict[str, int]:
        """Per-tier target counts: normalized ratio × max, rounded to nearest int."""
        return {
            tier: int(round(self.tier_ratios[tier] * self.max_negatives_per_anchor))
            for tier in TIER_NAMES
        }

    # ------------------------------------------------------------------ #
    # Per-anchor selection (Req 5, 6)
    # ------------------------------------------------------------------ #
    def select_negatives(
        self,
        anchor_cve: str,
        split_cve_ids: Sequence[str],
        present_ids: Optional[Set[str]] = None,
    ) -> List[str]:
        """Select negatives for one anchor CVE, updating :attr:`report`.

        Args:
            anchor_cve: The anchor CVE identifier.
            split_cve_ids: The CVE identifiers in the anchor's split — the
                candidate universe for the same-split random-sample fallback
                (Req 6.3, 6.4).
            present_ids: The set of CVE identifiers present among the converted
                ``CVE_View_Records`` — the validity universe for pooled negatives
                (Req 5.6). Defaults to the set of ``split_cve_ids`` when omitted.

        Returns:
            The ordered, de-duplicated list of selected negative CVE identifiers,
            excluding the anchor's own id and numbering at most
            ``max_negatives_per_anchor`` (Req 5.2, 5.4, 6.6, 6.8).
        """
        anchor_cve = str(anchor_cve).strip()
        if present_ids is None:
            present_ids = set(str(c) for c in split_cve_ids)

        rng = random.Random(f"{self.seed}:{anchor_cve}")
        max_n = self.max_negatives_per_anchor

        selected: List[str] = []
        selected_set: Set[str] = set()

        def add(candidates: Sequence[str], limit: Optional[int], tier: Optional[str]) -> int:
            """Add candidates (deduped, anchor-excluded, capped at max/limit).

            ``limit`` caps how many this call may add (``None`` = up to max).
            ``tier`` attributes added ids to a tier's selected-count, or ``None``
            for the random-sample fallback. Returns the number added.
            """
            added = 0
            for cid in candidates:
                if len(selected) >= max_n:
                    break
                if limit is not None and added >= limit:
                    break
                if cid in selected_set:
                    continue
                selected.append(cid)
                selected_set.add(cid)
                added += 1
                if tier is not None:
                    self.report.per_tier_selected_counts[tier] += 1
            return added

        # --- Read pools and build filtered, seeded-shuffled per-tier candidates.
        pool = self.ontology_adapter.get_pool(anchor_cve)
        tier_candidates: Dict[str, List[str]] = {}
        for tier in TIER_NAMES:
            raw = self._tier_list(pool, tier)
            filtered = self._filter_tier(raw, anchor_cve, present_ids)
            rng.shuffle(filtered)  # seeded random selection within the tier (Req 5.2)
            tier_candidates[tier] = filtered

        targets = self._tier_targets()

        # --- Primary draw: each tier up to its rounded ratio target (Req 5.5).
        tier_added: Dict[str, int] = {}
        for tier in TIER_NAMES:
            tier_added[tier] = add(tier_candidates[tier], targets[tier], tier)

        # --- Fallback usage accounting (Req 6.1, 6.2): a hard/medium tier that
        # could not meet its target triggers fallback filling from lower tiers.
        if tier_added["hard"] < targets["hard"]:
            self.report.per_tier_fallback_usage["hard"] += 1
        if tier_added["medium"] < targets["medium"]:
            self.report.per_tier_fallback_usage["medium"] += 1

        # --- Pooled fallback: supply all remaining pooled negatives before any
        # random draw, hardest-first (Req 6.1, 6.2, 6.3). This lets a lower tier
        # cover a higher tier's shortfall and drains the pools when they hold at
        # least max_n negatives (so the random path is only for truly sparse pools).
        if len(selected) < max_n:
            for tier in TIER_NAMES:
                if len(selected) >= max_n:
                    break
                add(tier_candidates[tier], None, tier)

        # --- Same-split random-sample fallback (Req 6.3, 6.4): only when pooled
        # negatives are exhausted and still short. Draw from other split records,
        # excluding the anchor and already-selected ids, seeded for reproducibility.
        if len(selected) < max_n:
            random_candidates = [
                cid
                for cid in (str(c) for c in split_cve_ids)
                if cid != anchor_cve and cid not in selected_set
            ]
            rng.shuffle(random_candidates)
            random_added = add(random_candidates, None, None)
            if random_added > 0:
                self.report.random_fallback_anchor_count += 1

        # --- Deficit accounting (Req 6.5): fewer unique candidates than the max.
        if len(selected) < max_n:
            self.report.deficit_anchor_count += 1

        self.report.anchors_processed += 1
        return selected

    # ------------------------------------------------------------------ #
    # Split-level driver + reporting (Req 6.7)
    # ------------------------------------------------------------------ #
    def select_for_split(
        self,
        split_records: Sequence[Mapping[str, Any]],
        present_ids: Optional[Set[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Select negatives for every anchor in a split and write the report.

        Args:
            split_records: The ``CVE_View_Records`` of the anchor's split. Each
                must carry a ``cve`` identifier.
            present_ids: Validity universe for pooled negatives (Req 5.6);
                defaults to the split's own ids when omitted.
            output_dir: When provided, ``negative_selection_report.json`` is
                written there on completion (Req 6.7).

        Returns:
            A mapping of each anchor ``cve`` to its selected negative ids.
        """
        self.reset_report()

        split_cve_ids = [
            str(rec.get("cve")).strip()
            for rec in split_records
            if rec.get("cve") is not None and str(rec.get("cve")).strip()
        ]
        if present_ids is None:
            present_ids = set(split_cve_ids)

        selections: Dict[str, List[str]] = {}
        for anchor in split_cve_ids:
            selections[anchor] = self.select_negatives(
                anchor, split_cve_ids, present_ids
            )

        if output_dir is not None:
            self.write_report(output_dir)

        return selections

    def reset_report(self) -> None:
        """Reset the accumulated report (called at the start of a split run)."""
        self.report = NegativeSelectionReport()

    def write_report(self, output_dir: str) -> Path:
        """Write ``negative_selection_report.json`` under ``output_dir`` (Req 6.7)."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / REPORT_FILENAME
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(self.report.to_dict(), handle, indent=2)
        logger.info("Wrote negative-selection report to %s", report_path)
        return report_path

    # ------------------------------------------------------------------ #
    # Candidate helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tier_list(pool: Optional[DenominatorPool], tier: str) -> List[str]:
        """Return the raw id list for ``tier`` from ``pool`` (empty when no pool)."""
        if pool is None:
            return []
        return list(getattr(pool, f"{tier}_negatives", []))

    def _filter_tier(
        self, raw_ids: Sequence[str], anchor_cve: str, present_ids: Set[str]
    ) -> List[str]:
        """Filter a tier's raw ids: drop the anchor, drop+count missing ids.

        Preserves source order and de-duplicates within the tier. The anchor's
        own id is excluded silently (Req 6.8); ids absent from the converted
        records are skipped and counted (Req 5.6).
        """
        seen: Set[str] = set()
        result: List[str] = []
        for raw in raw_ids:
            cid = str(raw).strip()
            if not cid or cid == anchor_cve:
                continue
            if cid not in present_ids:
                # Referenced negative absent from converted records (Req 5.6).
                self.report.skipped_missing_id_count += 1
                continue
            if cid in seen:
                continue
            seen.add(cid)
            result.append(cid)
        return result
