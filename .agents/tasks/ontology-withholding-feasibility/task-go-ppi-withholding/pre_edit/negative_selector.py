"""Grade-tiered negative selection for the GO/PPI domain.

Mirrors ``trials_domain/negative_selector.py``: same duck-typed contract, same
deterministic seeding, same curriculum ramp, same ``career_distances`` units. The
only differences are the tier names and which ontology scores the distance.

    hard tier  grade 1, ``weak_evidence``   -- some assay signal, not established
    easy tier  grade 0, ``no_interaction``  -- no recorded edge

Grade 1 is the point of the dataset. A pair with weak experimental evidence is
topically plausible: the two proteins have been assayed together and something
registered. Telling that apart from an established interaction is the hard
contrast, and it is the contrast where GO similarity was measured at rank-AUC
0.8195 -- far above the 0.485-0.607 that ESCO, ISCO, MeSH and CPC manage on their
own hard contrasts. If ontology-guided negative selection helps anywhere, it
should help here, which is exactly why this domain is worth the build.

Satisfies ``BatchProcessor._select_with_injected_selector``:
    select_batch_negatives(anchor_sample, candidate_negatives, max_negatives, epoch)
        -> Optional[Tuple[List[Dict], List[float]]]
Returns ``None`` to hand control back rather than train on nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_START_HARD_RATIO = 0.2
DEFAULT_END_HARD_RATIO = 0.6

#: Scale converting an ontology distance in ``[0, 1]`` to ``career_distances``,
#: matching ``_select_ontology_negatives``' ``d * 10.0`` convention so every
#: domain hands the loss engine the same units.
CAREER_DISTANCE_SCALE = 10.0

#: Fallback when the matcher cannot score a pair.
NEUTRAL_DISTANCE = 0.5


def _seeded_rng(training_seed: int, epoch: int, anchor_id: str) -> random.Random:
    """A per-(seed, epoch, anchor) RNG, independent of batch processing order."""
    digest = hashlib.sha256(
        f"{training_seed}|{epoch}|{anchor_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class GoPpiNegativeSelector:
    """Selects grade-tiered negatives for an anchor protein.

    Args:
        pools: ``{anchor_id: {"weak_evidence": [...], "no_interaction": [...]}}``.
        views: ``{protein_id: view_slot}`` supplying ``encoder_view`` and both facets.
        matcher: Optional :class:`GoMatcher` used to score ``career_distances`` and,
            when ``go_tiered`` is set, to order each tier.
        total_epochs: Denominator for the curriculum ramp.
        training_seed: Seed for deterministic selection.
        start_hard_ratio / end_hard_ratio: Curriculum endpoints for the grade-1 share.
        go_tiered: When ``False`` each tier is sampled uniformly; when ``True``
            each tier is ordered by GO simGIC distance to the anchor and drawn
            from the closest end. Both arms use the same pool, the same grade mix
            and the same label budget. Mirrors ``trials_mesh_tiered_negatives``
            and career's ``ontology_guided_negatives``.
        tier_sampling: How the ordered tier is drawn -- and the reason this
            argument exists is a CONFOUND worth being explicit about.

            ``"deterministic"`` takes a fixed prefix of the GO-ordered pool, which
            is what the trials implementation does and what this domain's first
            run used. It changes TWO things relative to the baseline at once:

              1. negatives now come from the GO-closest region (the intended
                 ontology effect), and
              2. the anchor receives THE SAME negatives at every epoch, whereas
                 the baseline reseeds on ``(seed, epoch, anchor)`` and therefore
                 draws a fresh sample each epoch.

            Over 15 epochs the baseline sees up to 15 different negative sets per
            anchor and the deterministic ontology arm sees exactly one. That is a
            large difference in effective negative diversity, entirely separate
            from anything the ontology knows, and it biases the comparison against
            the ontology arm. Any deficit measured this way cannot be attributed
            to ontology guidance.

            ``"stochastic"`` samples ``want`` negatives from the closest
            ``tier_window_frac`` of the ordered pool, so the ontology still decides
            WHICH REGION negatives are drawn from but per-epoch variety is
            preserved. This is the arm that isolates the ontology effect.
        tier_window_frac: Width of that window as a fraction of the tier's pool.
            0.34 makes it the closest tercile, matching the career path's
            ``ontology_negative_rank_tiers`` terciles.
        go_score_cap: Candidates scored per anchor per epoch. simGIC is O(|A|+|B|)
            rather than the MeSH matcher's O(|A|x|B|), so this can be far higher
            than trials' 100 without the 37s/batch problem that cap was fixing.
    """

    def __init__(
        self,
        pools: Dict[str, Dict[str, List[str]]],
        views: Dict[str, Dict[str, Any]],
        matcher=None,
        total_epochs: int = 10,
        training_seed: int = 42,
        start_hard_ratio: float = DEFAULT_START_HARD_RATIO,
        end_hard_ratio: float = DEFAULT_END_HARD_RATIO,
        go_tiered: bool = False,
        go_score_cap: int = 400,
        tier_sampling: str = "deterministic",
        tier_window_frac: float = 0.34,
    ) -> None:
        self.pools = pools
        self.views = views
        self.matcher = matcher
        self.go_tiered = bool(go_tiered)
        self.go_score_cap = int(go_score_cap)
        self.tier_sampling = str(tier_sampling)
        self.tier_window_frac = float(tier_window_frac)
        self.total_epochs = max(1, int(total_epochs))
        self.training_seed = int(training_seed)
        self.start_hard_ratio = float(start_hard_ratio)
        self.end_hard_ratio = float(end_hard_ratio)
        self.current_epoch = 0
        self._warned_missing: set = set()

        logger.info(
            "GoPpiNegativeSelector: %d anchor pools, %d indexed protein views, "
            "hard ratio %.2f -> %.2f over %d epochs, go_tiered=%s",
            len(pools), len(views), start_hard_ratio, end_hard_ratio,
            self.total_epochs, self.go_tiered,
        )

    # ------------------------------------------------------------------ build
    @classmethod
    def from_split_dir(
        cls,
        split_dir: Path,
        converted_dir: Path,
        split: str,
        matcher=None,
        **kwargs,
    ) -> "GoPpiNegativeSelector":
        """Load this split's pools plus only the protein views they reference."""
        pools: Dict[str, Dict[str, List[str]]] = {}
        needed: set = set()
        for row in _read_jsonl(Path(split_dir) / "negative_pools.jsonl"):
            if row.get("split") != split:
                continue
            pools[row["anchor_id"]] = {
                "weak_evidence": row.get("weak_evidence", []),
                "no_interaction": row.get("no_interaction", []),
            }
            needed.update(row.get("weak_evidence", []))
            needed.update(row.get("no_interaction", []))

        views: Dict[str, Dict[str, Any]] = {}
        for protein in _read_jsonl(Path(converted_dir) / "proteins.jsonl"):
            pid = protein["protein_id"]
            if pid not in needed:
                continue
            views[pid] = {
                "partner_id": pid,
                "encoder_view": protein["encoder_view"],
                "title": protein.get("gene", pid),
                "skill_uris": (sorted(matcher.terms_for_gene(pid)) if matcher
                               else protein.get("go_uris", [])),
                "coarse_uris": (sorted(matcher.coarse_for_gene(pid)) if matcher
                                else protein.get("coarse_uris", [])),
            }
        return cls(pools, views, matcher=matcher, **kwargs)

    @classmethod
    def from_config(cls, config, split: str = "train", matcher=None):
        """Build from a ``TrainingConfig``, honouring the tiering flag."""
        return cls.from_split_dir(
            Path(getattr(config, "go_ppi_split_dir", "preprocess/go_ppi_splits")),
            Path(getattr(config, "go_ppi_converted_dir", "preprocess/go_ppi")),
            split,
            matcher=matcher,
            total_epochs=int(getattr(config, "num_epochs", 10)),
            training_seed=int(getattr(config, "training_seed", 42)),
            start_hard_ratio=float(getattr(config, "go_ppi_start_hard_ratio", 0.2)),
            end_hard_ratio=float(getattr(config, "go_ppi_end_hard_ratio", 0.6)),
            go_tiered=bool(getattr(config, "go_ppi_go_tiered_negatives", False)),
            go_score_cap=int(getattr(config, "go_ppi_go_score_cap", 400)),
            tier_sampling=str(getattr(config, "go_ppi_tier_sampling", "deterministic")),
            tier_window_frac=float(getattr(config, "go_ppi_tier_window_frac", 0.34)),
        )

    # -------------------------------------------------------------- curriculum
    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch or 0)

    def hard_ratio(self, epoch: Optional[int] = None) -> float:
        """Grade-1 share for ``epoch``, ramped linearly and clamped to ``[0, 1]``."""
        e = self.current_epoch if epoch is None else int(epoch)
        if self.total_epochs <= 1:
            return self.end_hard_ratio
        frac = min(1.0, max(0.0, e / (self.total_epochs - 1)))
        ratio = self.start_hard_ratio + frac * (
            self.end_hard_ratio - self.start_hard_ratio)
        return min(1.0, max(0.0, ratio))

    # ---------------------------------------------------------------- selection
    def select_batch_negatives(
        self,
        anchor_sample,
        candidate_negatives: Sequence[Dict[str, Any]],
        max_negatives: int,
        epoch: Optional[int] = None,
    ) -> Optional[Tuple[List[Dict[str, Any]], List[float]]]:
        """Select this anchor's graded negatives.

        ``candidate_negatives`` is ignored -- it arrives empty from the domain-slot
        call site, and the graded pool is the point of this selector.
        """
        anchor_id = str(
            anchor_sample.metadata.get("resume_id")
            or anchor_sample.resume.get("protein_id")
            or ""
        ).strip()
        pool = self.pools.get(anchor_id)
        if not pool:
            if anchor_id not in self._warned_missing:
                self._warned_missing.add(anchor_id)
                logger.warning(
                    "GoPpiNegativeSelector: no graded pool for anchor %r; falling "
                    "back to the default negative-selection path", anchor_id)
            return None

        e = self.current_epoch if epoch is None else int(epoch)
        rng = _seeded_rng(self.training_seed, e, anchor_id)

        hard_pool = [p for p in pool["weak_evidence"] if p in self.views]
        easy_pool = [p for p in pool["no_interaction"] if p in self.views]
        if not hard_pool and not easy_pool:
            return None

        # The persisted split was made with BP annotations. Resolve each gene
        # against the selected aspect index so MF/CC/hybrid runs score the same
        # fixed pairs using the correct branch, without rebuilding the split.
        anchor_uris = (sorted(self.matcher.terms_for_gene(anchor_id)) if self.matcher
                       else anchor_sample.resume.get("skill_uris", []) or [])
        if self.matcher:
            anchor_sample.resume["skill_uris"] = anchor_uris
            anchor_sample.resume["coarse_uris"] = sorted(
                self.matcher.coarse_for_gene(anchor_id))
            positive_id = anchor_sample.job.get("partner_id")
            if positive_id:
                anchor_sample.job["skill_uris"] = sorted(
                    self.matcher.terms_for_gene(positive_id))
                anchor_sample.job["coarse_uris"] = sorted(
                    self.matcher.coarse_for_gene(positive_id))

        if self.tier_sampling == "random_window":
            # DIVERSITY CONTROL. Restrict each tier to a random window of the same
            # size the ontology arms use, chosen once per anchor (seeded WITHOUT
            # the epoch so it is fixed across training, exactly as the GO-closest
            # window is), then sample within it per epoch. This reproduces the
            # ontology arms' loss of negative diversity while using none of the
            # ontology's information, which is what makes the GO effect separable
            # from the narrowing effect. The GO ordering is deliberately skipped.
            wrng = _seeded_rng(self.training_seed, -1, anchor_id)
            hard_pool = self._fixed_random_window(hard_pool, wrng)
            easy_pool = self._fixed_random_window(easy_pool, wrng)
        elif self.go_tiered and self.matcher is not None and anchor_uris:
            hard_pool = self._go_ordered(anchor_uris, hard_pool)
            easy_pool = self._go_ordered(anchor_uris, easy_pool)

        ratio = self.hard_ratio(e)
        want_hard = min(len(hard_pool), int(round(max_negatives * ratio)))
        want_easy = min(len(easy_pool), max_negatives - want_hard)
        # Backfill from the other tier so the anchor gets its full budget wherever
        # the pool allows.
        if want_hard + want_easy < max_negatives:
            deficit = max_negatives - want_hard - want_easy
            if len(hard_pool) > want_hard:
                want_hard += min(deficit, len(hard_pool) - want_hard)
            elif len(easy_pool) > want_easy:
                want_easy += min(deficit, len(easy_pool) - want_easy)

        if self.tier_sampling == "random_window":
            # Pools are already restricted to their fixed random window.
            chosen = [(p, 1) for p in rng.sample(hard_pool, min(want_hard, len(hard_pool)))]
            chosen += [(p, 0) for p in rng.sample(easy_pool, min(want_easy, len(easy_pool)))]
        elif self.go_tiered and self.matcher is not None and anchor_uris:
            if self.tier_sampling == "stochastic":
                # Sample from the GO-closest WINDOW rather than taking a fixed
                # prefix. See the class docstring: the deterministic prefix changes
                # two things at once (ontology ordering AND per-epoch variety), so
                # it cannot be read as an ontology effect on its own.
                chosen = [(p, 1) for p in self._window_sample(hard_pool, want_hard, rng)]
                chosen += [(p, 0) for p in self._window_sample(easy_pool, want_easy, rng)]
            else:
                chosen = [(p, 1) for p in hard_pool[:want_hard]]
                chosen += [(p, 0) for p in easy_pool[:want_easy]]
        else:
            chosen = [(p, 1) for p in rng.sample(hard_pool, want_hard)]
            chosen += [(p, 0) for p in rng.sample(easy_pool, want_easy)]
        rng.shuffle(chosen)

        negatives: List[Dict[str, Any]] = []
        distances: List[float] = []
        for pid, grade in chosen:
            view = dict(self.views[pid])
            view["grade"] = grade
            view["original_label"] = "potential_fit" if grade == 1 else "no_fit"
            negatives.append(view)
            distances.append(
                CAREER_DISTANCE_SCALE * self._ontology_distance(anchor_uris, view))

        return negatives, distances

    def _fixed_random_window(self, pool: List[str], wrng: random.Random) -> List[str]:
        """A random subset of ``pool`` of the same size the GO window would take.

        Fixed per anchor (``wrng`` is seeded without the epoch), matching how the
        GO-closest window is also fixed across epochs. This is the control that
        isolates "the pool got narrower" from "the pool got GO-selected".
        """
        if not pool:
            return pool
        window = max(1, int(round(self.tier_window_frac * len(pool))))
        window = min(window, len(pool))
        return wrng.sample(pool, window)

    def _window_sample(self, ordered_pool: List[str], want: int,
                       rng: random.Random) -> List[str]:
        """Sample ``want`` items from the GO-closest window of an ordered pool.

        The window is ``max(want, tier_window_frac * len(pool))`` items from the
        closest end, sampled without replacement. This keeps the ontology in
        control of WHICH REGION negatives come from while restoring the
        per-epoch variety that a fixed prefix destroys.
        """
        if want <= 0 or not ordered_pool:
            return []
        window = max(want, int(round(self.tier_window_frac * len(ordered_pool))))
        window = min(window, len(ordered_pool))
        return rng.sample(ordered_pool[:window], min(want, window))

    def _go_ordered(self, anchor_uris: Sequence[str], pool: List[str]) -> List[str]:
        """Sort by GO distance, leaving unannotated candidates at the end."""
        if not pool:
            return pool
        candidates = pool[: self.go_score_cap] if self.go_score_cap else pool
        scored = [
            (not bool(self.views[p].get("skill_uris")),
             self._ontology_distance(anchor_uris, self.views[p]), p)
            for p in candidates
        ]
        scored.sort(key=lambda row: (row[0], row[1]))
        ordered = [p for _missing, _d, p in scored]
        seen = set(candidates)
        return ordered + [p for p in pool if p not in seen]

    def _ontology_distance(self, anchor_uris: Sequence[str], view: Dict[str, Any]) -> float:
        """``1 - simGIC`` for the ``career_distances`` field."""
        if self.matcher is None:
            return NEUTRAL_DISTANCE
        candidate_uris = view.get("skill_uris") or []
        if not anchor_uris or not candidate_uris:
            return NEUTRAL_DISTANCE
        try:
            return 1.0 - float(
                self.matcher.ontology_set_similarity(anchor_uris, candidate_uris))
        except Exception:
            return NEUTRAL_DISTANCE

    # ------------------------------------------------------------ diagnostics
    def pool_summary(self) -> Dict[str, Any]:
        """Per-tier pool sizes, for asserting the selector was wired correctly."""
        hard = [len([p for p in v["weak_evidence"] if p in self.views])
                for v in self.pools.values()]
        easy = [len([p for p in v["no_interaction"] if p in self.views])
                for v in self.pools.values()]
        return {
            # Key name kept as "topics" so run_orca_training._attach_domain's
            # zero-pool guard and the shared log lines work unchanged.
            "topics": len(self.pools),
            "anchors": len(self.pools),
            "indexed_views": len(self.views),
            "hard_total": sum(hard),
            "easy_total": sum(easy),
            "hard_min": min(hard) if hard else 0,
            "hard_median": sorted(hard)[len(hard) // 2] if hard else 0,
            "easy_min": min(easy) if easy else 0,
            "anchors_without_hard": sum(1 for h in hard if h == 0),
        }
