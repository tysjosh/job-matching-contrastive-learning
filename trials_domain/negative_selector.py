"""TrialsNegativeSelector — grade-aware negatives drawn from a topic's own qrels.

Why this exists
---------------
The shared in-batch / global-pool negative mechanism draws candidates from *other
samples' positives*. On TREC-CT that would mean a topic's negatives are other
topics' eligible trials — and the grade-1 judgments, the whole reason for using
this dataset, would never reach the loss. This selector instead draws each
anchor's negatives from **that topic's own graded judgments**.

The two tiers are the qrels grades, not a computed distance:

  * **hard** — grade 1, *ineligible*: the patient has the target condition but
    fails eligibility. Expert-annotated ambiguous negatives.
  * **easy** — grade 0, *not relevant*.

This is a stronger tier signal than the career domain's, where "hard" is a
tercile of a computed ontology distance and only correlates with true difficulty.
Here the tier *is* the human judgment. Measured separation between the tiers in
MeSH space is Cohen's d ≈ 0.69, so they are genuinely distinct populations and
not just relabeled noise.

Curriculum
----------
Mirrors the career path's ``linear_easy_to_hard`` schedule so results across the
two datasets are read the same way: the grade-1 share rises linearly from
:data:`DEFAULT_START_HARD_RATIO` to :data:`DEFAULT_END_HARD_RATIO` across
training. Early epochs learn the coarse relevant/irrelevant boundary; later
epochs concentrate on the eligibility boundary that the ontology cannot resolve.

Determinism
-----------
ORCA requires bit-for-bit reproducibility from ``training_seed`` alone. Selection
therefore uses a fresh RNG seeded from ``(training_seed, epoch, topic_id)`` on
every call rather than a shared mutable stream, so the negatives for a given
topic at a given epoch do not depend on batch ordering or on how many other
anchors were processed first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Grade-1 (ineligible) share of selected negatives at the first epoch.
DEFAULT_START_HARD_RATIO = 0.2

#: Grade-1 share at the final epoch.
DEFAULT_END_HARD_RATIO = 0.6

#: Scale converting an ontology distance in ``[0, 1]`` to the ``career_distances``
#: field, matching ``_select_ontology_negatives``' ``d * 10.0`` convention so both
#: domains hand the loss engine the same units.
CAREER_DISTANCE_SCALE = 10.0

#: Fallback ontology distance when the matcher cannot score a pair.
NEUTRAL_DISTANCE = 0.5


def _seeded_rng(training_seed: int, epoch: int, topic_id: str) -> random.Random:
    """A per-(seed, epoch, topic) RNG, independent of processing order."""
    digest = hashlib.sha256(
        f"{training_seed}|{epoch}|{topic_id}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class TrialsNegativeSelector:
    """Selects grade-tiered negatives for a patient topic.

    Satisfies the duck-typed contract ``BatchProcessor._select_with_injected_selector``
    expects: ``select_batch_negatives(anchor_sample, candidate_negatives,
    max_negatives, epoch)`` returning ``(negatives, career_distances)``.

    Args:
        pools: ``{topic_id: {"ineligible": [nct...], "not_relevant": [nct...]}}``.
        views: ``{nct_id: view_slot_dict}`` supplying ``encoder_view`` and the two
            ontology facets for each candidate.
        matcher: Optional ontology matcher used to score ``career_distances``. When
            ``None`` a neutral distance is emitted; the five-scalar ORCA features
            are captured separately by the batch processor either way.
        total_epochs: Denominator for the curriculum ramp.
        training_seed: Seed for deterministic selection.
        start_hard_ratio / end_hard_ratio: Curriculum endpoints for the grade-1 share.
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
        mesh_tiered: bool = False,
        mesh_score_cap: int = 100,
        tier_sampling: str = "uniform",
        tier_window_frac: float = 0.34,
        mesh_facet: str = "all",
        mesh_tier_scope: str = "both",
    ) -> None:
        self.pools = pools
        self.views = views
        self.matcher = matcher
        # Single factor for the learning-curve study: order the graded pool by
        # MeSH distance rather than sampling it uniformly. Off by default so the
        # judgment-substitution study's behaviour is unchanged.
        self.mesh_tiered = bool(mesh_tiered)
        self.mesh_score_cap = int(mesh_score_cap)
        self.tier_sampling = str(tier_sampling)
        self.tier_window_frac = float(tier_window_frac)
        self.mesh_facet = str(mesh_facet)
        self.mesh_tier_scope = str(mesh_tier_scope)
        if self.mesh_facet not in {"all", "disease"}:
            raise ValueError(f"unknown Trials mesh_facet={self.mesh_facet!r}")
        if self.mesh_tier_scope not in {"both", "ineligible", "not_relevant"}:
            raise ValueError(f"unknown Trials mesh_tier_scope={self.mesh_tier_scope!r}")
        if self.tier_sampling not in {"uniform", "deterministic", "stochastic",
                                    "random_window"}:
            raise ValueError(f"unknown Trials tier_sampling={self.tier_sampling!r}")
        if not 0 < self.tier_window_frac <= 1:
            raise ValueError("tier_window_frac must be in (0, 1]")
        self.total_epochs = max(1, int(total_epochs))
        self.training_seed = int(training_seed)
        self.start_hard_ratio = float(start_hard_ratio)
        self.end_hard_ratio = float(end_hard_ratio)
        self.current_epoch = 0
        self._warned_missing: set = set()
        # MeSH ranks do not change across epochs. Caching them makes an uncapped,
        # full-pool ranking cheaper than repeatedly scoring a capped prefix.
        self._mesh_order_cache: Dict[Tuple[str, str], List[str]] = {}

        logger.info(
            "TrialsNegativeSelector: %d topic pools, %d indexed trial views, "
            "hard ratio %.2f -> %.2f over %d epochs, mesh_tiered=%s, "
            "tier_sampling=%s, window=%.2f",
            len(pools), len(views), start_hard_ratio, end_hard_ratio,
            self.total_epochs, self.mesh_tiered, self.tier_sampling,
            self.tier_window_frac,
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
    ) -> "TrialsNegativeSelector":
        """Load pools for ``split`` plus the trial views they reference.

        Only trials actually referenced by this split's pools are indexed, so a
        train-split selector does not hold the test split's candidates in memory.
        """
        pools: Dict[str, Dict[str, List[str]]] = {}
        needed: set = set()
        for row in _read_jsonl(Path(split_dir) / "negative_pools.jsonl"):
            if row.get("split") != split:
                continue
            pools[row["topic_id"]] = {
                "ineligible": row.get("ineligible", []),
                "not_relevant": row.get("not_relevant", []),
            }
            needed.update(row.get("ineligible", []))
            needed.update(row.get("not_relevant", []))

        views: Dict[str, Dict[str, Any]] = {}
        for trial in _read_jsonl(Path(converted_dir) / "trials.jsonl"):
            nct_id = trial["nct_id"]
            if nct_id not in needed:
                continue
            views[nct_id] = {
                "nct_id": nct_id,
                "encoder_view": trial["encoder_view"],
                "title": trial.get("title", ""),
                "skill_uris": trial.get("mesh_uris", []),
                "coarse_uris": trial.get("condition_uris", []),
            }

        return cls(pools, views, matcher=matcher, **kwargs)

    @classmethod
    def from_config(cls, config, split: str = "train", matcher=None):
        """Build from a ``TrainingConfig``, honouring the tiering flag.

        Used by the learning-curve study so the single factor is driven from the
        config rather than from a call-site argument.
        """
        from pathlib import Path as _Path

        return cls.from_split_dir(
            _Path(getattr(config, "trials_split_dir", "preprocess/trec_ct_splits")),
            _Path(getattr(config, "trials_converted_dir", "preprocess/trec_ct")),
            split,
            matcher=matcher,
            total_epochs=int(getattr(config, "num_epochs", 10)),
            training_seed=int(getattr(config, "training_seed", 42)),
            start_hard_ratio=float(getattr(config, "trials_start_hard_ratio", 0.2)),
            end_hard_ratio=float(getattr(config, "trials_end_hard_ratio", 0.6)),
            mesh_tiered=bool(getattr(config, "trials_mesh_tiered_negatives", False)),
            mesh_score_cap=int(getattr(config, "trials_mesh_score_cap", 100)),
            tier_sampling=str(getattr(config, "trials_tier_sampling", "uniform")),
            tier_window_frac=float(
                getattr(config, "trials_tier_window_frac", 0.34)),
            mesh_facet=str(getattr(config, "trials_mesh_facet", "all")),
            mesh_tier_scope=str(getattr(config, "trials_mesh_tier_scope", "both")),
        )

    # -------------------------------------------------------------- curriculum
    def set_epoch(self, epoch: int) -> None:
        """Advance the curriculum. Called by the batch processor each epoch."""
        self.current_epoch = int(epoch or 0)

    def hard_ratio(self, epoch: Optional[int] = None) -> float:
        """Grade-1 share for ``epoch``, ramped linearly and clamped to ``[0, 1]``."""
        e = self.current_epoch if epoch is None else int(epoch)
        if self.total_epochs <= 1:
            return self.end_hard_ratio
        frac = min(1.0, max(0.0, e / (self.total_epochs - 1)))
        ratio = self.start_hard_ratio + frac * (
            self.end_hard_ratio - self.start_hard_ratio
        )
        return min(1.0, max(0.0, ratio))

    # ---------------------------------------------------------------- selection
    def select_batch_negatives(
        self,
        anchor_sample,
        candidate_negatives: Sequence[Dict[str, Any]],
        max_negatives: int,
        epoch: Optional[int] = None,
    ) -> Optional[Tuple[List[Dict[str, Any]], List[float]]]:
        """Select this topic's graded negatives.

        ``candidate_negatives`` (the in-batch jobs) is deliberately ignored: the
        graded pool is the point of this selector. Returns ``None`` when the topic
        has no pool, so the batch processor falls back to its own logic rather
        than training on nothing.
        """
        topic_id = str(
            anchor_sample.metadata.get("resume_id")
            or anchor_sample.resume.get("topic_id")
            or ""
        ).strip()
        pool = self.pools.get(topic_id)
        if not pool:
            if topic_id not in self._warned_missing:
                self._warned_missing.add(topic_id)
                logger.warning(
                    "TrialsNegativeSelector: no graded pool for topic %r; "
                    "falling back to the default negative-selection path",
                    topic_id,
                )
            return None

        e = self.current_epoch if epoch is None else int(epoch)
        rng = _seeded_rng(self.training_seed, e, topic_id)

        hard_pool = [n for n in pool["ineligible"] if n in self.views]
        easy_pool = [n for n in pool["not_relevant"] if n in self.views]
        if not hard_pool and not easy_pool:
            return None

        facet_key = "coarse_uris" if self.mesh_facet == "disease" else "skill_uris"
        anchor_uris = anchor_sample.resume.get(facet_key, []) or []
        mesh_active = bool(self.mesh_tiered and self.matcher is not None
                           and anchor_uris)
        ratio = self.hard_ratio(e)
        want_hard = min(len(hard_pool), int(round(max_negatives * ratio)))
        want_easy = min(len(easy_pool), max_negatives - want_hard)
        # Backfill from the other tier when one is short, so the anchor always
        # receives the requested number of negatives where the pool allows.
        if want_hard + want_easy < max_negatives:
            deficit = max_negatives - want_hard - want_easy
            if len(hard_pool) > want_hard:
                want_hard += min(deficit, len(hard_pool) - want_hard)
            elif len(easy_pool) > want_easy:
                want_easy += min(deficit, len(easy_pool) - want_easy)

        if (self.tier_sampling == "random_window" or
                (self.mesh_facet != "all" and self.mesh_tiered and
                 self.tier_sampling == "stochastic" and not mesh_active)):
            # Compute the grade allocation from the full pools above, then apply
            # equal-width windows. The minimum keeps small tiers from changing
            # the allocation before selection, which would break the match.
            wrng = _seeded_rng(self.training_seed, -1, topic_id)
            hard_pool = self._fixed_random_window(
                hard_pool, wrng, minimum=max_negatives)
            easy_pool = self._fixed_random_window(
                easy_pool, wrng, minimum=max_negatives)
            chosen = [(n, 1) for n in rng.sample(
                hard_pool, min(want_hard, len(hard_pool)))]
            chosen += [(n, 0) for n in rng.sample(
                easy_pool, min(want_easy, len(easy_pool)))]
        elif (mesh_active and self.mesh_tier_scope != "both" and
              self.tier_sampling == "stochastic"):
            # Restrict both tiers to equal-width windows. Only the selected
            # tier is MeSH-ranked; the other gets a fixed random window,
            # preserving candidate diversity and grade mix for the comparison.
            wrng = _seeded_rng(self.training_seed, -1, topic_id)
            if self.mesh_tier_scope == "ineligible":
                hard_pool = self._mesh_ordered(topic_id, "hard", anchor_uris, hard_pool)
                easy_pool = self._fixed_random_window(easy_pool, wrng,
                                                       minimum=max_negatives)
                chosen = [(n, 1) for n in self._window_sample(
                    hard_pool, want_hard, rng, minimum=max_negatives)]
                chosen += [(n, 0) for n in rng.sample(
                    easy_pool, min(want_easy, len(easy_pool)))]
            else:
                hard_pool = self._fixed_random_window(hard_pool, wrng,
                                                       minimum=max_negatives)
                easy_pool = self._mesh_ordered(topic_id, "easy", anchor_uris, easy_pool)
                chosen = [(n, 1) for n in rng.sample(
                    hard_pool, min(want_hard, len(hard_pool)))]
                chosen += [(n, 0) for n in self._window_sample(
                    easy_pool, want_easy, rng, minimum=max_negatives)]
        elif mesh_active:
            hard_pool = self._mesh_ordered(
                topic_id, "hard", anchor_uris, hard_pool)
            easy_pool = self._mesh_ordered(
                topic_id, "easy", anchor_uris, easy_pool)
            if self.tier_sampling == "stochastic":
                chosen = [(n, 1) for n in self._window_sample(
                    hard_pool, want_hard, rng, minimum=max_negatives)]
                chosen += [(n, 0) for n in self._window_sample(
                    easy_pool, want_easy, rng, minimum=max_negatives)]
            else:
                # Backward-compatible deterministic MeSH arm.
                chosen = [(n, 1) for n in hard_pool[:want_hard]]
                chosen += [(n, 0) for n in easy_pool[:want_easy]]
        else:
            chosen = [(n, 1) for n in rng.sample(hard_pool, want_hard)]
            chosen += [(n, 0) for n in rng.sample(easy_pool, want_easy)]
        rng.shuffle(chosen)

        negatives: List[Dict[str, Any]] = []
        distances: List[float] = []
        for nct_id, grade in chosen:
            view = dict(self.views[nct_id])
            view["grade"] = grade
            # Carried so downstream analysis can correlate the learned reliability
            # r_hat against the gold grade — the validation this dataset uniquely
            # supports. Inert for the InfoNCE denominator itself.
            view["original_label"] = "ineligible" if grade == 1 else "not_relevant"
            negatives.append(view)
            distances.append(
                CAREER_DISTANCE_SCALE * self._ontology_distance(anchor_uris, view)
            )

        return negatives, distances

    def _fixed_random_window(self, pool: List[str], wrng: random.Random,
                             minimum: int = 1) -> List[str]:
        """Return a fixed random subset with the ontology window's width."""
        if not pool:
            return pool
        window = max(int(minimum),
                     int(round(self.tier_window_frac * len(pool))))
        window = min(window, len(pool))
        return wrng.sample(pool, window)

    def _window_sample(self, ordered_pool: List[str], want: int,
                       rng: random.Random, minimum: int = 1) -> List[str]:
        """Sample within the MeSH-closest window, preserving epoch variety."""
        if want <= 0 or not ordered_pool:
            return []
        window = max(want, int(minimum),
                     int(round(self.tier_window_frac * len(ordered_pool))))
        window = min(window, len(ordered_pool))
        return rng.sample(ordered_pool[:window], min(want, window))

    def _mesh_ordered(self, topic_id: str, tier: str,
                      anchor_uris: Sequence[str], pool: List[str]) -> List[str]:
        """Return ``pool`` sorted by ascending MeSH distance to the anchor.

        Scoring is capped because set similarity is O(|A|x|B|) per pair; the cap is
        applied by taking a deterministic prefix of the already-sorted pool ids, so
        the same anchor always scores the same candidates.
        """
        if not pool:
            return pool
        cache_key = (str(topic_id), str(tier))
        cached = self._mesh_order_cache.get(cache_key)
        if cached is not None:
            return cached
        candidates = pool[: self.mesh_score_cap] if self.mesh_score_cap else pool
        scored = [
            (not bool(self.views[n].get("coarse_uris" if self.mesh_facet == "disease"
                                         else "skill_uris")),
             self._ontology_distance(anchor_uris, self.views[n]), n)
            for n in candidates
        ]
        if self.mesh_facet == "disease":
            scored.sort(key=lambda row: (row[0], row[1]))
        else:
            scored.sort(key=lambda row: row[1])
        ordered = [n for _missing, _d, n in scored]
        # Anything beyond the cap keeps its original order at the far end.
        result = ordered + [n for n in pool if n not in set(candidates)]
        self._mesh_order_cache[cache_key] = result
        return result

    def _ontology_distance(self, anchor_uris: Sequence[str], view: Dict[str, Any]) -> float:
        """``1 - set_similarity`` for the ``career_distances`` field."""
        if self.matcher is None:
            return NEUTRAL_DISTANCE
        facet_key = "coarse_uris" if self.mesh_facet == "disease" else "skill_uris"
        candidate_uris = view.get(facet_key) or []
        if not anchor_uris or not candidate_uris:
            return NEUTRAL_DISTANCE
        try:
            return 1.0 - float(
                self.matcher.ontology_set_similarity(anchor_uris, candidate_uris)
            )
        except Exception:
            return NEUTRAL_DISTANCE

    # ------------------------------------------------------------ diagnostics
    def pool_summary(self) -> Dict[str, Any]:
        """Per-tier pool sizes, for asserting the selector was wired correctly."""
        hard = [len([n for n in p["ineligible"] if n in self.views])
                for p in self.pools.values()]
        easy = [len([n for n in p["not_relevant"] if n in self.views])
                for p in self.pools.values()]
        return {
            "topics": len(self.pools),
            "indexed_views": len(self.views),
            "hard_total": sum(hard),
            "easy_total": sum(easy),
            "hard_min": min(hard) if hard else 0,
            "hard_median": sorted(hard)[len(hard) // 2] if hard else 0,
            "easy_min": min(easy) if easy else 0,
            "topics_without_hard": sum(1 for h in hard if h == 0),
        }
