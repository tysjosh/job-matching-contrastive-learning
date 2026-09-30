"""Corpus negative selectors: the ontology-substitutes-for-judgments contrast.

Both selectors draw negatives from **unjudged** trials, so they cost no
annotation. They differ in exactly one factor:

  * :class:`CorpusRandomNegativeSelector` — uniform sampling. The no-knowledge
    control.
  * :class:`CorpusMeshNegativeSelector` — tiered by MeSH distance to the topic,
    so "hard" means ontologically close. The ontology arm.

Both satisfy the duck-typed contract ``BatchProcessor`` expects from an injected
selector, and both attach through the domain-selector slot so the ORCA
orchestrator cannot clear them mid-run.

Why unjudged trials are legitimate negatives here
-------------------------------------------------
They are *not* known negatives — the track is explicit that unjudged pairs are not
negative labels. But that is precisely the low-budget setting: with no judgments
to consult you must guess, and the interesting question is whether the ontology
guesses better than chance. Some unjudged trials will be false negatives, and that
is a property of the regime being studied rather than a flaw in the design. It is
also exactly the false-negative problem ORCA was built for, which makes this the
natural place to reintroduce reliability estimation later.

The MeSH tier is a *proxy* for difficulty, unlike the graded arm where the tier is
a human judgment. That asymmetry is the point of the comparison: it measures how
much of the value of expert grading a cheap ontology proxy recovers.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Scale converting ontology distance in [0,1] to the career_distances field,
#: matching the convention used elsewhere in the pipeline.
CAREER_DISTANCE_SCALE = 10.0

#: Neutral distance when no ontology signal is available for a pair.
NEUTRAL_DISTANCE = 0.5

#: Curriculum endpoints for the share of negatives drawn from the ontologically
#: closest tier. Mirrors the graded selector's schedule so the arms differ only in
#: how the tier is defined, not in how it is scheduled.
DEFAULT_START_HARD_RATIO = 0.2
DEFAULT_END_HARD_RATIO = 0.6


def _seeded_rng(seed: int, epoch: int, topic_id: str) -> random.Random:
    """Per-(seed, epoch, topic) RNG, independent of batch ordering."""
    digest = hashlib.sha256(
        f"{seed}|{epoch}|{topic_id}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


class _CorpusSelectorBase:
    """Shared plumbing: view lookup, per-topic exclusions, epoch curriculum.

    Args:
        views: ``{nct_id: view_slot}`` for the unjudged candidate pool.
        judged: ``{topic_id: set(nct_id)}`` to exclude, so a topic never receives
            one of its own judged trials as a corpus negative. Without this the
            arms would leak label information and the comparison would be void.
        total_epochs / training_seed: curriculum and determinism controls.
    """

    def __init__(
        self,
        views: Dict[str, Dict[str, Any]],
        judged: Optional[Dict[str, set]] = None,
        total_epochs: int = 10,
        training_seed: int = 42,
    ) -> None:
        self.views = views
        self.judged = judged or {}
        self.total_epochs = max(1, int(total_epochs))
        self.training_seed = int(training_seed)
        self.current_epoch = 0
        self._pool_ids: List[str] = sorted(views)
        self._warned: set = set()

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch or 0)

    def _topic_id(self, anchor_sample) -> str:
        return str(
            anchor_sample.metadata.get("resume_id")
            or anchor_sample.resume.get("topic_id")
            or ""
        ).strip()

    def _candidates(self, topic_id: str) -> List[str]:
        """Pool ids minus this topic's judged trials."""
        exclude = self.judged.get(topic_id, set())
        if not exclude:
            return self._pool_ids
        return [n for n in self._pool_ids if n not in exclude]

    def _materialize(
        self, picked: Sequence[Tuple[str, float]]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        negatives, distances = [], []
        for nct_id, distance in picked:
            view = dict(self.views[nct_id])
            # Unjudged: no gold grade exists. Recorded explicitly rather than
            # defaulted to 0, so downstream analysis cannot mistake an unjudged
            # trial for an adjudicated not-relevant one.
            view["grade"] = None
            view["original_label"] = "unjudged"
            negatives.append(view)
            distances.append(CAREER_DISTANCE_SCALE * distance)
        return negatives, distances

    def pool_summary(self) -> Dict[str, Any]:
        return {
            "selector": type(self).__name__,
            "pool_size": len(self._pool_ids),
            "topics_with_exclusions": len(self.judged),
        }


class CorpusRandomNegativeSelector(_CorpusSelectorBase):
    """Uniform negatives from the unjudged corpus. The no-knowledge control."""

    def select_batch_negatives(
        self,
        anchor_sample,
        candidate_negatives: Sequence[Dict[str, Any]],
        max_negatives: int,
        epoch: Optional[int] = None,
    ) -> Optional[Tuple[List[Dict[str, Any]], List[float]]]:
        topic_id = self._topic_id(anchor_sample)
        pool = self._candidates(topic_id)
        if not pool:
            return None
        e = self.current_epoch if epoch is None else int(epoch)
        rng = _seeded_rng(self.training_seed, e, topic_id)
        picked = rng.sample(pool, min(max_negatives, len(pool)))
        # Neutral distance throughout: this arm has no ontology signal by
        # construction, and emitting a real distance would leak it back in.
        return self._materialize([(n, NEUTRAL_DISTANCE) for n in picked])


class CorpusMeshNegativeSelector(_CorpusSelectorBase):
    """MeSH-distance-tiered negatives from the unjudged corpus. The ontology arm.

    Scores a bounded random subsample of the pool by MeSH set distance to the
    topic, then draws by tercile with the same easy-to-hard curriculum the graded
    selector uses. Scoring is capped because set similarity is O(|A|x|B|) per pair
    and the pool can be large; the cap is applied by seeded subsample so it stays
    reproducible.
    """

    def __init__(
        self,
        views: Dict[str, Dict[str, Any]],
        matcher,
        judged: Optional[Dict[str, set]] = None,
        total_epochs: int = 10,
        training_seed: int = 42,
        start_hard_ratio: float = DEFAULT_START_HARD_RATIO,
        end_hard_ratio: float = DEFAULT_END_HARD_RATIO,
        score_cap: int = 400,
    ) -> None:
        super().__init__(views, judged, total_epochs, training_seed)
        self.matcher = matcher
        self.start_hard_ratio = float(start_hard_ratio)
        self.end_hard_ratio = float(end_hard_ratio)
        self.score_cap = int(score_cap)

    def hard_ratio(self, epoch: Optional[int] = None) -> float:
        e = self.current_epoch if epoch is None else int(epoch)
        if self.total_epochs <= 1:
            return self.end_hard_ratio
        frac = min(1.0, max(0.0, e / (self.total_epochs - 1)))
        r = self.start_hard_ratio + frac * (self.end_hard_ratio - self.start_hard_ratio)
        return min(1.0, max(0.0, r))

    def select_batch_negatives(
        self,
        anchor_sample,
        candidate_negatives: Sequence[Dict[str, Any]],
        max_negatives: int,
        epoch: Optional[int] = None,
    ) -> Optional[Tuple[List[Dict[str, Any]], List[float]]]:
        topic_id = self._topic_id(anchor_sample)
        pool = self._candidates(topic_id)
        if not pool:
            return None

        anchor_uris = anchor_sample.resume.get("skill_uris", []) or []
        e = self.current_epoch if epoch is None else int(epoch)
        rng = _seeded_rng(self.training_seed, e, topic_id)

        scoring_pool = (
            pool if len(pool) <= self.score_cap
            else rng.sample(pool, self.score_cap)
        )

        if not anchor_uris or self.matcher is None:
            # No ontology signal for this topic (7 of 125 are diagnostic
            # vignettes). Degrade to uniform rather than fabricating a tier, and
            # warn once so the degradation is visible in the run log.
            if topic_id not in self._warned:
                self._warned.add(topic_id)
                logger.warning(
                    "CorpusMeshNegativeSelector: no anchor MeSH URIs for topic "
                    "%r; falling back to uniform sampling for it", topic_id)
            picked = rng.sample(scoring_pool, min(max_negatives, len(scoring_pool)))
            return self._materialize([(n, NEUTRAL_DISTANCE) for n in picked])

        scored: List[Tuple[str, float]] = []
        for nct_id in scoring_pool:
            uris = self.views[nct_id].get("skill_uris") or []
            if not uris:
                scored.append((nct_id, NEUTRAL_DISTANCE))
                continue
            try:
                sim = float(self.matcher.ontology_set_similarity(anchor_uris, uris))
            except Exception:
                sim = 0.0
            scored.append((nct_id, 1.0 - sim))

        # Terciles of the realized distribution rather than absolute cut points:
        # the realized MeSH distance range is dataset-dependent, and fixed cuts
        # can leave the hard bucket permanently empty.
        scored.sort(key=lambda pair: pair[1])
        k = len(scored)
        t1, t2 = k // 3, (2 * k) // 3
        hard, medium, easy = scored[:t1] or scored[:1], scored[t1:t2], scored[t2:]

        ratio = self.hard_ratio(e)
        n_hard = min(len(hard), int(round(max_negatives * ratio)))
        n_easy = min(len(easy), max_negatives - n_hard)
        n_medium = max(0, max_negatives - n_hard - n_easy)
        n_medium = min(n_medium, len(medium))

        picked: List[Tuple[str, float]] = []
        picked += rng.sample(hard, n_hard)
        picked += rng.sample(medium, n_medium)
        picked += rng.sample(easy, n_easy)

        if len(picked) < max_negatives:
            chosen = {n for n, _ in picked}
            rest = [p for p in scored if p[0] not in chosen]
            rng.shuffle(rest)
            picked += rest[: max_negatives - len(picked)]

        rng.shuffle(picked)
        return self._materialize(picked[:max_negatives])
