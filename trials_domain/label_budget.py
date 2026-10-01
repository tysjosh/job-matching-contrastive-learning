"""Label-budget subsampling for the data-efficiency study.

The question this supports: **can an ontology substitute for relevance
judgments?** At a small annotation budget you cannot afford ~478 expert
adjudications per topic, so negatives must come from somewhere other than qrels.
MeSH distance over the 375,580-trial corpus is one such source, and TREC-CT is a
good testbed because 87.03% of the corpus is unjudged for any given topic.

Three arms, differing in exactly one factor at a time:

  ==================  =====================  ==============================
  arm                 positives              negatives
  ==================  =====================  ==============================
  ``full``            all judged grade-2     grade-tiered, from qrels
  ``low_random``      f x judged grade-2     uniform from the corpus
  ``low_ontology``    f x judged grade-2     MeSH-distance-tiered from corpus
  ==================  =====================  ==============================

``low_random`` vs ``low_ontology`` isolates the ontology. ``full`` vs
``low_ontology`` gives the label-efficiency ratio.

Why the negatives have to change with the budget
------------------------------------------------
On this dataset the graded negatives *are* labels — grade 1 and grade 0 are
expert adjudications, not derived quantities. Holding the graded pool fixed while
subsampling positives would spend the expensive resource freely while claiming to
economise on it, and the resulting "efficiency" number would not be defensible.
So a reduced budget reduces judgments of *every* grade, and the low-budget arms
draw negatives from the unjudged corpus instead.

Budget accounting
-----------------
:func:`budget_report` states the cost in judgments so the efficiency claim can be
quoted in the unit that actually costs money. The ontology arm additionally
requires MeSH plus entity linking, which is not free — it is simply a fixed cost
already paid for a curated ontology, rather than a per-query annotation cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Subsampling units. ``pair`` cuts individual judgments and keeps anchor
#: coverage — the annotation-cost question. ``anchor`` cuts whole topics and tests
#: generalization to unseen queries — the coverage question. They answer different
#: things; ``pair`` is the default because the judgment is the unit that costs
#: money, and because 10% of 60 topics is only 6 anchors, which is too few to
#: estimate anything stably.
UNITS = ("pair", "anchor")

ARMS = ("full", "low_random", "low_ontology")


def _rng(seed: int, *parts: Any) -> random.Random:
    """Deterministic RNG from a seed plus arbitrary key parts.

    Keyed rather than sequential so a subsample does not depend on iteration
    order or on how many other subsamples were drawn first.
    """
    key = "|".join(str(p) for p in (seed, *parts))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def subsample_records(
    records: Sequence[Dict[str, Any]],
    fraction: float,
    unit: str = "pair",
    seed: int = 42,
    group_key: str = "topic_id",
) -> List[Dict[str, Any]]:
    """Return a deterministic ``fraction`` of ``records``.

    ``unit="pair"`` samples within each group so every anchor keeps at least one
    positive where it had any — otherwise a small fraction would silently drop
    whole anchors and confound the pair-level regime with the anchor-level one.
    ``unit="anchor"`` samples whole groups.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return list(records)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["metadata"][group_key])].append(record)

    if unit == "anchor":
        groups = sorted(grouped)
        keep_n = max(1, int(round(len(groups) * fraction)))
        keep = set(_rng(seed, "anchor", fraction).sample(groups, keep_n))
        return [r for g in sorted(keep) for r in grouped[g]]

    out: List[Dict[str, Any]] = []
    for group in sorted(grouped):
        pool = grouped[group]
        keep_n = max(1, int(round(len(pool) * fraction)))
        picked = _rng(seed, "pair", fraction, group).sample(pool, min(keep_n, len(pool)))
        out.extend(picked)
    return out


def budget_report(
    records: Sequence[Dict[str, Any]],
    negative_pools: Optional[Dict[str, Dict[str, List[str]]]],
    arm: str,
) -> Dict[str, Any]:
    """Count the judgments an arm consumes, so cost is quoted in the real unit.

    ``full`` consumes every positive judgment *and* every graded negative
    judgment. The low-budget arms consume only the sampled positives; their
    negatives come from unjudged trials and cost nothing to annotate.
    """
    topics = {str(r["metadata"]["topic_id"]) for r in records}
    positives = len(records)

    graded_negatives = 0
    if arm == "full" and negative_pools:
        for topic in topics:
            pool = negative_pools.get(topic, {})
            graded_negatives += len(pool.get("ineligible", []))
            graded_negatives += len(pool.get("not_relevant", []))

    return {
        "arm": arm,
        "topics": len(topics),
        "positive_judgments": positives,
        "graded_negative_judgments": graded_negatives,
        "total_judgments": positives + graded_negatives,
        "negative_source": "qrels" if arm == "full" else "unjudged corpus",
        "needs_ontology": arm == "low_ontology",
    }


def build_arm(
    split_dir: Path,
    output_dir: Path,
    arm: str,
    fraction: float,
    unit: str = "pair",
    seed: int = 42,
) -> Dict[str, Any]:
    """Write one arm's training file and return its manifest."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    records = list(_read_jsonl(Path(split_dir) / "train.jsonl"))
    pools: Dict[str, Dict[str, List[str]]] = {}
    pools_path = Path(split_dir) / "negative_pools.jsonl"
    if pools_path.exists():
        for row in _read_jsonl(pools_path):
            if row.get("split") == "train":
                pools[str(row["topic_id"])] = {
                    "ineligible": row.get("ineligible", []),
                    "not_relevant": row.get("not_relevant", []),
                }

    effective_fraction = 1.0 if arm == "full" else fraction
    kept = subsample_records(records, effective_fraction, unit=unit, seed=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = budget_report(kept, pools, arm)
    manifest = {
        **report,
        "fraction": effective_fraction,
        "unit": unit,
        "seed": seed,
        "source_records": len(records),
        "kept_records": len(kept),
        "train_file": str(out_path),
        # Which negative selector the runner must wire for this arm.
        "negative_selector": {
            "full": "graded",
            "low_random": "corpus_random",
            "low_ontology": "corpus_mesh",
        }[arm],
    }
    (output_dir / "arm_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "arm=%s fraction=%.3f unit=%s seed=%d -> %d/%d records, "
        "%d judgments (%s negatives)",
        arm, effective_fraction, unit, seed, len(kept), len(records),
        report["total_judgments"], report["negative_source"],
    )
    return manifest
