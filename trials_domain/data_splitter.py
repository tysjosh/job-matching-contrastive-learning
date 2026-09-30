"""Build topic-disjoint train/validation/test splits from the converted TREC-CT data.

Split policy
------------
**Train and validation come from the 2021 topics; test is the full 2022 topic
set.** The two years were annotated in separate campaigns against the same April
2021 corpus snapshot, so holding out a whole year is a genuine distribution shift
rather than a random partition of one pool. Validation is carved out of 2021 by
topic so no topic's judgments appear in two splits — a random split over
*pairs* would leak, because every topic contributes dozens of positives and the
model would see the same query text on both sides.

Anchor scarcity
---------------
This dataset has 125 topics against career v7's ~619 distinct anchors, so anchor
diversity is roughly 8x lower even though the pair count is comparable (9,509
grade-2 pairs vs v7's 8,000 records). Each topic recurs across ~76 positives.
That is a real overfitting risk and the reason the split is topic-disjoint rather
than pair-level; it should also be expected to widen seed-to-seed variance
relative to the career runs.

Negatives: different shape for training vs evaluation
-----------------------------------------------------
**Train** emits grade-2 (eligible) pairs only. Its negatives are drawn at batch
time by :class:`~trials_domain.negative_selector.TrialsNegativeSelector` from
``negative_pools.jsonl``, because a topic's negatives are graded and must come
from that topic's own judgments — left to the default in-batch mechanism they
would be other topics' eligible trials and the grade-1 ambiguous negatives would
never reach the loss.

**Validation and test** additionally emit the graded negatives *as records*. This
is not optional: the embedding evaluation derives its metrics from the positive
and negative records in the file, so a positives-only test set yields
``auc_roc = nan`` with a degenerate confusion matrix (all true positives, nothing
to rank against). Career v7's test split carries all three grades as records for
exactly this reason.

Grade mapping
-------------
Each record carries the TREC grade under both its native name and the career
vocabulary the shared ordinal evaluator already understands:

  ===========  ==================  ==================  ===============
  TREC grade   ``trec_label``      ``original_label``  meaning
  ===========  ==================  ==================  ===============
  2            ``eligible``        ``good_fit``        a genuine match
  1            ``ineligible``      ``potential_fit``   topically right, not a match
  0            ``not_relevant``    ``no_fit``          unrelated
  ===========  ==================  ==================  ===============

The correspondence is exact and order-preserving, and the middle row is the
reason it is tight rather than merely convenient: a trial the patient has the
condition for but fails eligibility on is the same *kind* of label as a
"potential fit" resume-job pair — topically apt, not an actual match. Emitting
``original_label`` in the career vocabulary lets ``run_ordinal_evaluation.py``
consume trials runs unchanged; ``trec_grade`` and ``trec_label`` preserve the
unmapped truth for any analysis that needs it.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Fraction of 2021 topics held out for validation.
DEFAULT_VAL_FRACTION = 0.2

#: Grade -> qrels meaning, mirroring ``data_converter.GRADE_NAMES``.
GRADE_ELIGIBLE = 2
GRADE_INELIGIBLE = 1
GRADE_NOT_RELEVANT = 0

#: TREC grade -> its native label.
TREC_LABEL = {
    GRADE_ELIGIBLE: "eligible",
    GRADE_INELIGIBLE: "ineligible",
    GRADE_NOT_RELEVANT: "not_relevant",
}

#: TREC grade -> the career ordinal vocabulary the shared evaluator reads from
#: ``metadata['original_label']``. Order-preserving; see the module docstring for
#: why the middle mapping is a genuine correspondence and not a convenience.
ORDINAL_LABEL = {
    GRADE_ELIGIBLE: "good_fit",
    GRADE_INELIGIBLE: "potential_fit",
    GRADE_NOT_RELEVANT: "no_fit",
}

#: Binary label: only grade 2 counts as a positive.
BINARY_LABEL = {GRADE_ELIGIBLE: 1, GRADE_INELIGIBLE: 0, GRADE_NOT_RELEVANT: 0}

#: Default cap on grade-0 records per topic in the VALIDATION split. Validation
#: runs every epoch, and a topic's full grade-0 pool (median 308 in 2021) would
#: dominate epoch time for a set used only for model selection. Grade-1 is never
#: capped — it is the scarce, informative class.
DEFAULT_VAL_NOT_RELEVANT_CAP = 60


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _ontology_slot(record: Dict[str, Any], id_key: str, id_value: str) -> Dict[str, Any]:
    """Build a view slot carrying the encoder text and both ontology facets.

    ``skill_uris`` and ``coarse_uris`` are the key names the shared pipeline reads
    — ``skill_uris`` is what ``BatchProcessor`` looks for when scoring ontology
    negatives and capturing ORCA's ``d_esco``/``s_esco``, and ``coarse_uris`` is
    the generalized coarse facet consumed by the injected ``coarse_distance_fn``.
    Reusing the existing names (rather than renaming them to MeSH terminology)
    is deliberate: it keeps the ``orca`` package and the shared batch processor
    completely untouched.
    """
    return {
        id_key: id_value,
        "encoder_view": record["encoder_view"],
        "skill_uris": record["mesh_uris"],
        "coarse_uris": record["condition_uris"],
    }


def _build_record(
    topic_id: str,
    topic: Dict[str, Any],
    anchor: Dict[str, Any],
    nct_id: str,
    trial: Dict[str, Any],
    grade: int,
) -> Dict[str, Any]:
    """One (topic, trial) record at a given TREC grade."""
    candidate = _ontology_slot(trial, "nct_id", nct_id)
    candidate["title"] = trial["title"]
    candidate["grade"] = grade
    candidate["original_label"] = ORDINAL_LABEL[grade]
    return {
        "resume": anchor,
        "job": candidate,
        "label": BINARY_LABEL[grade],
        "metadata": {
            # Query-group identity. The shared pipeline groups by
            # metadata['resume_id']; the topic is the query, so it is the key.
            "resume_id": topic_id,
            "topic_id": topic_id,
            "nct_id": nct_id,
            "year": topic["year"],
            # Career vocabulary, so run_ordinal_evaluation.py works unchanged.
            "original_label": ORDINAL_LABEL[grade],
            # The unmapped truth, preserved for trials-specific analysis — in
            # particular for regressing learned reliability against the gold
            # grade, which is the validation this dataset uniquely supports.
            "trec_grade": grade,
            "trec_label": TREC_LABEL[grade],
            # 7 of 125 topics are diagnostic vignettes that never name the
            # disease, so their coarse condition signal is absent.
            "condition_signal_present": topic["condition_signal_present"],
        },
    }


def build_splits(
    converted_dir: Path,
    output_dir: Path,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 42,
    val_not_relevant_cap: int = DEFAULT_VAL_NOT_RELEVANT_CAP,
    test_not_relevant_cap: int = 0,
) -> Dict[str, Any]:
    """Emit train/validation/test JSONL plus per-topic graded negative pools.

    Args:
        val_not_relevant_cap: Max grade-0 records per topic in validation
            (``0`` = uncapped). Grade-1 is never capped.
        test_not_relevant_cap: Max grade-0 records per topic in test
            (``0`` = uncapped, the faithful TREC pool). Capping inflates ranking
            metrics by thinning the irrelevant tail, so the default keeps the
            full pool and pays the one-time encoding cost.
    """
    topics = {r["topic_id"]: r for r in _read_jsonl(converted_dir / "topics.jsonl")}
    trials = {r["nct_id"]: r for r in _read_jsonl(converted_dir / "trials.jsonl")}

    graded: Dict[str, Dict[int, List[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in _read_jsonl(converted_dir / "qrels.jsonl"):
        graded[row["topic_id"]][row["grade"]].append(row["nct_id"])

    # ---- topic-disjoint split ------------------------------------------
    topics_2021 = sorted(t for t, r in topics.items() if r["year"] == "2021")
    topics_2022 = sorted(t for t, r in topics.items() if r["year"] == "2022")

    rng = random.Random(seed)
    shuffled = list(topics_2021)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_topics = sorted(shuffled[:n_val])
    train_topics = sorted(shuffled[n_val:])

    assignment = {
        "train": train_topics,
        "validation": val_topics,
        "test": topics_2022,
    }
    overlap = set(train_topics) & set(val_topics)
    assert not overlap, f"train/validation topic leak: {overlap}"

    # ---- records --------------------------------------------------------
    # Train: positives only (negatives come from the selector at batch time).
    # Validation / test: all three grades as records, so the embedding evaluation
    # has something to rank against. A positives-only eval split produces
    # auc_roc = nan.
    grade_plan = {
        "train": {GRADE_ELIGIBLE: 0},
        "validation": {
            GRADE_ELIGIBLE: 0,
            GRADE_INELIGIBLE: 0,
            GRADE_NOT_RELEVANT: val_not_relevant_cap,
        },
        "test": {
            GRADE_ELIGIBLE: 0,
            GRADE_INELIGIBLE: 0,
            GRADE_NOT_RELEVANT: test_not_relevant_cap,
        },
    }

    counts: Dict[str, int] = {}
    grade_counts: Dict[str, Dict[str, int]] = {}
    missing_trials = 0
    for split, split_topics in assignment.items():
        records: List[Dict[str, Any]] = []
        per_grade: collections.Counter = collections.Counter()
        for topic_id in split_topics:
            topic = topics[topic_id]
            anchor = _ontology_slot(topic, "topic_id", topic_id)
            for grade, cap in grade_plan[split].items():
                ncts = sorted(graded[topic_id].get(grade, []))
                if cap and len(ncts) > cap:
                    # Deterministic per-(seed, split, topic, grade) subsample, so
                    # the cap does not depend on dict iteration order.
                    sub_rng = random.Random(f"{seed}|{split}|{topic_id}|{grade}")
                    ncts = sorted(sub_rng.sample(ncts, cap))
                for nct_id in ncts:
                    trial = trials.get(nct_id)
                    if trial is None:
                        missing_trials += 1
                        continue
                    records.append(
                        _build_record(topic_id, topic, anchor, nct_id, trial, grade)
                    )
                    per_grade[TREC_LABEL[grade]] += 1
        rng.shuffle(records)
        counts[split] = _write_jsonl(output_dir / f"{split}.jsonl", records)
        grade_counts[split] = dict(per_grade)

    # ---- graded negative pools -----------------------------------------
    pools = [
        {
            "topic_id": topic_id,
            "split": split,
            # Kept apart by grade so a selector can weight the ambiguous
            # (grade-1) negatives distinctly from the true (grade-0) ones.
            "ineligible": sorted(graded[topic_id].get(GRADE_INELIGIBLE, [])),
            "not_relevant": sorted(graded[topic_id].get(GRADE_NOT_RELEVANT, [])),
        }
        for split, split_topics in assignment.items()
        for topic_id in split_topics
    ]
    _write_jsonl(output_dir / "negative_pools.jsonl", pools)

    manifest = {
        "policy": "train+validation from 2021 topics, test = 2022 topics",
        "seed": seed,
        "val_fraction": val_fraction,
        "topics": {k: len(v) for k, v in assignment.items()},
        "records": counts,
        "records_by_grade": grade_counts,
        "grade_mapping": {
            str(g): {"trec_label": TREC_LABEL[g], "original_label": ORDINAL_LABEL[g],
                     "label": BINARY_LABEL[g]}
            for g in (GRADE_ELIGIBLE, GRADE_INELIGIBLE, GRADE_NOT_RELEVANT)
        },
        "val_not_relevant_cap": val_not_relevant_cap,
        "test_not_relevant_cap": test_not_relevant_cap,
        "missing_trials": missing_trials,
        "negative_pool_totals": {
            "ineligible": sum(len(p["ineligible"]) for p in pools),
            "not_relevant": sum(len(p["not_relevant"]) for p in pools),
        },
        "degraded_topics_per_split": {
            split: [t for t in split_topics
                    if not topics[t]["condition_signal_present"]]
            for split, split_topics in assignment.items()
        },
        "topic_assignment": assignment,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--converted-dir", type=Path, default=Path("preprocess/trec_ct"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("preprocess/trec_ct_splits")
    )
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-not-relevant-cap", type=int, default=DEFAULT_VAL_NOT_RELEVANT_CAP,
        help="max grade-0 records per topic in validation (0 = all)")
    parser.add_argument(
        "--test-not-relevant-cap", type=int, default=0,
        help="max grade-0 records per topic in test (0 = all; capping inflates "
             "ranking metrics by thinning the irrelevant tail)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = build_splits(
        args.converted_dir, args.output_dir, args.val_fraction, args.seed,
        val_not_relevant_cap=args.val_not_relevant_cap,
        test_not_relevant_cap=args.test_not_relevant_cap,
    )
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "topic_assignment"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
