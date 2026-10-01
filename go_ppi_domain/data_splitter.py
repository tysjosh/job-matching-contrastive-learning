"""Build protein-disjoint train/validation/test splits plus graded negative pools.

Emits into ``go_ppi_split_dir``:
    train.jsonl / validation.jsonl / test.jsonl
    negative_pools.jsonl   {anchor_id, split, weak_evidence: [...], no_interaction: [...]}
    split_manifest.json

FULL PROTEIN DISJOINTNESS -- the reason this splitter exists
------------------------------------------------------------
The career split in this project is stratified by LABEL over PAIR records, which
made it entity-overlapping: 379 of 381 test resumes also appear in train, because
8,000 pairs are drawn over only 643 resumes and 351 jobs. The per-arm deltas
survive that (both arms see the same leakage) but no absolute number from that
split is publishable.

This splitter partitions the PROTEIN SET into three disjoint groups and then keeps
only pairs whose BOTH endpoints fall in the same group. So a protein that appears
anywhere in train -- as an anchor or as a partner -- cannot appear anywhere in
validation or test, in any role. That is strictly stronger than trials' split
(topic-disjoint, but a trial can recur across splits) and it is the right default
for a network dataset, where the same entity is otherwise almost guaranteed to
appear on both sides of the split.

The cost is real and is reported in the manifest as ``pairs_dropped_cross_split``:
edges crossing a partition boundary are discarded, and in a dense interactome
that is most of them. With 2.38M candidate edges there is ample surplus, so
paying it is the right trade.

``--strategy anchor_disjoint`` keeps the weaker, trials-equivalent policy
(anchors disjoint, partners may recur) for the sake of a like-for-like comparison
against the trials setup. It is not the default and absolute numbers from it
should carry the same caveat career's do.

GRADE PLAN PER SPLIT
--------------------
Train is positives only: grade-1 and grade-0 negatives arrive through the
negative-selection seam at batch time, which is the whole mechanism under test.
Validation and test carry all three grades as records, because a positives-only
evaluation split has nothing to rank against.

Grade 0 is capped in validation (it runs every epoch and its pool is large) but
never in test. Grade 1 is never capped anywhere -- it is the scarce, informative
class and the one the hard contrast is made of.

Usage
    .venv/bin/python3 -m go_ppi_domain.data_splitter \
        --converted-dir preprocess/go_ppi --output-dir preprocess/go_ppi_splits
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from go_ppi_domain.data_converter import (
    BINARY_LABEL,
    GRADE_HIGH,
    GRADE_NONE,
    GRADE_WEAK,
    GRADE_NAMES,
    ORDINAL_LABEL,
)

logger = logging.getLogger(__name__)

DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.20
#: Max grade-0 records per anchor in validation (0 = uncapped).
DEFAULT_VAL_NO_INTERACTION_CAP = 20


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def _slot(protein: Dict[str, Any], id_key: str) -> Dict[str, Any]:
    """A view slot carrying the encoder text and both ontology facets.

    ``skill_uris`` / ``coarse_uris`` are the key names the shared pipeline reads:
    ``skill_uris`` is what ``BatchProcessor`` scores ontology negatives with and
    what ORCA's ``d_esco``/``s_esco`` capture uses, ``coarse_uris`` is the coarse
    facet consumed by the injected ``coarse_distance_fn``. Reusing them rather
    than renaming to GO terminology keeps ``contrastive_learning`` and ``orca``
    untouched.
    """
    return {
        id_key: protein["protein_id"],
        "encoder_view": protein["encoder_view"],
        "skill_uris": protein["go_uris"],
        "coarse_uris": protein["coarse_uris"],
    }


def _build_record(
    anchor_id: str,
    anchor_slot: Dict[str, Any],
    partner: Dict[str, Any],
    grade: int,
) -> Dict[str, Any]:
    candidate = _slot(partner, "partner_id")
    candidate["title"] = partner.get("gene", partner["protein_id"])
    candidate["grade"] = grade
    candidate["original_label"] = ORDINAL_LABEL[grade]
    return {
        "resume": anchor_slot,
        "job": candidate,
        "label": BINARY_LABEL[grade],
        "metadata": {
            # Query-group identity: the shared pipeline groups by
            # metadata['resume_id'], and the anchor protein is the query.
            "resume_id": anchor_id,
            "protein_id": anchor_id,
            "partner_id": partner["protein_id"],
            # Career vocabulary so run_ordinal_evaluation.py works unchanged.
            "original_label": ORDINAL_LABEL[grade],
            # The unmapped truth, for GO-specific analysis.
            "go_grade": grade,
            "go_grade_name": GRADE_NAMES[grade],
        },
    }


def build_splits(
    converted_dir: Path,
    output_dir: Path,
    strategy: str = "protein_disjoint",
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = 42,
    val_no_interaction_cap: int = DEFAULT_VAL_NO_INTERACTION_CAP,
    test_no_interaction_cap: int = 0,
) -> Dict[str, Any]:
    proteins = {r["protein_id"]: r for r in _read_jsonl(converted_dir / "proteins.jsonl")}
    pairs = list(_read_jsonl(converted_dir / "pairs.jsonl"))
    logger.info("loaded %d proteins, %d graded pairs", len(proteins), len(pairs))

    anchors = sorted({p["anchor_id"] for p in pairs})
    rng = random.Random(seed)

    if strategy == "protein_disjoint":
        # Partition ALL proteins, then keep only intra-partition pairs.
        all_ids = sorted(proteins)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_fraction))
        n_val = int(round(len(shuffled) * val_fraction))
        test_set = set(shuffled[:n_test])
        val_set = set(shuffled[n_test:n_test + n_val])
        train_set = set(shuffled[n_test + n_val:])
        group_of = {}
        for g, s in (("train", train_set), ("validation", val_set), ("test", test_set)):
            for pid in s:
                group_of[pid] = g

        kept: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        dropped = 0
        for p in pairs:
            ga = group_of.get(p["anchor_id"])
            gb = group_of.get(p["partner_id"])
            if ga is None or gb is None or ga != gb:
                dropped += 1
                continue
            kept[ga].append(p)
        assignment_note = (
            "proteins partitioned; a pair is kept only when BOTH endpoints fall in "
            "the same partition, so no protein appears in two splits in any role")
    elif strategy == "anchor_disjoint":
        shuffled = list(anchors)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_fraction))
        n_val = int(round(len(shuffled) * val_fraction))
        group_of = {}
        for pid in shuffled[:n_test]:
            group_of[pid] = "test"
        for pid in shuffled[n_test:n_test + n_val]:
            group_of[pid] = "validation"
        for pid in shuffled[n_test + n_val:]:
            group_of[pid] = "train"
        kept = collections.defaultdict(list)
        dropped = 0
        for p in pairs:
            g = group_of.get(p["anchor_id"])
            if g is None or p["partner_id"] not in proteins:
                dropped += 1
                continue
            kept[g].append(p)
        assignment_note = (
            "anchors disjoint only; partner proteins MAY recur across splits "
            "(trials-equivalent policy, weaker than the default)")
    else:
        raise SystemExit(f"unknown strategy {strategy!r}")

    grade_plan = {
        # Train: positives only. Negatives come from the selector at batch time.
        "train": {GRADE_HIGH: 0},
        "validation": {GRADE_HIGH: 0, GRADE_WEAK: 0,
                       GRADE_NONE: val_no_interaction_cap},
        "test": {GRADE_HIGH: 0, GRADE_WEAK: 0,
                 GRADE_NONE: test_no_interaction_cap},
    }

    counts: Dict[str, int] = {}
    grade_counts: Dict[str, Dict[str, int]] = {}
    anchors_per_split: Dict[str, List[str]] = {}

    for split in ("train", "validation", "test"):
        by_anchor: Dict[str, Dict[int, List[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for p in kept.get(split, []):
            by_anchor[p["anchor_id"]][p["grade"]].append(p["partner_id"])

        records: List[Dict[str, Any]] = []
        per_grade: collections.Counter = collections.Counter()
        for anchor_id in sorted(by_anchor):
            anchor_protein = proteins.get(anchor_id)
            if anchor_protein is None:
                continue
            anchor_slot = _slot(anchor_protein, "protein_id")
            for grade, cap in grade_plan[split].items():
                partner_ids = sorted(set(by_anchor[anchor_id].get(grade, [])))
                if cap and len(partner_ids) > cap:
                    sub = random.Random(f"{seed}|{split}|{anchor_id}|{grade}")
                    partner_ids = sorted(sub.sample(partner_ids, cap))
                for pid in partner_ids:
                    partner = proteins.get(pid)
                    if partner is None:
                        continue
                    records.append(
                        _build_record(anchor_id, anchor_slot, partner, grade))
                    per_grade[GRADE_NAMES[grade]] += 1
        rng.shuffle(records)
        counts[split] = _write_jsonl(output_dir / f"{split}.jsonl", records)
        grade_counts[split] = dict(per_grade)
        anchors_per_split[split] = sorted(by_anchor)

    # ---- graded negative pools ------------------------------------------
    pools: List[Dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        by_anchor = collections.defaultdict(lambda: collections.defaultdict(list))
        for p in kept.get(split, []):
            by_anchor[p["anchor_id"]][p["grade"]].append(p["partner_id"])
        for anchor_id in sorted(by_anchor):
            pools.append({
                "anchor_id": anchor_id,
                "split": split,
                # Kept apart by grade so the selector can ramp the ambiguous
                # (grade-1) share independently of the trivial (grade-0) one.
                "weak_evidence": sorted(set(by_anchor[anchor_id].get(GRADE_WEAK, []))),
                "no_interaction": sorted(set(by_anchor[anchor_id].get(GRADE_NONE, []))),
            })
    _write_jsonl(output_dir / "negative_pools.jsonl", pools)

    # ---- leakage audit ---------------------------------------------------
    def entities(split: str) -> set:
        s = set()
        for p in kept.get(split, []):
            s.add(p["anchor_id"])
            s.add(p["partner_id"])
        return s

    ents = {s: entities(s) for s in ("train", "validation", "test")}
    leakage = {
        "train_validation": len(ents["train"] & ents["validation"]),
        "train_test": len(ents["train"] & ents["test"]),
        "validation_test": len(ents["validation"] & ents["test"]),
    }
    if strategy == "protein_disjoint":
        for k, v in leakage.items():
            assert v == 0, f"protein leak between {k}: {v} shared proteins"

    manifest = {
        "strategy": strategy,
        "policy": assignment_note,
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "records": counts,
        "records_by_grade": grade_counts,
        "anchors": {k: len(v) for k, v in anchors_per_split.items()},
        "entities_per_split": {k: len(v) for k, v in ents.items()},
        "entity_overlap_between_splits": leakage,
        "pairs_dropped_cross_split": dropped,
        "pairs_total": len(pairs),
        "grade_mapping": {
            str(g): {"grade_name": GRADE_NAMES[g],
                     "original_label": ORDINAL_LABEL[g],
                     "label": BINARY_LABEL[g]}
            for g in (GRADE_HIGH, GRADE_WEAK, GRADE_NONE)
        },
        "val_no_interaction_cap": val_no_interaction_cap,
        "test_no_interaction_cap": test_no_interaction_cap,
        "negative_pool_totals": {
            "weak_evidence": sum(len(p["weak_evidence"]) for p in pools),
            "no_interaction": sum(len(p["no_interaction"]) for p in pools),
        },
        "anchors_without_weak_pool": sum(
            1 for p in pools if p["split"] == "train" and not p["weak_evidence"]),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("splits: %s", counts)
    logger.info("entity overlap between splits: %s", leakage)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--converted-dir", type=Path, default=Path("preprocess/go_ppi"))
    ap.add_argument("--output-dir", type=Path, default=Path("preprocess/go_ppi_splits"))
    ap.add_argument("--strategy", default="protein_disjoint",
                    choices=["protein_disjoint", "anchor_disjoint"])
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-no-interaction-cap", type=int,
                    default=DEFAULT_VAL_NO_INTERACTION_CAP)
    ap.add_argument("--test-no-interaction-cap", type=int, default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_splits(
        args.converted_dir, args.output_dir, strategy=args.strategy,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        seed=args.seed, val_no_interaction_cap=args.val_no_interaction_cap,
        test_no_interaction_cap=args.test_no_interaction_cap,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
