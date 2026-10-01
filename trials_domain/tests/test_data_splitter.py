"""Tests for the trials split builder.

The regression that motivates most of these: the eval splits originally emitted
grade-2 positives only, which made ``run_phase1_embedding_evaluation.py`` report
``auc_roc = nan`` with a confusion matrix of all true positives and nothing to
rank against. A split file that trains fine but cannot be evaluated is the kind of
defect that surfaces only at the end of a run, so the class balance of each split
is pinned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trials_domain.data_splitter import (
    BINARY_LABEL,
    GRADE_ELIGIBLE,
    GRADE_INELIGIBLE,
    GRADE_NOT_RELEVANT,
    ORDINAL_LABEL,
    TREC_LABEL,
    build_splits,
)


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def converted(tmp_path):
    """A miniature converted dataset: 5 topics in 2021, 3 in 2022."""
    root = tmp_path / "converted"
    topics, qrels, trials = [], [], []

    def topic(year, num):
        tid = f"{year}_{num}"
        topics.append({
            "topic_id": tid, "raw_topic_id": str(num), "year": year,
            "encoder_view": f"patient narrative {tid}",
            "narrative": f"patient narrative {tid}",
            "condition_uris": ["C01"], "intervention_uris": ["D01"],
            "mesh_uris": ["C01", "D01"],
            "condition_signal_present": True,
        })
        return tid

    counter = [0]

    def judge(tid, grade, n):
        for _ in range(n):
            counter[0] += 1
            nct = f"NCT{counter[0]:08d}"
            trials.append({
                "nct_id": nct, "encoder_view": f"trial {nct}", "title": f"T{nct}",
                "mesh_uris": ["C01"], "condition_uris": ["C01"],
                "intervention_uris": [],
            })
            qrels.append({"topic_id": tid, "nct_id": nct, "grade": grade})

    for num in range(1, 6):
        tid = topic("2021", num)
        judge(tid, GRADE_ELIGIBLE, 4)
        judge(tid, GRADE_INELIGIBLE, 3)
        judge(tid, GRADE_NOT_RELEVANT, 20)
    for num in range(1, 4):
        tid = topic("2022", num)
        judge(tid, GRADE_ELIGIBLE, 2)
        judge(tid, GRADE_INELIGIBLE, 2)
        judge(tid, GRADE_NOT_RELEVANT, 15)

    _write(root / "topics.jsonl", topics)
    _write(root / "qrels.jsonl", qrels)
    _write(root / "trials.jsonl", trials)
    return root


def _load(out, split):
    return [json.loads(l) for l in open(out / f"{split}.jsonl") if l.strip()]


# ------------------------------------------------------------ class balance
def test_train_is_positives_only(converted, tmp_path):
    out = tmp_path / "splits"
    build_splits(converted, out, val_fraction=0.2, seed=42)
    rows = _load(out, "train")
    assert rows
    assert {r["label"] for r in rows} == {1}
    assert {r["metadata"]["trec_grade"] for r in rows} == {GRADE_ELIGIBLE}


@pytest.mark.parametrize("split", ["validation", "test"])
def test_eval_splits_carry_all_three_grades(converted, tmp_path, split):
    """The regression guard: an eval split needs negatives or AUC is undefined."""
    out = tmp_path / "splits"
    build_splits(converted, out, val_fraction=0.2, seed=42)
    rows = _load(out, split)
    grades = {r["metadata"]["trec_grade"] for r in rows}
    assert grades == {GRADE_ELIGIBLE, GRADE_INELIGIBLE, GRADE_NOT_RELEVANT}
    labels = {r["label"] for r in rows}
    assert labels == {0, 1}, "both classes must be present or auc_roc is nan"
    assert sum(1 for r in rows if r["label"] == 0) > 0
    assert sum(1 for r in rows if r["label"] == 1) > 0


def test_test_split_keeps_the_full_judged_pool_by_default(converted, tmp_path):
    out = tmp_path / "splits"
    build_splits(converted, out, val_fraction=0.2, seed=42)
    rows = _load(out, "test")
    # 3 topics x (2 eligible + 2 ineligible + 15 not_relevant)
    assert len(rows) == 3 * 19
    assert sum(1 for r in rows if r["metadata"]["trec_grade"] == 0) == 45


def test_validation_caps_only_the_not_relevant_class(converted, tmp_path):
    out = tmp_path / "splits"
    build_splits(converted, out, val_fraction=0.2, seed=42,
                 val_not_relevant_cap=5)
    rows = _load(out, "validation")
    n_topics = len({r["metadata"]["topic_id"] for r in rows})
    by_grade = {g: sum(1 for r in rows if r["metadata"]["trec_grade"] == g)
                for g in (0, 1, 2)}
    assert by_grade[0] == 5 * n_topics, "grade-0 must be capped"
    assert by_grade[1] == 3 * n_topics, "grade-1 is scarce and never capped"
    assert by_grade[2] == 4 * n_topics


def test_cap_is_deterministic_across_runs(converted, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build_splits(converted, a, val_fraction=0.2, seed=42, val_not_relevant_cap=5)
    build_splits(converted, b, val_fraction=0.2, seed=42, val_not_relevant_cap=5)
    ids = lambda p: sorted(
        (r["metadata"]["topic_id"], r["metadata"]["nct_id"]) for r in _load(p, "validation")
    )
    assert ids(a) == ids(b)


# --------------------------------------------------------------- label mapping
def test_grade_mapping_is_order_preserving_and_consistent(converted, tmp_path):
    out = tmp_path / "splits"
    build_splits(converted, out, val_fraction=0.2, seed=42)
    for row in _load(out, "test"):
        grade = row["metadata"]["trec_grade"]
        assert row["metadata"]["trec_label"] == TREC_LABEL[grade]
        # Career vocabulary, so run_ordinal_evaluation.py works unchanged.
        assert row["metadata"]["original_label"] == ORDINAL_LABEL[grade]
        assert row["label"] == BINARY_LABEL[grade]
        assert row["job"]["original_label"] == ORDINAL_LABEL[grade]


def test_only_grade_two_is_a_binary_positive():
    assert BINARY_LABEL[GRADE_ELIGIBLE] == 1
    assert BINARY_LABEL[GRADE_INELIGIBLE] == 0
    assert BINARY_LABEL[GRADE_NOT_RELEVANT] == 0
    assert ORDINAL_LABEL[GRADE_INELIGIBLE] == "potential_fit"


# ------------------------------------------------------------ split integrity
def test_splits_are_topic_disjoint_and_year_partitioned(converted, tmp_path):
    out = tmp_path / "splits"
    manifest = build_splits(converted, out, val_fraction=0.2, seed=42)
    per = {s: {r["metadata"]["topic_id"] for r in _load(out, s)}
           for s in ("train", "validation", "test")}
    assert not per["train"] & per["validation"]
    assert not per["train"] & per["test"]
    assert not per["validation"] & per["test"]
    assert all(t.startswith("2021_") for t in per["train"] | per["validation"])
    assert all(t.startswith("2022_") for t in per["test"])
    assert manifest["missing_trials"] == 0


def test_negative_pools_still_emitted_for_training(converted, tmp_path):
    out = tmp_path / "splits"
    build_splits(converted, out, val_fraction=0.2, seed=42)
    pools = [json.loads(l) for l in open(out / "negative_pools.jsonl") if l.strip()]
    assert pools
    train_pools = [p for p in pools if p["split"] == "train"]
    assert train_pools
    assert all(p["ineligible"] and p["not_relevant"] for p in train_pools)


def test_manifest_reports_per_grade_counts(converted, tmp_path):
    out = tmp_path / "splits"
    manifest = build_splits(converted, out, val_fraction=0.2, seed=42)
    assert set(manifest["records_by_grade"]) == {"train", "validation", "test"}
    assert manifest["records_by_grade"]["train"] == {"eligible": 4 * len(
        manifest["topic_assignment"]["train"])}
    for split in ("validation", "test"):
        assert set(manifest["records_by_grade"][split]) == {
            "eligible", "ineligible", "not_relevant"}
