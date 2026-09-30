"""Scientific-integrity checks for the criterion pilot's actual data boundaries."""
import json
from pathlib import Path

import numpy as np
import pytest

from trials_domain.criterion_pilot import (
    INPUT_KEYS, LABELS, normalize_label, parse_evidence, parse_sentences,
    patient_splits, read_jsonl, review_sample,
)
from scripts.run_trials_criterion_pilot import metrics, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def test_exclusion_direction_is_not_inverted():
    assert normalize_label("inclusion", "included") == "met"
    assert normalize_label("exclusion", "excluded") == "not_met"
    assert normalize_label("exclusion", "not excluded") == "met"
    assert normalize_label("exclusion", "not enough information") == "unknown"
    with pytest.raises(ValueError):
        normalize_label("inclusion", "excluded")


def test_evidence_references_preserve_source_ids_and_reject_fabricated_ids():
    sentences = parse_sentences("0. A patient has a measurement of 1.2.\n1. Result unavailable.")
    assert sentences[0]["text"] == "A patient has a measurement of 1.2."
    assert parse_evidence("[1]", sentences) == [1]
    with pytest.raises(ValueError):
        parse_evidence("[2]", sentences)
    with pytest.raises(ValueError):
        parse_evidence('["1"]', sentences)


def test_split_invariant_to_row_order_and_repeated_patient_rows():
    ids = [f"patient-{i}" for i in range(53)]
    first = patient_splits(ids, 1729, .2, .2)
    assert first == patient_splits(list(reversed(ids)) * 3, 1729, .2, .2)
    assert [len(first[s]) for s in ("train", "development", "holdout")] == [32, 10, 11]
    assert len(set().union(*map(set, first.values()))) == 53


def test_rare_classes_remain_in_macro_f1_denominator():
    result = metrics(["met"] * 5, ["met"] * 5)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == .25
    assert result["not_met_recall"] is None
    result = metrics(["met", "not_met", "unknown", "not_applicable"], ["not_met", "met", "unknown", "not_applicable"])
    assert result["false_rejection_rate"] == 1.0
    assert result["false_clearance_rate"] == 1.0


def test_paired_cluster_bootstrap_returns_zero_for_identical_systems():
    result = paired_bootstrap(np.array(LABELS * 2), np.array(LABELS * 2), np.array(LABELS * 2),
                              ["patient-a"] * 4 + ["patient-b"] * 4, 25)
    assert result["unit"] == "patient"
    assert result["ci95"] == [0.0, 0.0]


@pytest.mark.skipif(not (ROOT / "preprocess/trials_criterion_pilot/split_manifest.json").exists(),
                    reason="Run preparation to enable actual-source integration checks")
def test_actual_prepared_data_has_no_gold_inputs_or_patient_leakage():
    folder = ROOT / "preprocess/trials_criterion_pilot"
    patient_sets, total, references = [], 0, {}
    for split in ("train", "development", "holdout"):
        rows = read_jsonl(folder / split / "inputs.jsonl")
        refs = read_jsonl(folder / split / "references.jsonl")
        assert {r["record_id"] for r in rows} == {r["record_id"] for r in refs}
        assert all(set(r) == INPUT_KEYS for r in rows)
        assert all(r["criterion_text"].strip() for r in rows)
        assert all(r["constraint_status"] in LABELS for r in refs)
        patient_sets.append({r["patient_id"] for r in rows})
        references.update({r["record_id"]: r for r in refs})
        total += len(rows)
    assert total == 1014
    assert not patient_sets[0] & patient_sets[1]
    assert not patient_sets[0] & patient_sets[2]
    assert not patient_sets[1] & patient_sets[2]
    assert "trialgpt-criterion-0883" not in references
    audit = json.loads((folder / "data_quality_audit.json").read_text())
    assert audit["quarantined_rows"] == 1
    assert audit["patients_crossing_source_training_flag"] == 44


@pytest.mark.skipif(not (ROOT / "preprocess/trials_criterion_pilot/split_manifest.json").exists(),
                    reason="Preparation required")
def test_review_sample_is_development_only_and_reference_free():
    folder = ROOT / "preprocess/trials_criterion_pilot"
    rows = read_jsonl(folder / "development/inputs.jsonl")
    sample = review_sample(rows, 120, 1729)
    assert len(sample) == len({r["record_id"] for r in sample}) == 120
    assert all(r["split"] == "development" for r in sample)
    assert all("expert_label_original" not in r for r in sample)
