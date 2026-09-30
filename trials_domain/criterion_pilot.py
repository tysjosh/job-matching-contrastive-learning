"""Auditable data preparation for exploratory patient–criterion experiments.

Gold annotations and historical model outputs never enter model-input files.
The audit flags review needs; it does not invent clinical reference labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

LABELS = ("met", "not_met", "unknown", "not_applicable")
LABEL_MAP = {
    "inclusion": {"included": "met", "not included": "not_met",
                  "not enough information": "unknown", "not applicable": "not_applicable"},
    "exclusion": {"not excluded": "met", "excluded": "not_met",
                  "not enough information": "unknown", "not applicable": "not_applicable"},
}
INPUT_KEYS = {"record_id", "patient_id", "trial_id", "trial_title", "patient_text",
              "patient_sentences", "criterion_type", "criterion_text", "patient_mesh_ids",
              "criterion_mesh_ids", "criterion_tags", "split", "source_revision"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_label(kind: str, label: str) -> str:
    try:
        return LABEL_MAP[kind][label.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Invalid label for {kind}: {label}") from exc


def parse_sentences(note: str) -> list[dict]:
    matches = list(re.finditer(r"(?m)^(\d+)\.\s*", note))
    if not matches or note[:matches[0].start()].strip():
        raise ValueError("Expected numbered patient sentences; refusing to invent evidence IDs")
    result = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        result.append({"sentence_id": int(match.group(1)), "text": note[match.end():end].strip()})
    if len({s["sentence_id"] for s in result}) != len(result):
        raise ValueError("Duplicate sentence identifiers")
    return result


def parse_evidence(raw: str, sentences: list[dict]) -> list[int]:
    ids = json.loads(raw)
    valid = {s["sentence_id"] for s in sentences}
    if not isinstance(ids, list) or any(type(i) is not int or i not in valid for i in ids):
        raise ValueError(f"Invalid evidence reference: {raw}")
    return sorted(set(ids))


def patient_splits(ids, seed: int, dev_fraction: float, holdout_fraction: float) -> dict:
    unique = sorted(set(ids))
    if not 0 < dev_fraction < 1 or not 0 < holdout_fraction < 1 or dev_fraction + holdout_fraction >= 1:
        raise ValueError("Invalid split fractions")
    random.Random(seed).shuffle(unique)
    ndev = max(1, int(len(unique) * dev_fraction))
    nholdout = max(1, math.ceil(len(unique) * holdout_fraction))
    ntrain = len(unique) - ndev - nholdout
    if ntrain < 1:
        raise ValueError("Not enough patient groups")
    return {"train": sorted(unique[:ntrain]), "development": sorted(unique[ntrain:ntrain + ndev]),
            "holdout": sorted(unique[ntrain + ndev:])}


def criterion_tags(text: str) -> list[str]:
    patterns = {
        "temporal": r"\b(within|prior|previous|past|months?|years?|weeks?|days?|history)\b",
        "numeric": r"\d|[<>≤≥]",
        "medication": r"\b(treatment|therapy|drug|medication|received|receiving|taking)\b",
        "negation": r"\b(no|not|without|except|excluding|absence)\b",
        "logical": r"\b(and|or|either|both|unless)\b",
    }
    tags = [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]
    return tags or ["other"]


def review_sample(inputs: list[dict], size: int, seed: int) -> list[dict]:
    """Round-robin coverage of input-only tags; no reference labels used."""
    buckets = defaultdict(list)
    for row in inputs:
        buckets[(row["criterion_type"], row["criterion_tags"][0])].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < min(size, len(inputs)):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < size:
                chosen.append(buckets[key].pop())
    return sorted(chosen, key=lambda r: r["record_id"])


def prepare(root: Path, config: dict, acquisition: dict) -> dict:
    from trials_domain.concept_extractor import extract_concepts
    from trials_domain.mesh_ontology import MeshIndex

    raw = root / config["raw_dir"] / "data/train-00000-of-00001.parquet"
    frame = pd.read_parquet(raw)
    required = {"annotation_id", "patient_id", "note", "trial_id", "trial_title", "criterion_type",
                "criterion_text", "gpt4_explanation", "explanation_correctness", "gpt4_sentences",
                "expert_sentences", "gpt4_eligibility", "expert_eligibility", "training"}
    if set(frame.columns) != required or len(frame) != config["expected_rows"]:
        raise ValueError("Unexpected source schema or row count")
    source_frame = frame.copy()
    if frame.annotation_id.duplicated().any():
        raise ValueError("Duplicate annotation ID")
    incomplete = frame.criterion_text.isna() | frame.criterion_text.fillna("").str.strip().eq("")
    quarantined = []
    for row in frame[incomplete].to_dict("records"):
        quarantined.append({"annotation_id": row["annotation_id"], "patient_id": row["patient_id"],
                            "trial_id": row["trial_id"], "reason": "missing_criterion_text",
                            "source_reference": "immutable raw parquet; no text or label imputed"})
    frame = frame[~incomplete].copy()
    if frame.isna().any().any():
        raise ValueError("Unexpected null outside quarantined criterion-text records")
    if frame.duplicated(["patient_id", "trial_id", "criterion_type", "criterion_text"]).any():
        raise ValueError("Duplicate patient-trial-criterion key")
    if frame.groupby("patient_id").note.nunique().max() != 1:
        raise ValueError("A patient has inconsistent notes")
    if set(frame.training.unique()) - {True, False}:
        raise ValueError("Unexpected source training flag")
    splits = patient_splits(frame.patient_id, config["split_seed"],
                            config["development_fraction"], config["holdout_fraction"])
    membership = {pid: split for split, pids in splits.items() for pid in pids}
    normalized_notes = defaultdict(set)
    for row in frame.itertuples():
        normalized_notes[normalize_text(row.note)].add(membership[row.patient_id])
    if any(len(groups) > 1 for groups in normalized_notes.values()):
        raise ValueError("Equivalent patient notes cross split boundaries")

    out = root / config["prepared_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "quarantine.jsonl", quarantined)
    old_manifest = out / "split_manifest.json"
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        if old["patients"] != splits or old["source_sha256"] != digest(raw):
            raise ValueError("Refusing to replace a different split; use a new version directory")
    print("Parsing the pinned MeSH 2021 source", flush=True)
    index = MeshIndex.from_descriptor_file(root / config["mesh_descriptor"])
    text_mesh = {}
    for text in sorted(set(frame.note) | set(frame.criterion_text)):
        text_mesh[text] = extract_concepts(text, index)

    inputs, references, historical, flags = [], [], [], []
    no_evidence_by_label = Counter()
    for row in frame.sort_values("annotation_id").to_dict("records"):
        rid = f"trialgpt-criterion-{row['annotation_id']:04d}"
        sentences = parse_sentences(row["note"])
        expert_ids = parse_evidence(row["expert_sentences"], sentences)
        model_ids = parse_evidence(row["gpt4_sentences"], sentences)
        label = normalize_label(row["criterion_type"], row["expert_eligibility"])
        historical_label = normalize_label(row["criterion_type"], row["gpt4_eligibility"])
        split = membership[row["patient_id"]]
        inputs.append({"record_id": rid, "patient_id": row["patient_id"], "trial_id": row["trial_id"],
                       "trial_title": row["trial_title"], "patient_text": row["note"],
                       "patient_sentences": sentences, "criterion_type": row["criterion_type"],
                       "criterion_text": row["criterion_text"], "split": split,
                       "patient_mesh_ids": text_mesh[row["note"]],
                       "criterion_mesh_ids": text_mesh[row["criterion_text"]],
                       "criterion_tags": criterion_tags(row["criterion_text"]),
                       "source_revision": config["dataset_revision"]})
        row_flags = []
        if not expert_ids:
            no_evidence_by_label[row["expert_eligibility"]] += 1
            row_flags.append("no_annotated_evidence")
            if label in {"met", "not_met"}:
                row_flags.append("decisive_label_without_annotated_evidence")
        if row["criterion_type"] == "exclusion" and label == "met" and not expert_ids:
            row_flags.append("review_absence_of_mention_policy")
        if not text_mesh[row["criterion_text"]]:
            row_flags.append("no_criterion_mesh_mapping")
        references.append({"record_id": rid, "expert_label_original": row["expert_eligibility"],
                           "constraint_status": label, "expert_sentence_ids": expert_ids,
                           "source_training_flag": row["training"], "review_flags": row_flags})
        historical.append({"record_id": rid, "prediction_origin": "source_cached_gpt4_not_rerun",
                           "predicted_status": historical_label, "predicted_sentence_ids": model_ids,
                           "source_explanation": row["gpt4_explanation"],
                           "expert_explanation_rating": row["explanation_correctness"]})
        for flag in row_flags:
            flags.append({"record_id": rid, "patient_id": row["patient_id"], "split": split, "flag": flag})
    if any(set(row) != INPUT_KEYS for row in inputs):
        raise ValueError("Model-input field allowlist violated")

    ref_index = {r["record_id"]: r for r in references}
    hist_index = {r["record_id"]: r for r in historical}
    split_summary = {}
    trial_sets = {}
    for split in splits:
        subset = [r for r in inputs if r["split"] == split]
        write_jsonl(out / split / "inputs.jsonl", subset)
        write_jsonl(out / split / "references.jsonl", [ref_index[r["record_id"]] for r in subset])
        write_jsonl(out / split / "historical_gpt4.jsonl", [hist_index[r["record_id"]] for r in subset])
        trial_sets[split] = {r["trial_id"] for r in subset}
        split_summary[split] = {"patients": len(splits[split]), "rows": len(subset),
                                "trials": len(trial_sets[split]),
                                "label_counts": dict(Counter(ref_index[r["record_id"]]["constraint_status"]
                                                             for r in subset))}
    overlap = {f"{a}__{b}": len(trial_sets[a] & trial_sets[b])
               for a, b in [("train", "development"), ("train", "holdout"), ("development", "holdout")]}

    # TREC integration is provenance and overlap auditing, not a fabricated criterion-label join.
    trec = {}
    criterion_trials = set(frame.trial_id)
    for year in (2021, 2022):
        trec_root = root / "trec-clinical-trials/raw" / str(year)
        topic_file = trec_root / f"topics{year}.xml"
        qrel_file = trec_root / f"qrels{year}.txt"
        if not topic_file.exists() or not qrel_file.exists():
            raise ValueError(f"Missing local TREC {year} source")
        topics = ET.parse(topic_file).getroot().findall("topic")
        qrels = [line.split() for line in qrel_file.read_text().splitlines() if line.strip()]
        topic_texts = {normalize_text(" ".join(t.itertext())) for t in topics}
        trec[str(year)] = {"topics": len(topics), "judgments": len(qrels),
                          "grade_counts": dict(Counter(q[3] for q in qrels)),
                          "shared_trial_ids_with_trialgpt_criteria": len(criterion_trials & {q[2] for q in qrels}),
                          "exact_normalized_note_overlap": len(topic_texts & set(normalized_notes)),
                          "topic_sha256": digest(topic_file), "qrels_sha256": digest(qrel_file),
                          "label_grain": "patient_trial_not_criterion",
                          "trial_corpus_date": "2021-04-27"}

    selected = review_sample([r for r in inputs if r["split"] == "development"],
                             config["review_sample_size"], config["split_seed"])
    review_rows = []
    for row in selected:
        review_rows.append({"record_id": row["record_id"], "patient_id": row["patient_id"],
                           "trial_id": row["trial_id"], "criterion_type": row["criterion_type"],
                           "criterion_text": row["criterion_text"], "patient_text": row["patient_text"],
                           "reviewer_id": "", "reviewed_constraint_status": "",
                           "reviewed_sentence_ids": "", "missing_information": "",
                           "conflicting_information": "", "ontology_mapping_valid": "",
                           "adjudication_notes": "", "review_status": "pending"})
    review_path = out / "review/development_review_blinded.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    if not review_path.exists():
        pd.DataFrame(review_rows).to_csv(review_path, index=False)
    else:
        existing = pd.read_csv(review_path)
        if existing.record_id.tolist() != [r["record_id"] for r in review_rows]:
            raise ValueError("Existing review sample differs; preserving reviewer work")
    write_jsonl(out / "review/development_reference_key.jsonl", [ref_index[r["record_id"]] for r in selected])
    write_jsonl(out / "audit_flags.jsonl", flags)
    write_jsonl(out / "mesh_vocabulary.jsonl", [
        {"mesh_id": ui, "name": index.ui_to_name[ui], "tree_numbers": index.ui_to_trees[ui]}
        for ui in sorted({ui for ids in text_mesh.values() for ui in ids})])
    manifest = {"schema_version": 1, "source_sha256": digest(raw), "source_revision": config["dataset_revision"],
                "split_seed": config["split_seed"], "patients": splits, "summary": split_summary,
                "patient_overlap": {key: 0 for key in overlap}, "trial_overlap": overlap,
                "source_training_flag_used_for_split": False,
                "split_design": "single seeded patient shuffle; no metric or label optimization",
                "holdout_policy": "prepared and profiled; not scored by the pilot; public historical data is not a pristine external test",
                "mesh_sha256": digest(root / config["mesh_descriptor"]),
                "review_ids": [r["record_id"] for r in selected]}
    write_json(old_manifest, manifest)
    write_json(out / "trec_inventory.json", trec)
    mixed = int((frame.groupby("patient_id").training.nunique() > 1).sum())
    audit = {"source_rows": len(source_frame), "rows": len(frame), "quarantined_rows": len(quarantined),
             "columns": len(frame.columns), "patients": frame.patient_id.nunique(),
             "trials": frame.trial_id.nunique(), "patient_trial_pairs": len(frame[["patient_id", "trial_id"]].drop_duplicates()),
             "source_null_cells": int(source_frame.isna().sum().sum()),
             "null_cells": int(frame.isna().sum().sum()), "duplicate_annotation_ids": 0,
             "duplicate_patient_trial_criterion_keys": 0, "invalid_evidence_references": 0,
             "empty_strings_by_column": {c: int(frame[c].eq("").sum()) for c in frame.columns if frame[c].dtype == object},
             "source_flag_counts": {str(k): int(v) for k, v in frame.training.value_counts().items()},
             "patients_crossing_source_training_flag": mixed,
             "source_label_counts": dict(Counter(frame.expert_eligibility)),
             "source_no_evidence_by_label": dict(no_evidence_by_label),
             "flag_counts": dict(Counter(r["flag"] for r in flags)),
             "criterion_mesh_coverage": sum(bool(r["criterion_mesh_ids"]) for r in inputs) / len(inputs),
             "patient_mesh_coverage": sum(bool(text_mesh[t]) for t in set(frame.note)) / frame.note.nunique(),
             "split_summary": split_summary, "trial_overlap": overlap,
             "license": acquisition["license_declared_in_card"],
             "license_file_empty": acquisition["license_file_empty"],
             "review_sample_size": len(selected), "review_status": "pending_clinical_review",
             "interpretation": "No annotated sentence is not proof of absent evidence or an incorrect label. Flags require review.",
             "temporal_limitation": "Static synthetic notes; both TREC years use the same 2021 trial snapshot.",
             "source_revision": config["dataset_revision"], "source_sha256": digest(raw)}
    # Convert pandas/NumPy scalar counts to native types.
    for key in ("patients", "trials"):
        audit[key] = int(audit[key])
    write_json(out / "data_quality_audit.json", audit)
    artifact_paths = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "file_manifest.json"
                            and "review" not in p.parts)
    write_json(out / "file_manifest.json", {str(p.relative_to(out)): digest(p) for p in artifact_paths})
    return {"prepared_dir": str(out), "rows": len(inputs), "split_summary": split_summary,
            "review_cases": len(selected), "source_training_patient_overlap": mixed}
