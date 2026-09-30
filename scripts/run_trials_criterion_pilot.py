#!/usr/bin/env python3
"""Run local exploratory criterion baselines. This command never scores holdout.

An optional pinned general-domain NLI encoder adds an actual entailment baseline.
It is NOT a fresh TrialGPT/generative-LLM reproduction or a clinical validator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trials_domain.criterion_pilot import LABELS, INPUT_KEYS, digest, read_jsonl, write_json, write_jsonl


def acquire_nli(config: dict) -> None:
    folder = ROOT / config["nli_dir"]
    manifest_path = folder / "acquisition_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != config["nli_model_revision"]:
            raise ValueError("NLI model revision changed")
        for item in manifest["files"]:
            if digest(folder / item["path"]) != item["sha256"]:
                raise ValueError("NLI checksum mismatch")
        print("Pinned local NLI model verified")
        return
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    # Safetensors only; avoid duplicate PyTorch and ONNX copies.
    for name in ["README.md", "config.json", "tokenizer_config.json", "tokenizer.json",
                 "special_tokens_map.json", "added_tokens.json", "model.safetensors"]:
        url = f"https://huggingface.co/{config['nli_model_id']}/resolve/{config['nli_model_revision']}/{name}"
        target = folder / name
        if target.exists():
            raise ValueError(f"Unmanifested download exists: {target}; inspect before resuming")
        with requests.get(url, timeout=90, stream=True) as response:
            response.raise_for_status()
            temporary = folder / f"{name}.partial"
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(target)
        files.append({"path": name, "sha256": digest(target), "bytes": target.stat().st_size, "url": url})
        print(f"Acquired NLI {name}: {target.stat().st_size:,} bytes", flush=True)
    write_json(manifest_path, {"model": config["nli_model_id"], "revision": config["nli_model_revision"],
                               "files": files, "use": "general-domain exploratory entailment baseline"})


def metrics(truth, prediction, probabilities=None) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    result = {"n": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
              "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
              "confusion_matrix_label_order": list(LABELS),
              "confusion_matrix": confusion_matrix(truth, prediction, labels=LABELS).tolist()}
    for label in LABELS:
        mask = truth == label
        result[f"{label}_n"] = int(mask.sum())
        result[f"{label}_recall"] = float(np.mean(prediction[mask] == label)) if mask.any() else None
    met, notmet = truth == "met", truth == "not_met"
    result["false_rejection_rate"] = float(np.mean(prediction[met] == "not_met")) if met.any() else None
    result["false_clearance_rate"] = float(np.mean(prediction[notmet] == "met")) if notmet.any() else None
    if probabilities is not None:
        confidence = np.max(probabilities, axis=1)
        # Fixed coverage diagnostic, not a calibrated clinical confidence claim.
        order = np.argsort(-confidence, kind="stable")
        result["risk_at_coverage"] = {}
        for coverage in (0.5, 0.8, 1.0):
            ix = order[:max(1, int(np.ceil(len(truth) * coverage))) ]
            result["risk_at_coverage"][str(coverage)] = float(np.mean(truth[ix] != prediction[ix]))
    return result


def paired_bootstrap(truth, baseline, variant, patients, replicates, seed=1729) -> dict:
    """Resample whole patients; pair the same resampled rows across systems."""
    truth, baseline, variant = map(np.asarray, (truth, baseline, variant))
    unique = sorted(set(patients))
    indices = {p: np.flatnonzero(np.asarray(patients) == p) for p in unique}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        selected = np.concatenate([indices[p] for p in rng.choice(unique, size=len(unique), replace=True)])
        a = f1_score(truth[selected], baseline[selected], labels=LABELS, average="macro", zero_division=0)
        b = f1_score(truth[selected], variant[selected], labels=LABELS, average="macro", zero_division=0)
        differences.append(b - a)
    return {"unit": "patient", "patients": len(unique), "replicates": replicates,
            "metric": "four_class_macro_f1_delta", "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "caveat": "exploratory development interval; few patients and rare violations; not multiplicity adjusted"}


def prepare_features(config: dict, inputs: list[dict], use_nli: bool) -> tuple:
    import torch
    from sentence_transformers import SentenceTransformer
    from trials_domain.mesh_ontology import MeshIndex, MeshMatcher
    from trials_domain.concept_extractor import extract_concepts

    torch.set_num_threads(4)
    torch.manual_seed(1729)
    cache_dir = ROOT / "embedding_cache/trials_criterion_pilot"
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({
        "inputs": inputs, "config": config, "use_nli": use_nli,
        "code": digest(Path(__file__)), "mesh": digest(ROOT / config["mesh_descriptor"]),
    }, sort_keys=True).encode()).hexdigest()
    cached = cache_dir / f"features_{signature[:20]}.npz"
    cached_evidence = cached.with_suffix(".json")
    if cached.exists() and cached_evidence.exists():
        loaded = np.load(cached, allow_pickle=False)
        details = json.loads(cached_evidence.read_text())
        print("Using matching immutable feature cache", flush=True)
        return {key: loaded[key] for key in loaded.files}, details["evidence"], details["metadata"]

    texts = sorted({r["criterion_text"] for r in inputs} |
                   {s["text"] for r in inputs for s in r["patient_sentences"]})
    print(f"Encoding {len(texts):,} unique sentences/criteria with cached MPNet (CPU)", flush=True)
    encoder = SentenceTransformer(config["text_model_path"], device="cpu", local_files_only=True)
    encoder.max_seq_length = config["max_sequence_length"]
    lengths = [len(encoder.tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    embeddings = encoder.encode(texts, batch_size=config["sentence_batch_size"],
                                normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    embedding = dict(zip(texts, embeddings))
    evidence, vectors, similarities, premises = [], [], [], []
    for row in inputs:
        criterion = embedding[row["criterion_text"]]
        sentence_vectors = np.stack([embedding[s["text"]] for s in row["patient_sentences"]])
        scores = sentence_vectors @ criterion
        selected = np.argsort(-scores, kind="stable")[:config["top_evidence_sentences"]]
        pooled = sentence_vectors[selected].mean(axis=0)
        vector = np.concatenate([criterion, pooled, abs(criterion - pooled), criterion * pooled])
        vectors.append(vector)
        similarities.append([float(scores.max()), float(scores[selected].mean()),
                             float(row["criterion_type"] == "exclusion")])
        chosen = [row["patient_sentences"][int(i)] for i in selected]
        premises.append(" ".join(s["text"] for s in chosen))
        evidence.append({"record_id": row["record_id"], "sentence_ids": [s["sentence_id"] for s in chosen],
                         "cosine_scores": [float(scores[i]) for i in selected],
                         "origin": "MPNet_top3_no_gold_evidence_used"})
    del encoder
    nli, nli_truncated, label_order = np.empty((len(inputs), 0)), 0, {}
    if use_nli:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        folder = ROOT / config["nli_dir"]
        if not (folder / "acquisition_manifest.json").exists():
            raise ValueError("Run --acquire-nli first, or explicitly use --without-nli")
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(folder), local_files_only=True)
        model.eval()
        label_order = {str(k): v for k, v in model.config.id2label.items()}
        nli_rows = []
        hypotheses = ["The patient meets this condition: " + r["criterion_text"] for r in inputs]
        nli_truncated = sum(len(tokenizer(a, b, truncation=False)["input_ids"]) > config["nli_max_length"]
                            for a, b in zip(premises, hypotheses))
        print(f"NLI criterion evidence evaluation: {len(inputs)} rows", flush=True)
        for start in range(0, len(inputs), config["nli_batch_size"]):
            batch = tokenizer(premises[start:start + config["nli_batch_size"]],
                              hypotheses[start:start + config["nli_batch_size"]], padding=True,
                              truncation="longest_first", max_length=config["nli_max_length"], return_tensors="pt")
            with torch.inference_mode():
                nli_rows.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
            if start % 100 == 0:
                print(f"NLI {min(start + config['nli_batch_size'], len(inputs))}/{len(inputs)}", flush=True)
        nli = np.concatenate(nli_rows)
        del model

    print("Calculating concept and hierarchy controls", flush=True)
    index = MeshIndex.from_descriptor_file(ROOT / config["mesh_descriptor"])
    matcher = MeshMatcher(index, cache_size=100_000, set_sim_cache_size=10_000)
    uis = sorted(index.ui_to_trees)
    shuffled = np.random.default_rng(config["split_seed"]).permutation(uis)
    random_index = MeshIndex(index.ui_to_name, {ui: index.ui_to_trees[other] for ui, other in zip(uis, shuffled)},
                             {}, index.term_to_ui, "descriptor-tree permutation negative control")
    random_matcher = MeshMatcher(random_index, cache_size=100_000, set_sim_cache_size=10_000)
    concept, hierarchy, shuffled_hierarchy, gated = [], [], [], []
    for row, premise, text_info in zip(inputs, premises, similarities):
        a, b = set(extract_concepts(premise, index)), set(row["criterion_mesh_ids"])
        coverage = len(a & b) / len(b) if b else 0.0
        jaccard = len(a & b) / len(a | b) if a | b else 0.0
        concept.append([coverage, jaccard, float(bool(a)), float(bool(b))])
        score = matcher.ontology_set_similarity(sorted(a), sorted(b)) if a and b else 0.0
        random_score = random_matcher.ontology_set_similarity(sorted(a), sorted(b)) if a and b else 0.0
        hierarchy.append([score])
        shuffled_hierarchy.append([random_score])
        gate = bool(a and b and text_info[0] >= config["ontology_gate_min_cosine"])
        gated.append([score if gate else 0.0])
    base = np.concatenate([np.asarray(vectors), np.asarray(similarities), nli], axis=1)
    arrays = {"base": base, "concept": np.asarray(concept), "hierarchy": np.asarray(hierarchy),
              "shuffled_hierarchy": np.asarray(shuffled_hierarchy), "gated_hierarchy": np.asarray(gated),
              "nli_probabilities": nli}
    metadata = {"feature_cache_signature": signature, "text_model": config["text_model_id"],
                "text_revision": config["text_model_revision"], "text_max_length": config["max_sequence_length"],
                "unique_texts": len(texts), "text_truncated_count": sum(n > config["max_sequence_length"] for n in lengths),
                "max_untruncated_text_tokens": max(lengths), "evidence_top_k": config["top_evidence_sentences"],
                "gold_evidence_used_for_features": False, "nli_used": use_nli,
                "nli_model": config["nli_model_id"] if use_nli else None,
                "nli_revision": config["nli_model_revision"] if use_nli else None,
                "nli_id2label": label_order, "nli_truncated_pairs": nli_truncated,
                "hierarchy_control": "fixed random descriptor-to-tree permutation; exact concept identity preserved",
                "gate": "MeSH mapping on both sides and top text cosine >= prespecified 0.25; heuristic, not clinically validated",
                "ontology_is_not_patient_evidence": True}
    np.savez_compressed(cached, **arrays)
    write_json(cached_evidence, {"evidence": evidence, "metadata": metadata})
    return arrays, evidence, metadata


def run(config: dict, use_nli: bool) -> dict:
    import joblib
    prepared, output = ROOT / config["prepared_dir"], ROOT / config["results_dir"]
    output.mkdir(parents=True, exist_ok=True)
    # Intentional allowlist: neither holdout inputs nor holdout reference files are opened.
    train = read_jsonl(prepared / "train/inputs.jsonl")
    dev = read_jsonl(prepared / "development/inputs.jsonl")
    for row in train + dev:
        if set(row) != INPUT_KEYS:
            raise ValueError("Unsafe or unexpected model input fields")
    train_ref = {r["record_id"]: r for r in read_jsonl(prepared / "train/references.jsonl")}
    dev_ref = {r["record_id"]: r for r in read_jsonl(prepared / "development/references.jsonl")}
    if {r["patient_id"] for r in train} & {r["patient_id"] for r in dev}:
        raise ValueError("Patient leakage")
    arrays, evidence, feature_meta = prepare_features(config, train + dev, use_nli)
    ytrain = np.asarray([train_ref[r["record_id"]]["constraint_status"] for r in train])
    ydev = np.asarray([dev_ref[r["record_id"]]["constraint_status"] for r in dev])
    patients = [r["patient_id"] for r in dev]
    ntrain = len(train)
    variants = {"text": ["base"], "text_concepts": ["base", "concept"],
                "text_concepts_hierarchy": ["base", "concept", "hierarchy"],
                "text_concepts_shuffled_hierarchy": ["base", "concept", "shuffled_hierarchy"],
                "text_concepts_gated_hierarchy": ["base", "concept", "gated_hierarchy"]}
    scores, predictions, probabilities, prediction_rows = {}, {}, {}, []
    for name, fields in variants.items():
        X = np.concatenate([arrays[field] for field in fields], axis=1)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(
            C=config["classifier_C"], class_weight="balanced", max_iter=2000, random_state=1729))
        classifier.fit(X[:ntrain], ytrain)
        prediction = classifier.predict(X[ntrain:])
        original_prob = classifier.predict_proba(X[ntrain:])
        classes = classifier[-1].classes_.tolist()
        prob = original_prob[:, [classes.index(label) for label in LABELS]]
        probabilities[name], predictions[name] = prob, prediction
        scores[name] = metrics(ydev, prediction, prob)
        scores[name]["feature_dimension"] = X.shape[1]
        scores[name]["classifier_iterations"] = classifier[-1].n_iter_.tolist()
        model_path = output / "models" / f"{name}.joblib"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(classifier, model_path)
        for row, pred, p in zip(dev, prediction, prob):
            prediction_rows.append({"record_id": row["record_id"], "patient_id": row["patient_id"],
                                    "system": name, "predicted_status": str(pred),
                                    "probabilities": dict(zip(LABELS, map(float, p))),
                                    "origin": "fresh_local_frozen_encoders_plus_train_only_linear_classifier"})
        print(f"{name}: development macro-F1={scores[name]['macro_f1']:.4f}", flush=True)
    for name in variants:
        if name != "text":
            scores[name]["paired_delta_vs_text"] = paired_bootstrap(
                ydev, predictions["text"], predictions[name], patients, config["bootstrap_replicates"])
    majority = Counter(ytrain).most_common(1)[0][0]
    scores["train_majority"] = metrics(ydev, [majority] * len(dev))
    source = {r["record_id"]: r for r in read_jsonl(prepared / "development/historical_gpt4.jsonl")}
    source_predictions = [source[r["record_id"]]["predicted_status"] for r in dev]
    scores["historical_gpt4_reference"] = metrics(ydev, source_predictions)
    scores["historical_gpt4_reference"]["caveat"] = (
        "Source-cached predictions, not rerun or compute-matched; experts assessed these same outputs during dataset construction.")
    write_jsonl(output / "development_predictions.jsonl", prediction_rows)
    dev_evidence = evidence[ntrain:]
    write_jsonl(output / "development_retrieved_evidence.jsonl", dev_evidence)
    supported, evidence_precision, evidence_recall = [], [], []
    for row, retrieved in zip(dev, dev_evidence):
        gold, predicted = set(dev_ref[row["record_id"]]["expert_sentence_ids"]), set(retrieved["sentence_ids"])
        if gold:
            supported.append(row["record_id"])
            evidence_precision.append(len(gold & predicted) / len(predicted))
            evidence_recall.append(len(gold & predicted) / len(gold))
    slice_results = {}
    slices = {"all": np.ones(len(dev), dtype=bool)}
    for kind in ("inclusion", "exclusion"):
        slices[kind] = np.asarray([r["criterion_type"] == kind for r in dev])
    slices["no_annotated_evidence"] = np.asarray([not dev_ref[r["record_id"]]["expert_sentence_ids"] for r in dev])
    slices["no_criterion_mesh_mapping"] = np.asarray([not r["criterion_mesh_ids"] for r in dev])
    for name, mask in slices.items():
        if mask.any():
            slice_results[name] = {system: metrics(ydev[mask], pred[mask]) for system, pred in predictions.items()}
    result = {"status": "exploratory_development_only", "holdout_scored": False,
              "train_rows": len(train), "development_rows": len(dev),
              "train_patients": len(set(r["patient_id"] for r in train)), "development_patients": len(set(patients)),
              "labels": list(LABELS), "metrics": scores, "slices": slice_results,
              "evidence_retrieval": {"evaluated_only_where_gold_evidence_nonempty": True,
                                     "n": len(supported), "precision": float(np.mean(evidence_precision)),
                                     "recall": float(np.mean(evidence_recall)), "identical_across_variants": True},
              "features": feature_meta, "config": config,
              "versions": {name: importlib.metadata.version(name) for name in
                           ["torch", "transformers", "sentence-transformers", "scikit-learn", "pandas", "pyarrow"]},
              "input_hashes": {f"{split}/{file}": digest(prepared / split / file) for split in
                               ("train", "development") for file in ("inputs.jsonl", "references.jsonl")},
              "code_sha256": digest(Path(__file__)),
              "limitations": ["General-domain frozen encoders; no fresh generative clinical LLM run",
                              "10 development patients and only four not_met criteria",
                              "Source unknown-label conventions need clinical review",
                              "No new expert-reviewed challenge cases or clinician workflow study",
                              "No trial-level aggregation metrics inferred from partial criterion annotations"]}
    write_json(output / "development_results.json", result)
    print("Saved development results; holdout was not scored", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/trials_criterion_pilot.json")
    parser.add_argument("--acquire-nli", action="store_true")
    parser.add_argument("--without-nli", action="store_true", help="Explicit embedding-only fallback; reported as such")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.acquire_nli:
        acquire_nli(config)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        run(config, not args.without_nli)


if __name__ == "__main__":
    main()
