#!/usr/bin/env python3
"""Confirmatory patient-group nested evaluation for TrialGPT criteria.

This script is deliberately separate from the exploratory pilot.  It uses
frozen text/NLI/MeSH features, performs all model and fusion-weight selection
inside training folds, reports continuous as well as hard-label metrics, and
scores the prepared holdout only after the nested evaluation is complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, log_loss)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trials_domain.criterion_pilot import LABELS, digest, read_jsonl, write_json, write_jsonl


DEFAULT_CONFIG = ROOT / "config/trials_criterion_pilot.json"
DEFAULT_OUTPUT = ROOT / "results/trials_criterion_nested"
CLASS_INDEX = {label: i for i, label in enumerate(LABELS)}
METRIC_LABELS = tuple(sorted(LABELS))

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponent / exponent.sum(axis=1, keepdims=True)


def aligned_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Return probabilities in the fixed LABELS order."""
    raw = model.predict_proba(X)
    classes = model[-1].classes_.tolist()
    result = np.zeros((len(X), len(LABELS)), dtype=float)
    for source_index, label in enumerate(classes):
        result[:, CLASS_INDEX[str(label)]] = raw[:, source_index]
    # If a pathological training fold lacks a class, keep probabilities valid.
    row_sums = result.sum(axis=1, keepdims=True)
    return np.divide(result, row_sums, out=np.full_like(result, 1.0 / len(LABELS)), where=row_sums > 0)


def aligned_log_probabilities(model, X: np.ndarray) -> np.ndarray:
    probabilities = np.clip(aligned_probabilities(model, X), 1e-8, 1.0)
    return np.log(probabilities)


def fit_classifier(X: np.ndarray, y: np.ndarray, C: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=1200,
                           solver="lbfgs", random_state=seed),
    ).fit(X, y)


def global_score_normalization(train_scores: np.ndarray, scores: np.ndarray,
                               target_std: float | None = None) -> np.ndarray:
    """Normalize a score block using training-only statistics.

    A global (rather than class-wise) affine transformation preserves the
    base model's argmax at fusion weight zero.
    """
    mean = float(train_scores.mean())
    std = float(train_scores.std())
    if std < 1e-8:
        std = 1.0
    if target_std is None:
        target_std = std
    return (scores - mean) * (target_std / std)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = np.asarray([LABELS[i] for i in np.argmax(probabilities, axis=1)])
    y_true = np.asarray(y_true)
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=LABELS,
                                   average="macro", zero_division=0)),
        "log_loss": float(log_loss(
            y_true, probabilities[:, [CLASS_INDEX[label] for label in METRIC_LABELS]],
            labels=list(METRIC_LABELS))),
        "multiclass_brier": float(np.mean(np.sum(
            (probabilities - np.eye(len(LABELS))[np.asarray([CLASS_INDEX[str(x)] for x in y_true])]) ** 2,
            axis=1))),
        "confusion_matrix_label_order": list(LABELS),
    }
    for label in LABELS:
        mask = y_true == label
        result[f"{label}_n"] = int(mask.sum())
        result[f"{label}_recall"] = float(np.mean(predictions[mask] == label)) if mask.any() else None
    ap_values = []
    for label in LABELS:
        target = (y_true == label).astype(int)
        if target.sum() and target.sum() < len(target):
            ap = float(average_precision_score(target, probabilities[:, CLASS_INDEX[label]]))
            result[f"{label}_average_precision"] = ap
            ap_values.append(ap)
        else:
            result[f"{label}_average_precision"] = None
    result["macro_average_precision"] = float(np.mean(ap_values)) if ap_values else None
    top_two = np.sort(probabilities, axis=1)[:, -2:]
    result["mean_top2_margin"] = float(np.mean(top_two[:, 1] - top_two[:, 0]))
    result["label_counts"] = dict(Counter(map(str, y_true)))
    return result


def quick_metric_values(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Only the two quantities used during inner selection."""
    predictions = np.asarray([LABELS[i] for i in np.argmax(probabilities, axis=1)])
    macro = float(f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0))
    loss = float(log_loss(y_true, probabilities[:, [CLASS_INDEX[label] for label in METRIC_LABELS]],
                          labels=list(METRIC_LABELS)))
    return macro, loss


def metric_value(y_true: np.ndarray, probabilities: np.ndarray, name: str) -> float:
    if name == "macro_f1":
        return quick_metric_values(y_true, probabilities)[0]
    if name == "log_loss":
        return quick_metric_values(y_true, probabilities)[1]
    if name == "multiclass_brier":
        one_hot = np.eye(len(LABELS))[np.asarray([CLASS_INDEX[str(x)] for x in y_true])]
        return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    raise ValueError(name)


def choose_C(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             inner_splits: list[tuple[np.ndarray, np.ndarray]],
             candidates: list[float], seed: int) -> tuple[float, dict]:
    scores = {}
    for C in candidates:
        fold_f1, fold_loss = [], []
        for train_idx, valid_idx in inner_splits:
            model = fit_classifier(X[train_idx], y[train_idx], C, seed)
            probabilities = aligned_probabilities(model, X[valid_idx])
            macro, loss = quick_metric_values(y[valid_idx], probabilities)
            fold_f1.append(macro)
            fold_loss.append(loss)
        scores[str(C)] = {"macro_f1_mean": float(np.mean(fold_f1)),
                          "macro_f1_sd": float(np.std(fold_f1)),
                          "log_loss_mean": float(np.mean(fold_loss))}
    selected = sorted(candidates, key=lambda C: (-scores[str(C)]["macro_f1_mean"],
                                                   scores[str(C)]["log_loss_mean"], C))[0]
    return float(selected), {"selected": float(selected), "candidates": scores}


def choose_fusion_lambda(base_X: np.ndarray, onto_X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray, inner_splits: list[tuple[np.ndarray, np.ndarray]],
                         base_C: float, onto_C: float, lambdas: list[float], seed: int) -> tuple[float, dict]:
    records = []
    for train_idx, valid_idx in inner_splits:
        base_model = fit_classifier(base_X[train_idx], y[train_idx], base_C, seed)
        onto_model = fit_classifier(onto_X[train_idx], y[train_idx], onto_C, seed)
        base_train = aligned_log_probabilities(base_model, base_X[train_idx])
        base_valid = aligned_log_probabilities(base_model, base_X[valid_idx])
        onto_train = aligned_log_probabilities(onto_model, onto_X[train_idx])
        onto_valid = aligned_log_probabilities(onto_model, onto_X[valid_idx])
        base_scale = max(float(base_train.std()), 1e-8)
        onto_valid_scaled = global_score_normalization(onto_train, onto_valid, base_scale)
        for value in lambdas:
            probabilities = softmax(base_valid + float(value) * onto_valid_scaled)
            macro, loss = quick_metric_values(y[valid_idx], probabilities)
            records.append({"lambda": float(value), "macro_f1": macro, "log_loss": loss})
    aggregate = {}
    for value in lambdas:
        subset = [r for r in records if r["lambda"] == float(value)]
        aggregate[str(value)] = {
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in subset])),
            "macro_f1_sd": float(np.std([r["macro_f1"] for r in subset])),
            "log_loss_mean": float(np.mean([r["log_loss"] for r in subset])),
        }
    selected = sorted(lambdas, key=lambda value: (-aggregate[str(value)]["macro_f1_mean"],
                                                    aggregate[str(value)]["log_loss_mean"], value))[0]
    return float(selected), {"selected": float(selected), "candidates": aggregate}


def load_split(root: Path, prepared: str, split: str) -> tuple[list[dict], dict[str, dict]]:
    folder = root / prepared / split
    inputs = read_jsonl(folder / "inputs.jsonl")
    references = {r["record_id"]: r for r in read_jsonl(folder / "references.jsonl")}
    return inputs, references


def paired_patient_bootstrap(records: list[dict], system_a: str, system_b: str,
                             metric_name: str, replicates: int, seed: int) -> dict:
    patients = sorted({r["patient_id"] for r in records})
    by_patient = {p: [r for r in records if r["patient_id"] == p] for p in patients}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        selected = [row for p in rng.choice(patients, len(patients), replace=True) for row in by_patient[p]]
        y = np.asarray([r["true_label"] for r in selected])
        a = np.asarray([r["probabilities"][system_a] for r in selected])
        b = np.asarray([r["probabilities"][system_b] for r in selected])
        differences.append(metric_value(y, b, metric_name) - metric_value(y, a, metric_name))
    return {"metric": metric_name, "unit": "patient", "replicates": replicates,
            "ci95": [float(x) for x in np.quantile(differences, [0.025, 0.975])],
            "mean": float(np.mean(differences))}


def evaluate_outer(X: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
                   row_ids: list[str], train_idx: np.ndarray, test_idx: np.ndarray,
                   seed: int, fold: int, config: dict) -> tuple[list[dict], dict]:
    inner_splitter = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True,
                                          random_state=seed + 1000 + fold)
    inner_splits = list(inner_splitter.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]))
    inner_splits = [(train_idx[a], train_idx[b]) for a, b in inner_splits]
    C_candidates = [float(x) for x in config["classifier_C_candidates"]]
    lambda_candidates = [float(x) for x in config["fusion_lambda_candidates"]]

    base_C, base_tuning = choose_C(X["base"], y, groups, inner_splits, C_candidates, seed)
    systems = {
        "text_nli": None,
        "concept_overlap_fusion": "concept",
        "mesh_hierarchy_fusion": "hierarchy",
        "gated_hierarchy_fusion": "gated_hierarchy",
        "shuffled_hierarchy_control": "shuffled_hierarchy",
    }
    models = {}
    tuning = {"base_C": base_tuning}
    for name, field in systems.items():
        if field is None:
            model = fit_classifier(X["base"][train_idx], y[train_idx], base_C, seed)
            models[name] = {"base": model, "lambda": 0.0}
            continue
        onto_C, onto_tuning = choose_C(X[field], y, groups, inner_splits, C_candidates, seed)
        selected_lambda, lambda_tuning = choose_fusion_lambda(
            X["base"], X[field], y, groups, inner_splits, base_C, onto_C,
            lambda_candidates, seed)
        models[name] = {"base": fit_classifier(X["base"][train_idx], y[train_idx], base_C, seed),
                        "onto": fit_classifier(X[field][train_idx], y[train_idx], onto_C, seed),
                        "lambda": selected_lambda, "field": field}
        tuning[name] = {"onto_C": onto_tuning, "fusion_lambda": lambda_tuning}

    predictions = {}
    for name, model_info in models.items():
        base_model = model_info["base"]
        base_test_prob = aligned_probabilities(base_model, X["base"][test_idx])
        if model_info["lambda"] == 0.0:
            predictions[name] = base_test_prob
            continue
        onto_model = model_info["onto"]
        base_train_log = aligned_log_probabilities(base_model, X["base"][train_idx])
        base_test_log = aligned_log_probabilities(base_model, X["base"][test_idx])
        onto_train_log = aligned_log_probabilities(onto_model, X[model_info["field"]][train_idx])
        onto_test_log = aligned_log_probabilities(onto_model, X[model_info["field"]][test_idx])
        onto_test_scaled = global_score_normalization(onto_train_log, onto_test_log,
                                                      max(float(base_train_log.std()), 1e-8))
        predictions[name] = softmax(base_test_log + model_info["lambda"] * onto_test_scaled)

    rows = []
    for local, row_index in enumerate(test_idx):
        per_system = {name: {"predicted_label": LABELS[int(np.argmax(prob[local]))],
                             "probabilities": {label: float(prob[local, i]) for i, label in enumerate(LABELS)}}
                      for name, prob in predictions.items()}
        rows.append({"record_id": str(row_ids[row_index]), "row_index": int(row_index),
                     "patient_id": str(groups[row_index]), "true_label": str(y[row_index]),
                     "seed": int(seed), "outer_fold": int(fold), "systems": per_system})
    fold_metrics = {name: metrics(y[test_idx], prob) for name, prob in predictions.items()}
    fold_info = {"seed": int(seed), "outer_fold": int(fold),
                 "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
                 "train_patients": int(len(set(groups[train_idx]))),
                 "test_patients": int(len(set(groups[test_idx]))),
                 "train_label_counts": dict(Counter(map(str, y[train_idx]))),
                 "test_label_counts": dict(Counter(map(str, y[test_idx]))),
                 "selected_tuning": tuning, "metrics": fold_metrics}
    return rows, fold_info


def flatten_predictions(rows: list[dict]) -> list[dict]:
    flattened = []
    for row in rows:
        for system, result in row["systems"].items():
            flattened.append({"record_id": row["record_id"], "row_index": row["row_index"],
                              "patient_id": row["patient_id"], "true_label": row["true_label"],
                              "seed": row["seed"], "outer_fold": row["outer_fold"],
                              "system": system, **result})
    return flattened


def write_summary(path: Path, aggregate: dict, bootstrap: dict) -> None:
    systems = sorted(aggregate)
    fields = ["system", "n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss",
              "multiclass_brier", "macro_average_precision", "mean_top2_margin"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for system in systems:
            row = {field: aggregate[system].get(field) for field in fields}
            row["system"] = system
            writer.writerow(row)
    write_json(path.with_name("paired_bootstrap.json"), bootstrap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update({"outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
                   "seeds": args.seeds,
                   "classifier_C_candidates": config.get("classifier_C_candidates", [0.03, 0.1, 0.3]),
                   "fusion_lambda_candidates": config.get("fusion_lambda_candidates", [0.0, 0.25, 0.5, 1.0, 2.0]),
                   "nested_protocol": "patient-group stratified nested CV; all tuning inside outer training folds",
                   "primary_metric": "macro_f1"})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    prepared = config["prepared_dir"]
    train_inputs, train_refs = load_split(ROOT, prepared, "train")
    dev_inputs, dev_refs = load_split(ROOT, prepared, "development")
    hold_inputs, hold_refs = load_split(ROOT, prepared, "holdout")
    development_inputs = train_inputs + dev_inputs
    all_inputs = development_inputs + hold_inputs
    all_refs = {**train_refs, **dev_refs, **hold_refs}
    if len({r["patient_id"] for r in development_inputs} & {r["patient_id"] for r in hold_inputs}):
        raise ValueError("Patient leakage between development pool and holdout")
    for row in all_inputs:
        if row["record_id"] not in all_refs:
            raise ValueError(f"Missing reference for {row['record_id']}")

    # Reuse the audited feature builder.  It uses no gold evidence or labels.
    from scripts.run_trials_criterion_pilot import prepare_features
    arrays, feature_evidence, feature_meta = prepare_features(config, all_inputs, use_nli=True)
    y = np.asarray([all_refs[r["record_id"]]["constraint_status"] for r in development_inputs])
    groups = np.asarray([r["patient_id"] for r in development_inputs])
    n_development = len(development_inputs)
    X = {key: value for key, value in arrays.items()
         if key in {"base", "concept", "hierarchy", "shuffled_hierarchy", "gated_hierarchy"}}
    if any(len(value) != n_development + len(hold_inputs) for value in X.values()):
        raise ValueError("Feature row count mismatch")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", config)
    write_json(output / "feature_metadata.json", feature_meta)
    write_json(output / "data_manifest.json", {
        "development_rows": len(development_inputs), "development_patients": len(set(groups)),
        "holdout_rows": len(hold_inputs), "holdout_patients": len(set(r["patient_id"] for r in hold_inputs)),
        "development_input_hashes": {"train_inputs": digest(ROOT / prepared / "train/inputs.jsonl"),
                                      "train_references": digest(ROOT / prepared / "train/references.jsonl"),
                                      "development_inputs": digest(ROOT / prepared / "development/inputs.jsonl"),
                                      "development_references": digest(ROOT / prepared / "development/references.jsonl")},
        "holdout_input_hashes": {"inputs": digest(ROOT / prepared / "holdout/inputs.jsonl"),
                                 "references": digest(ROOT / prepared / "holdout/references.jsonl")},
        "labels": list(LABELS), "holdout_policy": "not read until nested development evaluation finished",
    })

    all_outer_rows, fold_infos = [], []
    for seed in args.seeds:
        splitter = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(n_development), y, groups))
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"outer seed={seed} fold={fold + 1}/{args.outer_folds}", flush=True)
            rows, info = evaluate_outer(X, y, groups,
                                        [r["record_id"] for r in development_inputs],
                                        train_idx, test_idx, seed, fold, config)
            all_outer_rows.extend(rows)
            fold_infos.append(info)
            print("  " + ", ".join(f"{name}={values['macro_f1']:.4f}" for name, values in info["metrics"].items()), flush=True)

    flat = flatten_predictions(all_outer_rows)
    write_jsonl(output / "outer_predictions.jsonl", flat)
    write_json(output / "outer_fold_results.json", fold_infos)
    aggregate = {}
    systems = sorted({row["system"] for row in flat})
    for system in systems:
        subset = [r for r in flat if r["system"] == system]
        truth = np.asarray([r["true_label"] for r in subset])
        probabilities = np.asarray([[r["probabilities"][label] for label in LABELS] for r in subset])
        aggregate[system] = metrics(truth, probabilities)
    bootstrap = {}
    paired_records = {}
    for row in all_outer_rows:
        paired_records.setdefault((row["record_id"], row["seed"], row["outer_fold"]), {
            "record_id": row["record_id"], "patient_id": row["patient_id"], "true_label": row["true_label"],
            "probabilities": {}})["probabilities"].update(
                {system: np.asarray([value for value in result["probabilities"].values()])
                 for system, result in row["systems"].items()})
    # Convert per-fold rows into one patient/bootstrap table per system pair.
    pair_rows = []
    for item in paired_records.values():
        pair_rows.append({"record_id": item["record_id"], "patient_id": item["patient_id"],
                          "true_label": item["true_label"], "probabilities": item["probabilities"]})
    baseline = "text_nli"
    for system in systems:
        if system != baseline:
            bootstrap[system] = {
                metric_name: paired_patient_bootstrap(pair_rows, baseline, system, metric_name,
                                                      int(config.get("bootstrap_replicates", 2000)), 1729)
                for metric_name in ("macro_f1", "log_loss", "multiclass_brier")
            }
    write_summary(output / "outer_summary.csv", aggregate, bootstrap)

    holdout_result = None
    if not args.skip_holdout:
        # This is the only holdout scoring path.  No tuning is performed here.
        X_development = {key: value[:n_development] for key, value in X.items()}
        X_holdout = {key: value[n_development:] for key, value in X.items()}
        outer_seed = int(args.seeds[0])
        inner_splitter = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True,
                                              random_state=outer_seed + 9000)
        inner_splits = list(inner_splitter.split(np.zeros(n_development), y, groups))
        inner_splits = [(a, b) for a, b in inner_splits]
        C_candidates = [float(x) for x in config["classifier_C_candidates"]]
        base_C, base_tuning = choose_C(X_development["base"], y, groups, inner_splits,
                                       C_candidates, outer_seed)
        system_fields = {"text_nli": None, "concept_overlap_fusion": "concept",
                         "mesh_hierarchy_fusion": "hierarchy", "gated_hierarchy_fusion": "gated_hierarchy",
                         "shuffled_hierarchy_control": "shuffled_hierarchy"}
        hold_probabilities = {}
        hold_tuning = {"base_C": base_tuning}
        for name, field in system_fields.items():
            base_model = fit_classifier(X_development["base"], y, base_C, outer_seed)
            if field is None:
                hold_probabilities[name] = aligned_probabilities(base_model, X_holdout["base"])
                continue
            onto_C, onto_tuning = choose_C(X_development[field], y, groups, inner_splits,
                                           C_candidates, outer_seed)
            lam, lam_tuning = choose_fusion_lambda(X_development["base"], X_development[field],
                                                   y, groups, inner_splits,
                                                   base_C, onto_C, config["fusion_lambda_candidates"], outer_seed)
            onto_model = fit_classifier(X_development[field], y, onto_C, outer_seed)
            base_train_log = aligned_log_probabilities(base_model, X_development["base"])
            base_hold_log = aligned_log_probabilities(base_model, X_holdout["base"])
            onto_train_log = aligned_log_probabilities(onto_model, X_development[field])
            onto_hold_log = aligned_log_probabilities(onto_model, X_holdout[field])
            onto_hold_scaled = global_score_normalization(onto_train_log, onto_hold_log,
                                                          max(float(base_train_log.std()), 1e-8))
            hold_probabilities[name] = softmax(base_hold_log + lam * onto_hold_scaled)
            hold_tuning[name] = {"onto_C": onto_tuning, "fusion_lambda": lam_tuning}
        hold_y = np.asarray([all_refs[r["record_id"]]["constraint_status"] for r in hold_inputs])
        holdout_result = {"policy": "single final score after protocol freeze", "n": len(hold_inputs),
                          "patients": len(set(r["patient_id"] for r in hold_inputs)),
                          "tuning": hold_tuning,
                          "metrics": {name: metrics(hold_y, prob) for name, prob in hold_probabilities.items()}}
        hold_rows = []
        for i, row in enumerate(hold_inputs):
            for name, prob in hold_probabilities.items():
                hold_rows.append({"record_id": row["record_id"], "patient_id": row["patient_id"],
                                  "true_label": str(hold_y[i]), "system": name,
                                  "predicted_label": LABELS[int(np.argmax(prob[i]))],
                                  "probabilities": {label: float(prob[i, j]) for j, label in enumerate(LABELS)}})
        write_jsonl(output / "holdout_predictions.jsonl", hold_rows)
        write_json(output / "holdout_results.json", holdout_result)

    final = {"status": "completed_nested_patient_group_evaluation", "holdout_scored": holdout_result is not None,
             "protocol": config, "outer_aggregate": aggregate, "paired_bootstrap": bootstrap,
             "holdout": holdout_result, "feature_metadata": feature_meta,
             "code_sha256": digest(Path(__file__))}
    write_json(output / "nested_results.json", final)
    print(f"Saved nested evaluation to {output}", flush=True)


if __name__ == "__main__":
    main()
