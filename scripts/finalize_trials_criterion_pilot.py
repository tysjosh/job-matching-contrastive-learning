#!/usr/bin/env python3
"""Independently verify saved pilot outputs and build its reviewable handoff.

Does not score holdout, change predictions, tune models, or alter source labels.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trials_domain.criterion_pilot import LABELS, digest, read_jsonl, write_json
from scripts.run_trials_criterion_pilot import metrics

NAMES = {
    "text": "Text + NLI",
    "text_concepts": "+ Concept overlap",
    "text_concepts_hierarchy": "+ MeSH hierarchy",
    "text_concepts_shuffled_hierarchy": "+ Shuffled hierarchy",
    "text_concepts_gated_hierarchy": "+ Gated hierarchy",
    "train_majority": "Training majority",
    "historical_gpt4_reference": "Historical GPT-4 reference",
}


def main() -> None:
    config = json.loads((ROOT / "config/trials_criterion_pilot.json").read_text())
    prep, out = ROOT / config["prepared_dir"], ROOT / config["results_dir"]
    results = json.loads((out / "development_results.json").read_text())
    audit = json.loads((prep / "data_quality_audit.json").read_text())
    split = json.loads((prep / "split_manifest.json").read_text())
    dev = read_jsonl(prep / "development/inputs.jsonl")
    ref = {r["record_id"]: r for r in read_jsonl(prep / "development/references.jsonl")}
    ids = [r["record_id"] for r in dev]
    truth = np.array([ref[rid]["constraint_status"] for rid in ids])
    saved = read_jsonl(out / "development_predictions.jsonl")
    systems = sorted({r["system"] for r in saved})
    pred = {}
    for system in systems:
        rows = {r["record_id"]: r for r in saved if r["system"] == system}
        if len(rows) != len(ids) or set(rows) != set(ids):
            raise ValueError("Prediction coverage mismatch")
        pred[system] = np.array([rows[rid]["predicted_status"] for rid in ids])
        recalculated = metrics(truth, pred[system])
        for metric in ("accuracy", "macro_f1", "false_rejection_rate", "false_clearance_rate"):
            if not np.isclose(recalculated[metric], results["metrics"][system][metric]):
                raise ValueError(f"Metric mismatch: {system} {metric}")
        if not all(np.isclose(sum(r["probabilities"].values()), 1) for r in rows.values()):
            raise ValueError("Invalid probability sums")
    feature_path = (ROOT / "embedding_cache/trials_criterion_pilot" /
                    f"features_{results['features']['feature_cache_signature'][:20]}.npz")
    features = np.load(feature_path, allow_pickle=False)
    train_ref = read_jsonl(prep / "train/references.jsonl")
    ytrain = [r["constraint_status"] for r in train_ref]
    classifier = joblib.load(out / "models/text.joblib")
    train_pred = classifier.predict(features["base"][:len(ytrain)])
    diagnostics = {"train_macro_f1": float(f1_score(ytrain, train_pred, labels=LABELS, average="macro", zero_division=0)),
                   "development_macro_f1": results["metrics"]["text"]["macro_f1"],
                   "base_feature_dimension": features["base"].shape[1],
                   "predictions_changed_vs_text": {s: int(np.count_nonzero(pred[s] != pred["text"])) for s in systems},
                   "hierarchy_vs_shuffled_mean_absolute_difference": float(np.abs(features["hierarchy"] - features["shuffled_hierarchy"]).mean()),
                   "gate_changed_rows_train_and_development": int(np.count_nonzero(features["hierarchy"] - features["gated_hierarchy"])),
                   "prediction_metrics_independently_recomputed": True, "holdout_scored": False}
    if diagnostics["hierarchy_vs_shuffled_mean_absolute_difference"] == 0:
        raise ValueError("Hierarchy control has no actual feature contrast")
    write_json(out / "verification.json", diagnostics)

    prior_root = ROOT / "results/publication_phase1_fixedval"
    prior_files = [ROOT / "config/lc_trials_baseline.json", ROOT / "scripts/evaluate_publication_fusion.py",
                   prior_root / "trials_fusion_learning_curve_ndcg_selected.json",
                   prior_root / "tables_ndcg_selected/trials_primary_table.md"]
    prior_files += sorted((prior_root / "checkpoints").glob("trials_baseline_*/best_checkpoint.pt"))
    prior_files += sorted((prior_root / "checkpoints").glob("trials_baseline_*/training_config.json"))
    prior_snapshot = {str(p.relative_to(ROOT)): {"sha256": digest(p), "bytes": p.stat().st_size}
                      for p in prior_files if p.is_file()}
    prior_manifest = out / "prior_phase1_baseline_manifest.json"
    if prior_manifest.exists():
        if json.loads(prior_manifest.read_text()) != prior_snapshot:
            raise ValueError("Earlier baseline changed since pilot snapshot; investigate without overwriting")
    else:
        write_json(prior_manifest, prior_snapshot)

    display_rows = []
    for order, (system, label) in enumerate(NAMES.items()):
        score = results["metrics"][system]
        row = {"order": order, "system": label, "accuracy": round(score["accuracy"], 4),
               "macro_f1": round(score["macro_f1"], 4), "unknown_recall": round(score["unknown_recall"], 4),
               "violation_recall": round(score["not_met_recall"], 4),
               "origin": "Source predictions; not rerun" if system.startswith("historical") else "Fresh local baseline",
               "development_patients": len(split["patients"]["development"]), "criteria": len(ids),
               "violation_criteria": score["not_met_n"]}
        display_rows.append(row)
    pd.DataFrame(display_rows).to_csv(out / "development_comparison.csv", index=False)
    sqlite_path = out / "development_comparison.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.row_factory = sqlite3.Row
        pd.DataFrame(display_rows).to_sql("comparison", connection, if_exists="replace", index=False)
        query_rows = [dict(row) for row in connection.execute(
            "SELECT `order`, system, accuracy, macro_f1, unknown_recall, violation_recall, origin, "
            "development_patients, criteria, violation_criteria FROM comparison ORDER BY macro_f1 DESC, `order` ASC"
        ).fetchall()]

    # Written after inspecting predictions; explicitly an exploratory interpretation.
    no_effect = all(value == 0 for value in diagnostics["predictions_changed_vs_text"].values())
    conclusion = ("Ontology features changed no development classifications in this pilot. "
                  if no_effect else "The saved development comparisons are exploratory. ")
    conclusion += (f"The text baseline reached training macro-F1 {diagnostics['train_macro_f1']:.4f} "
                   f"but development macro-F1 {diagnostics['development_macro_f1']:.4f}, showing a large generalization gap. "
                   "This weak baseline cannot establish that ontology reasoning is ineffective. "
                   "The immediate next step is a stronger criterion evaluator and clinical review, before larger training grids.")
    sections = [
        ("summary", "Technical summary", conclusion, "results"),
        ("scope", "What was evaluated", (
            f"The source contains {audit['source_rows']:,} patient–criterion annotations. One missing criterion was quarantined, "
            f"leaving {audit['rows']:,} usable rows from {audit['patients']} synthetic patient cases. "
            "The partition contains 32 training patients (574 rows), 10 development patients (222 rows), "
            "and 11 held-out patients (218 rows). No held-out predictions were computed. "
            "Macro-F1 averages met, not met, unknown and not applicable equally; accuracy weights each criterion equally. "
            "For exclusion criteria, excluded maps to not met, and not excluded maps to met. These are criterion decisions, not clinical enrollment decisions."), "audit"),
        ("quality", "Evidence conventions require clinical review", (
            f"The source training flag crosses {audit['patients_crossing_source_training_flag']} patients between its true and false values. "
            f"Our seeded patient partition eliminates that overlap. {audit['flag_counts']['no_annotated_evidence']} usable rows "
            "have no annotated supporting sentence, including 365 not-excluded labels. "
            "A missing annotation is not proof that the source label is incorrect. We retain the original labels and create a separate review flag. "
            "MeSH dictionary mapping covers 79.2% of criteria; mappings are automatic, not clinically validated. "
            "These findings are high-priority risks for uncertainty evaluation, not grounds for automatic relabeling."), "audit"),
        ("results_intro", "The local ontology variants made identical classifications", (
            "The following table reports the same 222 development criteria for every system. "
            "The historical GPT-4 row is contextual: it reuses published predictions that experts assessed during dataset construction. "
            "It is neither a fresh run nor a compute-matched comparison. Only four development criteria are violations; "
            "the fresh local models recovered none. The apparent accuracy must be read with this class imbalance."), "results"),
        ("mechanism", "The feature controls worked, but the classifier did not generalize", (
            f"The text classifier has {diagnostics['base_feature_dimension']:,} features and only 574 training examples. "
            f"Its training macro-F1 is {diagnostics['train_macro_f1']:.4f}, compared with {diagnostics['development_macro_f1']:.4f} on different patients. "
            f"The true and shuffled hierarchy features differ by {diagnostics['hierarchy_vs_shuffled_mean_absolute_difference']:.4f} in mean absolute value. "
            f"The gate changes {diagnostics['gate_changed_rows_train_and_development']} training/development feature rows. "
            "Thus identical classifications are not explained by an empty ontology feature or identical negative control. "
            "The matched prediction deltas have a zero-width bootstrap interval because the predictions are identical on this sample; "
            "that does not imply certainty about the ontology effect in other populations."), "results"),
        ("methods", "Model and evaluation specification", (
            "A frozen local MPNet encoder retrieves the three most similar patient sentences for each criterion. "
            "A frozen DeBERTa NLI encoder evaluates that same evidence against the criterion. "
            "All five trainable variants use these common features and a class-balanced logistic classifier "
            "with C=0.1 and scaling fitted only on training patients. Ontology additions are normalized concept overlap, "
            "MeSH best-match hierarchy similarity, a fixed descriptor-tree permutation, or a prespecified mapping/coverage gate. "
            "The gate requires concepts on both sides and text cosine at least 0.25; it is a heuristic, not a validated uncertainty mechanism. "
            "No expert evidence or historical GPT-4 output enters model features. No encoded sentence or NLI pair was truncated. "
            "The 1,000 paired bootstrap replicates resample whole development patients. The raw data, model revisions, "
            "configuration, features, predictions and original Phase 1 artifacts are hashed for reproducibility."), "results"),
        ("limits", "Limits on the current evidence", (
            "This is an exploratory local feasibility run, not a publication-ready clinical model. "
            "The encoders were not trained as a generative clinical reasoning system; there is no fresh TrialGPT-style run. "
            "The development cohort is small, and no new expert-reviewed challenge set or clinician workflow study has been completed. "
            "The train and held-out partitions share one trial identifier, so the split is patient-disjoint but not fully protocol-disjoint. "
            "Public historical cases may also be present in prior model training. TREC labels remain at patient–trial grain, "
            "and both TREC years share a 2021 trial snapshot. No trial-ranking, treatment-effect or clinical-efficiency claim follows from this criterion pilot."), None),
        ("next", "Next decision and review package", (
            "The 120-case blinded development review sheet is ready; all reviewer fields remain pending. "
            "Reviewers should judge evidence sufficiency, missing/conflicting facts and ontology mappings independently of source predictions. "
            "Use adjudication to define a separate strict-evidence reference without overwriting benchmark labels. "
            "Before scaling, test a stronger criterion evaluator or a smaller feature model selected through training-only patient-group cross-validation. "
            "A fresh generative-model comparison requires a configured local or approved API runtime. "
            "After this development step, lock the method before scoring held-out patients. "
            "Questions remaining: which clinician will adjudicate the review set, and which generative model will form the main benchmark?"), None),
    ]
    now = datetime.now(timezone.utc).isoformat()
    title = "Trials criterion pilot: data ready, model needs improvement"
    blocks = [{"id": "title", "type": "markdown", "body": "# " + title}]
    for sid, heading, body, source_id in sections:
        block = {"id": sid, "type": "markdown", "body": f"## {heading}\n\n{body}"}
        if source_id:
            block["sourceId"] = source_id
        blocks.append(block)
        if sid == "results_intro":
            blocks.append({"id": "comparison", "type": "table", "tableId": "comparison"})
            blocks.append({"id": "comparison_chart", "type": "chart", "chartId": "comparison_chart"})
    sources = [
        {"id": "audit", "label": "Pinned TrialGPT annotation audit",
         "path": "preprocess/trials_criterion_pilot/data_quality_audit.json",
         "query": {"language": "python", "description": "Schema, label, evidence and split checks",
                   "tables_used": ["TrialGPT-Criterion-Annotations@1cfcafde94a1560a33b4addc1638664fe28fc059"],
                   "filters": ["one missing criterion quarantined; no label imputation"]}},
        {"id": "results", "label": "Local development evaluation and independent verification",
         "path": "results/trials_criterion_pilot/development_results.json",
         "query": {"language": "python", "description": "Saved predictions independently recomputed in finalize_trials_criterion_pilot.py",
                   "tables_used": ["development_comparison.sqlite:comparison", "development_results.json", "verification.json", "development_predictions.jsonl"],
                   "sql": "SELECT `order`, system, accuracy, macro_f1, unknown_recall, violation_recall, origin, development_patients, criteria, violation_criteria FROM comparison ORDER BY macro_f1 DESC, `order` ASC",
                   "filters": ["development patients only; held-out set not scored"]}},
    ]
    artifact = {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title,
                "generatedAt": now, "description": "Exploratory methods audit and baseline comparison",
                "blocks": blocks, "sources": sources, "cards": [], "filters": [],
                "charts": [{"id": "comparison_chart", "title": "Development macro-F1 by system",
                            "subtitle": "222 criteria from 10 patients; higher is better; four-class macro-F1",
                            "type": "bar", "dataset": "comparison", "sourceId": "results",
                            "valueFormat": "number",
                            "encodings": {"x": {"field": "system", "type": "nominal", "label": "System"},
                                          "y": {"field": "macro_f1", "type": "quantitative", "label": "Macro-F1"}}}],
                "tables": [{"id": "comparison", "title": "Development criterion classification",
                            "subtitle": "222 criteria from 10 patients; four violation examples",
                            "dataset": "comparison", "sourceId": "results", "defaultSort": {"field": "macro_f1", "direction": "desc"},
                            "columns": [{"field": "system", "label": "System", "type": "text"},
                                        {"field": "accuracy", "label": "Accuracy", "format": "number"},
                                        {"field": "macro_f1", "label": "Macro-F1", "format": "number"},
                                        {"field": "unknown_recall", "label": "Unknown recall", "format": "number"},
                                        {"field": "violation_recall", "label": "Violation recall", "format": "number"}]}]},
                "snapshot": {"version": 1, "generatedAt": now, "status": "ready", "datasets": {"comparison": query_rows},
                             "accessIssues": []}, "sources": sources}
    write_json(out / "artifact.json", artifact)
    write_json(out / "report_notes.json", {
        "delivery": "portable HTML in local Codex runtime", "audience": "technical",
        "chart_omission": "Exact audit lookup is primary; the five controlled variants have identical classifications and F1. A bar chart would add no information.",
        "review_claims": "No completed human review is claimed. All blanks remain blank.",
        "required_structure": "technical summary; findings/table; scope; models; limitations; next decision/questions",
        "clinical_review_status": "pending", "fresh_generative_comparison_status": "runtime_not_configured",
        "skill_influence": "Data-quality workflow separated immutable source, quarantine, predictions, gold labels and pending reviewer judgments."})
    print(json.dumps({"comparison": display_rows, "verification": diagnostics,
                      "artifact": str(out / "artifact.json")}, indent=2))


if __name__ == "__main__":
    main()
