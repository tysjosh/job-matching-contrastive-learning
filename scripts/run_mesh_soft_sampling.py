#!/usr/bin/env python3
"""V2 MeSH development study. 2021 only; no 2022 evaluation switch.

Default: fixed historical 2021 split. --folds 3 creates patient-disjoint 2021
development folds (not nested CV or a new independent confirmation set).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ("uniform", "text", "mesh", "hybrid")
SEEDS = (13, 21, 42, 87, 123)


def config_for(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    cfg = json.loads((ROOT / "config/lc_trials_ontneg_stoch.json").read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(trials_soft_guidance=arm, trials_soft_mix=.5,
               trials_soft_temperature=.2, trials_common_validation=True,
               trials_mesh_tiered_negatives=arm in ("mesh", "hybrid"),
               trials_mesh_facet="disease", trials_mesh_tier_scope="not_relevant",
               mesh_similarity_mode="exact", trials_tier_sampling="uniform",
               loss_type="infonce", ontology_weight=0., use_ot_distance=False,
               fixed_validation_negatives=True, validation_negative_seed=1729,
               validation_negative_epoch=14)
    return cfg


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prepare_folds(out, folds):
    """Rebuild uncapped 2021 pairs from converted qrels; never use 2022 rows."""
    if folds == 1:
        return [(ROOT / "preprocess/trec_ct_lc/frac_100/train.jsonl",
                 ROOT / "preprocess/trec_ct_lc/validation_positive.jsonl",
                 ROOT / "preprocess/trec_ct_splits")]
    source = ROOT / "preprocess/trec_ct"
    topics = [r for r in read_rows(source / "topics.jsonl") if str(r["year"]) == "2021"]
    ids = {r["topic_id"] for r in topics}
    from trials_domain.data_splitter import _build_record, _ontology_slot
    topic_map = {r["topic_id"]: r for r in topics}
    trials = {r["nct_id"]: r for r in read_rows(source / "trials.jsonl")}
    qrels = [r for r in read_rows(source / "qrels.jsonl") if r["topic_id"] in ids]
    pool_map = {t: {"topic_id":t, "ineligible":[], "not_relevant":[]} for t in ids}
    records = []
    seen = set()
    for row in qrels:
        topic, nct, grade = row["topic_id"], row["nct_id"], row["grade"]
        if (topic, nct) in seen:
            raise ValueError(f"duplicate judgment: {topic} {nct}")
        seen.add((topic, nct))
        if grade not in (0, 1, 2):
            raise ValueError(f"invalid grade: {grade}")
        anchor = _ontology_slot(topic_map[topic], "topic_id", topic)
        records.append(_build_record(topic, topic_map[topic], anchor, nct, trials[nct], grade))
        if grade != 2:
            pool_map[topic]["ineligible" if grade==1 else "not_relevant"].append(nct)
    pools = [dict(p, ineligible=sorted(p["ineligible"]), not_relevant=sorted(p["not_relevant"]))
             for _, p in sorted(pool_map.items())]
    shuffled = sorted(ids)
    random.Random(42).shuffle(shuffled)
    splits = []
    for fold in range(folds):
        validation_ids = set(shuffled[fold::folds])
        split = out / "data" / f"fold_{fold}"
        training = [r for r in records if r["resume"]["topic_id"] not in validation_ids
                    and r["job"]["grade"] == 2]
        validation = [r for r in records if r["resume"]["topic_id"] in validation_ids]
        write_rows(split / "train.jsonl", training)
        write_rows(split / "validation.jsonl", validation)
        write_rows(split / "validation_positive.jsonl", [r for r in validation if r["job"]["grade"] == 2])
        write_rows(split / "negative_pools.jsonl", [dict(p, split="validation" if p["topic_id"] in validation_ids else "train") for p in pools])
        (split / "fold_manifest.json").write_text(json.dumps({"validation_topics": sorted(validation_ids), "train_topics": sorted(ids-validation_ids), "source_year": 2021, "split_seed":42}, indent=2))
        splits.append((split / "train.jsonl", split / "validation_positive.jsonl", split))
    return splits


def summarize(results):
    controls = {(r["fold"],r["seed"]):r for r in results if r["arm"]=="uniform"}
    summary = {"study":"MeSH V2 2021 development only", "runs":results,
               "primary_metric":"patient-macro eligible_map", "arms":{}}
    for arm in sorted({r["arm"] for r in results}):
        runs = [r for r in results if r["arm"]==arm]
        pairs, topic_deltas = [], {}
        for row in runs:
            control = controls[(row["fold"],row["seed"])]
            a, b = row["per_patient"], control["per_patient"]
            if set(a["per_topic"]) != set(b["per_topic"]):
                raise ValueError("mismatched evaluation patients")
            delta = a["macro"]["eligible_map"]-b["macro"]["eligible_map"]
            pairs.append({"fold":row["fold"],"seed":row["seed"],"eligible_map_delta":delta,
                          "pooled_auc_delta":row["pooled_auc"]-control["pooled_auc"]})
            for topic, metrics in a["per_topic"].items():
                if "eligible_map" in metrics:
                    topic_deltas.setdefault(topic,[]).append(
                        metrics["eligible_map"]-b["per_topic"][topic]["eligible_map"])
        # Average seeds within each patient before bootstrapping patients.
        values = [statistics.mean(v) for _,v in sorted(topic_deltas.items())]
        rng = random.Random(1729)
        boots = sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(2000))
        summary["arms"][arm] = {"n_runs":len(runs),"paired":pairs,
            "mean_patient_map":statistics.mean(r["per_patient"]["macro"]["eligible_map"] for r in runs),
            "patient_mean_delta":statistics.mean(values),"patient_bootstrap_ci95":[boots[49],boots[1949]],
            "bootstrap_note":"Seeds averaged within patient; descriptive development CI, not selection-adjusted.",
            "patients":len(values)}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one epoch, small training subset, isolated output")
    ap.add_argument("--folds", type=int, choices=(1, 3, 5), default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = ap.parse_args()
    if args.smoke and args.evaluate:
        ap.error("smoke runs cannot produce study evaluation summaries")
    if args.evaluate and "uniform" not in args.arms:
        ap.error("evaluation requires the paired uniform control")
    name = "mesh_soft_sampling_smoke" if args.smoke else "mesh_soft_sampling"
    out = ROOT / "results" / name / f"folds_{args.folds}"
    out.mkdir(parents=True, exist_ok=True)
    splits = prepare_folds(out, args.folds)
    seeds = [42] if args.smoke else args.seeds
    print(f"V2 MeSH: {len(splits)*len(seeds)*len(args.arms)} runs; output={out}")
    results = []
    for fold, (train, val, split) in enumerate(splits):
        if args.smoke:
            smoke_train = out / f"smoke_train_{fold}.jsonl"
            write_rows(smoke_train, read_rows(train)[:64])
            train = smoke_train
        for seed in seeds:
            for arm in args.arms:
                run = out / f"fold_{fold}" / f"{arm}_s{seed}"
                config = config_for(arm)
                config["trials_split_dir"] = str(split)
                if args.smoke:
                    config["num_epochs"] = 1
                fingerprint = hashlib.sha256(json.dumps({"config":config,"train":hashlib.sha256(train.read_bytes()).hexdigest(),"val":hashlib.sha256(val.read_bytes()).hexdigest(),"seed":seed},sort_keys=True).encode()).hexdigest()
                run.mkdir(parents=True, exist_ok=True)
                cfgpath = run / "requested_config.json"
                completed = run / "lc_manifest.json"
                stamp = run / "protocol.sha256"
                if completed.exists() and (not stamp.exists() or stamp.read_text()!=fingerprint):
                    raise RuntimeError(f"existing run has a different protocol: {run}")
                cfgpath.write_text(json.dumps(config, indent=2)+"\n")
                cmd = [sys.executable, str(ROOT/"scripts/run_learning_curve_point.py"),
                       "--domain", "trials", "--config", str(cfgpath), "--train-file", str(train),
                       "--validation-file", str(val), "--output-dir", str(run), "--seed", str(seed)]
                if args.execute and not completed.exists():
                    stamp.write_text(fingerprint)
                    print(f"[train] {run.name} fold={fold}", flush=True)
                    with (run / "train.log").open("w") as log:
                        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                elif not args.evaluate:
                    print(("[complete] " if completed.exists() else "[plan] ")+" ".join(cmd))
                if args.evaluate:
                    if not completed.exists():
                        raise RuntimeError(f"incomplete run: {run}")
                    from eval_learning_curve import _load_eval_module, evaluate_one
                    if "ev" not in locals():
                        ev = _load_eval_module()
                    result = evaluate_one(ev, run/"best_checkpoint.pt", run/"training_config.json",
                                          split/"validation.jsonl", run/"eval_validation2021")
                    if not result or not result.get("per_patient"):
                        raise RuntimeError(f"missing patient evaluation: {run}")
                    results.append({"fold":fold,"seed":seed,"arm":arm,"pooled_auc":result["metrics"]["auc_roc"],"per_patient":result["per_patient"]})
    if results:
        summary = summarize(results)
        (out/"validation2021_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
