#!/usr/bin/env python3
"""Score export and gate 2 (protocol_trials_fusion.md §2, §5.2, §6.4). CPU only.

Modes
-----
``--gate-reproduce``  Gate 2: re-score the 15 published checkpoints (W = 0..1.0, nDCG
    selection, read-only embedding sources) and compare leaf by leaf with the published
    fusion JSON under the tiered tolerance; zero-shot macro pooled AUC must be 0.795 (3 dp).
    Also reports (non-gating) the exploratory +0.013 estimate on the restricted grid.
    With ``--dev-local`` it runs without pins as a NON-CONFIRMATORY code check; output
    must be under ``results/trials_fusion_confirm/dev_local/`` or /tmp.
default  Export per-pair scores for every model (cohort 2022: the passing retrained runs of
    §9 A1 + zero-shot; cohort 2021: per outer fold, passing seeds + zero-shot):
    text, φ_hier, φ_exact, φ_shuf, grades, patient and nct ids for validation and test,
    plus ``selected_w.json`` (per model × φ variant × normaliser) and shuffle draw counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trials_domain import fusion_confirm as fc  # noqa: E402

VARIANTS = ("hierarchy", "exact", "shuffled")


def load_config():
    from contrastive_learning.data_structures import TrainingConfig
    return TrainingConfig.from_json(str(fc.absp(fc.BASE_CONFIG_REL)))


def phis_for(prep: Dict[str, Any], matchers: Dict[str, Any], group_of: Dict[str, int],
             seed: int, max_draws: int) -> Dict[str, Any]:
    """φ hierarchy / exact / shuffled (group-aware derangement, rng created once per file)."""
    out = {"hierarchy": fc.phi_scores(matchers["hierarchy"], prep["topic_uris"], prep["trial_uris"])}
    if "exact" in matchers:
        out["exact"] = fc.phi_scores(matchers["exact"], prep["topic_uris"], prep["trial_uris"])
        mapping, draws = fc.group_derangement(list(set(prep["queries"])), group_of, seed, max_draws)
        shuf = fc.shuffled_topic_uris(prep["queries"], prep["topic_uris"], mapping)
        out["shuffled"] = fc.phi_scores(matchers["hierarchy"], shuf, prep["trial_uris"])
        out["shuffle_draws"] = draws
        out["shuffle_mapping"] = mapping
    return out


def selection_record(val_text, val_phi, prep_val, grid) -> Dict[str, Any]:
    rec = {}
    g, q = prep_val["grades"], prep_val["queries"]
    tz = fc.zapply(val_text, fc.zstats(val_text))
    pz = fc.zapply(val_phi, fc.zstats(val_phi))
    w, curve = fc.select_weight(tz, pz, g, q, grid)
    rec["global_z"] = {"w": w, "curve": curve, "text_stats": list(fc.zstats(val_text)),
                       "phi_stats": list(fc.zstats(val_phi))}
    w2, curve2 = fc.select_weight(fc.within_patient_z(val_text, q), fc.within_patient_z(val_phi, q), g, q, grid)
    rec["within_patient_z"] = {"w": w2, "curve": curve2}
    return rec


# --------------------------------------------------------------------------- gate 2
def gate_reproduce(args, pins: Optional[Dict[str, Any]]) -> int:
    import torch  # noqa: F401
    E = fc.epf()
    out = fc.refuse_existing(args.out)
    if pins is None:
        parsed = fc.parse_protocol(fc.absp(fc.PROTOCOL_REL).read_text(encoding="utf-8"))
        ckpts = parsed["checkpoints"]
        test_dir, val_dir = parsed["test_dir"], parsed["validation_dir"]
        expected_digest = parsed["manifest_digest"]
        tol = fc.GATE2_TOLERANCE_ABS
    else:
        ckpts = pins["cohort2022"]["existing_checkpoints"]
        test_dir = pins["published_prepared_cache"]["test_dir"]
        val_dir = pins["published_prepared_cache"]["validation_dir"]
        expected_digest = pins["published_prepared_cache"]["manifest_digest"]
        tol = pins["analysis"]["gate2_tolerance_abs"]
    published = json.loads(fc.absp(fc.PUBLISHED_FUSION_REL).read_text(encoding="utf-8"))
    pub_runs = {(r["fraction"], r["seed"]): r for r in published["runs"]}
    cache_sha = fc.file_sha256(fc.ORIGINAL_CACHE_REL)

    config = load_config()
    matcher = fc.build_matchers(config)["hierarchy"]
    frozen = fc.load_frozen_cache(fc.ORIGINAL_CACHE_REL)
    val = fc.load_split(fc.VAL2022_REL, frozen, published_chunk_dir=val_dir)
    test = fc.load_split(fc.TEST2022_REL, frozen, published_chunk_dir=test_dir)
    if fc.manifest_digest(test_dir) != expected_digest:
        raise fc.ProtocolError("published test chunk digest mismatch")
    val_phi = fc.phi_scores(matcher, val["topic_uris"], val["trial_uris"])
    test_phi = fc.phi_scores(matcher, test["topic_uris"], test["trial_uris"])
    ont_stats = E.z_stats(val_phi)
    vpz, tpz = E.z_apply(val_phi, ont_stats), E.z_apply(test_phi, ont_stats)
    grid = list(fc.GATE_WEIGHT_GRID)
    patients = fc.sort_topics(set(test["queries"]))

    comparisons, runs, explo = [], [], {m: [] for m in ("pooled", "ap")}
    for row in ckpts:
        F, S = row["fraction"], row["seed"]
        if fc.file_sha256(row["path"]) != row["sha256"]:
            raise fc.ProtocolError(f"checkpoint {row['path']} sha256 mismatch")
        vt = E.text_scores(val, fc.absp(row["path"]), config)
        tt = E.text_scores(test, fc.absp(row["path"]), config)
        ts = E.z_stats(vt)
        vtz, ttz = E.z_apply(vt, ts), E.z_apply(tt, ts)
        curve = {str(w): E.validation_selection_value(fc.fuse(vtz, vpz, w), val, "ndcg@10") for w in grid}
        w = max(grid, key=lambda x: curve[str(x)])
        tf = fc.fuse(ttz, tpz, w)
        text_res, fused_res = E.evaluate_score(ttz, test), E.evaluate_score(tf, test)
        delta = {"global_auc": {n: fused_res["contrasts"]["global_auc"][n] - text_res["contrasts"]["global_auc"][n]
                                for n in ("hard", "easy", "middle_vs_easy", "pooled")},
                 "ranking_macro": {n: fused_res["ranking"]["macro"][n] - text_res["ranking"]["macro"][n]
                                   for n in fused_res["ranking"]["macro"]}}
        run = E.json_safe({"fraction": F, "seed": S, "checkpoint": str(Path(row["path"]).as_posix()),
                           "selected_weight": w, "validation_selection_curve": curve,
                           "validation_stats": {"text": {"mean": ts[0], "sd": ts[1]},
                                                "ontology": {"mean": ont_stats[0], "sd": ont_stats[1]}},
                           "test": {"text": text_res, "fusion": fused_res, "delta": delta}})
        cmp_ = fc.compare_gate2_run(pub_runs[(F, S)], run, tol)
        cmp_.update({"fraction": F, "seed": S, "selected_weight": w,
                     "published_selected_weight": pub_runs[(F, S)]["selected_weight"]})
        comparisons.append(cmp_)
        runs.append({"fraction": F, "seed": S, "selected_weight": w})
        # exploratory estimator on the restricted grid (non-gating code check)
        w_r, _ = fc.select_weight(vtz, vpz, val["grades"], val["queries"], fc.WEIGHT_GRID)
        mt = fc.per_patient_metrics(ttz, test["grades"], test["queries"], patients)
        mf = fc.per_patient_metrics(fc.fuse(ttz, tpz, w_r), test["grades"], test["queries"], patients)
        for m in explo:
            explo[m].append(mf[m] - mt[m])
        print(f"f{F} s{S}: w={w} (published {pub_runs[(F, S)]['selected_weight']}) "
              f"{'PASS' if cmp_['passed'] else 'FAIL'} max={cmp_['max_abs_by_class']}", flush=True)

    zs = fc.zero_shot_scores(test)
    zs_macro = E.contrast_metrics(zs, test["grades"], test["queries"])["macro_query_auc"]["pooled"]
    rng = np.random.default_rng([31415, 2022])
    C, P = fc.draw_indices(rng, len(ckpts), len(patients), 2000)
    explo_out = {}
    for m, rows in explo.items():
        s = fc.summarize(np.asarray(rows), C, P)
        explo_out[m] = {k: v for k, v in s.items() if k != "_T"}
    max_by_class: Dict[str, float] = {}
    for c in comparisons:
        for k, v in c["max_abs_by_class"].items():
            max_by_class[k] = max(max_by_class.get(k, 0.0), v)
    passed = all(c["passed"] for c in comparisons) and round(zs_macro, 3) == 0.795
    if fc.file_sha256(fc.ORIGINAL_CACHE_REL) != cache_sha:
        raise fc.ProtocolError("frozen cache changed during gate 2")
    result = {
        "schema": "trials_fusion_confirm/gate_reproduction/v1", "created_utc": fc.utc_now(),
        "confirmatory": pins is not None,
        "note": None if pins is not None else "NON-CONFIRMATORY local code check (Mac CPU); not gate 2",
        "pins": None if pins is None else {"path": fc.rel(fc.absp(args.pins)), "sha256": fc.file_sha256(args.pins)},
        "passed": passed, "tolerance_abs": tol, "selected_weight_rule": "exact",
        "leaf_classes": fc.GATE2_LEAF_CLASSES, "max_abs_by_class": max_by_class,
        "zero_shot_macro_pooled_auc": zs_macro, "zero_shot_gate": round(zs_macro, 3) == 0.795,
        "runs": comparisons,
        "exploratory_reproduction_restricted_grid": {
            "note": "Δ = fused − text, hierarchy φ, w on {0..0.5} by validation nDCG@10, two-level bootstrap "
                    "B=2000 rng [31415,2022]; findings §B1 reported pooled +0.013 [+0.006,+0.020], AP +0.017",
            **explo_out},
        "embedding_sources": {"validation": val["embedding_sources"], "test": test["embedding_sources"]},
        "device": "cpu", "run_env": fc.run_env_record(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc.json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(f"gate 2 {'PASSED' if passed else 'FAILED'}; max abs {max_by_class}; zero-shot {zs_macro:.4f}; "
          f"exploratory pooled {explo_out['pooled']['point']:+.4f} {explo_out['pooled']['ci95']} -> {out}")
    return 0 if passed else 2


# --------------------------------------------------------------------------- scoring
def subset_prepared(prep: Dict[str, Any], keep: set) -> Dict[str, Any]:
    """Records of ``keep`` patients only (dev-local smoke: never touch unexposed 2021 outcomes)."""
    idx = [i for i, q in enumerate(prep["queries"]) if q in keep]
    out = dict(prep)
    for k in ("left_index", "right_index", "grades", "queries", "nct_ids", "condition_signal"):
        out[k] = np.asarray(prep[k])[idx]
    out["topic_uris"] = [prep["topic_uris"][i] for i in idx]
    out["trial_uris"] = [prep["trial_uris"][i] for i in idx]
    out["records_scored"] = len(idx)
    out["subset_note"] = "dev-local: restricted to exposed_topics_closed patients"
    return out


def dev_local_restrict(val: Dict[str, Any], test: Dict[str, Any], closed, group_of, seed: int, max_draws: int):
    """Dev-local 2021 smoke: restrict BOTH the inner-validation and the test file to
    ``exposed_topics_closed`` patients, so no confirmatory 2021 quantity (test or
    validation side) is ever computed locally (§9 A0). Aborts before any scoring if a
    restricted file is empty or its group-aware derangement is infeasible; on the real
    folds this holds for every outer fold, so 2021 dev smokes need a synthetic fold."""
    closed = set(closed)
    out = []
    for split, prep in (("validation", val), ("test", test)):
        sub = subset_prepared(prep, closed)
        topics = sorted(set(sub["queries"]))
        if not topics:
            raise fc.ProtocolError(f"dev-local: {split} file has no exposed_topics_closed patient")
        try:
            fc.group_derangement(topics, group_of, seed, max_draws)
        except fc.ProtocolError as exc:
            raise fc.ProtocolError(f"dev-local: restricted {split} file {topics} cannot be deranged ({exc}); "
                                   "use a synthetic fold for 2021 smokes") from None
        out.append(sub)
    return out[0], out[1]


def passed_runs(ledger: fc.Ledger) -> Dict[str, Dict[str, Any]]:
    out = {}
    for e in ledger.entries():
        if e.get("status") == "passed":
            out[e["run"]] = e
    return out


def write_model(path: Path, val, test, val_text, test_text, phis_val, phis_test) -> None:
    arrays = {}
    for split, prep, text, phis in (("validation", val, val_text, phis_val), ("test", test, test_text, phis_test)):
        arrays[f"{split}_text"] = np.asarray(text, dtype=np.float64)
        for v in VARIANTS:
            arrays[f"{split}_phi_{v}"] = np.asarray(phis[v], dtype=np.float64)
        arrays[f"{split}_grades"] = np.asarray(prep["grades"], dtype=np.int64)
        arrays[f"{split}_patients"] = np.asarray(prep["queries"]).astype(str)
        arrays[f"{split}_nct_ids"] = np.asarray(prep["nct_ids"]).astype(str)
        arrays[f"{split}_condition_signal"] = np.asarray(prep["condition_signal"], dtype=bool)
    np.savez(path, **arrays)


def score_group(group_dir: Path, val, test, models: List[Dict[str, Any]], matchers, group_of, config,
                pins) -> Dict[str, Any]:
    """Score one (cohort, fold-or-all) group: every model on its validation and test file."""
    E = fc.epf()
    a = pins["analysis"]
    phis_val = phis_for(val, matchers, group_of, a["shuffle_seed"], a["shuffle_max_draws"])
    phis_test = phis_for(test, matchers, group_of, a["shuffle_seed"], a["shuffle_max_draws"])
    group_dir.mkdir(parents=True, exist_ok=False)
    selected = {"validation": {"path": val["path"], "sha256": val["sha256"], "embedding_sources": val["embedding_sources"]},
                "test": {"path": test["path"], "sha256": test["sha256"], "embedding_sources": test["embedding_sources"]},
                "shuffle_draws": {"validation": phis_val["shuffle_draws"], "test": phis_test["shuffle_draws"]},
                "shuffle_mapping": {"validation": phis_val["shuffle_mapping"], "test": phis_test["shuffle_mapping"]},
                "models": {}}
    for m in models:
        if m["kind"] == "zeroshot":
            vt, tt = fc.zero_shot_scores(val), fc.zero_shot_scores(test)
        else:
            if fc.file_sha256(m["checkpoint"]) != m["sha256"]:
                raise fc.ProtocolError(f"{m['checkpoint']} sha256 differs from the ledger/pins")
            vt = E.text_scores(val, fc.absp(m["checkpoint"]), config).astype(np.float64)
            tt = E.text_scores(test, fc.absp(m["checkpoint"]), config).astype(np.float64)
        write_model(group_dir / f"{m['name']}.npz", val, test, vt, tt, phis_val, phis_test)
        selected["models"][m["name"]] = {
            "kind": m["kind"], "checkpoint": m.get("checkpoint"), "sha256": m.get("sha256"),
            "fraction": m.get("fraction"), "seed": m.get("seed"),
            "variants": {v: selection_record(vt, phis_val[v], val, a["weight_grid"]) for v in VARIANTS}}
        print(f"  scored {group_dir.name}/{m['name']}: w(hier)={selected['models'][m['name']]['variants']['hierarchy']['global_z']['w']}", flush=True)
    (group_dir / "selected_w.json").write_text(json.dumps(fc.json_safe(selected), indent=2) + "\n", encoding="utf-8")
    return selected


def score_all(args, pins) -> int:
    from sentence_transformers import SentenceTransformer

    out = fc.refuse_existing(args.out)
    if args.dev_local:
        fc.require_dev_local(out)
    ledger = fc.Ledger(args.ledger)
    passed = passed_runs(ledger)
    excluded = sorted({e["run"] for e in ledger.entries() if e.get("status") == "excluded"})
    original_sha = pins["inputs"][fc.ORIGINAL_CACHE_REL]
    if fc.file_sha256(fc.ORIGINAL_CACHE_REL) != original_sha:
        raise fc.ProtocolError("original frozen cache sha256 changed")
    config = load_config()
    matchers = fc.build_matchers(config)
    frozen = fc.load_frozen_cache(fc.ORIGINAL_CACHE_REL)
    group_of = fc.group_lookup(pins["cohort2021"]["components"])
    revision, snap_path = fc.resolve_mpnet_snapshot(pins["environment"]["mpnet_snapshot"])
    encoder_factory = lambda: SentenceTransformer(snap_path, device="cpu")  # noqa: E731
    fresh_root = f"{pins['results_dir']}/prepared_embeddings/trials"
    manifest = {"schema": "trials_fusion_confirm/scores/v1", "created_utc": fc.utc_now(),
                "confirmatory": pins["confirmatory"],
                "pins": {"path": fc.rel(fc.absp(args.pins)), "sha256": fc.file_sha256(args.pins)},
                "ledger": {"path": fc.rel(ledger.path), "sha256": fc.file_sha256(ledger.path)},
                "excluded_runs": excluded, "groups": {}, "device": "cpu", "run_env": fc.run_env_record()}
    out.mkdir(parents=True)
    cohorts = args.cohorts.split(",")

    if "2022" in cohorts:
        pc = pins["published_prepared_cache"]
        val = fc.load_split(pins["cohort2022"]["validation"], frozen, published_chunk_dir=pc["validation_dir"])
        test = fc.load_split(pins["cohort2022"]["test"], frozen, published_chunk_dir=pc["test_dir"])
        # §9 A1: the confirmatory cohort-2022 population is the 19 retrained runs only;
        # the 15 published checkpoints are scored by gate 2 (--gate-reproduce) and nowhere else.
        models = []
        for r in pins["cohort2022"]["new_runs"]:
            e = passed.get(f"2022:f{r['fraction']}_s{r['seed']}")
            if e is not None:
                models.append({"name": f"f{r['fraction']}_s{r['seed']}", "kind": "new",
                               "checkpoint": f"{r['output_dir']}/best_checkpoint.pt",
                               "sha256": e["best_checkpoint_sha256"], "fraction": r["fraction"], "seed": r["seed"]})
        models.append({"name": "zeroshot", "kind": "zeroshot"})
        manifest["groups"]["2022/all"] = len(score_group(out / "2022" / "all", val, test, models, matchers,
                                                         group_of, config, pins)["models"])
    if "2021" in cohorts:
        cv = pins["cohort2021"]
        folds = [int(k) for k in args.folds.split(",")] if args.folds else range(cv["n_folds"])
        for k in folds:
            d = f"{cv['fold_dir']}/fold_{k}"
            val = fc.load_split(f"{d}/validation.jsonl", frozen, fresh_encode_root=fresh_root, encoder_factory=encoder_factory)
            test = fc.load_split(f"{d}/test.jsonl", frozen, fresh_encode_root=fresh_root, encoder_factory=encoder_factory)
            if args.dev_local:
                val, test = dev_local_restrict(val, test, cv["exposed_topics_closed"], group_of,
                                               pins["analysis"]["shuffle_seed"], pins["analysis"]["shuffle_max_draws"])
                manifest.setdefault("dev_local_subset", {})[f"fold_{k}"] = {
                    "validation": fc.sort_topics(set(val["queries"])),
                    "test": fc.sort_topics(set(test["queries"]))}
            models = []
            for s in cv["seeds"]:
                e = passed.get(f"2021:k{k}_s{s}")
                if e is not None:
                    models.append({"name": f"s{s}", "kind": "new", "seed": s,
                                   "checkpoint": f"{e['output_dir']}/best_checkpoint.pt",
                                   "sha256": e["best_checkpoint_sha256"]})
            models.append({"name": "zeroshot", "kind": "zeroshot"})
            manifest["groups"][f"2021/fold_{k}"] = len(score_group(out / "2021" / f"fold_{k}", val, test, models,
                                                                   matchers, group_of, config, pins)["models"])
    if fc.file_sha256(fc.ORIGINAL_CACHE_REL) != original_sha:
        raise fc.ProtocolError("original frozen cache sha256 changed during scoring")
    (out / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote scores to {out}: {manifest['groups']}")
    return 0


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pins", type=Path, default=None)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cpu")
    ap.add_argument("--gate-reproduce", action="store_true")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dev-local", action="store_true", help="non-confirmatory (Mac) mode")
    ap.add_argument("--cohorts", default="2022,2021", help="--dev-local only")
    ap.add_argument("--folds", default=None, help="--dev-local only, e.g. 0")
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    fc.require_cpu_scoring(args.device)
    if not args.dev_local and (args.cohorts != "2022,2021" or args.folds):
        raise fc.ProtocolError("--cohorts/--folds are --dev-local only")

    pins = None
    if args.pins is not None:
        pins = fc.load_verified_pins(args.pins)
        if pins.get("confirmatory") is not (not args.dev_local):
            raise fc.ProtocolError("dev pins need --dev-local and vice versa")
        bad = fc.verify_pinned_hashes(pins)
        if bad:
            raise fc.ProtocolError(f"pinned hash mismatch, refusing to score: {bad[:10]}")
    elif not (args.dev_local and args.gate_reproduce):
        raise fc.ProtocolError("--pins is required (only the --dev-local gate code check runs without pins)")
    if args.dev_local:
        fc.require_dev_local(args.out)

    if args.gate_reproduce:
        return gate_reproduce(args, pins if not args.dev_local else None)
    if args.ledger is None:
        raise fc.ProtocolError("--ledger is required for scoring")
    return score_all(args, pins)


if __name__ == "__main__":
    sys.exit(main())
