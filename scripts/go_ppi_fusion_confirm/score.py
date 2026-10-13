#!/usr/bin/env python3
"""Gate 2 and the score export (protocol_go_ppi_fusion.md §2, §5.2, §6.4). CPU only.

Modes
-----
``--gate-reproduce``  Gate 2 parts (``--parts``, default all):
    2a  the 15 published checkpoints with the published φ U on the published unmasked files,
        grid 0..1.0, under both rules, leaf by leaf against the two published fusion JSONs
        (tiered tolerances, near-tie ``selected_weight`` exception), plus the zero-shot check;
    2b  record alignment (masked vs published splits, cohort-A fractions, protein URIs);
    2c  typed-parser check (and the pinned is_a-only index re-derived);
    2d  derived inputs exist, are pinned and have the asserted counts;
    2e  φ path equality (X, P, N, S, S_P, UX vs direct ``GoMatcher`` calls, label-free);
    2f  AZ-X and GZ-P on the 15 published checkpoints against
        ``results/go_ppi_fusion_best/test_results.json`` (F3_P, PUB_ndcg_P).
    ``--dev-local`` (Mac dev smoke, §6.5) runs only 2a and/or 2f, without pins, writing under
    ``results/go_ppi_fusion_confirm_dev/`` or /tmp. It is NOT gate 2.
default  Export per-record scores once: cohort A (passing retrained runs + zero-shot), cohort B
    (per fold: passing seeds + zero-shot), the HuRI cohort (cohort-A models + zero-shot) and the
    descriptive A-pub arm (15 published checkpoints on the published files). Each group gets
    ``phis_<file>.npz`` (grades, anchors, partners, every φ variant, flags), one ``<model>.npz``
    of text scores per file, and ``selected_w.json`` (per model × arm: w, curve, top-two gap,
    GZ z-stats) plus the shuffle draw counts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402

GATE_PARTS = ("2a", "2b", "2c", "2d", "2e", "2f")
DEV_PARTS = ("2a", "2f")
GATE_GRID = list(fc.WEIGHT_GRID)


# --------------------------------------------------------------------------- shared loading
class Ctx:
    """Indexes, matchers, n_PMID and the read-only text sources."""

    def __init__(self, pins: Optional[Dict[str, Any]], need_isa: bool = True):
        self.pins = pins
        self.pruned = fc.load_index(fc.PRUNED_INDEX_REL)
        self.unpruned = fc.load_index(fc.UNPRUNED_INDEX_REL)
        isa = None
        if pins is not None and need_isa and pins["analysis"].get("T_computed"):
            isa = fc.load_index(pins["derived"]["isa_index"]["path"])
        self.matchers = fc.build_matchers(self.pruned, self.unpruned, isa)
        self.n_pmid: Dict[str, int] = {}
        if pins is not None:
            self.n_pmid = json.loads(fc.absp(f"{pins['derived']['phi_dir']}/n_pmid.json").read_text())["n_pmid"]
        self.config = fc.load_base_config()
        self._frozen = None
        self._proteins_u = None

    @property
    def frozen(self):
        if self._frozen is None:
            self._frozen = fc.load_frozen_cache(fc.ORIGINAL_CACHE_REL)
        return self._frozen

    @property
    def unpruned_uris(self) -> Dict[str, List[str]]:
        if self._proteins_u is None:
            self._proteins_u = {p["protein_id"]: list(p.get("go_uris") or [])
                                for p in fc.read_jsonl(fc.UNPRUNED_PROTEINS_REL)}
        return self._proteins_u


def published_prepared(ctx: Ctx, split: str, prep: Dict[str, Any]) -> Dict[str, Any]:
    """A published unmasked file (gate 2, A-pub): frozen cache + published chunks; pruned URIs
    from the aligned masked file, unpruned URIs from the published records."""
    path = f"{fc.PUB_SPLIT_DIR}/{split}.jsonl"
    p = fc.load_prepared(path, chunk_dir=prep[split]["dir"], frozen_cache=ctx.frozen)
    masked = fc.read_jsonl(f"{fc.MASKED_SPLIT_DIR}/{split}.jsonl")
    fc.align_records(fc.read_jsonl(path), masked, f"{split}: published vs masked")
    p["uris_a_u"], p["uris_b_u"] = p["uris_a"], p["uris_b"]
    p["uris_a"] = [list(r["resume"].get("skill_uris") or []) for r in masked]
    p["uris_b"] = [list(r["job"].get("skill_uris") or []) for r in masked]
    p["uris_a_shuf"] = None
    return p


def masked_prepared(ctx: Ctx, path: str, shuffle_id: str, aligned_published: Optional[str] = None) -> Dict[str, Any]:
    """A confirmatory masked file: texts only from the fresh remote-CPU encodes; pruned URIs
    from the records; unpruned URIs from the aligned published file (cohort A) or
    ``preprocess/go_ppi/proteins.jsonl`` (cohort B, HuRI); the pinned shuffle map."""
    pins = ctx.pins
    p = fc.load_prepared(path, chunk_dir=pins["masked_encodes"]["dir"])
    if aligned_published is not None:
        pub = fc.read_jsonl(aligned_published)
        fc.align_records(pub, fc.read_jsonl(path), f"{path} vs {aligned_published}")
        p["uris_a_u"] = [list(r["resume"].get("skill_uris") or []) for r in pub]
        p["uris_b_u"] = [list(r["job"].get("skill_uris") or []) for r in pub]
    else:
        u = ctx.unpruned_uris
        p["uris_a_u"] = [u[a] for a in p["anchors"]]
        p["uris_b_u"] = [u[b] for b in p["partners"]]
    sm = json.loads(fc.absp(f"{pins['derived']['phi_dir']}/shuffle_maps/{shuffle_id}.json").read_text())
    if sm["file"] != path or sm["file_sha256"] != p["sha256"]:
        raise fc.ProtocolError(f"shuffle map {shuffle_id} was built for {sm['file']}, not {path}")
    p["uris_a_shuf"] = fc.apply_shuffle(p["queries"], p["uris_a"], sm["mapping"])
    p["shuffle"] = {"id": shuffle_id, "draws": [s["draws"] for s in sm["strata"]], "n_units": sm["n_units"]}
    return p


def text_for(ctx: Ctx, model: Dict[str, Any], prep: Dict[str, Any]) -> np.ndarray:
    if model["kind"] == "zeroshot":
        return fc.zero_shot_scores(prep)
    if fc.file_sha256(model["checkpoint"]) != model["sha256"]:
        raise fc.ProtocolError(f"{model['checkpoint']} sha256 differs from the ledger/pins")
    return fc.text_scores(prep, model["checkpoint"], ctx.config)


def arrays(prep: Dict[str, Any], text: np.ndarray, phis: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {"text": text, "phi": phis, "grades": prep["grades"], "queries": prep["queries"],
            "ai": prep.setdefault("_ai", fc.AnchorIndex(prep["queries"], prep["grades"]))}


# --------------------------------------------------------------------------- gate 2
def gate_2a_2f(ctx: Ctx, parsed: Dict[str, Any], tol, near_ties, parts) -> Dict[str, Any]:
    E = fc.epf()
    out: Dict[str, Any] = {}
    val = published_prepared(ctx, "validation", parsed["prepared"])
    test = published_prepared(ctx, "test", parsed["prepared"])
    for split, spec in parsed["prepared"].items():
        if not fc.check_prepared_dir(spec)["ok"]:
            raise fc.ProtocolError(f"published prepared cache {split} digest mismatch")
    ckpts = parsed["checkpoints"]
    grid_order = [(F, S) for F in fc.FRACTIONS for S in fc.PUBLISHED_SEEDS]
    if [(r["fraction"], r["seed"]) for r in ckpts] != grid_order:
        raise fc.ProtocolError("published checkpoint rows are not in the published grid order")
    texts = []
    for r in ckpts:
        if fc.file_sha256(r["path"]) != r["sha256"]:
            raise fc.ProtocolError(f"checkpoint {r['path']} sha256 mismatch")
        texts.append((fc.text_scores(val, r["path"], ctx.config), fc.text_scores(test, r["path"], ctx.config)))
    if "2a" in parts:
        U = ctx.matchers["U"]
        vphi, tphi = fc.phi_array(U, val["uris_a_u"], val["uris_b_u"]), fc.phi_array(U, test["uris_a_u"], test["uris_b_u"])
        ostats = E.z_stats(vphi)
        vpz, tpz = E.z_apply(vphi, ostats), E.z_apply(tphi, ostats)
        ties = {tuple(c[:3]) for c in near_ties}
        rules = {}
        for rule, path in fc.PUBLISHED_FUSION_REL.items():
            pub = {(r["fraction"], r["seed"]): r for r in json.loads(fc.absp(path).read_text())["runs"]}
            comps, events = [], []
            for r, (vt, tt) in zip(ckpts, texts):
                F, S = r["fraction"], r["seed"]
                ts = E.z_stats(vt)
                vtz, ttz = E.z_apply(vt, ts), E.z_apply(tt, ts)
                curve = {str(w): E.validation_selection_value(fc.fuse(vtz, vpz, w), val, rule) for w in GATE_GRID}
                w = max(GATE_GRID, key=lambda x: curve[str(x)])
                pw = pub[(F, S)]["selected_weight"]
                exception = (w != pw and (rule, F, S) in ties and pw in fc.top_two_weights(curve))
                w_eval = pw if exception else w
                tf = fc.fuse(ttz, tpz, w_eval)
                tr, fr = E.evaluate_score(ttz, test), E.evaluate_score(tf, test)
                delta = {"global_auc": {n: fr["contrasts"]["global_auc"][n] - tr["contrasts"]["global_auc"][n]
                                        for n in ("hard", "easy", "middle_vs_easy", "pooled")},
                         "ranking_macro": {n: fr["ranking"]["macro"][n] - tr["ranking"]["macro"][n]
                                           for n in fr["ranking"]["macro"]}}
                run = fc.json_safe({"fraction": F, "seed": S, "checkpoint": r["path"], "selected_weight": w,
                                    "validation_selection_curve": curve,
                                    "validation_stats": {"text": {"mean": ts[0], "sd": ts[1]},
                                                         "ontology": {"mean": ostats[0], "sd": ostats[1]}},
                                    "test": {"text": tr, "fusion": fr, "delta": delta}})
                c = fc.compare_gate2_run(pub[(F, S)], run, tol, weight_exception=exception)
                c.update({"fraction": F, "seed": S, "selected_weight": w, "published_selected_weight": pw,
                          "validation_top2_gap": fc.top_two_gap(curve.values()), "near_tie_exception": exception})
                if exception:
                    events.append({"fraction": F, "seed": S, "computed": w, "published": pw,
                                   "test_leaves_compared_at": pw})
                comps.append(c)
                print(f"2a {rule} f{F} s{S}: w={w} (published {pw}) {'PASS' if c['passed'] else 'FAIL'} "
                      f"max={c['max_abs_by_class']}", flush=True)
            mx: Dict[str, float] = {}
            for c in comps:
                for k, v in c["max_abs_by_class"].items():
                    mx[k] = max(mx.get(k, 0.0), v)
            rules[rule] = {"passed": all(c["passed"] for c in comps), "runs": comps, "near_tie_events": events,
                           "max_abs_by_class": mx}
        zs = fc.zero_shot_scores(test)
        con = E.contrast_metrics(zs, test["grades"], test["queries"])["macro_query_auc"]
        nd = E.ranking_metrics(zs, test["grades"], test["queries"])["macro"]["ndcg@10"]
        zgot = {"hard": con["hard"], "pooled": con["pooled"], "ndcg10": nd}
        zok = all(round(zgot[k], 3) == v for k, v in fc.GATE2_ZERO_SHOT.items())
        out["2a"] = {"passed": all(r["passed"] for r in rules.values()) and zok, "rules": rules,
                     "zero_shot": {"computed": zgot, "expected_3dp": fc.GATE2_ZERO_SHOT, "passed": zok},
                     "tolerance_abs": tol, "leaf_classes": fc.GATE2_LEAF_CLASSES, "near_tie_cells": near_ties}
    if "2f" in parts:
        best = json.loads(fc.absp(fc.BEST_TEST_RESULTS_REL).read_text())
        phis_v = {"X": fc.phi_array(ctx.matchers["X"], val["uris_a"], val["uris_b"]),
                  "P": fc.phi_array(ctx.matchers["P"], val["uris_a"], val["uris_b"])}
        phis_t = {"X": fc.phi_array(ctx.matchers["X"], test["uris_a"], test["uris_b"]),
                  "P": fc.phi_array(ctx.matchers["P"], test["uris_a"], test["uris_b"])}
        res = {}
        for arm, label in fc.GATE2F_VARIANTS.items():
            ref = best["variants"][label]
            rows, events, ok = [], [], True
            for i, (r, (vt, tt)) in enumerate(zip(ckpts, texts)):
                v, t = arrays(val, vt, phis_v), arrays(test, tt, phis_t)
                st = fc.fit_arm(arm, v)
                pw = float(ref["theta"][i])
                exception = (st["w"] != pw and st["top2_gap"] < fc.GATE2_NEAR_TIE_GAP
                             and pw in fc.top_two_weights(st["curve"]))
                w_ok = st["w"] == pw or exception
                st_eval = dict(st, w=pw) if exception else st
                fm, tm = fc.anchor_metrics(fc.apply_arm(arm, st_eval, t), t["ai"]), fc.anchor_metrics(tt, t["ai"])
                deltas, diffs = {}, {}
                for m, key in fc.GATE2F_METRICS.items():
                    d = fm[m] - tm[m]
                    deltas[m] = float(np.nanmean(d))
                    diffs[m] = abs(deltas[m] - float(ref["delta"][key]["per_checkpoint"][i]))
                row_ok = w_ok and all(x <= tol["rank"] for x in diffs.values())
                ok &= row_ok
                rows.append({"fraction": r["fraction"], "seed": r["seed"], "w": st["w"], "published_w": pw,
                             "validation_top2_gap": st["top2_gap"], "near_tie_exception": exception,
                             "delta": deltas, "abs_diff": diffs, "passed": row_ok})
                if exception:
                    events.append(rows[-1])
                print(f"2f {arm} f{r['fraction']} s{r['seed']}: w={st['w']} (best {pw}) "
                      f"max|Δ diff|={max(diffs.values()):.2e} {'PASS' if row_ok else 'FAIL'}", flush=True)
            res[arm] = {"reference": f"{fc.BEST_TEST_RESULTS_REL}:variants.{label}", "passed": ok, "runs": rows,
                        "near_tie_events": events, "max_abs_diff": max(max(r["abs_diff"].values()) for r in rows)}
        out["2f"] = {"passed": all(v["passed"] for v in res.values()), "arms": res,
                     "tolerance": {"selected_w": "exact except near-tie (<1e-4)", "delta": tol["rank"]}}
    out["_embedding_sources"] = {"validation": val["embedding_sources"], "test": test["embedding_sources"]}
    return out


def gate_2b(pins) -> Dict[str, Any]:
    res = {}
    for split in ("train", "validation", "test"):
        fc.align_records(fc.read_jsonl(f"{fc.PUB_SPLIT_DIR}/{split}.jsonl"),
                         fc.read_jsonl(f"{fc.MASKED_SPLIT_DIR}/{split}.jsonl"), split)
        res[split] = "aligned"
    data = pins["data_dir"]
    for F, n in fc.LC_ROW_COUNTS.items():
        pub = [fc.record_key(r) for r in fc.read_jsonl(f"{fc.LC_DIR}/frac_{F}/train.jsonl")]
        mine = [fc.record_key(r) for r in fc.read_jsonl(f"{data}/cohortA/frac_{F}/train.jsonl")]
        if pub != mine or len(mine) != n:
            raise fc.ProtocolError(f"cohort-A frac_{F} keys differ from the published fraction")
        res[f"frac_{F}"] = n
    pub = [fc.record_key(r) for r in fc.read_jsonl(f"{fc.LC_DIR}/validation_positive.jsonl")]
    mine = [fc.record_key(r) for r in fc.read_jsonl(f"{data}/cohortA/validation_positive.jsonl")]
    if pub != mine or len(mine) != fc.VALIDATION_POSITIVE_COUNT:
        raise fc.ProtocolError("cohort-A validation_positive keys differ")
    res["validation_positive"] = len(mine)
    pruned = fc.load_index(fc.PRUNED_INDEX_REL)
    res["protein_uris"] = fc.check_protein_uris(fc.read_jsonl(fc.MASKED_PROTEINS_REL), pruned)
    if not res["protein_uris"]["ok"]:
        raise fc.ProtocolError("pruned record URIs differ from the pruned index direct")
    return {"passed": True, **res}


def gate_2c(pins) -> Dict[str, Any]:
    pruned = fc.load_index(fc.PRUNED_INDEX_REL)
    tc = fc.typed_check(pruned, fc.OBO_REL)
    out = {"passed": True, "typed_check": tc, "T_computed": bool(tc["passed"]),
           "note": "a typed-check failure only removes T (no stop)"}
    if tc["passed"] != bool(pins["analysis"].get("T_computed")):
        raise fc.ProtocolError("typed check result differs from the pinned T_computed")
    if tc["passed"]:
        isa = fc.build_typed_index(pruned, fc.OBO_REL, ("is_a",))
        pinned = fc.load_index(pins["derived"]["isa_index"]["path"])
        same = isa.ancestors == pinned.ancestors and isa.ic == pinned.ic and isa.closure == pinned.closure
        out["pinned_isa_index_rederived_equal"] = same
        out["passed"] = same
    return out


def gate_2d(pins) -> Dict[str, Any]:
    bad = fc.verify_pinned_hashes(pins, ("derived", "configs", "masked", "inputs", "chunks"))
    d = pins["derived"]
    checks = {"hashes": not bad}
    cv = json.loads(fc.absp(d["cv"]["manifest"]).read_text())
    checks["group_counts_14_31"] = cv["group_counts"]["groups"] == 14 and cv["group_counts"]["proteins"] == 31
    proteins = {p["protein_id"]: p for p in fc.read_jsonl(fc.MASKED_PROTEINS_REL)}
    pairs = fc.read_jsonl(fc.PAIRS_REL)
    groups = fc.view_groups(proteins)
    folds = fc.assign_protein_folds(proteins, groups, cv["n_folds"], cv["fold_seed"])
    checks["folds_rederived_equal"] = folds == cv["folds"]
    ident = True
    for k in range(cv["n_folds"]):
        _, m = fc.build_fold(proteins, pairs, folds, k, cv["fold_seed"], groups=groups)
        written = {n: fc.file_sha256(f"{d['cv']['fold_dir']}/fold_{k}/{n}") for n in m["rows_sha256"]}
        ident &= written == m["rows_sha256"]
    checks["two_build_identity"] = ident
    checks["bclean_sha256"] = fc.list_sha256(cv["bclean"]["anchors"]) == d["bclean"]["sha256"]
    checks["bclean_min_hard_defined"] = d["bclean"]["hard_defined"] >= pins["analysis"]["bclean_min_hard_defined"]
    n, t = pins["analysis"]["huri_expected_anchors"]
    hu = d["huri"]
    checks["huri_anchors"] = abs(hu["counts"]["anchors"] - n) <= t
    g1 = {}
    for key, path in (("huri_observed_only", pins["huri"]["test"]), ("unrestricted", pins["huri"]["test_grade1_unrestricted"])):
        g1[key] = sum(1 for r in fc.read_jsonl(path) if r["job"]["grade"] == 1)
    checks["huri_grade1_counts"] = g1 == hu["grade1_counts"]
    checks["masked_encodes_digest"] = fc.manifest_digest(pins["masked_encodes"]["dir"]) == pins["masked_encodes"]["digest"]
    rlcp = fc.load_script_module("scripts/run_learning_curve_point.py", "run_learning_curve_point_gate")
    audit_ok = True
    for p in pins["configs"]["derived"]:
        audit_ok &= rlcp.report_factor("go_ppi", fc.absp(fc.BASE_CONFIG_REL), fc.absp(p))["unexpected_fields"] \
            == fc.EXPECTED_UNEXPECTED_FIELDS
    checks["config_audit_sets"] = audit_ok
    return {"passed": all(checks.values()), "checks": checks, "hash_mismatches": bad[:20]}


def gate_2e(ctx: Ctx) -> Dict[str, Any]:
    from go_ppi_domain.go_ontology import GoIndex, GoMatcher
    out = {}
    direct_idx = {"pruned": GoIndex.load(fc.absp(fc.PRUNED_INDEX_REL)), "unpruned": GoIndex.load(fc.absp(fc.UNPRUNED_INDEX_REL))}
    modes = {"X": ("pruned", "exact"), "P": ("pruned", "simgic"), "N": ("pruned", "ancestor"),
             "S": ("pruned", "exact"), "S_P": ("pruned", "simgic"), "UX": ("unpruned", "exact")}
    for split, sid in (("validation", "A_validation"), ("test", "A_test")):
        f = masked_prepared(ctx, f"{fc.MASKED_SPLIT_DIR}/{split}.jsonl", sid, f"{fc.PUB_SPLIT_DIR}/{split}.jsonl")
        mine = fc.compute_phis(f, ctx.matchers, ctx.n_pmid, ("X", "P", "N", "S", "S_P", "UX"))
        res = {}
        for v, (which, mode) in modes.items():
            m = GoMatcher(direct_idx[which], similarity_mode=mode)
            A = f["uris_a_shuf"] if v in ("S", "S_P") else (f["uris_a_u"] if which == "unpruned" else f["uris_a"])
            B = f["uris_b_u"] if which == "unpruned" else f["uris_b"]
            ref = []
            for a, b in zip(A, B):
                try:
                    ref.append(float(m.ontology_set_similarity(a, b)) if a and b else 0.0)
                except Exception:
                    ref.append(0.0)
            res[v] = bool(np.array_equal(mine[v], np.asarray(ref)))
        out[split] = res
    return {"passed": all(all(r.values()) for r in out.values()), "equal": out, "files": "cohort A masked"}


def gate_reproduce(args, pins: Optional[Dict[str, Any]]) -> int:
    out = fc.refuse_existing(args.out)
    parts = tuple(p.strip() for p in args.parts.split(",")) if args.parts else (DEV_PARTS if args.dev_local else GATE_PARTS)
    if args.dev_local and set(parts) - set(DEV_PARTS):
        raise fc.ProtocolError(f"--dev-local runs only {DEV_PARTS} (§6.5)")
    parsed = fc.parse_protocol(fc.absp(fc.PROTOCOL_REL if pins is None else pins["protocol"]["path"]).read_text())
    if pins is None:
        tol = dict(fc.GATE2_TOLERANCE_ABS)
        pub = {r: json.loads(fc.absp(p).read_text()) for r, p in fc.PUBLISHED_FUSION_REL.items()}
        near = fc.near_tie_cells(pub)
    else:
        tol, near = pins["analysis"]["gate2_tolerance_abs"], pins["analysis"]["gate2_near_tie_cells"]
    t0 = time.time()
    caches = {p: fc.file_sha256(p) for p in (fc.ORIGINAL_CACHE_REL, fc.MASKED_CACHE_REL) if fc.absp(p).exists()}
    ctx = Ctx(pins)
    res: Dict[str, Any] = {}
    if {"2a", "2f"} & set(parts):
        res.update(gate_2a_2f(ctx, parsed, tol, near, parts))
    if "2b" in parts:
        res["2b"] = gate_2b(pins)
    if "2c" in parts:
        res["2c"] = gate_2c(pins)
    if "2d" in parts:
        res["2d"] = gate_2d(pins)
    if "2e" in parts:
        res["2e"] = gate_2e(ctx)
    for p, h in caches.items():
        if fc.file_sha256(p) != h:
            raise fc.ProtocolError(f"{p} changed during gate 2")
    sources = res.pop("_embedding_sources", None)
    passed = all(res[p]["passed"] for p in parts)
    result = {"schema": "go_ppi_fusion_confirm/gate_reproduction/v1", "created_utc": fc.utc_now(),
              "confirmatory": pins is not None and not args.dev_local,
              "note": None if pins is not None else "NON-CONFIRMATORY Mac dev smoke (§6.5); not gate 2",
              "pins": None if pins is None else {"path": fc.rel(fc.absp(args.pins)), "sha256": fc.file_sha256(args.pins)},
              "parts": list(parts), "passed": passed,
              "parts_passed": {p: res[p]["passed"] for p in parts}, "results": res,
              "embedding_sources": sources, "device": "cpu", "seconds": round(time.time() - t0, 1),
              "run_env": fc.run_env_record()}
    fc.write_json(out, result)
    print(f"gate 2 {'PASSED' if passed else 'FAILED'} {result['parts_passed']} -> {fc.show(out)} ({result['seconds']}s)")
    return 0 if passed else 2


# --------------------------------------------------------------------------- score export
def passed_runs(ledger: fc.Ledger) -> Dict[str, Dict[str, Any]]:
    return {e["run"]: e for e in ledger.entries() if e.get("status") == "passed"}


def file_flags(prep: Dict[str, Any], huri_edges: set, mention: Optional[np.ndarray] = None,
               string_grade2: Optional[set] = None) -> Dict[str, np.ndarray]:
    a, b = prep["anchors"], prep["partners"]
    out = {"empty_anchor": np.asarray([len(u) == 0 for u in prep["uris_a"]], dtype=bool),
           "huri_edge": np.asarray([tuple(sorted((x, y))) in huri_edges for x, y in zip(a, b)], dtype=bool)}
    if mention is not None:
        out["mention"] = mention
    if string_grade2 is not None:
        out["string_grade2"] = np.asarray([(x, y) in string_grade2 for x, y in zip(a, b)], dtype=bool)
    return out


def export_group(gdir: Path, ctx: Ctx, files: Dict[str, Dict[str, Any]], models: List[Dict[str, Any]],
                 arms, flags: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    gdir.mkdir(parents=True, exist_ok=False)
    phis = {}
    meta = {"files": {}, "models": {}}
    for name, prep in files.items():
        phis[name] = fc.compute_phis(prep, ctx.matchers, ctx.n_pmid)
        arr = {"grades": prep["grades"], "anchors": np.asarray(prep["queries"]).astype(str),
               "partners": np.asarray(prep["partners"]).astype(str)}
        arr.update({f"phi_{k}": v for k, v in phis[name].items()})
        arr.update({f"flag_{k}": v for k, v in flags.get(name, {}).items()})
        np.savez(gdir / f"phis_{name}.npz", **arr)
        meta["files"][name] = {"path": prep["path"], "sha256": prep["sha256"], "records": int(len(prep["grades"])),
                               "embedding_sources": prep["embedding_sources"], "shuffle": prep.get("shuffle"),
                               "phi_variants": sorted(phis[name])}
    for m in models:
        texts = {name: text_for(ctx, m, prep) for name, prep in files.items()}
        np.savez(gdir / f"{m['name']}.npz", **{f"{k}_text": v for k, v in texts.items()})
        rec = {k: m.get(k) for k in ("kind", "checkpoint", "sha256", "fraction", "seed", "k")}
        if "validation" in files and arms:
            v = arrays(files["validation"], texts["validation"], phis["validation"])
            rec["arms"] = {a: fc.fit_arm(a, v) for a in arms if fc.arm_available(a, phis["validation"])}
            print(f"  {gdir.name}/{m['name']}: w(AZ-X)={rec['arms']['AZ-X']['w']} "
                  f"w(GZ-P)={rec['arms'].get('GZ-P', {}).get('w')}", flush=True)
        meta["models"][m["name"]] = rec
    fc.write_json(gdir / "selected_w.json", meta)
    return meta


def score_all(args, pins) -> int:
    out = fc.refuse_existing(args.out)
    ledger = fc.Ledger(args.ledger)
    passed = passed_runs(ledger)
    excluded = sorted({e["run"] for e in ledger.entries() if e.get("status") == "excluded"})
    t0 = time.time()
    ctx = Ctx(pins)
    caches = {p: pins["inputs"][p] for p in (fc.ORIGINAL_CACHE_REL, fc.MASKED_CACHE_REL)}
    hu = json.loads(fc.absp(pins["derived"]["huri"]["manifest"]).read_text())
    edges = {tuple(e) for e in hu["mapped_edges"]}
    sg2 = {tuple(x) for x in hu["string_grade2_positives"]}
    mention_doc = json.loads(fc.absp(f"{pins['derived']['phi_dir']}/mention_flags_A_test.json").read_text())
    manifest = {"schema": "go_ppi_fusion_confirm/scores/v1", "created_utc": fc.utc_now(), "confirmatory": True,
                "pins": {"path": fc.rel(fc.absp(args.pins)), "sha256": fc.file_sha256(args.pins)},
                "ledger": {"path": fc.rel(ledger.path), "sha256": fc.file_sha256(ledger.path)},
                "excluded_runs": excluded, "groups": {}, "device": "cpu", "run_env": fc.run_env_record()}
    out.mkdir(parents=True)
    arms = [a for a in fc.COHORT_ARMS if a != "AZ-T" or pins["analysis"].get("T_computed")]

    # cohort A (+ HuRI with the same models)
    a_models = []
    for r in pins["cohortA"]["new_runs"]:
        e = passed.get(r["run"])
        if e is not None:
            a_models.append({"name": f"f{r['fraction']}_s{r['seed']}", "kind": "new", "fraction": r["fraction"],
                             "seed": r["seed"], "checkpoint": f"{r['output_dir']}/best_checkpoint.pt",
                             "sha256": e["best_checkpoint_sha256"]})
    a_models.append({"name": "zeroshot", "kind": "zeroshot"})
    ca = pins["cohortA"]
    files = {s: masked_prepared(ctx, ca[s], f"A_{s}", ca["aligned_published"][s]) for s in ("validation", "test")}
    mention = np.zeros(len(files["test"]["grades"]), dtype=bool)
    if mention_doc["file_sha256"] != files["test"]["sha256"]:
        raise fc.ProtocolError("mention flags were computed on a different cohort-A test file")
    mention[mention_doc["flagged_record_indices"]] = True
    flags = {s: file_flags(files[s], edges, mention if s == "test" else None) for s in files}
    manifest["groups"]["A/all"] = len(export_group(out / "A" / "all", ctx, files, a_models, arms, flags)["models"])
    hfiles = {"test": masked_prepared(ctx, pins["huri"]["test"], "huri_test"),
              "test_g1u": masked_prepared(ctx, pins["huri"]["test_grade1_unrestricted"], "huri_test_grade1_unrestricted")}
    hflags = {s: file_flags(hfiles[s], edges, None, sg2) for s in hfiles}
    manifest["groups"]["huri/all"] = len(export_group(out / "huri" / "all", ctx, hfiles, a_models, [], hflags)["models"])
    print(f"cohort A + HuRI exported ({time.time() - t0:.0f}s)", flush=True)

    # cohort B
    cb = pins["cohortB"]
    for k in range(cb["n_folds"]):
        d = f"{cb['fold_dir']}/fold_{k}"
        files = {s: masked_prepared(ctx, f"{d}/{s}.jsonl", f"B_fold{k}_{s}") for s in ("validation", "test")}
        models = []
        for S in cb["seeds"]:
            e = passed.get(f"B:k{k}_s{S}")
            if e is not None:
                models.append({"name": f"s{S}", "kind": "new", "seed": S, "k": k,
                               "checkpoint": f"{e['output_dir']}/best_checkpoint.pt", "sha256": e["best_checkpoint_sha256"]})
        models.append({"name": "zeroshot", "kind": "zeroshot"})
        flags = {s: file_flags(files[s], edges) for s in files}
        manifest["groups"][f"B/fold_{k}"] = len(export_group(out / "B" / f"fold_{k}", ctx, files, models, arms, flags)["models"])
    print(f"cohort B exported ({time.time() - t0:.0f}s)", flush=True)

    # A-pub (descriptive): 15 published checkpoints on the published unmasked files
    prep = {s: {"dir": pins["published_prepared_cache"][s]["dir"]} for s in ("validation", "test")}
    files = {s: published_prepared(ctx, s, prep) for s in ("validation", "test")}
    models = [{"name": f"f{r['fraction']}_s{r['seed']}", "kind": "published", "fraction": r["fraction"],
               "seed": r["seed"], "checkpoint": r["path"], "sha256": r["sha256"]} for r in pins["published_checkpoints"]]
    flags = {s: file_flags(files[s], edges) for s in files}
    manifest["groups"]["apub/all"] = len(export_group(out / "apub" / "all", ctx, files, models, list(fc.APUB_ARMS), flags)["models"])
    for p, h in caches.items():
        if fc.file_sha256(p) != h:
            raise fc.ProtocolError(f"{p} sha256 changed during scoring")
    manifest["seconds"] = round(time.time() - t0, 1)
    fc.write_json(out / "score_manifest.json", manifest)
    print(f"wrote scores to {fc.rel(out)}: {manifest['groups']} ({manifest['seconds']}s)")
    return 0


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pins", type=Path, default=None)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gate-reproduce", action="store_true")
    ap.add_argument("--parts", default=None, help=f"gate parts, comma-separated from {GATE_PARTS}")
    ap.add_argument("--dev-local", action="store_true", help="NON-CONFIRMATORY Mac dev smoke (gate 2a/2f only)")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    fc.require_cpu(args.device, "gate 2 and scoring")
    if args.dev_local:
        if not args.gate_reproduce or args.pins is not None:
            raise fc.ProtocolError("--dev-local is only for --gate-reproduce without pins (§6.5)")
        fc.require_dev(args.out)
        return gate_reproduce(args, None)
    if args.pins is None:
        raise fc.ProtocolError("--pins is required (only the --dev-local gate code check runs without pins)")
    pins = fc.load_verified_pins(args.pins)
    if not pins.get("confirmatory"):
        raise fc.ProtocolError("non-confirmatory pins")
    bad = fc.verify_pinned_hashes(pins)
    if bad:
        raise fc.ProtocolError(f"pinned hash mismatch, refusing to run: {bad[:10]}")
    if args.gate_reproduce:
        return gate_reproduce(args, pins)
    if args.ledger is None:
        raise fc.ProtocolError("--ledger is required for scoring")
    return score_all(args, pins)


if __name__ == "__main__":
    sys.exit(main())
