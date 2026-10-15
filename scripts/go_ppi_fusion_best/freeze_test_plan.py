#!/usr/bin/env python3
"""Freeze the test-scoring plan (validation only, before any test scoring).

Runs in search mode, so no test file can be opened. It:

1. pins the variants to be scored on test: the 3 frozen finalists, each finalist's
   counterpart with the other phi version (pruned <-> unpruned, same family and
   settings; all already in ``search_log.jsonl``), and the published linear form
   under both published weight rules, unpruned and pruned;
2. declares the headline winner before test: the frozen leak-free slot F2 (the
   highest CV composite among leak-pruned variants), checked against the log;
3. runs the review's fold sensitivity (finding 1): the same cross-fitting with FIT
   records that share no protein (anchor or partner) with the held-out fold. It
   first checks that the re-implementation reproduces the logged anchor-grouped CV.
   Finalists are NOT changed by it; it is reported only;
4. writes ``results/go_ppi_fusion_best/test_plan.json``.

Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_best/freeze_test_plan.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_fusion import guard  # noqa: E402

guard.enter_search_mode()  # before anything else is loaded

import numpy as np  # noqa: E402

from go_ppi_fusion import data as D  # noqa: E402
from go_ppi_fusion import final as FN  # noqa: E402
from go_ppi_fusion import metrics as M  # noqa: E402
from go_ppi_fusion import phi as PHI  # noqa: E402
from go_ppi_fusion import search as S  # noqa: E402
from go_ppi_fusion.fusion import Inputs  # noqa: E402

OUT_DIR = ROOT / "results" / "go_ppi_fusion_best"
LOG = OUT_DIR / "search_log.jsonl"
FINALISTS = OUT_DIR / "finalists.json"
PLAN = OUT_DIR / "test_plan.json"

#: label -> search-log id. Counterparts differ from their finalist only in the phi version.
VARIANTS = {
    "PUB_ndcg_U": "f17bbc1bc2eb",    # published form, nDCG@10 rule (current best), unpruned
    "PUB_ndcg_P": "6fe6ca5035e9",    # same, pruned phi
    "PUB_pooled_U": "481aed91f2aa",  # published form, global pooled-AUC rule, unpruned
    "PUB_pooled_P": "9aa88b50f639",  # same, pruned phi
    "F1_U": "4aa815d4ed1c",          # finalist F1 (HGB stacker, rich set), unpruned
    "F1_P": "73de490782c5",          # F1 with every GO input pruned
    "F2_P": "57515cc164be",          # finalist F2 (HGB stacker, BP simGIC), pruned
    "F2_U": "de30798e61e4",          # F2 with every GO input unpruned
    "F3_U": "f4598215fb81",          # finalist F3 (anchor-z, exact BP), unpruned
    "F3_P": "239fcb00f573",          # F3 with pruned phi
}
WINNER = "F2_P"
ANNOTATION_BINS = [[0, 0], [1, 2], [3, 5], [6, 10], [11, 25], [26, 10_000]]


def spec_id(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def swap_version(spec, frm, to):
    def sw(x):
        if isinstance(x, list):
            return [sw(v) for v in x]
        return x[: -len(frm)] + to if isinstance(x, str) and x.endswith("_" + frm) else x
    return {k: sw(v) if k in ("phi", "flags", "counts") else v for k, v in spec.items()}


def main() -> int:
    t0 = time.time()
    if PLAN.exists():
        print(f"{PLAN} exists; the plan is frozen once", file=sys.stderr)
        return 1
    fin = json.loads(FINALISTS.read_text())
    rows = {r["id"]: r for r in map(json.loads, LOG.read_text().splitlines())}
    specs = {}
    for label, vid in VARIANTS.items():
        spec = rows[vid]["spec"]
        assert spec_id(spec) == vid, label
        specs[label] = spec
    by_slot = {f["slot"]: f for f in fin["finalists"]}
    assert specs["F1_U"] == by_slot["F1"]["spec"]
    assert specs["F2_P"] == by_slot["F2"]["spec"]
    assert specs["F3_U"] == by_slot["F3"]["spec"]
    assert specs["PUB_ndcg_U"] == fin["reference_published"]["spec"]
    # counterparts are exact version swaps of their finalist
    assert specs["F1_P"] == swap_version(specs["F1_U"], "unpruned", "pruned")
    assert specs["F2_U"] == swap_version(specs["F2_P"], "pruned", "unpruned")
    assert specs["F3_P"] == swap_version(specs["F3_U"], "unpruned", "pruned")
    assert specs["PUB_ndcg_P"] == swap_version(specs["PUB_ndcg_U"], "unpruned", "pruned")
    assert specs["PUB_pooled_P"] == swap_version(specs["PUB_pooled_U"], "unpruned", "pruned")
    assert {**specs["PUB_ndcg_U"], "select_on": "global_pooled_auc"} == specs["PUB_pooled_U"]
    # winner = highest CV composite among leak-pruned variants (= frozen slot F2)
    best_pruned = max((r for r in rows.values() if r["leak_pruned"]),
                      key=lambda r: r["cv_delta"]["composite"])
    assert best_pruned["id"] == VARIANTS[WINNER] == by_slot["F2"]["id"]

    config = D.load_config()
    frozen = D.load_frozen_cache(config)
    val = D.load_split("validation", config, frozen, with_pruned=True)
    grid = D.checkpoint_grid()
    texts = [D.text_scores(val, r["checkpoint"], config) for r in grid]
    phis = PHI.load_phis(PHI.SCRATCH / "phi_validation.npz")
    inp = Inputs("validation", val.queries, phis, M.QueryIndex(val.queries, val.grades), val.grades)
    folds = S.anchor_folds(inp.qi)
    rec_fold = folds[inp.qi.inverse]
    anchor_masks = [rec_fold != s for s in range(S.N_FOLDS)]
    pd_masks = FN.protein_disjoint_masks(val, inp.qi, folds)
    fit_share = [float(pd[a].sum() / a.sum()) for pd, a in zip(pd_masks, anchor_masks)]
    print(f"loaded validation ({time.time() - t0:.0f}s); protein-disjoint FIT share per fold "
          f"{[round(x, 3) for x in fit_share]}", flush=True)

    # the re-implementation must reproduce the logged anchor-grouped CV
    repro = {}
    for label in ("F3_U", "F2_P"):
        r = FN.cv_with_fit_masks(specs[label], texts, inp, folds, anchor_masks)
        logged = rows[VARIANTS[label]]["cv_delta"]
        diff = max(abs(r["cv_delta"][k] - logged[k]) for k in S.REPORT)
        repro[label] = diff
        print(f"repro {label}: max |cv - logged| = {diff:.2e}", flush=True)
        assert diff < 1e-9, (label, diff)

    sens = {}
    for label, spec in specs.items():
        t1 = time.time()
        r = FN.cv_with_fit_masks(spec, texts, inp, folds, pd_masks)
        sens[label] = {"protein_disjoint_cv_delta": r["cv_delta"],
                       "anchor_grouped_cv_delta": rows[VARIANTS[label]]["cv_delta"]}
        c, a = r["cv_delta"], rows[VARIANTS[label]]["cv_delta"]
        print(f"{label:13s} composite anchor-grouped {a['composite']:+.4f} -> protein-disjoint "
              f"{c['composite']:+.4f}; hard {a['auc_hard']:+.4f} -> {c['auc_hard']:+.4f}; "
              f"ndcg {a['ndcg@10']:+.4f} -> {c['ndcg@10']:+.4f} ({time.time() - t1:.0f}s)",
              flush=True)

    plan = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finalists_sha256": hashlib.sha256(FINALISTS.read_bytes()).hexdigest(),
        "search_log_sha256": hashlib.sha256(LOG.read_bytes()).hexdigest(),
        "variants": {k: {"id": VARIANTS[k], "spec": specs[k],
                         "leak_pruned": rows[VARIANTS[k]]["leak_pruned"],
                         "cv_delta": rows[VARIANTS[k]]["cv_delta"]} for k in VARIANTS},
        "winner": {"label": WINNER, "id": VARIANTS[WINNER],
                   "rule": ("declared before test: the frozen leak-free finalist slot F2, i.e. the "
                            "highest CV composite over all leak-pruned variants of the search; "
                            "it stays the headline whatever the test numbers show")},
        "current_best": "PUB_ndcg_U",
        "test_scoring": {
            "fit": "go_ppi_fusion.search.final_fit on all validation (z stats, stacker fit, theta)",
            "score": "go_ppi_fusion.search.final_score on test, once",
            "delta": "variant - text, same checkpoint, per test anchor",
            "bootstrap": {"estimator": "trials_domain.fusion_confirm draw_indices/summarize",
                          "B": FN.B, "seed": FN.BOOT_SEED,
                          "common_random_numbers": True},
            "contrasts": ["each variant - PUB_ndcg_U", "unpruned - pruned per pair",
                          "winner - shuffled", "winner - exact"],
            "by_fraction": "mean Delta over the 3 seeds per training fraction",
            "by_annotation": {"count": "anchor direct BP terms, pruned (na_P_pruned)",
                              "bins": ANNOTATION_BINS},
        },
        "diagnostics_winner_only": {
            "shuffled": {"rule": "protocol_go_ppi_fusion.md section 2.3, rules (a)-(e)",
                         "seed": FN.SHUFFLE_SEED, "strata": FN.N_STRATA,
                         "max_draws": FN.MAX_DRAWS,
                         "fit": "own stacker fit on the deranged validation file"},
            "exact": {"spec_change": "phi simgic_P_pruned -> exact_P_pruned, own fit"},
        },
        "fold_sensitivity": {
            "note": ("review finding 1. FIT records for held-out fold k exclude every record whose "
                     "anchor or partner appears in fold k (as anchor or partner). theta selection "
                     "for linear variants still averages over the other folds' anchors. Finalists "
                     "are not changed by this; reported only."),
            "fit_share_vs_anchor_grouped": fit_share,
            "reimplementation_max_abs_diff_vs_log": repro,
            "results": sens,
        },
        "seconds": round(time.time() - t0, 1),
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote {PLAN} ({plan['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
