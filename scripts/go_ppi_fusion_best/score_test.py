#!/usr/bin/env python3
"""One-time test scoring of the frozen GO/PPI fusion plan.

Reads ``test_plan.json`` and ``test_plan_amendment.json`` (both written before any
test scoring), freezes every planned variant on ALL of validation
(``search.final_fit``), scores the test split once (``search.final_score``) and
reports Delta = variant - text on the same checkpoint per test anchor, with the
trials two-level bootstrap over checkpoints x anchors (B = 2000, seed 20260101,
common random numbers). Then, for the declared winner only, the shuffled-GO
derangement control and the hierarchy-vs-exact arm. Also: by training fraction,
by anchor annotation count, and paired contrasts (vs the current best; unpruned -
pruned).

``--dry-run`` runs the whole pipeline in search mode with validation standing in
for test (no test file can be opened) and writes to /tmp; use it to check the code
path. The real run refuses to start if ``test_results.json`` already exists.

Usage:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_best/score_test.py --dry-run
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_best/score_test.py
"""
from __future__ import annotations

import argparse
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

DRY = "--dry-run" in sys.argv
if DRY:
    guard.enter_search_mode()  # validation stands in for test; test cannot be opened

import numpy as np  # noqa: E402

from go_ppi_fusion import data as D  # noqa: E402
from go_ppi_fusion import final as FN  # noqa: E402
from go_ppi_fusion import metrics as M  # noqa: E402
from go_ppi_fusion import phi as PHI  # noqa: E402
from go_ppi_fusion import search as S  # noqa: E402
from go_ppi_fusion.fusion import Inputs  # noqa: E402

OUT_DIR = ROOT / "results" / "go_ppi_fusion_best"
PLAN = OUT_DIR / "test_plan.json"
AMEND = OUT_DIR / "test_plan_amendment.json"
FINALISTS = OUT_DIR / "finalists.json"
REPRO = OUT_DIR / "reproduction.json"
OUT = Path("/tmp/goppi_fusion_best/dryrun_results.json") if DRY else OUT_DIR / "test_results.json"
DELTAS = PHI.SCRATCH / ("dryrun_deltas.npz" if DRY else "scored_deltas.npz")
SIMGIC_ARM = {"family": "anchor_z", "phi": "simgic_P_pruned", "select_on": "global_pooled_auc",
              "step": 0.1, "scope": "checkpoint"}
SIMGIC_ARM_ID = "2ef31536a048"
BIN_METRICS = ("auc_hard", "auc_pooled", "average_precision", "ndcg@10")
PAIRS = [("PUB_ndcg_U", "PUB_ndcg_P"), ("PUB_pooled_U", "PUB_pooled_P"), ("F1_U", "F1_P"),
         ("F2_U", "F2_P"), ("F3_U", "F3_P")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_id(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def theta_json(th):
    return S._json_theta(th)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.parse_args()
    t0 = time.time()
    if OUT.exists() and not DRY:
        print(f"{OUT} exists: test is scored once", file=sys.stderr)
        return 1
    plan = json.loads(PLAN.read_text())
    amend = json.loads(AMEND.read_text())
    assert amend["test_plan_sha256"] == sha(PLAN), "plan changed after the amendment"
    assert plan["finalists_sha256"] == sha(FINALISTS), "finalists changed after the plan"
    assert spec_id(SIMGIC_ARM) == SIMGIC_ARM_ID
    winner = amend["winner"]["label"]
    variants = {k: v["spec"] for k, v in plan["variants"].items()}
    for k, v in plan["variants"].items():
        assert spec_id(v["spec"]) == v["id"], k
    assert winner == "F3_P" and variants[winner]["phi"] == "exact_P_pruned"

    config = D.load_config()
    frozen = D.load_frozen_cache(config)
    val = D.load_split("validation", config, frozen, with_pruned=True)
    tgt = val if DRY else D.load_split("test", config, frozen, with_pruned=True)
    grid = D.checkpoint_grid()
    texts_val = [D.text_scores(val, r["checkpoint"], config) for r in grid]
    texts_tgt = texts_val if DRY else [D.text_scores(tgt, r["checkpoint"], config) for r in grid]
    phis_val = PHI.load_phis(PHI.SCRATCH / "phi_validation.npz")
    phis_tgt = phis_val if DRY else PHI.compute_phis(tgt)
    inp_val = Inputs("validation", val.queries, phis_val, M.QueryIndex(val.queries, val.grades),
                     val.grades)
    qi = M.QueryIndex(tgt.queries, tgt.grades)
    inp_tgt = Inputs(tgt.name, tgt.queries, phis_tgt, qi, tgt.grades)
    print(f"scored split={tgt.name} records={tgt.n} anchors={qi.Q} ({time.time() - t0:.0f}s)",
          flush=True)

    text_pa = FN.per_anchor(texts_tgt, qi)
    C, P = FN.draws(len(grid), qi.Q)
    # anchor annotation count (pruned direct BP terms), per query
    na = phis_tgt["na_P_pruned"][qi.idx[:, 0]]   # constant within an anchor (checked by derange)
    bins = plan["test_scoring"]["by_annotation"]["bins"]
    bin_cols = [np.flatnonzero((na >= lo) & (na <= hi)) for lo, hi in bins]
    bin_P = [FN.draws(len(grid), max(len(c), 1), seed=FN.BOOT_SEED + 1 + i)[1]
             for i, c in enumerate(bin_cols)]
    fractions = [r["fraction"] for r in grid]

    def score_variant(spec, inp_fit, inp_score, texts_fit):
        fitted = S.final_fit(spec, texts_fit, inp_fit)
        scores = [S.final_score(spec, f, t, inp_score) for f, t in zip(fitted, texts_tgt)]
        return scores, [theta_json(f["theta"]) for f in fitted]

    def report(label, spec, scores, thetas):
        Dv = FN.deltas(FN.per_anchor(scores, qi), text_pa)
        summ = FN.summarize_all(Dv, C, P)
        g_hard = [M.global_auc(s, tgt.grades, (2,), (1,)) - M.global_auc(t, tgt.grades, (2,), (1,))
                  for s, t in zip(scores, texts_tgt)]
        by_frac = {str(f): {k: float(np.mean([summ[k]["per_checkpoint"][i]
                                              for i, ff in enumerate(fractions) if ff == f]))
                            for k in FN.METRICS} for f in D.FRACTIONS}
        by_ann = []
        for (lo, hi), cols, Pb in zip(bins, bin_cols, bin_P):
            row = {"bin": [lo, hi], "n_anchors": int(len(cols))}
            if len(cols):
                row.update(FN.summarize_all(Dv, C, Pb, cols=cols, metrics=BIN_METRICS))
                for k in BIN_METRICS:
                    row[k].pop("per_checkpoint")
            by_ann.append(row)
        out = {"spec": spec, "theta": thetas, "delta": summ,
               "global_hard_auc_delta": {"point": float(np.mean(g_hard)),
                                         "per_checkpoint": [float(x) for x in g_hard],
                                         "n_ck_positive": int(sum(x > 0 for x in g_hard))},
               "by_fraction": by_frac, "by_annotation": by_ann}
        h, n = summ["auc_hard"], summ["ndcg@10"]
        print(f"{label:16s} hard {h['point']:+.4f} [{h['ci95'][0]:+.4f},{h['ci95'][1]:+.4f}] "
              f"{h['n_ck_positive']}/15  pooled {summ['auc_pooled']['point']:+.4f}  "
              f"AP {summ['average_precision']['point']:+.4f}  nDCG {n['point']:+.4f} "
              f"[{n['ci95'][0]:+.4f},{n['ci95'][1]:+.4f}]  theta {thetas}", flush=True)
        return out, Dv

    results, Ds = {}, {}
    for label, spec in variants.items():
        scores, thetas = score_variant(spec, inp_val, inp_tgt, texts_val)
        results[label], Ds[label] = report(label, spec, scores, thetas)
    if not DRY:
        rep = json.loads(REPRO.read_text())["rules"]
        assert results["PUB_ndcg_U"]["theta"] == rep["ndcg@10"]["selected_weights"]
        assert results["PUB_pooled_U"]["theta"] == rep["pooled_auc"]["selected_weights"]

    # ------------------------------------------------ winner diagnostics
    diag = {}
    scores, thetas = score_variant(SIMGIC_ARM, inp_val, inp_tgt, texts_val)
    diag["simgic_arm"], Ds["W_simgic_arm"] = report("W_simgic_arm", SIMGIC_ARM, scores, thetas)
    from go_ppi_domain.go_ontology import GoMatcher
    matcher = GoMatcher(PHI.load_index("P", "pruned"), similarity_mode="simgic")
    man_val = FN.derange_anchors(val, matcher)
    man_tgt = man_val if DRY else FN.derange_anchors(tgt, matcher)
    dval = FN.deranged_split(val, man_val["mapping"])
    dtgt = FN.deranged_split(tgt, man_tgt["mapping"])
    ph_dval = PHI.compute_phis(dval, versions=("pruned",), resnik=False)
    ph_dtgt = ph_dval if DRY else PHI.compute_phis(dtgt, versions=("pruned",), resnik=False)
    inp_dval = Inputs("validation", dval.queries, ph_dval, inp_val.qi, val.grades)
    inp_dtgt = Inputs(tgt.name, dtgt.queries, ph_dtgt, qi, tgt.grades)
    wspec = variants[winner]
    scores, thetas = score_variant(wspec, inp_dval, inp_dtgt, texts_val)
    diag["shuffled"], Ds["W_shuffled"] = report("W_shuffled", wspec, scores, thetas)
    # same control at the winner's own (real-validation) weights: shows what deranged GO
    # does to the ranking when it is not allowed to select w = 0
    fitted = S.final_fit(wspec, texts_val, inp_val)
    scores = [S.final_score(wspec, f, t, inp_dtgt) for f, t in zip(fitted, texts_tgt)]
    diag["shuffled_fixed_w"], Ds["W_shuffled_fixed_w"] = report(
        "W_shuffled_fixed_w", wspec, scores, [theta_json(f["theta"]) for f in fitted])
    diag["shuffle_manifest"] = {
        "validation": {k: v for k, v in man_val.items() if k != "mapping"},
        "scored": {k: v for k, v in man_tgt.items() if k != "mapping"},
        "phi_change": {"exact_P_pruned_records_changed_frac": float(np.mean(
            ph_dtgt["exact_P_pruned"] != phis_tgt["exact_P_pruned"]))},
    }

    # ------------------------------------------------ paired contrasts
    def contrast(a, b):
        Dc = {k: Ds[a][k] - Ds[b][k] for k in FN.METRICS}
        s = FN.summarize_all(Dc, C, P)
        for k in s:
            s[k].pop("per_checkpoint")
        return s

    contrasts = {f"{k} - PUB_ndcg_U": contrast(k, "PUB_ndcg_U") for k in variants if k != "PUB_ndcg_U"}
    for u, p in PAIRS:
        contrasts[f"{u} - {p}"] = contrast(u, p)
    contrasts[f"{winner} - W_shuffled"] = contrast(winner, "W_shuffled")
    contrasts[f"{winner} - W_shuffled_fixed_w"] = contrast(winner, "W_shuffled_fixed_w")
    contrasts[f"W_simgic_arm - {winner}"] = contrast("W_simgic_arm", winner)

    text_level = {k: float(np.mean([M.macro({k: text_pa[k][c]})[k] for c in range(len(grid))]))
                  for k in FN.METRICS}
    out = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": DRY,
        "scored_split": {"name": tgt.name, "path": tgt.meta["path"], "sha256": tgt.meta["sha256"],
                         "pruned_sha256": tgt.meta.get("pruned_sha256"), "records": int(tgt.n),
                         "anchors": int(qi.Q)},
        "plan_sha256": sha(PLAN), "amendment_sha256": sha(AMEND),
        "winner": winner, "current_best": plan["current_best"],
        "bootstrap": {"B": FN.B, "seed": FN.BOOT_SEED, "estimator": "trials_domain.fusion_confirm",
                      "annotation_bin_anchor_seeds": [FN.BOOT_SEED + 1 + i for i in range(len(bins))]},
        "text_level": text_level,
        "variants": results, "diagnostics": diag, "contrasts": contrasts,
        "anchor_annotation_count_quartiles": [float(x) for x in np.percentile(na, [25, 50, 75])],
        "seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    np.savez_compressed(DELTAS, **{f"{lab}|{k}": v for lab, d in Ds.items() for k, v in d.items()})
    print(f"wrote {OUT} ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
