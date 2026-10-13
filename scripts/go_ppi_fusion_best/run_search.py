#!/usr/bin/env python3
"""Validation-only search for the best GO/PPI inference-time fusion.

Enters search mode first, so no test file can be opened by this process
(``go_ppi_fusion.guard``). Scores the 15 published text-only checkpoints on the
published validation split, computes every phi variant on it, evaluates every
variant with anchor-grouped 5-fold cross-fitting (``go_ppi_fusion.search``),
appends one JSON line per variant to ``search_log.jsonl`` and freezes at most 3
finalists in ``finalists.json`` by the rule in ``FINALIST_RULE``.

Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_best/run_search.py
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
from go_ppi_fusion import metrics as M  # noqa: E402
from go_ppi_fusion import phi as PHI  # noqa: E402
from go_ppi_fusion import search as S  # noqa: E402
from go_ppi_fusion.fusion import Inputs  # noqa: E402

OUT_DIR = ROOT / "results" / "go_ppi_fusion_best"
LOG = OUT_DIR / "search_log.jsonl"
FINALISTS = OUT_DIR / "finalists.json"
SCRATCH = PHI.SCRATCH
PUBLISHED_SPEC = {"family": "linear", "phi": "simgic_P_unpruned", "missing": "zero",
                  "select_on": "ndcg@10", "step": 0.1, "scope": "checkpoint"}

FINALIST_RULE = {
    "criterion": ("cv_delta.composite: mean over the 15 checkpoints of the mean of "
                  "Delta within-anchor hard AUC, Delta within-anchor pooled AUC, Delta AP "
                  "and Delta nDCG@10, each from anchor-grouped 5-fold cross-fitting on the "
                  "published validation split (697 anchors; fold seed 20261012). "
                  "Delta = variant - text on the same checkpoint."),
    "slots": [
        "F1: highest criterion over all logged variants",
        "F2: highest criterion among variants whose GO inputs are all leak-pruned "
        "(pruned index + pruned record URIs; with/from partner leak removed), if not F1",
        "F3: highest criterion among variants with at most one tuned parameter and no "
        "learned model (simplest robust form), if not already chosen; else next overall",
    ],
    "ties": "higher criterion wins; exact ties -> fewer tuned parameters, then earlier log order",
    "test_set": "never read by this step; finalists are frozen before any test scoring",
}


def phi_names(version: str):
    names = []
    for a in "PFC":
        for mode in ("simgic", "ancestor", "exact", "icdirect"):
            names.append(f"{mode}_{a}_{version}")
    names += [f"resnik_P_{version}", f"isa_simgic_P_{version}",
              f"simgic_mean_{version}", f"simgic_max_{version}"]
    return names


def spec_id(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def is_pruned(spec):
    phis = spec["phi"] if isinstance(spec["phi"], list) else [spec["phi"]]
    extra = spec.get("flags", []) + spec.get("counts", [])
    return all(p.endswith("_pruned") for p in phis + extra)


def n_tuned(spec, n_thetas):
    if spec["family"] == "stacker":
        return 99
    return {"linear": 1, "rank": 1, "rrf": 1, "anchor_z": 1}.get(spec["family"], 2)


class Runner:
    def __init__(self, texts, inp, grid):
        self.texts, self.inp, self.grid = texts, inp, grid
        self.folds = S.anchor_folds(inp.qi)
        self.done = {}
        self.order = []

    def run(self, spec, stage):
        sid = spec_id(spec)
        if sid in self.done:
            return self.done[sid]
        t0 = time.time()
        res = S.evaluate_variant(spec, self.texts, self.inp,
                                 S.scope_groups(spec.get("scope", "checkpoint"), self.grid),
                                 self.folds)
        row = {"id": sid, "stage": stage, "spec": spec, "n_thetas": res["n_thetas"],
               "leak_pruned": is_pruned(spec), "seconds": round(time.time() - t0, 1),
               **res["summary"],
               "theta_all": [r["theta_all"] for r in res["per_checkpoint"]],
               "theta_folds": [r["theta_folds"] for r in res["per_checkpoint"]],
               "cv_delta_per_checkpoint": [r["cv_delta"] for r in res["per_checkpoint"]]}
        with LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        self.done[sid] = row
        self.order.append(sid)
        c = row["cv_delta"]
        print(f"[{stage}] {sid} {spec['family']:8s} {str(spec['phi'])[:60]:60s} "
              f"cv comp {c['composite']:+.4f} hard {c['auc_hard']:+.4f} "
              f"ndcg {c['ndcg@10']:+.4f} ap {c['average_precision']:+.4f} "
              f"({row['seconds']}s)", flush=True)
        return row

    def top(self, pred, k):
        rows = [self.done[s] for s in self.order if pred(self.done[s])]
        rows.sort(key=lambda r: -r["cv_delta"]["composite"])
        return rows[:k]


def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()  # one complete log per search run
    config = D.load_config()
    frozen = D.load_frozen_cache(config)
    val = D.load_split("validation", config, frozen, with_pruned=True)
    grid = D.checkpoint_grid()
    texts = [D.text_scores(val, r["checkpoint"], config) for r in grid]
    phi_cache = SCRATCH / "phi_validation.npz"
    if phi_cache.exists():
        phis = PHI.load_phis(phi_cache)
    else:
        phis = PHI.compute_phis(val)
        PHI.save_phis(phi_cache, phis)
    inp = Inputs("validation", val.queries, phis, M.QueryIndex(val.queries, val.grades),
                 val.grades)
    print(f"validation: {val.n} records, {inp.qi.Q} anchors; {len(phis)} phi arrays; "
          f"{time.time() - t_start:.0f}s", flush=True)
    R = Runner(texts, inp, grid)

    # Sanity: the published variant, in-sample, must equal the published validation curve max.
    pub = R.run(dict(PUBLISHED_SPEC), "0_published")

    # Stage 1: phi screen under the published fusion form.
    for version in ("unpruned", "pruned"):
        for name in phi_names(version):
            R.run({**PUBLISHED_SPEC, "phi": name}, "1_phi")

    # Stage 2: inner selection metric and grid resolution.
    for name in ("simgic_P_unpruned", "simgic_P_pruned"):
        for sel in ("ndcg@10", "average_precision", "auc_pooled", "global_pooled_auc", "composite"):
            for step in (0.1, 0.05):
                R.run({**PUBLISHED_SPEC, "phi": name, "select_on": sel, "step": step}, "2_select")
    best_sel = max(("ndcg@10", "average_precision", "auc_pooled", "global_pooled_auc", "composite"),
                   key=lambda s: np.mean([R.done[spec_id({**PUBLISHED_SPEC, "phi": n, "select_on": s,
                                                          "step": 0.1})]["cv_delta"]["composite"]
                                          for n in ("simgic_P_unpruned", "simgic_P_pruned")]))
    print("stage 2 best inner metric:", best_sel, flush=True)

    linear1 = lambda r: r["spec"]["family"] == "linear" and r["stage"] in ("0_published", "1_phi")
    top_u = [r["spec"]["phi"] for r in R.top(lambda r: linear1(r) and not r["leak_pruned"], 3)]
    top_p = [r["spec"]["phi"] for r in R.top(lambda r: linear1(r) and r["leak_pruned"], 3)]
    tops = list(dict.fromkeys(top_u + top_p + ["simgic_P_unpruned", "simgic_P_pruned"]))
    print("top phis:", tops, flush=True)
    base = {**PUBLISHED_SPEC, "select_on": best_sel}

    # Stage 3: missing handling and weight scope.
    for name in tops:
        for missing in ("zero", "impute", "text_only"):
            for scope in ("checkpoint", "fraction", "global"):
                R.run({**base, "phi": name, "missing": missing, "scope": scope}, "3_missing_scope")

    # Stage 4: other fusion forms.
    for name in tops:
        R.run({"family": "rank", "phi": name, "select_on": best_sel, "step": 0.1,
               "scope": "checkpoint"}, "4_form")
        for k in (5, 60):
            R.run({"family": "rrf", "phi": name, "k": k, "select_on": best_sel, "step": 0.1,
                   "scope": "checkpoint"}, "4_form")
        R.run({"family": "anchor_z", "phi": name, "select_on": best_sel, "step": 0.1,
               "scope": "checkpoint"}, "4_form")
        R.run({"family": "coverage", "phi": name, "missing": "impute", "select_on": best_sel,
               "step": 0.1, "c_grid": [2, 5, 10], "scope": "checkpoint"}, "4_form")

    # Stage 5: two GO signals (BP + MF / CC / Resnik), per version.
    for version in ("unpruned", "pruned"):
        p1 = f"simgic_P_{version}"
        for p2 in (f"simgic_F_{version}", f"simgic_C_{version}", f"resnik_P_{version}",
                   f"exact_P_{version}"):
            for missing in ("zero", "impute"):
                R.run({"family": "linear2", "phi": [p1, p2], "missing": missing,
                       "select_on": best_sel, "step": 0.1, "scope": "checkpoint"}, "5_two_phi")

    # Stage 6: learned stackers, fit on validation folds only.
    for version in ("unpruned", "pruned"):
        sets = {
            "P": [f"simgic_P_{version}"],
            "PFC": [f"simgic_{a}_{version}" for a in "PFC"],
            "rich": [f"simgic_{a}_{version}" for a in "PFC"]
            + [f"resnik_P_{version}", f"exact_P_{version}", f"isa_simgic_P_{version}"],
        }
        flags = [f"miss_{a}_{version}" for a in "PFC"]
        counts = [f"na_P_{version}", f"nb_P_{version}"]
        for set_name, phis_set in sets.items():
            for model, extra in (("lr", {"C": 1.0}), ("lr", {"C": 0.1}),
                                 ("pairwise", {"C": 1.0}), ("hgb", {})):
                for with_counts in (False, True):
                    if set_name == "P" and model == "hgb" and not with_counts:
                        continue
                    R.run({"family": "stacker", "model": model, **extra, "phi": phis_set,
                           "flags": flags, "counts": counts if with_counts else [],
                           "interactions": True, "scope": "checkpoint",
                           "set": set_name}, "6_stacker")

    # ---------------------------------------------------------------- finalists
    rows = [R.done[s] for s in R.order]

    def key(r):
        return (-r["cv_delta"]["composite"], n_tuned(r["spec"], r["n_thetas"]),
                R.order.index(r["id"]))

    ranked = sorted(rows, key=key)
    chosen = []
    f1 = ranked[0]
    chosen.append(("F1", f1))
    f2 = next((r for r in ranked if r["leak_pruned"]), None)
    if f2 is not None and f2["id"] != f1["id"]:
        chosen.append(("F2", f2))
    ids = {r["id"] for _, r in chosen}
    f3 = next((r for r in ranked if n_tuned(r["spec"], r["n_thetas"]) <= 1 and r["id"] not in ids),
              None)
    if f3 is None:
        f3 = next((r for r in ranked if r["id"] not in ids), None)
    if f3 is not None:
        chosen.append(("F3", f3))
    pub_row = R.done[spec_id(dict(PUBLISHED_SPEC))]
    finalists = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule": FINALIST_RULE,
        "n_variants_logged": len(rows),
        "validation": {"path": val.meta["path"], "sha256": val.meta["sha256"],
                       "pruned_sha256": val.meta.get("pruned_sha256"),
                       "records": int(val.n), "anchors": int(inp.qi.Q),
                       "fold_seed": S.FOLD_SEED, "n_folds": S.N_FOLDS},
        "stage2_inner_metric": best_sel,
        "reference_published": {"id": pub_row["id"], "spec": pub_row["spec"],
                                "cv_delta": pub_row["cv_delta"],
                                "insample_delta": pub_row["insample_delta"]},
        "finalists": [{"slot": slot, "id": r["id"], "spec": r["spec"],
                       "leak_pruned": r["leak_pruned"], "n_thetas": r["n_thetas"],
                       "cv_delta": r["cv_delta"], "insample_delta": r["insample_delta"],
                       "cv_n_positive_composite": r["cv_delta_n_positive_composite"],
                       "theta_all": r["theta_all"]} for slot, r in chosen],
        "test_scoring": ("NOT run in this step. Report step: freeze each finalist with "
                         "go_ppi_fusion.search.final_fit on all validation, score test once "
                         "with final_score, Delta vs text on the same checkpoint."),
        "top10_by_criterion": [{"id": r["id"], "family": r["spec"]["family"],
                                "phi": r["spec"]["phi"], "cv_composite": r["cv_delta"]["composite"],
                                "leak_pruned": r["leak_pruned"]} for r in ranked[:10]],
        "seconds": round(time.time() - t_start, 1),
    }
    FINALISTS.write_text(json.dumps(finalists, indent=2) + "\n")
    print(json.dumps({"finalists": [(s, r["id"], r["spec"], r["cv_delta"]) for s, r in chosen],
                      "published_cv": pub_row["cv_delta"]}, indent=1))
    print(f"logged {len(rows)} variants in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
