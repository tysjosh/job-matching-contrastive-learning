#!/usr/bin/env python3
"""Gate: reproduce the published GO/PPI fusion on the 15 checkpoints exactly.

Read-only. Re-scores both published selection rules (pooled_auc and ndcg@10)
with the published functions from ``scripts/evaluate_publication_fusion.py``
(``z_stats``, ``validation_selection_value``, ``contrast_metrics``,
``ranking_metrics``, ``text_scores``) on inputs prepared by the read-only port
``go_ppi_fusion.data``, and compares every leaf with the published JSONs:
selected weight, validation stats, validation curve, and for test text and
fusion the global AUCs, macro within-anchor AUCs, query counts, ranking macro
and every per-anchor ranking row. It also checks the vectorised metrics in
``go_ppi_fusion.metrics`` against the published ones on the real test scores.

This is the only script in the study that reads test data before the report
step, and it only scores the already-published method.

Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_best/reproduce_published.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from go_ppi_fusion import data as D  # noqa: E402
from go_ppi_fusion import metrics as M  # noqa: E402
from scripts import evaluate_publication_fusion as epf  # noqa: E402

PUBLISHED = {
    "pooled_auc": ROOT / "results" / "publication_phase1_fixedval" / "go_ppi_fusion_learning_curve.json",
    "ndcg@10": ROOT / "results" / "publication_phase1_fixedval"
    / "go_ppi_fusion_learning_curve_ndcg_selected.json",
}
OUT = ROOT / "results" / "go_ppi_fusion_best" / "reproduction.json"
WEIGHTS = [i / 10 for i in range(11)]


def published_phi(split: D.Split) -> np.ndarray:
    """Exactly the published phi: unpruned BP simGIC over record skill_uris."""
    from go_ppi_domain.go_ontology import GoIndex, GoMatcher
    m = GoMatcher(GoIndex.load(ROOT / "preprocess" / "go_ppi" / "go_index_P.pkl"),
                  alpha=0.5, max_hops=12, similarity_mode="simgic")
    vals = []
    for a, b in zip(split.uris_a, split.uris_b):
        try:
            vals.append(float(m.ontology_set_similarity(a, b)) if a and b else 0.0)
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=float)


class Diff:
    """Track max |a - b| and exact equality over compared leaves."""

    def __init__(self):
        self.max_abs = 0.0
        self.n = 0
        self.n_exact = 0
        self.worst = None

    def add(self, a, b, where):
        self.n += 1
        if a is None or b is None:
            ok = a is None and b is None
            if not ok:
                self.max_abs, self.worst = float("inf"), where
            self.n_exact += ok
            return
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                self.max_abs, self.worst = float("inf"), where
            else:
                self.n_exact += 1
            return
        a, b = float(a), float(b)
        if np.isnan(a) and np.isnan(b):
            self.n_exact += 1
            return
        d = abs(a - b)
        if d == 0.0:
            self.n_exact += 1
        if d > self.max_abs:
            self.max_abs, self.worst = d, where

    def tree(self, a, b, where):
        if isinstance(b, dict):
            for k in b:
                self.tree(a.get(k) if isinstance(a, dict) else None, b[k], f"{where}.{k}")
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                self.max_abs, self.worst = float("inf"), where + "[len]"
                return
            for i, (x, y) in enumerate(zip(a, b)):
                self.tree(x, y, f"{where}[{i}]")
        else:
            self.add(a, b, where)

    def as_dict(self):
        return {"leaves": self.n, "exact": self.n_exact, "max_abs_diff": self.max_abs,
                "worst_leaf": self.worst}


def main() -> int:
    t0 = time.time()
    config = D.load_config()
    frozen = D.load_frozen_cache(config)
    val = D.load_split("validation", config, frozen, with_pruned=False)
    test = D.load_split("test", config, frozen, with_pruned=False)
    print("prepared", val.meta, test.meta, flush=True)
    val_phi, test_phi = published_phi(val), published_phi(test)
    val_prep = {**val.prepared(), "ontology": val_phi}
    test_prep = {**test.prepared(), "ontology": test_phi}
    ostats = epf.z_stats(val_phi)
    vz_o, tz_o = epf.z_apply(val_phi, ostats), epf.z_apply(test_phi, ostats)
    qi_test = M.QueryIndex(test.queries, test.grades)

    report = {"inputs": {"validation": val.meta, "test": test.meta,
                         "phi_stats_validation": list(ostats)},
              "rules": {}}
    fast_check = Diff()
    for rule, path in PUBLISHED.items():
        pub = json.loads(path.read_text())
        pruns = {(r["fraction"], r["seed"]): r for r in pub["runs"]}
        d_weight, d_stats, d_curve, d_test = Diff(), Diff(), Diff(), Diff()
        weights_match = []
        deltas = []
        for row in D.checkpoint_grid():
            key = (row["fraction"], row["seed"])
            pr = pruns[key]
            vt = D.text_scores(val, row["checkpoint"], config)
            tt = D.text_scores(test, row["checkpoint"], config)
            tstats = epf.z_stats(vt)
            vz_t, tz_t = epf.z_apply(vt, tstats), epf.z_apply(tt, tstats)
            curve = {str(w): epf.validation_selection_value(
                (1 - w) * vz_t + w * vz_o, val_prep, rule) for w in WEIGHTS}
            w_sel = max(WEIGHTS, key=lambda w: curve[str(w)])
            weights_match.append(w_sel == pr["selected_weight"])
            d_weight.add(w_sel, pr["selected_weight"], f"{key}.selected_weight")
            d_stats.tree({"text": {"mean": tstats[0], "sd": tstats[1]},
                          "ontology": {"mean": ostats[0], "sd": ostats[1]}},
                         pr["validation_stats"], f"{key}.validation_stats")
            d_curve.tree(curve, pr["validation_selection_curve"], f"{key}.curve")
            fused = (1 - w_sel) * tz_t + w_sel * tz_o
            mine = {}
            for arm, s in (("text", tz_t), ("fusion", fused)):
                c = epf.contrast_metrics(s, test.grades, test.queries)
                r = epf.ranking_metrics(s, test.grades, test.queries)
                mine[arm] = {"contrasts": c, "ranking": r}
                d_test.tree(epf.json_safe(c), pr["test"][arm]["contrasts"], f"{key}.{arm}.contrasts")
                d_test.tree(epf.json_safe(r), pr["test"][arm]["ranking"], f"{key}.{arm}.ranking")
                # vectorised metrics vs published on the real test scores
                pq = M.macro(M.per_query(s, qi_test))
                for name in ("hard", "easy", "middle_vs_easy", "pooled"):
                    fast_check.add(pq[f"auc_{name}"], c["macro_query_auc"][name],
                                   f"{key}.{arm}.auc_{name}")
                for name in M.RANKING:
                    fast_check.add(pq[name], r["macro"][name], f"{key}.{arm}.{name}")
            deltas.append({
                "fraction": key[0], "seed": key[1], "selected_weight": w_sel,
                **{f"within_{n}": mine["fusion"]["contrasts"]["macro_query_auc"][n]
                   - mine["text"]["contrasts"]["macro_query_auc"][n]
                   for n in ("hard", "pooled", "easy")},
                "global_hard": mine["fusion"]["contrasts"]["global_auc"]["hard"]
                - mine["text"]["contrasts"]["global_auc"]["hard"],
                **{n: mine["fusion"]["ranking"]["macro"][n] - mine["text"]["ranking"]["macro"][n]
                   for n in ("average_precision", "ndcg@10", "precision@10")},
                "text_within_hard": mine["text"]["contrasts"]["macro_query_auc"]["hard"],
                "text_within_pooled": mine["text"]["contrasts"]["macro_query_auc"]["pooled"],
            })
            print(f"{rule} f{key[0]} s{key[1]} w={w_sel} match={weights_match[-1]}", flush=True)
        mean = {k: float(np.mean([d[k] for d in deltas]))
                for k in deltas[0] if k not in ("fraction", "seed")}
        by_frac = {str(f): float(np.mean([d["within_hard"] for d in deltas if d["fraction"] == f]))
                   for f in D.FRACTIONS}
        report["rules"][rule] = {
            "published_file": str(path.relative_to(ROOT)),
            "published_sha256": D.file_sha256(path),
            "selected_weights_match": f"{sum(weights_match)}/{len(weights_match)}",
            "selected_weights": [d["selected_weight"] for d in deltas],
            "diff_selected_weight": d_weight.as_dict(),
            "diff_validation_stats": d_stats.as_dict(),
            "diff_validation_curve": d_curve.as_dict(),
            "diff_test_leaves": d_test.as_dict(),
            "mean_delta_15": mean,
            "within_hard_delta_by_fraction": by_frac,
            "per_checkpoint": deltas,
        }
    report["vectorised_metrics_vs_published"] = fast_check.as_dict()
    gate = all(r["selected_weights_match"] == "15/15"
               and r["diff_test_leaves"]["max_abs_diff"] == 0.0
               for r in report["rules"].values())
    report["gate_passed"] = bool(gate)
    report["gate_rule"] = ("all 15 selected weights equal AND every published test leaf "
                           "(contrasts, ranking macro, per-anchor rows) bit-identical")
    report["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(epf.json_safe(report), indent=2) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_checkpoint"}
                      for k, v in report["rules"].items()}, indent=1, default=str))
    print("vectorised vs published:", report["vectorised_metrics_vs_published"])
    print("GATE", "PASSED" if gate else "FAILED", f"({report['seconds']} s)")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
