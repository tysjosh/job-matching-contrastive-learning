#!/usr/bin/env python3
"""Size-matched control for the protein-disjoint fold sensitivity (validation only).

``freeze_test_plan.py`` found that protein-disjoint FIT sets keep only 15-20% of
the anchor-grouped FIT records, so a drop under protein-disjoint fitting mixes two
causes: partner overlap and a 5x smaller fit set. This control refits every
planned variant on a random record subset of the anchor-grouped FIT set with the
same size as the protein-disjoint one (seed 0 per fold). Partner overlap is kept,
only the size shrinks. Runs in search mode before any test scoring; it changes
neither the finalists nor the declared winner.

Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_ppi_fusion_best/fold_sensitivity_sizematched.py
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

from go_ppi_fusion import guard  # noqa: E402

guard.enter_search_mode()

import numpy as np  # noqa: E402

from go_ppi_fusion import data as D  # noqa: E402
from go_ppi_fusion import final as FN  # noqa: E402
from go_ppi_fusion import metrics as M  # noqa: E402
from go_ppi_fusion import phi as PHI  # noqa: E402
from go_ppi_fusion import search as S  # noqa: E402
from go_ppi_fusion.fusion import Inputs  # noqa: E402

OUT_DIR = ROOT / "results" / "go_ppi_fusion_best"
PLAN = OUT_DIR / "test_plan.json"
OUT = OUT_DIR / "fold_sensitivity_sizematched.json"
SEED = 0


def main() -> int:
    t0 = time.time()
    plan = json.loads(PLAN.read_text())
    config = D.load_config()
    frozen = D.load_frozen_cache(config)
    val = D.load_split("validation", config, frozen, with_pruned=True)
    texts = [D.text_scores(val, r["checkpoint"], config) for r in D.checkpoint_grid()]
    phis = PHI.load_phis(PHI.SCRATCH / "phi_validation.npz")
    inp = Inputs("validation", val.queries, phis, M.QueryIndex(val.queries, val.grades), val.grades)
    folds = S.anchor_folds(inp.qi)
    rec_fold = folds[inp.qi.inverse]
    pd_masks = FN.protein_disjoint_masks(val, inp.qi, folds)
    rng = np.random.default_rng(SEED)
    sm_masks = []
    for s in range(S.N_FOLDS):
        pool = np.flatnonzero(rec_fold != s)
        keep = rng.choice(pool, size=int(pd_masks[s].sum()), replace=False)
        m = np.zeros(inp.n, bool)
        m[keep] = True
        sm_masks.append(m)
    res = {}
    for label, v in plan["variants"].items():
        r = FN.cv_with_fit_masks(v["spec"], texts, inp, folds, sm_masks)
        res[label] = {"size_matched_random_cv_delta": r["cv_delta"]}
        a = v["cv_delta"]
        p = plan["fold_sensitivity"]["results"][label]["protein_disjoint_cv_delta"]
        c = r["cv_delta"]
        print(f"{label:13s} composite: anchor-grouped {a['composite']:+.4f} | size-matched "
              f"random {c['composite']:+.4f} | protein-disjoint {p['composite']:+.4f}; hard "
              f"{a['auc_hard']:+.4f} | {c['auc_hard']:+.4f} | {p['auc_hard']:+.4f}", flush=True)
    out = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": SEED,
           "fit_records_per_fold": [int(m.sum()) for m in sm_masks],
           "note": ("random record subset of each anchor-grouped FIT set, same size as the "
                    "protein-disjoint FIT set; validation only, before test scoring"),
           "results": res, "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
