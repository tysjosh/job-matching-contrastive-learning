#!/usr/bin/env python3
"""Diagnose the CVSS Stage-2 regression collapse.

Reads the CVSS experiment eval reports and prints, per experiment, the
band-from-score diagnostics that reveal whether the predicted priority_score
collapsed to a near-constant (all predictions in one band):

  - severity_order + score_cut_points (the band boundaries the metric derived)
  - per-class support / precision / recall / f1 (where predictions landed)
  - mae_priority_score (huge => predictions far from the true scores)
  - the predicted-band histogram, reconstructed from per-class tp+fp

Run on the box:
    python scripts/diagnose_cvss_head.py
    python scripts/diagnose_cvss_head.py --root cve_domain/runs/experiments_cvss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def diagnose(root: Path, exps):
    for e in exps:
        p = root / e / "eval" / "evaluation_report.json"
        if not p.exists():
            print(f"\n[{e}] no eval report at {p}")
            continue
        rep = json.load(open(p))
        rk = rep.get("ranking", {})
        bfs = rep.get("classification", {}).get("priority_band_from_score", {})
        print(f"\n===== {e} =====")
        print(f"  ranking: ndcg={rk.get('ndcg')}  map={rk.get('map')}  spearman={rk.get('spearman_rho')}")
        if bfs.get("skipped"):
            print(f"  band_from_score SKIPPED: {bfs.get('reason')}"); continue
        print(f"  severity_order : {bfs.get('severity_order')}")
        print(f"  score_cutpoints: {bfs.get('score_cut_points')}")
        print(f"  macro_f1       : {bfs.get('macro_f1')}   accuracy: {bfs.get('accuracy')}")
        print(f"  mae_priority_score: {bfs.get('mae_priority_score')}")
        per = bfs.get("per_class", {})
        # Reconstruct predicted count per class = tp + fp. tp = recall*support (approx via precision).
        print(f"  {'class':<10} {'support':>8} {'prec':>7} {'recall':>7} {'f1':>7}")
        for c, m in per.items():
            print(f"  {c:<10} {m.get('support',0):>8} {m.get('precision',0):>7.3f} "
                  f"{m.get('recall',0):>7.3f} {m.get('f1',0):>7.3f}")
        # A class with recall≈1 and every other class recall≈0 == everything predicted that class.
        nonzero = [c for c, m in per.items() if (m.get('recall') or 0) > 0.5]
        print(f"  -> classes receiving (recall>0.5): {nonzero}  "
              f"(single class => predictions collapsed to it)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cve_domain/runs/experiments_cvss")
    ap.add_argument("--exps", default="E0,E1,E4,E7")
    args = ap.parse_args()
    diagnose(Path(args.root), [x.strip() for x in args.exps.split(",") if x.strip()])


if __name__ == "__main__":
    main()
