#!/usr/bin/env python3
"""ER-SCHED confound test: is the ORCA gain reliability, or the 4-phase schedule?

ER-SCHED sets ``orca_r_min=1.0`` so ``clamp(r, 1.0, 1.0) == 1.0`` for every
negative, which reduces the reliability-calibrated denominator to Standard
InfoNCE bit-for-bit, and ``orca_eta_rel=0.0`` so the reliability BCE term cannot
regularize the shared projection head either. What remains is the same model
trained under ORCA's exact 4-phase schedule.

So ER-SCHED vs ER-DEN isolates reliability, and ER-SCHED vs the single-phase
E4-* baselines isolates the schedule. Reads results from the Hugging Face results
dataset; the baselines come from the local run tree.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = "olukotunjosh/cdcl-orca-results"
ROOT = Path(__file__).resolve().parents[1]

#: Config fields that must hold for ER-SCHED to actually neutralize reliability.
NEUTRALIZED = {"orca_r_min": 1.0, "orca_eta_rel": 0.0}


def _fetch(path: str) -> Optional[dict]:
    from huggingface_hub import hf_hub_download

    try:
        return json.load(open(hf_hub_download(REPO, path, repo_type="dataset")))
    except Exception:
        return None


def _hf_auc(variant: str, seed: str) -> Optional[float]:
    d = _fetch(f"orca/{variant}__cnamuangtoun__s{seed}"
               f"/phase1_evaluation/phase1_evaluation_results.json")
    return None if d is None else d["metrics"]["auc_roc"]


def _local_auc(variant: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pat = str(ROOT / "results" / "research_runs"
              / f"{variant}__cnamuangtoun__s*"
              / "phase1_evaluation" / "phase1_evaluation_results.json")
    for p in sorted(glob.glob(pat)):
        seed = re.search(r"__s(\d+)", p).group(1)
        out[seed] = json.load(open(p))["metrics"]["auc_roc"]
    return out


def _summary(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.4f}±{sd:.4f} (n={len(vals)})"


def _paired(a: List[float], b: List[float]) -> str:
    """Paired t-test if scipy is available, else the mean delta."""
    diff = [x - y for x, y in zip(a, b)]
    mean = st.mean(diff)
    if len(diff) < 2:
        return f"Δ={mean:+.4f} (n={len(diff)}, no test possible)"
    sd = st.stdev(diff)
    try:
        from scipy import stats

        t, p = stats.ttest_rel(a, b)
        return f"Δ={mean:+.4f} sd={sd:.4f}  t={t:+.2f} p={p:.4f}"
    except Exception:
        return f"Δ={mean:+.4f} sd={sd:.4f} (scipy unavailable)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", default=["13", "21", "42", "87", "123"])
    args = ap.parse_args(argv)

    # ---- 1. validity: does ER-SCHED's config really neutralize reliability? ----
    print("=" * 70)
    print("1. VALIDITY CHECK — is reliability actually off in ER-SCHED?")
    print("=" * 70)
    sched_cfg = _fetch("orca/ER-SCHED__cnamuangtoun__s42/training_config.json")
    den_cfg = _fetch("orca/ER-DEN__cnamuangtoun__s42/training_config.json")
    if sched_cfg is None or den_cfg is None:
        print("  could not fetch configs from HF")
        return 1

    ok = True
    for key, want in NEUTRALIZED.items():
        got = sched_cfg.get(key)
        good = got == want
        ok &= good
        print(f"  {key:22s} = {got!r:8} (need {want!r})  {'OK' if good else 'FAIL'}")
    if not ok:
        print("\n  ER-SCHED does not neutralize reliability; the test is invalid.")
        return 1

    diffs = {k: (den_cfg.get(k), sched_cfg.get(k))
             for k in set(den_cfg) | set(sched_cfg)
             if den_cfg.get(k) != sched_cfg.get(k)}
    print(f"\n  ER-DEN vs ER-SCHED differ ONLY in: {diffs}")
    if set(diffs) - set(NEUTRALIZED):
        print("  WARNING: they differ in fields beyond the reliability switches, "
              "so the contrast is not single-factor.")

    # ---- 2. the comparison, on ER-SCHED's available seeds ----
    print("\n" + "=" * 70)
    print("2. THE CONFOUND TEST")
    print("=" * 70)

    sched = {s: v for s in args.seeds if (v := _hf_auc("ER-SCHED", s)) is not None}
    matched = sorted(sched, key=lambda s: args.seeds.index(s))
    print(f"  ER-SCHED seeds available: {matched}\n")

    rows = {}
    for variant in ("ER-SCHED", "ER-DEN", "ER-EXT", "ER-NOONT", "ER-NOONTW"):
        vals = [_hf_auc(variant, s) for s in matched]
        if all(v is not None for v in vals):
            rows[variant] = vals
            print(f"  {variant:12s} {_summary(vals)}   per-seed="
                  f"{[round(v, 4) for v in vals]}")

    baselines = {}
    for variant in ("E4-InfoNCE", "E4-OSCAR-Skill"):
        local = _local_auc(variant)
        if local:
            baselines[variant] = local
            print(f"  {variant:12s} {_summary(list(local.values()))}   "
                  f"[local, single-phase]")

    # ---- 3. decomposition ----
    print("\n" + "=" * 70)
    print("3. DECOMPOSITION")
    print("=" * 70)
    if "ER-DEN" in rows and "ER-SCHED" in rows:
        print(f"  reliability effect   ER-DEN - ER-SCHED : "
              f"{_paired(rows['ER-DEN'], rows['ER-SCHED'])}")
    for name, local in baselines.items():
        shared = [s for s in matched if s in local]
        if shared and "ER-SCHED" in rows:
            a = [sched[s] for s in shared]
            b = [local[s] for s in shared]
            print(f"  schedule effect      ER-SCHED - {name:14s}: {_paired(a, b)}"
                  f"  [seeds {shared}]")
        elif local and "ER-SCHED" in rows:
            delta = st.mean(rows["ER-SCHED"]) - st.mean(list(local.values()))
            print(f"  schedule effect      ER-SCHED - {name:14s}: "
                  f"Δ={delta:+.4f} (unpaired, disjoint seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
