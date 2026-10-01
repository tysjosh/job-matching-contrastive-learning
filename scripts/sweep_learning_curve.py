#!/usr/bin/env python3
"""Sweep a learning curve: arms x fractions x seeds, one training run per point.

Runs points sequentially in subprocesses. Sequential is deliberate: these are
CPU-bound frozen-encoder runs on one machine, so parallelism would trade wall
clock for contention, and the trials label-budget queue is already occupying
cores.

Resumable. A point whose output directory already holds ``lc_manifest.json`` is
skipped, so an interrupted sweep can be relaunched without repeating work and
without a shell wrapper.

Seeds matter here more than usual: the existing career learning-curve runs were
all unseeded (``training_seed: null``), which is why their single-run deltas
(+0.0226 -> +0.0076 -> +0.0153 -> +0.0187 -> -0.0004, non-monotonic) cannot be
separated from the +/-0.013 seed spread measured on the 5-seed full-data
comparison. Three seeds per point is the minimum that lets the curve be read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "scripts" / "run_learning_curve_point.py"

ARMS: Dict[str, Dict[str, Path]] = {
    "career": {
        "baseline": ROOT / "config" / "lc_career_baseline.json",
        # Ontology-guided NEGATIVE SELECTION. Came back a clean null across
        # 5/10/15/20% with 3 seeds: pooled delta -0.0007, t=-0.163, p=0.87.
        "ontneg": ROOT / "config" / "lc_career_ontneg_only.json",
        # Sample-level ontology WEIGHTING, negatives left random. Also null:
        # pooled delta +0.0039, t=+0.933, p=0.37.
        "ontweight": ROOT / "config" / "lc_career_ontweight_only.json",
        # All four mechanisms WITH rank tiers. Pooled +0.0051, t=1.44, p=0.18.
        # Not a faithful reproduction of April: rank tiers postdate those runs.
        "ontfull": ROOT / "config" / "lc_career_ontfull.json",
        # Faithful reproduction of April: absolute tiers, so the hard bucket is
        # empty (measured: 0/12,000 pairs at d<=0.3, 98.3% land in 'easy').
        "ontfull_abs": ROOT / "config" / "lc_career_ontfull_abs.json",
    },
    "trials": {
        "baseline": ROOT / "config" / "lc_trials_baseline.json",
        "ontneg": ROOT / "config" / "lc_trials_ontneg_only.json",
        # Diversity-matched pair: both arms use a fixed 34% tier window and
        # resample within it each epoch; only the method choosing that window
        # (random versus MeSH-nearest) differs.
        "randwin": ROOT / "config" / "lc_trials_randwin.json",
        "ontneg_stoch": ROOT / "config" / "lc_trials_ontneg_stoch.json",
    },
    # The HIGH-CEILING arm. Career (ESCO 0.558 / ISCO 0.485), trials (MeSH
    # 0.526-0.607) and patents (CPC 0.527) all sit near chance on their hard
    # contrast, so a null there cannot separate "injection does not work" from
    # "this ontology has no signal". GO measures 0.8195 on the same style of
    # contrast, and a smoke checkpoint puts the MODEL at 0.7584 on it — below the
    # ceiling, so there is real headroom for the ontology to close. That headroom
    # is the precondition the career and trials runs never had.
    "go_ppi": {
        "baseline": ROOT / "config" / "lc_go_ppi_baseline.json",
        # DETERMINISTIC prefix of the GO-ordered pool (the trials design).
        # Confounded: also removes per-epoch negative variety. Measured
        # -0.0092 on the hard contrast, 0/3 seeds improved.
        "ontneg": ROOT / "config" / "lc_go_ppi_ontneg_only.json",
        # STOCHASTIC draw from the GO-closest tercile. This is the arm that
        # isolates ontology ordering from the diversity confound.
        "ontneg_stoch": ROOT / "config" / "lc_go_ppi_ontneg_stoch.json",
        # DIVERSITY CONTROL: a RANDOM window of the same width, no GO. Measured
        # negative diversity over 15 epochs, unique negatives per anchor from a
        # ~54-candidate pool: baseline 46.4, randwin 18.3, ontneg_stoch 18.4,
        # ontneg 10.0. randwin and ontneg_stoch are matched, so the difference
        # between THEM is the ontology and nothing else.
        "randwin": ROOT / "config" / "lc_go_ppi_randwin.json",
    },
}

#: Per-domain default learning-curve directory (the ``frac_<pct>`` parent).
LC_DIRS = {
    "career": ROOT / "preprocess" / "learning_curve_v7",
    "trials": ROOT / "preprocess" / "trec_ct_lc",
    "go_ppi": ROOT / "preprocess" / "go_ppi_lc",
}


def career_files(lc_dir: Path, pct: int):
    d = lc_dir / f"frac_{pct}"
    # Trials and GO/PPI evaluation splits contain all three relevance grades for
    # downstream AUC/ranking evaluation.  Contrastive validation loss, however,
    # expects positive anchors and obtains their negatives through the domain
    # selector.  Feeding grade-0/1 records directly would treat those negative
    # pairs as positives and select checkpoints on the wrong objective.
    positive_validation = lc_dir / "validation_positive.jsonl"
    validation = (positive_validation if positive_validation.exists()
                  else d / "validation.jsonl")
    return d / "train.jsonl", validation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", default="career", choices=sorted(ARMS))
    ap.add_argument("--lc-dir", type=Path, default=None,
                    help="the frac_<pct> parent; defaults per domain via LC_DIRS")
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT / "results" / "lc_single_factor")
    ap.add_argument("--fractions", nargs="+", type=int,
                    default=[10, 25, 50, 75, 100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 21])
    ap.add_argument("--arms", nargs="+", default=["baseline", "ontneg"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.lc_dir is None:
        args.lc_dir = LC_DIRS[args.domain]
    configs = ARMS[args.domain]
    # Order: fraction outermost, then seed, then arm. Keeping the two arms of a
    # (fraction, seed) cell adjacent means an interrupted sweep still yields
    # complete PAIRS, which are what the contrast needs — a sweep ordered by arm
    # would finish every baseline before any ontology run and leave nothing
    # comparable if stopped early.
    points: List[Dict] = []
    for pct in args.fractions:
        for seed in args.seeds:
            for arm in args.arms:
                points.append({"pct": pct, "seed": seed, "arm": arm})

    print(f"domain={args.domain}  points={len(points)}  "
          f"(arms={args.arms} fractions={args.fractions} seeds={args.seeds})")

    done = skipped = failed = 0
    for i, point in enumerate(points, 1):
        pct, seed, arm = point["pct"], point["seed"], point["arm"]
        out = args.results_dir / f"{args.domain}_{arm}_f{pct}_s{seed}"
        tag = f"[{i}/{len(points)}] {args.domain} {arm} frac={pct}% seed={seed}"

        if (out / "lc_manifest.json").exists():
            print(f"{tag}  SKIP (already complete)")
            skipped += 1
            continue

        train, val = career_files(args.lc_dir, pct)
        if not train.exists():
            print(f"{tag}  SKIP (missing {train})")
            skipped += 1
            continue

        # Audit the contrast the caller actually requested.  The historical
        # baseline/ontneg default remains unchanged, while pairs such as the
        # diversity-matched randwin/ontneg_stoch GO contrast no longer get
        # audited against an unrelated third arm.
        if len(args.arms) == 2:
            other = args.arms[1] if arm == args.arms[0] else args.arms[0]
        else:
            other = "ontneg" if arm == "baseline" else "baseline"
        cmd = [
            sys.executable, str(RUNNER),
            "--domain", args.domain,
            "--config", str(configs[arm]),
            "--compare-config", str(configs[other]),
            "--train-file", str(train),
            "--validation-file", str(val),
            "--output-dir", str(out),
            "--seed", str(seed),
            "--fraction", str(pct / 100.0),
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]

        if args.dry_run:
            print(f"{tag}\n    {' '.join(cmd)}")
            continue

        out.mkdir(parents=True, exist_ok=True)
        log = out / "train.log"
        print(f"{tag}  running -> {log}", flush=True)
        started = time.time()
        with open(log, "w", encoding="utf-8") as handle:
            rc = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT))
        mins = (time.time() - started) / 60
        if rc == 0 and (out / "lc_manifest.json").exists():
            print(f"{tag}  OK in {mins:.1f} min", flush=True)
            done += 1
        else:
            print(f"{tag}  FAILED rc={rc} after {mins:.1f} min — see {log}",
                  flush=True)
            failed += 1

    print(f"\nsweep finished: {done} run, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
