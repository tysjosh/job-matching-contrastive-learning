"""Gate G4 (remote GPU): our trainer reproduces DGZ's MLP and DeepGOZero (protocol
§10 G4, design §7, plan item 13).

--write-specs   writes the four gate run specs (BP/MF x V0/V4; InterPro, DGZ-rule
                labels = condition DGZ/all, selection best_valid_loss, <= 256 epochs,
                lambda 1, seed 42, DGZ's EL formulas) into
                results/go_function/specs/gates/. --fallback adds seeds 13 and 21 of
                the same cells; --diagnostic adds V4 with the corrected NF4 radius
                (protocol D10, reported only). Specs are write-or-verify.
(default)       compare: reads eval.json (``dgz_faithful``: class AUC, Fmax) of each
                seed-42 gate run and checks it against config/go_function/dgz_published.json
                (V0 = MLP, V4 = DeepGOZero) within +-0.01. A cell that misses passes if
                the mean over seeds 42, 13 and 21 is within +-0.01 (3-seed fallback).
                Writes results/go_function/gates/gate_trainer.json and pin G4 when every
                cell passes.

Exit codes: 0 ok; 2 fatal; 3 miss (protocol stop: ask the user); 4 runs missing
(e.g. the seed-13/21 fallback runs are needed but not yet run).

Usage (the runs themselves are remote: run.py --spec <spec> --device cuda):
  .venv/bin/python scripts/go_function/gate_trainer.py --write-specs
  .venv/bin/python scripts/go_function/gate_trainer.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_REFUSED, EXIT_STOP  # noqa: E402
from go_function.pins import (canonical_json, record_pin, sha256_bytes,  # noqa: E402
                              write_or_verify_file)
from go_function.runtime import FatalError, repo_path, repo_rel, setup_logging  # noqa: E402

SPEC_DIR = ROOT / "results" / "go_function" / "specs" / "gates"
RUN_DIR = "results/go_function/gates/runs"
GATE_OUT = ROOT / "results" / "go_function" / "gates" / "gate_trainer.json"
PUBLISHED = ROOT / "config" / "go_function" / "dgz_published.json"
CELLS = (("bp", "V0"), ("bp", "V4"), ("mf", "V0"), ("mf", "V4"))
PUBLISHED_MODEL = {"V0": "mlp", "V4": "deepgozero"}
METRICS = ("auc", "fmax")
SEED, FALLBACK_SEEDS = 42, (13, 21)
TOL = 0.01
EPOCHS = 256


def run_name(aspect: str, variant: str, seed: int, nf4_radius: str = "dgz") -> str:
    return f"{aspect}_{variant}" + ("_nf4class" if nf4_radius == "class" else "") + f"_s{seed}"


def gate_spec(study: dict, aspect: str, variant: str, seed: int, nf4_radius: str = "dgz") -> dict:
    return {"aspect": aspect, "feature": "interpro", "variant": variant, "regime": "DGZ",
            "rule": "W2", "retention_draw": 1, "pack": None, "k": "all", "seed": seed,
            "epochs": EPOCHS, "lambda_el": None if variant == "V0" else float(study["lambda_el"]),
            "batch_size": study["batch_size"], "lr": study["lr"],
            "milestones": list(study["milestones"]), "gamma": study["gamma"],
            "selection": "best_valid_loss",
            "out_dir": f"{RUN_DIR}/{run_name(aspect, variant, seed, nf4_radius)}",
            "mode": "gate", "nf4_radius": nf4_radius}


def write_specs(study: dict, fallback: bool, diagnostic: bool, spec_dir: Path, log) -> dict:
    from go_function.train import validate_spec
    items = [(a, v, SEED, "dgz") for a, v in CELLS]
    if fallback:
        items += [(a, v, s, "dgz") for a, v in CELLS for s in FALLBACK_SEEDS]
    if diagnostic:
        items += [(a, "V4", SEED, "class") for a in ("bp", "mf")]
    out = {}
    for a, v, s, rad in items:
        spec = gate_spec(study, a, v, s, rad)
        validate_spec(spec, study)                     # the specs run.py will accept
        sub = "diagnostic/" if rad == "class" else ""
        path = spec_dir / f"{sub}{run_name(a, v, s, rad)}.json"
        try:
            key = repo_rel(path)
        except ValueError:                             # tests: spec_dir outside the repo
            key = path.relative_to(spec_dir).as_posix()
        out[key] = write_or_verify_file(path, canonical_json(spec))
        log.info("%s: %s", key, out[key])
    return out


def read_eval(out_dir: str):
    d = repo_path(out_dir)
    if not (d / "DONE.json").exists():
        return None
    ev = json.loads((d / "eval.json").read_text())
    if "dgz_faithful" not in ev:
        raise FatalError(f"{out_dir}/eval.json has no dgz_faithful section (not a gate run)")
    return ev["dgz_faithful"]


def compare(study: dict, published: dict, reader=read_eval) -> dict:
    """Per cell: seed-42 values vs published; the 3-seed mean when seed 42 misses."""
    cells = {}
    for a, v in CELLS:
        pub = {m: float(published["values"][a][PUBLISHED_MODEL[v]][m]["value"]) for m in METRICS}
        runs = {s: reader(gate_spec(study, a, v, s)["out_dir"]) for s in (SEED,) + FALLBACK_SEEDS}
        e = {"published": pub, "published_model": PUBLISHED_MODEL[v]}
        if runs[SEED] is None:
            e.update({"status": "missing", "missing_seeds": [SEED]})
            cells[f"{a}_{v}"] = e
            continue
        s42 = {m: float(runs[SEED][m]) for m in METRICS}
        d42 = {m: s42[m] - pub[m] for m in METRICS}
        e.update({"seed42": s42, "diff_seed42": d42})
        if all(abs(d) <= TOL + 1e-12 for d in d42.values()):
            e["status"] = "pass"
        else:
            miss = [s for s in FALLBACK_SEEDS if runs[s] is None]
            if miss:
                e.update({"status": "needs_fallback", "missing_seeds": miss})
            else:
                mean = {m: sum(float(runs[s][m]) for s in (SEED,) + FALLBACK_SEEDS) / 3
                        for m in METRICS}
                dm = {m: mean[m] - pub[m] for m in METRICS}
                e.update({"mean_3_seeds": mean, "diff_mean_3_seeds": dm,
                          "status": "pass_3_seed" if all(abs(d) <= TOL + 1e-12 for d in dm.values())
                          else "miss"})
        cells[f"{a}_{v}"] = e
    return cells


def main(argv=None, *, spec_dir: Path = None, out_path: Path = None, pins_path: Path = None,
         study_path: Path = None, reader=read_eval) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-specs", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="also seeds 13 and 21")
    ap.add_argument("--diagnostic", action="store_true", help="also V4 with nf4_radius 'class'")
    args = ap.parse_args(argv)
    from go_function.train import load_study
    log = setup_logging("gate_trainer")
    try:
        study = load_study(study_path)
        if args.write_specs:
            res = write_specs(study, args.fallback, args.diagnostic,
                              Path(spec_dir) if spec_dir is not None else SPEC_DIR, log)
            print(json.dumps(res, sort_keys=True))
            return EXIT_OK
        published = json.loads(PUBLISHED.read_text())
        cells = compare(study, published, reader)
        for k, e in cells.items():
            log.info("%s: %s %s", k, e["status"], e.get("diff_seed42", ""))
        status = {e["status"] for e in cells.values()}
        if status & {"missing", "needs_fallback"}:
            log.error("gate runs missing: %s", {k: e["missing_seeds"] for k, e in cells.items()
                                                if "missing_seeds" in e})
            return EXIT_REFUSED
        out_path = Path(out_path) if out_path is not None else GATE_OUT
        doc = {"schema": 1, "gate": "G4", "tolerance": TOL, "metrics": list(METRICS),
               "cells": cells, "passed": status <= {"pass", "pass_3_seed"}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not doc["passed"]:
            out_path.with_name(f"{out_path.stem}.FAILED.json").write_bytes(canonical_json(doc))
            log.error("G4 miss: %s", sorted(k for k, e in cells.items() if e["status"] == "miss"))
            return EXIT_STOP
        b = canonical_json(doc)
        log.info("%s: %s", out_path.name, write_or_verify_file(out_path, b))
        log.info("pin G4: %s", record_pin("G4", {"gate_trainer.json": sha256_bytes(b)},
                                          pins_path=pins_path))
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
