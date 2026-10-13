#!/usr/bin/env python3
"""Resumable training driver for the 34 new runs (protocol_go_ppi_fusion.md §3b, §3c, §5.3, §6.3).

Issues exactly the §6.3 ``run_learning_curve_point.py --domain go_ppi`` commands in the
fixed order (cohort A: F 10, 25, 50, 75, 100, seeds 42, 13, 21, then 87, 123 at F 25 and 100;
cohort B: k 0..4, seeds 13, 21, 42), with ``--compare-config config/lc_go_ppi_baseline.json``
and ``--device cuda``, sets ``PYTHONHASHSEED=0`` in the child, applies gate 3 to each run and
appends to the append-only ``run_ledger.json``. On restart it skips runs with a
``passed``/``excluded`` entry and treats an output dir without a passing entry as a failed
attempt (renamed ``.failed1``; a second failure -> excluded).

Runs are never sharded: the order is part of the condition because the cache copy grows.
``--max-runs N`` stops after N attempted runs; a later call resumes in order.
``--dry-run`` prints the plan and ledger state and trains nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402

RUNNER = "scripts/run_learning_curve_point.py"


def interpreter() -> str:
    """Repo-relative interpreter for the recorded command (no machine paths in the ledger)."""
    try:
        return Path(sys.executable).relative_to(fc.ROOT).as_posix()
    except ValueError:
        return "python"


def plan(pins: Dict[str, Any], cohort: str, device: str = "cuda") -> List[Dict[str, Any]]:
    """The fixed-order run list with the exact §6.3 commands."""
    derived = pins["configs"]["derived"]
    order = pins["analysis"]["training_order"]
    rows = pins["cohortA"]["new_runs"] if cohort == "A" else pins["cohortB"]["runs"]
    runs = []
    for r in rows:
        if r["config"] not in derived:
            raise fc.ProtocolError(f"derived config {r['config']} not pinned")
        run = dict(r, cohort=cohort, order_index=order.index(r["run"]),
                   expected_unexpected=derived[r["config"]]["expected_unexpected_fields"])
        run["command"] = [interpreter(), RUNNER, "--domain", "go_ppi", "--config", r["config"],
                          "--compare-config", fc.BASE_CONFIG_REL, "--train-file", r["train_file"],
                          "--validation-file", r["validation_file"], "--fraction", r["fraction_arg"],
                          "--seed", str(r["seed"]), "--device", device, "--output-dir", r["output_dir"]]
        runs.append(run)
    return sorted(runs, key=lambda r: r["order_index"])


def cache_state(path: str) -> Dict[str, Any]:
    p = fc.absp(path)
    if not p.exists():
        return {"entries": None, "sha256": None}
    return {"entries": len(fc.load_frozen_cache(p)), "sha256": fc.file_sha256(p)}


def originals_ok(pins: Dict[str, Any]) -> bool:
    return all(fc.file_sha256(p) == pins["inputs"][p] for p in (fc.ORIGINAL_CACHE_REL, fc.MASKED_CACHE_REL))


def make_attempt(pins: Dict[str, Any]):
    results_dir = pins["results_dir"]
    gpu_name = pins["environment"]["gpu_name"]
    logs = fc.absp(results_dir) / "train_logs"

    def attempt(run: Dict[str, Any], n: int) -> Dict[str, Any]:
        env = dict(os.environ, PYTHONHASHSEED="0", HF_HUB_OFFLINE="1")
        before = cache_state(fc.CACHE_COPY_REL)
        logs.mkdir(parents=True, exist_ok=True)
        log_path = fc.refuse_existing(logs / f"{run['run'].replace(':', '_')}.attempt{n}.log")
        t0 = time.time()
        print(f"[{run['order_index']}] {run['run']} attempt {n}: {' '.join(run['command'])}", flush=True)
        with open(log_path, "w", encoding="utf-8") as handle:
            code = subprocess.call([sys.executable] + run["command"][1:], cwd=fc.ROOT, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        after = cache_state(fc.CACHE_COPY_REL)
        ok = originals_ok(pins)
        fails = fc.gate3_failures(run["output_dir"], code, run["expected_unexpected"], gpu_name, ok)
        out = fc.absp(run["output_dir"])
        mpath, ckpt = out / "lc_manifest.json", out / "best_checkpoint.pt"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        entry = {"run": run["run"], "order_index": run["order_index"], "attempt": n,
                 "status": "passed" if not fails else "failed", "gate3_failures": fails,
                 "command": run["command"], "exit_code": code, "output_dir": run["output_dir"],
                 "best_checkpoint_sha256": fc.file_sha256(ckpt) if ckpt.exists() else None,
                 "lc_manifest_sha256": fc.file_sha256(mpath) if mpath.exists() else None,
                 "unexpected_fields": (manifest.get("single_factor_audit") or {}).get("unexpected_fields"),
                 "device": manifest.get("device"), "wall_time_s": round(wall, 1),
                 "cache_copy": {"entries_before": before["entries"], "sha256_before": before["sha256"],
                                "entries_after": after["entries"], "sha256_after": after["sha256"]},
                 "original_caches_unchanged": ok, "PYTHONHASHSEED": env["PYTHONHASHSEED"],
                 "log": fc.rel(log_path), "time_utc": fc.utc_now()}
        print(f"    -> {entry['status']} in {wall:.0f}s {fails if fails else ''}", flush=True)
        return entry

    return attempt


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pins", required=True, type=Path)
    ap.add_argument("--cohort", required=True, choices=["A", "B"])
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and ledger state only")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    pins = fc.load_verified_pins(args.pins)
    if not pins.get("confirmatory"):
        raise fc.ProtocolError("train.py runs only under confirmatory pins")
    ledger = fc.Ledger(f"{pins['results_dir']}/run_ledger.json")
    runs = plan(pins, args.cohort, "cuda")
    if args.dry_run:
        for r in runs:
            print(r["run"], ledger.status(r["run"]) if ledger.path.exists() else "-", " ".join(r["command"]))
        return 0
    if args.device != "cuda":
        raise fc.ProtocolError("confirmatory training is --device cuda (§6.3)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise fc.ProtocolError('CUDA_VISIBLE_DEVICES="" hides the GPU; training runs without the mask (§6)')
    bad = fc.verify_pinned_hashes(pins, ("inputs", "code", "configs", "derived"))
    if bad:
        raise fc.ProtocolError(f"pinned hash mismatch, refusing to train: {bad[:10]}")
    if args.cohort == "B":
        if not ledger.path.exists():
            raise fc.ProtocolError("run --cohort A first (it creates the ledger)")
        for r in plan(pins, "A"):
            st = ledger.status(r["run"])
            if not (st["passed"] or st["excluded"]):
                raise fc.ProtocolError(f"cohort A run {r['run']} not finished; fixed order (§3b)")
    elif not ledger.path.exists():
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        ledger.path.write_text(fc.Ledger._render([]), encoding="utf-8")
    attempt = make_attempt(pins)
    attempted, summary = 0, {}
    for r in runs:
        if args.max_runs is not None and attempted >= args.max_runs:
            break
        result = fc.run_with_resume(ledger, r, attempt)
        summary[r["run"]] = result
        if not result.startswith("skipped"):
            attempted += 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
