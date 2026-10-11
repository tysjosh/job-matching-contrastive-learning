#!/usr/bin/env python3
"""Resumable training driver for the 4 + 15 new runs (protocol_trials_fusion.md §6.3).

Issues exactly the §6.3 ``run_learning_curve_point.py`` commands in the fixed §3b order
(2022: F 25 then 100, each S 87 then 123; 2021: k 0..4, each S 13, 21, 42), verifies
gate 3 per run and appends to the append-only ``run_ledger.json``. On restart it skips
runs with a ``passed``/``excluded`` entry and treats an output dir without a passing
entry as a failed attempt (renamed ``.failed1``/``.failed2``; two failures -> excluded).

Runs are deliberately not sharded across processes or GPUs: the order is part of the
condition because the cache copy grows across runs (§3b). ``--max-runs N`` stops after
N attempted runs so the grid can be advanced in chunks; a later call resumes in order.

``--dev-local`` (Mac smoke only): reads ``dev_pins.json``, allows ``--device cpu|auto``,
``--epochs`` and ``--runs``, and checks the device block against the resolved device.
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
from trials_domain import fusion_confirm as fc  # noqa: E402

RUNNER = "scripts/run_learning_curve_point.py"


def interpreter() -> str:
    """Repo-relative interpreter for the recorded command (no machine paths in the ledger)."""
    try:
        return Path(sys.executable).relative_to(fc.ROOT).as_posix()
    except ValueError:
        return "python"


def exec_command(command: List[str]) -> List[str]:
    """The recorded command with its interpreter swapped for this process's own."""
    return [sys.executable] + list(command[1:])


def plan(pins: Dict[str, Any], cohort: str, device: str, epochs=None) -> List[Dict[str, Any]]:
    """The fixed-order run list with the exact §6.3 commands."""
    derived = pins["configs"]["derived"]

    def cfg(name: str) -> str:
        hits = [p for p in derived if p.endswith("/" + name)]
        if len(hits) != 1:
            raise fc.ProtocolError(f"derived config {name} not pinned")
        return hits[0]

    runs = []
    order = pins["analysis"]["training_order"]
    if cohort == "2022":
        for r in pins["cohort2022"]["new_runs"]:
            F, S = r["fraction"], r["seed"]
            rid = f"2022:f{F}_s{S}"
            runs.append({"run": rid, "cohort": "2022", "fraction": F, "seed": S, "output_dir": r["output_dir"],
                         "config": cfg("lc_trials_baseline_cache.json"),
                         "train_file": f"preprocess/trec_ct_lc/frac_{F}/train.jsonl",
                         "validation_file": "preprocess/trec_ct_lc/validation_positive.jsonl",
                         "fraction_arg": {25: "0.25", 100: "1.0"}[F],
                         "expected_unexpected": derived[cfg("lc_trials_baseline_cache.json")]["expected_unexpected_fields"]})
    else:
        cv = pins["cohort2021"]
        for k in range(cv["n_folds"]):
            for S in cv["seeds"]:
                c = cfg(f"lc_trials_baseline_cv2021_fold{k}.json")
                runs.append({"run": f"2021:k{k}_s{S}", "cohort": "2021", "k": k, "seed": S,
                             "output_dir": f"{cv['run_dirs'][f'fold_{k}']}/trials_baseline_full_s{S}",
                             "config": c, "train_file": f"{cv['fold_dir']}/fold_{k}/train.jsonl",
                             "validation_file": f"{cv['fold_dir']}/fold_{k}/validation_positive.jsonl",
                             "fraction_arg": "1.0", "expected_unexpected": derived[c]["expected_unexpected_fields"]})
    for r in runs:
        r["order_index"] = order.index(r["run"])
        r["command"] = [interpreter(), RUNNER, "--domain", "trials", "--config", r["config"],
                        "--compare-config", fc.BASE_CONFIG_REL, "--train-file", r["train_file"],
                        "--validation-file", r["validation_file"], "--fraction", r["fraction_arg"],
                        "--seed", str(r["seed"]), "--device", device, "--output-dir", r["output_dir"]]
        if epochs is not None:
            r["command"] += ["--epochs", str(epochs)]
    return sorted(runs, key=lambda r: r["order_index"])


def cache_state(path: str) -> Dict[str, Any]:
    p = fc.absp(path)
    if not p.exists():
        return {"entries": None, "sha256": None}
    return {"entries": len(fc.load_frozen_cache(p)), "sha256": fc.file_sha256(p)}


def make_attempt(pins, results_dir: str, gpu_name, require_device: str):
    cache_copy = f"{results_dir}/cache/trials_text_embeddings.pt"
    original_sha = pins["inputs"][fc.ORIGINAL_CACHE_REL]
    logs = fc.absp(results_dir) / "train_logs"

    def attempt(run: Dict[str, Any], n: int) -> Dict[str, Any]:
        env = dict(os.environ, PYTHONHASHSEED="0", HF_HUB_OFFLINE="1")
        before = cache_state(cache_copy)
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{run['run'].replace(':', '_')}.attempt{n}.log"
        fc.refuse_existing(log_path)
        t0 = time.time()
        print(f"[{run['order_index']}] {run['run']} attempt {n}: {' '.join(run['command'])}", flush=True)
        with open(log_path, "w", encoding="utf-8") as handle:
            code = subprocess.call(exec_command(run["command"]), cwd=fc.ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        after = cache_state(cache_copy)
        original_ok = fc.file_sha256(fc.ORIGINAL_CACHE_REL) == original_sha
        fails = fc.gate3_failures(run["output_dir"], code, run["expected_unexpected"], gpu_name,
                                  original_ok, require_device)
        out = fc.absp(run["output_dir"])
        manifest = json.loads((out / "lc_manifest.json").read_text()) if (out / "lc_manifest.json").exists() else {}
        entry = {"run": run["run"], "order_index": run["order_index"], "attempt": n,
                 "status": "passed" if not fails else "failed", "gate3_failures": fails,
                 "command": run["command"], "exit_code": code, "output_dir": run["output_dir"],
                 "best_checkpoint_sha256": fc.file_sha256(out / "best_checkpoint.pt") if (out / "best_checkpoint.pt").exists() else None,
                 "lc_manifest_sha256": fc.file_sha256(out / "lc_manifest.json") if (out / "lc_manifest.json").exists() else None,
                 "single_factor_audit_unexpected_fields": (manifest.get("single_factor_audit") or {}).get("unexpected_fields"),
                 "device": manifest.get("device"), "wall_time_s": round(wall, 1),
                 "cache_copy": {"entries_before": before["entries"], "sha256_before": before["sha256"],
                                "entries_after": after["entries"], "sha256_after": after["sha256"]},
                 "original_cache_unchanged": original_ok, "PYTHONHASHSEED": env["PYTHONHASHSEED"],
                 "log": fc.rel(log_path), "time_utc": fc.utc_now()}
        print(f"    -> {entry['status']} in {wall:.0f}s {fails if fails else ''}", flush=True)
        return entry

    return attempt


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pins", required=True, type=Path)
    ap.add_argument("--cohort", required=True, choices=["2022", "2021"])
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and ledger state only")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dev-local", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="--dev-local only")
    ap.add_argument("--runs", default=None, help="--dev-local only: comma-separated run ids")
    args = ap.parse_args(argv)
    fc.set_root(args.root)

    pins = fc.load_verified_pins(args.pins)
    if pins.get("confirmatory") is not (not args.dev_local):
        raise fc.ProtocolError("dev pins are only usable with --dev-local, and --dev-local only with dev pins")
    if not args.dev_local and (args.device != "cuda" or args.epochs is not None or args.runs):
        raise fc.ProtocolError("confirmatory training is --device cuda, full epochs, all runs (§6.3)")
    if args.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise fc.ProtocolError('CUDA_VISIBLE_DEVICES="" hides the GPU; training runs with no override (§6)')
    bad = fc.verify_pinned_hashes(pins, ("inputs", "code", "configs"))
    if bad:
        raise fc.ProtocolError(f"pinned hash mismatch, refusing to train: {bad[:10]}")
    results_dir = pins["results_dir"]
    if args.dev_local:
        fc.require_dev_local(results_dir)
    require = args.device
    if args.dev_local and args.device == "auto":
        import torch
        require = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = pins["environment"].get("gpu_name")
    ledger = fc.Ledger(f"{results_dir}/run_ledger.json")
    if args.cohort == "2021" and not args.dev_local:
        if not ledger.path.exists():
            raise fc.ProtocolError("run --cohort 2022 first (it creates the ledger)")
        if not args.dev_local:
            for r in plan(pins, "2022", args.device):
                st = ledger.status(r["run"])
                if not (st["passed"] or st["excluded"]):
                    raise fc.ProtocolError(f"cohort 2022 run {r['run']} not finished; fixed order (§3b)")
    if not ledger.path.exists() and not args.dry_run and (args.cohort == "2022" or args.dev_local):
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        ledger.path.write_text(fc.Ledger._render([]), encoding="utf-8")

    runs = plan(pins, args.cohort, args.device, args.epochs)
    if args.runs:
        keep = set(args.runs.split(","))
        runs = [r for r in runs if r["run"] in keep]
    if args.dry_run:
        for r in runs:
            print(r["run"], ledger.status(r["run"]) if ledger.path.exists() else "-", " ".join(r["command"]))
        return 0
    attempt = make_attempt(pins, results_dir, gpu_name, require)
    attempted = 0
    summary = {}
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
