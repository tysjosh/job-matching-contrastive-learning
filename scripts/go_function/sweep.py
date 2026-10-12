"""Resumable sweep driver for the GO-function grid (design §12.3, §13.1; protocol
§10 G8-G10; plan item 15). Modelled on scripts/go_withholding/sweep.py (read, not
imported).

Subcommands::

    sweep.py plan    [--smoke [--feature F]] [filters] [--shard i/N]
    sweep.py run     --max-runs M [--workers N] [--device auto|cuda|cpu] [filters]
                     [--shard i/N] [--smoke [--feature F]] [--dry-run]
    sweep.py status  [--release-stale] [--smoke [--feature F]] [--shard i/N]
    sweep.py rescore --max-runs M [--device ...] [--smoke [--feature F]]
    sweep.py pin     [--accept-content-pin]

filters: --aspect bp|mf, --feature esm2_650m|interpro, --seeds ..., --variants ...

Grid. The specs written by materialize.py (results/go_function/specs/<aspect>/
<feature>/index.json) in seed-major order (all cells of seed 13 first, then 21,
42, 87, 123; within a seed: bp before mf, esm2_650m before interpro, index
order). Run dir: results/go_function/runs/<aspect>/<feature>/<run_name>/ with
run_name = <variant>__<regime>__<rule>__<pack|all>__k<k>[__d2]__s<seed>.
``--shard i/N`` keeps run index r (0-based, over the FULL ordered grid,
independent of filters) iff r % N == i - 1, so shards are disjoint and
exhaustive; it needs the spec indexes of all four (aspect, feature) pairs.

Runs. Each run is a ``scripts/go_function/run.py --spec <spec> --device <d>``
subprocess (argument list, cwd = repo root); --device is forwarded. ``--workers N``
(default 4) runs up to N subprocesses at once on the one GPU; ``--max-runs M``
(required) bounds the runs started by one invocation, so the full grid is a
repeated explicit decision.

- Claim: ``<run_dir>.CLAIM`` (a sibling file, because run.py refuses a non-empty
  out dir) created with os.open(O_CREAT | O_EXCL) and holding pid and host. A live
  claim (or one from another host) means the run is in progress elsewhere and it is
  skipped; a claim of a dead pid on this host is stale and released by
  ``status --release-stale`` (the partial dir is renamed <run_dir>.interrupted<n>).
- Resume: a run with DONE.json is never re-run.
- Failures: a failed attempt (FAILED.json stage preflight/train, or a crash) is
  renamed <run_dir>.attempt<n> and retried once; a second failure leaves
  FAILED.json and the run is missing for the analysis. An out-of-memory failure
  (CUDA OOM, or the process killed) lowers the invocation's worker count by one
  before its retry. A FAILED.json with stage eval is left for ``rescore``.
- Refusals: a run refused by the study's state (run.py exit 4, or FAILED.json
  error Refused: norm unset, a pin or the code pin missing) uses no attempt; its
  dir is renamed <run_dir>.refused<n>, the run stays pending and the invocation
  stops (exit 4). PREFLIGHT_STOP (3) consecutive runs that end failed at stage
  preflight also stop the invocation (exit 2), so a systemic preflight error
  (e.g. a pin or code-pin mismatch) cannot consume the grid's attempts.
- Claims record the host's boot id; a same-host claim from an earlier boot is
  stale even if its pid was reused.
- ``rescore`` re-runs only the evaluation (run.py --rescore) of runs whose
  FAILED.json has stage eval, from eval_inputs.npz or a kept checkpoint.
- ``pin`` hashes every .py under go_function/ and scripts/go_function/ and every
  .json under config/go_function/ into results/go_function/code_pin.json and records
  pin G8 (code pin, spec-set hashes, git HEAD). It needs the study code committed
  (clean ``git status`` for those paths; read-only git commands only) or
  ``--accept-content-pin``.
- ``--smoke`` (G9): the cell list BP p01 x k {0, 5} x {V0, V3, V3c} x seed 13, TP/W2.
  With --feature interpro (default; G9a, local CPU): 2 epochs and
  max_train_batches 40, mode smoke. With --feature esm2_650m (G9b, remote): E*
  epochs, plus V4 and V6 (TP/W2, k 0 and 5) and V4 under DGZ with W2, WDGZrel and
  W2drop at k 0. Specs in results/go_function/smoke/specs/, runs in
  results/go_function/smoke/runs/<aspect>/<feature>/. After the runs,
  results/go_function/smoke/smoke_report_<feature>.json records seconds and peak
  memory per run and the projected grid wall time (an estimate); for esm2_650m a
  projection above 10 days is a protocol stop (exit 3).

Every path written is repo-relative; subprocess logs are scrubbed of absolute
paths (results/go_function/<runs|smoke/runs>/logs/).

Exit codes: 0 ok; 2 fatal (bad specs, worker errors); 3 protocol stop (G9b
projection); 4 refused by state (no specs, no code pin, a refused run, dirty tree
for pin).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function.constants import (EXIT_FATAL, EXIT_OK, EXIT_REFUSED, EXIT_STOP,  # noqa: E402
                                   SEEDS)
from go_function.pins import (assert_writable_target, canonical_json, record_pin,  # noqa: E402
                              sha256_bytes, sha256_file, write_or_verify_file)
from go_function.runtime import DEVICES, FatalError, repo_path, repo_rel, setup_logging  # noqa: E402

ASPECTS = ("bp", "mf")
FEATURES = ("esm2_650m", "interpro")
RESULTS = ROOT / "results" / "go_function"
STUDY = ROOT / "config" / "go_function" / "study.json"
RUN_PY = "scripts/go_function/run.py"
SMOKE_INTERPRO = {"epochs": 2, "max_train_batches": 40}
PROJECTION_LIMIT_DAYS = 10.0
PREFLIGHT_STOP = 3          # consecutive runs failed at stage preflight -> stop the invocation
GRID_RUNS = 2782 + 3157

log = setup_logging("sweep")


class Refused(RuntimeError):
    pass


class Stop(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_materialize():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "go_function_materialize", ROOT / "scripts" / "go_function" / "materialize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, obj: Any) -> None:
    assert_writable_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(canonical_json(obj))
    tmp.replace(path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def next_free(path: Path, suffix: str) -> Path:
    n = 1
    while path.with_name(f"{path.name}{suffix}{n}").exists():
        n += 1
    return path.with_name(f"{path.name}{suffix}{n}")


# ================================================================ grid
def load_grid(spec_root: Path = None, require_all: bool = False) -> List[Dict[str, Any]]:
    """Every materialised run, seed-major, with its index over the full grid."""
    spec_root = Path(spec_root) if spec_root is not None else RESULTS / "specs"
    items, missing = [], []
    for ai, a in enumerate(ASPECTS):
        for fi, f in enumerate(FEATURES):
            idx = _read_json(spec_root / a / f / "index.json")
            if idx is None:
                missing.append(f"{a}/{f}")
                continue
            for pos, e in enumerate(idx["specs"]):
                items.append({"name": e["name"], "aspect": a, "feature": f, "seed": e["seed"],
                              "block": e["block"], "variant": e["name"].split("__")[0],
                              "spec": repo_rel(spec_root / a / f / f"{e['name']}.json"),
                              "out_dir": e["out_dir"], "_order": (ai, fi, pos)})
    if require_all and missing:
        raise Refused(f"spec indexes missing for {missing} (materialize.py first)")
    if not items:
        raise Refused("no materialised specs (materialize.py, G8)")
    items.sort(key=lambda r: (SEEDS.index(r["seed"]) if r["seed"] in SEEDS else len(SEEDS),
                              r["_order"]))
    for i, r in enumerate(items):
        r["index"] = i
        del r["_order"]
    return items


def smoke_cells(feature: str) -> List[Tuple[str, str, str, int]]:
    """(variant, regime, rule, k) of the G9 smoke (BP, p01, seed 13)."""
    cells = [(v, "TP", "W2", k) for k in (0, 5) for v in ("V0", "V3", "V3c")]
    if feature == "esm2_650m":
        cells += [(v, "TP", "W2", k) for k in (0, 5) for v in ("V4", "V6")]
        cells += [("V4", "DGZ", r, 0) for r in ("W2", "WDGZrel", "W2drop")]
    return cells


def smoke_grid(feature: str, study: Dict[str, Any] = None, smoke_root: Path = None
               ) -> List[Dict[str, Any]]:
    """Write (or verify) the smoke specs and return their run items."""
    from go_function.train import validate_spec
    mat = _load_materialize()
    study = study if study is not None else json.loads(STUDY.read_text())
    smoke_root = Path(smoke_root) if smoke_root is not None else RESULTS / "smoke"
    if feature == "interpro":
        epochs, mtb = SMOKE_INTERPRO["epochs"], SMOKE_INTERPRO["max_train_batches"]
    else:
        epochs = (study.get("epochs_star") or {}).get(f"bp_{feature}")
        if epochs is None:
            raise Refused(f"E* for bp_{feature} is not pinned (G7) - needed by the G9b smoke")
        mtb = None
    items = []
    for i, (v, regime, rule, k) in enumerate(smoke_cells(feature)):
        name = mat.run_name(v, regime, rule, "p01", k, 13)
        out_dir = repo_rel(smoke_root / "runs" / "bp" / feature / name)
        spec = mat.make_spec(study, aspect="bp", feature=feature, variant=v, regime=regime,
                             rule=rule, pack="p01", k=k, seed=13, epochs=epochs,
                             out_dir=out_dir, mode="smoke", max_train_batches=mtb)
        validate_spec(spec, study)
        path = smoke_root / "specs" / "bp" / feature / f"{name}.json"
        write_or_verify_file(path, canonical_json(spec))
        items.append({"name": name, "aspect": "bp", "feature": feature, "seed": 13,
                      "block": "smoke", "variant": v, "spec": repo_rel(path),
                      "out_dir": out_dir, "index": i})
    return items


def parse_shard(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text or "")
    if not m:
        raise argparse.ArgumentTypeError(f"--shard must be i/N, got {text!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 1 <= i <= n:
        raise argparse.ArgumentTypeError(f"--shard needs 1 <= i <= N, got {text!r}")
    return i, n


def in_shard(index: int, shard: Optional[Tuple[int, int]]) -> bool:
    return shard is None or index % shard[1] == shard[0] - 1


def select(items: Sequence[Dict[str, Any]], *, aspect=None, feature=None, seeds=None,
           variants=None, shard=None) -> List[Dict[str, Any]]:
    return [r for r in items
            if (aspect is None or r["aspect"] == aspect)
            and (feature is None or r["feature"] == feature)
            and (seeds is None or r["seed"] in seeds)
            and (variants is None or r["variant"] in variants)
            and in_shard(r["index"], shard)]


# ================================================================ run state
def claim_path(run_dir: Path) -> Path:
    return run_dir.with_name(f"{run_dir.name}.CLAIM")


def attempt_dirs(run_dir: Path) -> List[Path]:
    return sorted(p for p in run_dir.parent.glob(f"{run_dir.name}.attempt*") if p.is_dir())


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


@functools.lru_cache(maxsize=1)
def boot_id() -> Optional[str]:
    """An id of the current boot of this host (Linux boot_id; macOS kern.boottime),
    or None when it cannot be read. A claim written in an earlier boot is stale even
    if its pid has been reused."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True,
                             timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"sec\s*=\s*(\d+)", out)
    return m.group(1) if m else None


def claim_state(run_dir: Path) -> Optional[str]:
    """None (no claim) | live | stale (this host: dead pid, or a claim from an
    earlier boot)."""
    p = claim_path(run_dir)
    if not p.exists():
        return None
    c = _read_json(p) or {}
    if c.get("host") != socket.gethostname():
        return "live"                     # another host: cannot probe, hands off
    boot = c.get("boot")
    if boot is not None and boot != boot_id():
        return "stale"                    # written before a reboot: the pid may be reused
    return "live" if _pid_alive(c.get("pid", -1)) else "stale"


def run_state(run_dir: Path, own_claim: bool = False) -> str:
    """absent | done | in_progress | stale | failed_eval | failed_once | failed |
    interrupted."""
    run_dir = Path(run_dir)
    if (run_dir / "DONE.json").exists():
        return "done"
    c = None if own_claim else claim_state(run_dir)
    if c == "live":
        return "in_progress"
    if c == "stale":
        return "stale"
    f = _read_json(run_dir / "FAILED.json")
    if f is not None:
        if f.get("stage") == "eval":
            return "failed_eval"
        return "failed" if attempt_dirs(run_dir) else "failed_once"
    if run_dir.exists():
        return "interrupted"
    return "absent"


PENDING = ("absent", "interrupted", "failed_once")


def claim(run_dir: Path) -> bool:
    p = claim_path(run_dir)
    assert_writable_target(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "boot": boot_id(),
                   "started": _now(), "run": run_dir.name}, fh)
    return True


def release(run_dir: Path) -> None:
    try:
        claim_path(run_dir).unlink()
    except FileNotFoundError:
        pass


# ================================================================ subprocesses
_ABS = re.compile(r"/Users/[^\s'\"]*")


def scrub(text: str) -> str:
    text = text.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")
    return _ABS.sub("<path>", text)


def run_subprocess(cmd: List[str], log_path: Path) -> int:
    """Argument-list subprocess (cwd = repo root); output scrubbed into log_path.
    Tests replace this function."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(scrub(line))
            fh.flush()
        return proc.wait()


def runner_cmd(item: Dict[str, Any], device: str, rescore: bool = False) -> List[str]:
    cmd = [sys.executable, RUN_PY, "--spec", item["spec"], "--device", device]
    return cmd + (["--rescore"] if rescore else [])


def logs_dir(item: Dict[str, Any]) -> Path:
    """<runs root>/logs/<aspect>/<feature>/ beside the run dirs."""
    run_dir = repo_path(item["out_dir"])
    return run_dir.parents[2] / "logs" / item["aspect"] / item["feature"]


def is_oom(failed: Optional[dict], rc: int) -> bool:
    if rc in (-9, 137):
        return True
    if not failed:
        return False
    text = f"{failed.get('error', '')} {failed.get('message', '')}".lower()
    return "outofmemory" in text or "out of memory" in text


def execute(item: Dict[str, Any], device: str, max_attempts: int = 2) -> str:
    """One run's lifecycle: done | failed | failed_eval | oom (retry pending) |
    refused | skipped."""
    run_dir = repo_path(item["out_dir"])
    if not claim(run_dir):
        return "skipped"
    try:
        st = run_state(run_dir, own_claim=True)
        if st not in PENDING:
            return "skipped"
        if st == "interrupted":
            os.replace(run_dir, next_free(run_dir, ".interrupted"))
        elif st == "failed_once":
            os.replace(run_dir, next_free(run_dir, ".attempt"))
        while True:
            attempt = len(attempt_dirs(run_dir)) + 1
            t0 = time.time()
            rc = run_subprocess(runner_cmd(item, device),
                                logs_dir(item) / f"{item['name']}.attempt{attempt}.log")
            log.info("%s attempt %d: exit %d in %.1f s", item["name"], attempt, rc, time.time() - t0)
            if (run_dir / "DONE.json").exists():
                return "done"
            failed = _read_json(run_dir / "FAILED.json")
            if rc == EXIT_REFUSED or (failed or {}).get("error") == "Refused":
                # refused by the study's state (missing pin, norm or code pin): not a
                # run failure, so no attempt is used; the run stays pending
                if run_dir.exists():
                    os.replace(run_dir, next_free(run_dir, ".refused"))
                return "refused"
            if failed is None:
                failed = {"stage": "unknown", "error": f"exit code {rc}", "by": "sweep",
                          "message": "run.py ended without DONE.json or FAILED.json"}
                _write_json(run_dir / "FAILED.json", failed)
            if failed.get("stage") == "eval":
                return "failed_eval"
            if attempt >= max_attempts:
                return "failed"
            oom = is_oom(failed, rc)
            os.replace(run_dir, next_free(run_dir, ".attempt"))
            if oom:
                return "oom"
    finally:
        release(run_dir)


def failed_stage(item: Dict[str, Any]) -> Optional[str]:
    return (_read_json(repo_path(item["out_dir"]) / "FAILED.json") or {}).get("stage")


def run_items(todo: Sequence[Dict[str, Any]], device: str, workers: int) -> Dict[str, Any]:
    """Run ``todo`` with up to ``workers`` concurrent subprocesses; an OOM lowers
    the limit by one and re-queues the run for its last attempt. A refused run, or
    PREFLIGHT_STOP consecutive runs that end failed at stage preflight (a systemic
    problem such as a pin mismatch), stops the invocation: no new run is started."""
    queue, retry = deque(todo), deque()
    limit = workers
    results: Dict[str, str] = {}
    errors: List[str] = []
    stop = False
    preflight_streak, preflight_stop = 0, False
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures: Dict[Any, Dict[str, Any]] = {}

        def submit() -> bool:
            if retry:
                item = retry.popleft()
            elif queue:
                item = queue.popleft()
            else:
                return False
            futures[ex.submit(execute, item, device)] = item
            return True

        while len(futures) < limit and submit():
            pass
        while futures:
            finished, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for f in finished:
                item = futures.pop(f)
                try:
                    res = f.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{item['name']}: {type(e).__name__}: {scrub(str(e))}")
                    stop = True
                    continue
                results[item["name"]] = res
                if res == "failed" and failed_stage(item) == "preflight":
                    preflight_streak += 1
                    if preflight_streak >= PREFLIGHT_STOP and not preflight_stop:
                        preflight_stop = True
                        stop = True
                        errors.append(f"{preflight_streak} consecutive runs failed at stage "
                                      f"preflight (last {item['name']}): a study-state problem "
                                      f"is likely; invocation stopped")
                elif res in ("done", "failed", "failed_eval"):
                    preflight_streak = 0
                if res == "oom":
                    limit = max(1, limit - 1)
                    log.warning("%s: out of memory; workers -> %d for its retry", item["name"],
                                limit)
                    retry.append(item)
                if res == "refused":
                    stop = True
            while not stop and len(futures) < limit and submit():
                pass
    summary = {k: sum(v == k for v in results.values())
               for k in ("done", "failed", "failed_eval", "skipped", "refused")}
    summary.update({"started": len(results), "workers_final": limit, "errors": errors,
                    "results": dict(sorted(results.items()))})
    return summary


# ================================================================ commands
def _items(args, require_all: bool = False) -> List[Dict[str, Any]]:
    if args.smoke:
        return smoke_grid(args.feature or "interpro")
    return load_grid(require_all=require_all)


def _sel(args, items):
    return select(items, aspect=getattr(args, "aspect", None),
                  feature=None if args.smoke else getattr(args, "feature", None),
                  seeds=getattr(args, "seeds", None), variants=getattr(args, "variants", None),
                  shard=getattr(args, "shard", None))


def cmd_plan(args) -> int:
    items = _items(args)
    sel = _sel(args, items)
    states = {r["name"] + r["feature"] + r["aspect"]: run_state(repo_path(r["out_dir"]))
              for r in sel}
    pending = [r["name"] for r in sel
               if states[r["name"] + r["feature"] + r["aspect"]] in PENDING]
    counts: Dict[str, int] = {}
    for v in states.values():
        counts[v] = counts.get(v, 0) + 1
    print(json.dumps({"n_grid": len(items), "n_selected": len(sel), "states": counts,
                      "n_pending": len(pending), "next": pending[:20],
                      "shard": None if args.shard is None else f"{args.shard[0]}/{args.shard[1]}"},
                     indent=1, sort_keys=True))
    return EXIT_OK


def _code_pin_present() -> bool:
    return (RESULTS / "code_pin.json").exists()


def cmd_run(args) -> int:
    if not args.smoke and not _code_pin_present():
        raise Refused("results/go_function/code_pin.json is missing: run `sweep.py pin` (G8)")
    items = _items(args, require_all=args.shard is not None and not args.smoke)
    sel = _sel(args, items)
    todo = [r for r in sel if run_state(repo_path(r["out_dir"])) in PENDING][:args.max_runs]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "would_run": [r["spec"] for r in todo]}, indent=1))
        return EXIT_OK
    log.info("running %d of %d selected runs with %d workers on %s", len(todo), len(sel),
             args.workers, args.device)
    summary = run_items(todo, args.device, args.workers)
    code = EXIT_OK
    if args.smoke:
        report = smoke_report(items, args.feature or "interpro", args.workers)
        summary["smoke_report"] = report["path"]
        if report["stop"]:
            log.error("G9b projection %.1f days exceeds %.0f days: stop and ask the user",
                      report["projection_days"], PROJECTION_LIMIT_DAYS)
            code = EXIT_STOP
    print(json.dumps(summary, indent=1, sort_keys=True))
    if summary["errors"]:
        return EXIT_FATAL
    if summary["refused"]:
        return EXIT_REFUSED
    return code


def smoke_report(items: Sequence[Dict[str, Any]], feature: str, workers: int) -> Dict[str, Any]:
    runs = []
    for r in items:
        d = repo_path(r["out_dir"])
        rj = _read_json(d / "run.json") if (d / "DONE.json").exists() else None
        if rj is None:
            runs.append({"name": r["name"], "state": run_state(d)})
            continue
        res = rj.get("resources") or {}
        runs.append({"name": r["name"], "state": "done",
                     "seconds": rj["wall_seconds"]["total"],
                     "seconds_train": rj["wall_seconds"].get("train"),
                     "seconds_eval": rj["wall_seconds"].get("eval"),
                     "max_rss_bytes": res.get("max_rss_bytes"),
                     "cuda_max_memory_allocated_bytes": res.get("cuda_max_memory_allocated"),
                     "device": rj["environment"]["device"]})
    secs = [x["seconds"] for x in runs if x.get("seconds") is not None]
    mean = sum(secs) / len(secs) if secs else None
    days = (mean * GRID_RUNS / max(1, workers) / 86400.0) if mean is not None else None
    stop = feature == "esm2_650m" and days is not None and days > PROJECTION_LIMIT_DAYS
    doc = {"schema": 1, "gate": "G9a" if feature == "interpro" else "G9b", "feature": feature,
           "runs": runs, "n_done": len(secs), "mean_seconds_per_run": mean,
           "max_rss_bytes": max([x.get("max_rss_bytes") or 0 for x in runs] or [0]),
           "projection": {"grid_runs": GRID_RUNS, "workers": workers, "days": days,
                          "label": "estimate: mean smoke seconds per run x grid runs / workers"
                                   + ("; G9a uses reduced batches on CPU and does not project "
                                      "the remote grid" if feature == "interpro" else "")},
           "protocol_stop": stop}
    path = RESULTS / "smoke" / f"smoke_report_{feature}.json"
    _write_json(path, doc)
    return {"path": repo_rel(path), "stop": stop, "projection_days": days}


def cmd_status(args) -> int:
    items = _items(args)
    sel = _sel(args, items)
    counts: Dict[str, int] = {}
    stale, released = [], []
    for r in sel:
        d = repo_path(r["out_dir"])
        st = run_state(d)
        counts[st] = counts.get(st, 0) + 1
        if st == "stale":
            stale.append(r["name"])
            if args.release_stale:
                release(d)
                if d.exists() and not (d / "DONE.json").exists() and not (d / "FAILED.json").exists():
                    os.replace(d, next_free(d, ".interrupted"))
                released.append(r["name"])
    out = {"n_selected": len(sel), "states": counts, "stale_claims": stale[:50],
           "released": released[:50], "n_released": len(released)}
    print(json.dumps(out, indent=1, sort_keys=True))
    return EXIT_OK


def cmd_rescore(args) -> int:
    items = _items(args)
    todo = [r for r in _sel(args, items)
            if run_state(repo_path(r["out_dir"])) == "failed_eval"][:args.max_runs]
    res = {}
    for r in todo:
        d = repo_path(r["out_dir"])
        if not claim(d):
            res[r["name"]] = "skipped"
            continue
        try:
            rc = run_subprocess(runner_cmd(r, args.device, rescore=True),
                                logs_dir(r) / f"{r['name']}.rescore.log")
            res[r["name"]] = "done" if (d / "DONE.json").exists() else f"failed (exit {rc})"
        finally:
            release(d)
    print(json.dumps({"rescored": res}, indent=1, sort_keys=True))
    return EXIT_OK if all(v == "done" for v in res.values()) else EXIT_FATAL


def code_files() -> Dict[str, str]:
    files = {}
    for base, pat in (("go_function", "*.py"), ("scripts/go_function", "*.py"),
                      ("config/go_function", "*.json")):
        for p in sorted((ROOT / base).rglob(pat)):
            if "__pycache__" in p.parts:
                continue
            files[repo_rel(p)] = sha256_file(p)
    return files


def git_committed() -> Optional[str]:
    """HEAD if the study paths are committed and clean (read-only git), else None."""
    try:
        st = subprocess.run(["git", "status", "--porcelain", "--", "go_function",
                             "scripts/go_function", "config/go_function"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head if not st.strip() else None


def cmd_pin(args) -> int:
    head = git_committed()
    if head is None and not args.accept_content_pin:
        raise Refused("study code is not committed: commit go_function/, scripts/go_function/, "
                      "config/go_function/ or pass --accept-content-pin")
    sets = {}
    for a in ASPECTS:
        for f in FEATURES:
            idx = _read_json(RESULTS / "specs" / a / f / "index.json")
            if idx is None:
                raise Refused(f"spec index {a}/{f} missing: run materialize.py first")
            sets[f"{a}/{f}"] = idx["spec_set_sha256"]
    doc = {"schema": 1, "files": code_files(), "git_head": head,
           "content_pin_accepted": head is None}
    b = canonical_json(doc)
    res = write_or_verify_file(RESULTS / "code_pin.json", b)
    pin = record_pin("G8", {"code_pin.json": sha256_bytes(b), "git_head": head,
                            "spec_sets": sets})
    print(json.dumps({"code_pin.json": res, "n_files": len(doc["files"]), "pin_G8": pin,
                      "git_head": head}, indent=1, sort_keys=True))
    return EXIT_OK


# ================================================================ CLI
def _max_runs(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("--max-runs must be >= 1")
    return n


def _workers(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("--workers must be >= 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, filters=True):
        p.add_argument("--smoke", action="store_true", help="the G9 smoke cell list")
        p.add_argument("--feature", choices=FEATURES)
        p.add_argument("--shard", type=parse_shard)
        if filters:
            p.add_argument("--aspect", choices=ASPECTS)
            p.add_argument("--seeds", type=int, nargs="+")
            p.add_argument("--variants", nargs="+")

    p = sub.add_parser("plan")
    common(p)
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--max-runs", type=_max_runs, required=True)
    p.add_argument("--workers", type=_workers, default=4)
    p.add_argument("--device", choices=DEVICES, default="auto")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("status")
    common(p)
    p.add_argument("--release-stale", action="store_true")
    p = sub.add_parser("rescore")
    common(p)
    p.add_argument("--max-runs", type=_max_runs, required=True)
    p.add_argument("--device", choices=DEVICES, default="auto")
    p = sub.add_parser("pin")
    p.add_argument("--accept-content-pin", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {"plan": cmd_plan, "run": cmd_run, "status": cmd_status, "rescore": cmd_rescore,
                "pin": cmd_pin}[args.cmd](args)
    except Refused as e:
        log.error("%s", e)
        return EXIT_REFUSED
    except Stop as e:
        log.error("%s", e)
        return EXIT_STOP
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())
