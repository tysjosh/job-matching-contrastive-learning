"""Resumable sweep driver for the GO/PPI withholding grid (design 12.3, protocol 3/7/10).

Subcommands::

    sweep.py pin
    sweep.py plan   [--seeds ...] [--packs ...] [--ks ...] [--cells ...] [--shard i/N]
    sweep.py run    --max-runs N  (required, 1 <= N <= 100) [same filters] [--shard i/N]
                    [--workers W] [--device auto|cuda|cpu] [--dry-run]
    sweep.py status [--shard i/N]
    sweep.py check-group --run-dirs <baseline> <loss_only> <neg_only> <both>
    sweep.py clear-group --cond <cond_id> --seed <s> --amendment <id>
    sweep.py rescore --run-id <run_id> [--device ...]

There is no mode that runs the whole grid: every ``run`` executes at most
``--max-runs`` (<= 100) runs, so the full grid is an explicit, repeated user
decision. Enumeration is seed-major (13, 21, 42, 87, 123), then condition
(``all``, then the table's packs in ``pack_id`` order x k ascending), then cell
(baseline, loss_only, neg_only, both); ``run_id = <pack>__k<k>__<cell>__s<seed>``
and the run dir is ``results/go_withholding/runs/<run_id>/``.

Per run: the runner subprocess (argument list, ``cwd=ROOT``), then
``evaluate_withholding.py score --relation``; ``DONE.json`` =
``{lc_manifest_sha256, metrics_sha256, code_sha256, finished}``. Training failure:
the dir is renamed ``<run_dir>.attempt<n>`` and retrained once; a second failure
writes ``FAILED.json`` (``stage: train``). Evaluation-only failure: ``eval/`` is
renamed ``eval.attempt<n>`` and re-scored once without retraining; a second
failure writes ``FAILED.json`` (``stage: eval``, with ``lc_manifest_sha256``,
plan decision 4c). Group checks (``check_pending_groups``) run at the start of
``run`` and ``status``, after every completed run and at the end of ``run``;
``GROUP_OK.json`` / ``GROUP_FAILED.json`` follow design 12.3 exactly.

Remote GPU execution (user decision B):

* ``--shard i/N`` (1 <= i <= N) partitions the grid by (condition, seed) group:
  group number g (seed-major order over the FULL grid, independent of filters)
  belongs to shard ``g % N + 1``. Shards are disjoint and exhaustive, keep a
  group's four cells together and stay seed-major, so N processes (or machines)
  can share one ``runs/`` dir without overlap.
* ``--workers W`` runs up to W runs of one invocation concurrently (threads, each
  driving its own subprocesses; one GPU is shared). ``--max-runs`` still bounds
  the total runs started by the invocation. Group checks stay in the main thread.
  Parallel workers or shards need a warm frozen-encoder cache (the
  ``embedding_cache_path`` of the baseline config), because concurrent runs would
  otherwise race on writing it: run ``run --max-runs 1`` once first.
* Per-run claim marker ``RUNNING.json`` (pid, host, start time), created with
  ``O_EXCL``. A run dir without ``DONE.json``/``FAILED.json`` whose marker is
  absent, unreadable or held by a dead process on this host is an interrupted
  run: it is renamed ``<run_id>.interrupted<n>`` (never deleted, never analysed)
  and retrained from scratch. A live marker means the run is in progress
  elsewhere and is skipped. A run with ``DONE.json`` is never touched again
  (re-running is a no-op).
* ``--device`` is forwarded to the runner and the evaluator.

Exit codes: 0 ok; 2 refused (argparse, pins, preflight, rule violated); 3
``rescore`` scoring failed again; 4 grid stopped / group mismatch.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding as wh  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("sweep")

RUNNER = ROOT / "scripts" / "run_learning_curve_point.py"
EVALUATOR = ROOT / "scripts" / "go_withholding" / "evaluate_withholding.py"
MAX_RUNS_CAP = 100
MAX_WORKERS = 16
#: Free disk required before ``run`` starts (design 12.3). Read at call time.
#: Checked on every invocation, which writes at most MAX_RUNS_CAP runs
#: (~100 x 3.2 MB = 0.32 GB), so 3 GB leaves ~9x headroom per invocation; the
#: full grid's ~7.7 GB is never written by one invocation.
MIN_FREE_BYTES = 3 * 10 ** 9
INTERRUPTED = ".interrupted"
#: Renamed dirs: never a run id, skipped everywhere.
IGNORED_SUFFIXES = (*wc.RUN_DIR_IGNORE, INTERRUPTED)
REQUIRED_PINS = ("G1", "G1-leak", "G2-targets", "G2-covariate", "G2-cache", "G2-validation",
                 "G3", "G4-lambda", "G4-configs", "G5")
CELL_FILES = tuple(f"{c}.json" for c in wc.CELLS)
K_ORDER: Tuple[Any, ...] = tuple(wc.KS)

_LOG_LOCK = threading.Lock()


class SweepError(RuntimeError):
    def __init__(self, msg: str, code: int = 2):
        super().__init__(msg)
        self.code = code


# ================================================================ paths
def runs_dir() -> Path:
    return Path(wc.STUDY_RES) / "runs"


def log_path() -> Path:
    return Path(wc.STUDY_RES) / "sweep_log.jsonl"


def code_pin_path() -> Path:
    return Path(wc.STUDY_RES) / "code_pin.json"


def eval_cache_dir() -> Path:
    return Path(wc.STUDY_RES) / "eval_cache"


def table_path() -> Path:
    return Path(wc.TARGETS) / "target_table.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(wc.relativize_json(obj), indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def log_event(obj: Dict[str, Any]) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(wc.relativize_json(obj), sort_keys=True) + "\n"
    with _LOG_LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


def read_log() -> List[Dict[str, Any]]:
    path = log_path()
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def next_free(path: Path, suffix: str) -> Path:
    """``<path><suffix><n>`` with the smallest unused n >= 1."""
    path = Path(path)
    n = 1
    while path.with_name(f"{path.name}{suffix}{n}").exists():
        n += 1
    return path.with_name(f"{path.name}{suffix}{n}")


# ================================================================ subprocesses
def run_subprocess(cmd: List[str]) -> int:
    """Argument-list subprocess with ``cwd=ROOT`` (tests replace this)."""
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


PROBE = """
import json, sys, importlib.util
sys.path.insert(0, {root!r}); sys.path.insert(0, {root!r} + "/scripts")
import go_ppi_domain.record_adapter, go_ppi_domain.withholding, go_ppi_domain.withholding_diagnostics
import contrastive_learning.trainer, trials_domain.label_budget_runner, run_phase1_embedding_evaluation
import scripts.run_learning_curve_point
spec = importlib.util.spec_from_file_location("evaluate_withholding", {ev!r})
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from go_ppi_domain import withholding_constants as wc
print(json.dumps(wc.imported_repo_modules()))
"""


def import_probe() -> List[str]:
    """Repo modules loaded by importing the runner and evaluator (design 13, G5)."""
    out = subprocess.run([sys.executable, "-c", PROBE.format(root=str(ROOT), ev=str(EVALUATOR))],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SweepError(f"import probe failed: {out.stderr.strip()[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


# ================================================================ grid
def load_table() -> Dict[str, Any]:
    from scripts.go_withholding import make_conditions as mc

    try:
        return mc.load_table(table_path())["table"]
    except (OSError, RuntimeError, ValueError) as exc:
        raise SweepError(f"target table: {exc}") from exc


def grid(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every run in seed-major order with its group number (full grid)."""
    out = []
    g = 0
    for seed in wc.SEEDS:
        for cond in wh.condition_ids(table):
            pack, k = wh.parse_cond_id(cond)
            for cell in wc.CELLS:
                out.append({"run_id": wc.run_id_for(cond, cell, seed), "cond": cond,
                            "pack": pack, "k": k, "cell": cell, "seed": seed, "group": g})
            g += 1
    if len(out) != int(table["n_runs"]):
        raise SweepError(f"grid has {len(out)} runs != target_table n_runs {table['n_runs']}")
    return out


def groups_of(table: Dict[str, Any]) -> List[Tuple[str, int]]:
    return [(cond, seed) for seed in wc.SEEDS for cond in wh.condition_ids(table)]


def parse_shard(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text or "")
    if not m:
        raise argparse.ArgumentTypeError(f"--shard must be i/N, got {text!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 1 <= i <= n:
        raise argparse.ArgumentTypeError(f"--shard needs 1 <= i <= N, got {text!r}")
    return i, n


def in_shard(group: int, shard: Optional[Tuple[int, int]]) -> bool:
    return shard is None or group % shard[1] == shard[0] - 1


def select(runs: List[Dict[str, Any]], seeds=None, packs=None, ks=None, cells=None,
           shard: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    kset = None if ks is None else {("all" if str(k) == "all" else int(k)) for k in ks}
    return [r for r in runs
            if (seeds is None or r["seed"] in seeds) and (packs is None or r["pack"] in packs)
            and (kset is None or r["k"] in kset) and (cells is None or r["cell"] in cells)
            and in_shard(r["group"], shard)]


# ================================================================ run state
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def run_state(run_dir: Path) -> str:
    """``absent`` | ``done`` | ``failed`` | ``in_progress`` | ``interrupted``."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return "absent"
    done = _read_json(run_dir / "DONE.json")
    if done is not None and "lc_manifest_sha256" in done:
        return "done"
    if (run_dir / "FAILED.json").is_file():
        return "failed"
    claim = _read_json(run_dir / "RUNNING.json")
    if claim is not None:
        same_host = claim.get("host") == socket.gethostname()
        if not same_host:
            return "in_progress"            # cannot probe another host's pid: hands off
        if int(claim.get("pid", -1)) != os.getpid() and _pid_alive(claim.get("pid", -1)):
            return "in_progress"
    return "interrupted"


def claim_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    fd = os.open(run_dir / "RUNNING.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": _now()}, fh)


def release_run(run_dir: Path) -> None:
    try:
        (Path(run_dir) / "RUNNING.json").unlink()
    except FileNotFoundError:
        pass


# ================================================================ preflights
def _latest(pins: List[dict], gate: str) -> dict:
    found = [e for e in pins if e.get("gate") == gate]
    if not found:
        raise SweepError(f"protocol pins have no {gate} entry")
    return found[-1]


def _check_sha(path: Path, expected: Optional[str], what: str) -> None:
    if not Path(path).is_file():
        raise SweepError(f"{what}: {wc.rel_to_root(path)} does not exist")
    got = wc.sha256_file(path)
    if got != expected:
        raise SweepError(f"{what}: {wc.rel_to_root(path)} sha256 {got} != pin {expected}")


def _free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def pilot_dirs() -> List[str]:
    root = runs_dir()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir()
                  and ((_read_json(d / "lc_manifest.json") or {}).get("pilot") is True))


def pin_preflight(table: Dict[str, Any]) -> Dict[str, Any]:
    """Design 12.3 preflight: disk, every pin present and still matching, code pin, no pilot."""
    from scripts.go_withholding import regenerate_conditions as rg

    free = _free_bytes(runs_dir())
    if free < MIN_FREE_BYTES:
        raise SweepError(f"only {free / 1e9:.2f} GB free (< {MIN_FREE_BYTES / 1e9:.0f} GB)")
    pins = wc.read_pins()
    for gate in REQUIRED_PINS:
        _latest(pins, gate)
    if _latest(pins, "G1-leak").get("pass") is not True:
        raise SweepError("G1-leak pin is not a pass")
    try:
        rg.verify_base(Path(wc.SPLITS), Path(wc.CONVERTED))
        t = rg.verify_table(table_path())
    except RuntimeError as exc:
        raise SweepError(f"pin mismatch: {exc}") from exc
    if t["table"] != table:
        raise SweepError("target table changed while loading")
    cov = _latest(pins, "G2-covariate")
    _check_sha(Path(wc.TARGETS) / "text_covariate.json", cov.get("text_covariate_sha256"),
               "G2-covariate")
    cache = _latest(pins, "G2-cache")
    sha16 = str(cache.get("test_sha256", ""))[:16]
    npz = eval_cache_dir() / f"test_{sha16}.npz"
    _check_sha(npz, cache.get("cache_npz_sha256"), "G2-cache")
    _check_sha(eval_cache_dir() / f"test_{sha16}.meta.json", cache.get("meta_sha256"),
               "G2-cache meta")
    val = _latest(pins, "G2-validation")
    vdir = Path(wc.VALIDATION_TF)
    _check_sha(vdir / "manifest.json", val.get("manifest_sha256"), "G2-validation")
    for name, sha in (val.get("sha256s") or {}).items():
        _check_sha(vdir / name, sha, "G2-validation")
    g3 = _latest(pins, "G3")
    conds = wh.condition_ids(table)
    if int(g3.get("n_conditions", -1)) != int(table["n_conditions"]) or \
            sorted(g3.get("manifests", {})) != sorted(conds) or \
            len(conds) != int(table["n_conditions"]):
        raise SweepError(f"G3 pin does not list exactly the table's {table['n_conditions']} "
                         f"conditions")
    for cond, sha in g3["manifests"].items():
        _check_sha(Path(wc.CONDITIONS) / cond / wh.MANIFEST, sha, f"G3 {cond}")
    lam = _latest(pins, "G4-lambda")
    _check_sha(wc.lambda_choice_path(), lam.get("lambda_choice_sha256"), "G4-lambda")
    cfg = _latest(pins, "G4-configs")
    files = cfg.get("files") or {}
    if sorted(files) != sorted(CELL_FILES):
        raise SweepError(f"last G4-configs pin lists {sorted(files)}, need exactly "
                         f"{sorted(CELL_FILES)}")
    for name, sha in files.items():
        _check_sha(Path(wc.CONFIG_DIR) / name, sha, "G4-configs")
    _check_sha(Path(wc.CONFIG_DIR) / "configs.sha256.json", cfg.get("configs_sha256_json"),
               "G4-configs")
    g5 = _latest(pins, "G5")
    code_pin = _read_json(code_pin_path())
    if code_pin is None:
        raise SweepError(f"{wc.rel_to_root(code_pin_path())} missing (run `sweep.py pin`)")
    current = wc.code_sha256()[0]
    if current != code_pin.get("code_sha256") or g5.get("code_sha256") != code_pin.get("code_sha256"):
        raise SweepError(f"code_sha256 {current} != code pin {code_pin.get('code_sha256')} "
                         f"(re-pinning is an amendment)")
    pilots = pilot_dirs()
    if pilots:
        raise SweepError(f"pilot run dir(s) under runs/: {pilots[:5]}")
    return {"code_sha256": code_pin["code_sha256"], "cache": npz}


def stopped_groups(table: Dict[str, Any]) -> List[Tuple[str, int, Path]]:
    """Groups with a ``GROUP_FAILED.json`` in any of their (unrenamed) run dirs."""
    out = []
    for cond, seed in groups_of(table):
        for rid in wc.group_run_ids(cond, seed):
            p = runs_dir() / rid / "GROUP_FAILED.json"
            if p.is_file():
                out.append((cond, seed, p.parent))
                break
    return out


def stopped_message(stopped) -> str:
    return "; ".join(f"grid stopped: group {c},s{s} failed the identity check; see "
                     f"{wc.rel_to_root(p)}/GROUP_FAILED.json; fix under an amendment "
                     f"(protocol §13)" for c, s, p in stopped)


def embedding_cache_warm() -> bool:
    raw = _read_json(Path(wc.CONFIG_DIR) / "baseline.json") or {}
    path = wc.expand_root_token(raw.get("embedding_cache_path") or "")
    return bool(path) and Path(path).is_file()


# ================================================================ group checks
def _code_hash(ctx: Dict[str, Any]) -> str:
    if "code_sha256" not in ctx:
        pin = _read_json(code_pin_path())
        ctx["code_sha256"] = (pin or {}).get("code_sha256") or wc.code_sha256()[0]
    return ctx["code_sha256"]


def _post_checks_ok(lc: Dict[str, Any]) -> bool:
    a = (lc.get("withholding") or {}).get("assertions") or {}
    post = {k: v for k, v in a.items() if k.startswith("post_")}
    return bool(post) and all(v == "ok" for v in post.values())


def group_checks(manifests: List[Dict[str, Any]], run_ids: List[str],
                 require_epochs: Optional[int] = wc.GRID_EPOCHS) -> Dict[str, Any]:
    """Design 12.3 group checks from the four ``lc_manifest.json`` (cell order)."""
    fails: List[str] = []
    ns = [m.get("negative_selection") or {} for m in manifests]
    sel = [n.get("selection_sha256_by_epoch") or {} for n in ns]
    by = dict(zip(wc.CELLS, sel))
    for i, (m, cell) in enumerate(zip(manifests, wc.CELLS)):
        got = (m.get("withholding") or {}).get("cell")
        if got != cell:
            fails.append(f"{run_ids[i]}: cell {got!r} != {cell!r} (cell order)")
        if m.get("pilot") is True:
            fails.append(f"{run_ids[i]}: pilot run")
        if require_epochs is not None and m.get("epochs") != require_epochs:
            fails.append(f"{run_ids[i]}: epochs {m.get('epochs')} != {require_epochs}")
    if len({(m.get("withholding") or {}).get("cond_id") for m in manifests}) != 1 or \
            len({m.get("seed") for m in manifests}) != 1:
        fails.append("the four runs differ in cond_id or seed")
    if not by["baseline"] or by["baseline"] != by["loss_only"]:
        fails.append("selection_sha256_by_epoch: baseline != loss_only")
    if not by["neg_only"] or by["neg_only"] != by["both"]:
        fails.append("selection_sha256_by_epoch: neg_only != both")
    epochs = set(by["baseline"]) | set(by["neg_only"])
    if not any(by["baseline"].get(e) != by["neg_only"].get(e) for e in epochs):
        fails.append("selection_sha256_by_epoch: baseline == neg_only in every epoch")
    un = [n.get("unannotated_anchor_calls_by_epoch") for n in ns]
    if any(u != un[0] for u in un):
        fails.append("unannotated_anchor_calls_by_epoch differ across cells")
    if not all(n.get("unannotated_anchor_window") is True for n in ns):
        fails.append("unannotated_anchor_window is not true in all four")
    for rid, n in zip(run_ids, ns):
        if n.get("selector_none_returns") != 0 or n.get("selector_exceptions") != 0:
            fails.append(f"{rid}: selector_none_returns/exceptions != 0")
    ok = [{str(e.get("epoch")): e.get("ok_batches") for e in (m.get("epoch_stats") or [])}
          for m in manifests]
    if not ok[0] or any(o != ok[0] for o in ok):
        fails.append("ok_batches per epoch differ across cells")
    if len({m.get("train_records") for m in manifests}) != 1:
        fails.append("train_records differ across cells")
    for rid, m, cell in zip(run_ids, manifests, wc.CELLS):
        applied = int((m.get("go_relation_loss") or {}).get("applied_batches", 0) or 0)
        if cell in ("loss_only", "both") and applied <= 0:
            fails.append(f"{rid}: go_relation_loss.applied_batches == 0 in a loss-on cell")
        if cell in ("baseline", "neg_only") and applied != 0:
            fails.append(f"{rid}: go_relation_loss.applied_batches {applied} in a loss-off cell")
    return {"failures": fails,
            "selection_sha256_by_epoch": {"baseline_loss_only": by["baseline"],
                                          "neg_only_both": by["neg_only"]}}


def _supersede_ok(d: Path) -> None:
    src = d / "GROUP_OK.json"
    if not src.exists():
        return
    dst = d / "GROUP_OK.superseded.json"
    n = 2
    while dst.exists():
        dst = d / f"GROUP_OK.superseded{n}.json"
        n += 1
    os.replace(src, dst)


def write_group_result(dirs: List[Path], payload: Dict[str, Any], failed: bool) -> None:
    for d in dirs:
        if failed:
            _write_atomic(d / "GROUP_FAILED.json", payload)
            _supersede_ok(d)
        else:
            _write_atomic(d / "GROUP_OK.json", payload)


def check_group_dirs(dirs: List[Path], ctx: Dict[str, Any],
                     require_epochs: Optional[int] = wc.GRID_EPOCHS) -> List[str]:
    """Drift pre-check (dirs with DONE.json) then the group checks; writes markers."""
    run_ids = [d.name for d in dirs]
    shas, drift = {}, []
    for d in dirs:
        shas[d.name] = wc.sha256_file(d / "lc_manifest.json")
        done = _read_json(d / "DONE.json")
        if done is not None and done.get("lc_manifest_sha256") != shas[d.name]:
            drift.append(f"lc_manifest_changed_after_done: {d.name}")
    base = {"run_ids": run_ids, "lc_manifest_sha256": shas, "checked": _now(),
            "code_sha256": _code_hash(ctx)}
    if drift:
        write_group_result(dirs, {**base, "selection_sha256_by_epoch": None, "failures": drift},
                           failed=True)
        logger.error("group %s: manifest drift %s -> GROUP_FAILED.json", run_ids[0], drift)
        return drift
    manifests = [json.loads((d / "lc_manifest.json").read_text(encoding="utf-8")) for d in dirs]
    res = group_checks(manifests, run_ids, require_epochs)
    payload = {**base, "selection_sha256_by_epoch": res["selection_sha256_by_epoch"]}
    if res["failures"]:
        write_group_result(dirs, {**payload, "failures": res["failures"]}, failed=True)
        logger.error("group %s failed the identity check: %s", run_ids[0], res["failures"])
        return res["failures"]
    write_group_result(dirs, payload, failed=False)
    return []


def group_status(dirs: List[Path]) -> str:
    if any((d / "GROUP_FAILED.json").is_file() for d in dirs):
        return "group_failed"
    if any((d / "FAILED.json").is_file() for d in dirs):
        return "failed_member"
    if all(_read_json(d / "DONE.json") is not None for d in dirs):
        return "ok" if all(wc.group_ok_valid(dirs)) else "eligible"
    return "incomplete"


def check_pending_groups(table: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
    """Check every eligible group (design 12.3); returns the newly failed groups."""
    failed = []
    for cond, seed in groups_of(table):
        dirs = [runs_dir() / r for r in wc.group_run_ids(cond, seed)]
        if group_status(dirs) != "eligible":
            continue
        if check_group_dirs(dirs, ctx):
            failed.append(f"{cond},s{seed}")
            log_event({"event": "group_failed", "cond_id": cond, "seed": seed, "time": _now()})
        else:
            log_event({"event": "group_ok", "cond_id": cond, "seed": seed, "time": _now()})
    return failed


# ================================================================ per run
def runner_cmd(item: Dict[str, Any], device: Optional[str]) -> List[str]:
    cond = Path(wc.CONDITIONS) / item["cond"]
    run = runs_dir() / item["run_id"]
    cfg = Path(wc.CONFIG_DIR) / f"{item['cell']}.json"
    cmd = [sys.executable, str(RUNNER), "--domain", "go_ppi", "--config", str(cfg),
           "--train-file", str(cond / "train.jsonl"), "--go-ppi-split-dir", str(cond),
           "--withholding-manifest", str(cond / wh.MANIFEST), "--study-cell", item["cell"],
           "--go-diagnostics", str(run / "go_diagnostics.jsonl"), "--output-dir", str(run),
           "--seed", str(item["seed"]),
           "--compare-config", str(Path(wc.CONFIG_DIR) / "baseline.json")]
    if device:
        cmd += ["--device", device]
    return cmd


def eval_cmd(run_dir: Path, cache: Path, device: Optional[str]) -> List[str]:
    cmd = [sys.executable, str(EVALUATOR), "score", "--run-dir", str(run_dir),
           "--cache", str(cache), "--target-table", str(table_path()), "--relation"]
    if device:
        cmd += ["--device", device]
    return cmd


def training_problem(run_dir: Path, rc: int) -> Optional[str]:
    if rc != 0:
        return f"runner exit {rc}"
    lc = _read_json(run_dir / "lc_manifest.json")
    if lc is None:
        return "lc_manifest.json missing"
    if lc.get("pilot") is True:
        return "pilot run"
    if lc.get("epochs") != wc.GRID_EPOCHS:
        return f"lc_manifest epochs {lc.get('epochs')} != {wc.GRID_EPOCHS}"
    if not _post_checks_ok(lc):
        return "post-training checks not all ok"
    return None


def write_done(run_dir: Path, code_hash: str) -> None:
    _write_atomic(run_dir / "DONE.json", {
        "lc_manifest_sha256": wc.sha256_file(run_dir / "lc_manifest.json"),
        "metrics_sha256": wc.sha256_file(run_dir / "eval" / "metrics.json"),
        "code_sha256": code_hash, "finished": _now()})


def _attempt_line(run_id: str, stage: str, attempt: int, rc: int, t0: float) -> None:
    log_event({"run_id": run_id, "stage": stage, "attempt": attempt, "exit_code": rc,
               "time": _now(), "wall_seconds": round(time.time() - t0, 1)})


def score_once(run_dir: Path, cache: Path, device: Optional[str]) -> int:
    rc = run_subprocess(eval_cmd(run_dir, cache, device))
    if rc == 0 and not (run_dir / "eval" / "metrics.json").is_file():
        return 1
    return rc


def execute_run(item: Dict[str, Any], ctx: Dict[str, Any], device: Optional[str]) -> str:
    """One run's full lifecycle; returns ``done`` | ``failed_train`` | ``failed_eval``
    | ``skipped`` (the run dir was claimed or finished elsewhere after selection)."""
    rid = item["run_id"]
    run_dir = runs_dir() / rid
    if run_dir.exists():
        # Recheck right before the rename: an overlapping invocation may have
        # claimed or finished this run since it was selected (review R5).
        state = run_state(run_dir)
        if state != "interrupted":
            logger.warning("%s: %s since selection; not renamed, skipped", rid, state)
            return "skipped"
        dst = next_free(run_dir, INTERRUPTED)
        os.replace(run_dir, dst)
        log_event({"event": "interrupted_restart", "run_id": rid, "renamed_to": dst.name,
                   "time": _now()})
        logger.warning("%s: interrupted run renamed to %s; restarting", rid, dst.name)
    claim_run(run_dir)
    try:
        for attempt in (1, 2):
            t0 = time.time()
            rc = run_subprocess(runner_cmd(item, device))
            _attempt_line(rid, "train", attempt, rc, t0)
            problem = training_problem(run_dir, rc)
            if problem is None:
                break
            logger.warning("%s: training attempt %d failed: %s", rid, attempt, problem)
            if attempt == 1:
                release_run(run_dir)
                os.replace(run_dir, next_free(run_dir, ".attempt"))
                claim_run(run_dir)
            else:
                _write_atomic(run_dir / "FAILED.json", {
                    "run_id": rid, "stage": "train", "attempts": 2, "exit_code": rc,
                    "reason": problem, "time": _now()})
                return "failed_train"
        cache = ctx["cache"]
        for attempt in (1, 2):
            t0 = time.time()
            rc = score_once(run_dir, cache, device)
            _attempt_line(rid, "eval", attempt, rc, t0)
            if rc == 0:
                break
            logger.warning("%s: scoring attempt %d failed (exit %d)", rid, attempt, rc)
            if attempt == 1:
                if (run_dir / "eval").exists():
                    os.replace(run_dir / "eval", next_free(run_dir / "eval", ".attempt"))
            else:
                _write_atomic(run_dir / "FAILED.json", {
                    "run_id": rid, "stage": "eval", "attempts": 2, "exit_code": rc,
                    "lc_manifest_sha256": wc.sha256_file(run_dir / "lc_manifest.json"),
                    "time": _now()})
                return "failed_eval"
        write_done(run_dir, _code_hash(ctx))
        return "done"
    finally:
        release_run(run_dir)


# ================================================================ commands
def _filters(args) -> Dict[str, Any]:
    return {"seeds": args.seeds, "packs": args.packs, "ks": args.ks, "cells": args.cells,
            "shard": args.shard}


def cmd_plan(args) -> int:
    table = load_table()
    stopped = stopped_groups(table)
    if stopped:
        logger.warning(stopped_message(stopped))
    full = grid(table)
    sel = select(full, **_filters(args))
    states = {r["run_id"]: run_state(runs_dir() / r["run_id"]) for r in sel}
    pending = [r["run_id"] for r in sel if states[r["run_id"]] in ("absent", "interrupted")]
    out = {"n_runs": int(table["n_runs"]), "n_conditions": int(table["n_conditions"]),
           "n_selected": len(sel), "shard": None if args.shard is None else
           f"{args.shard[0]}/{args.shard[1]}",
           "n_done": sum(v == "done" for v in states.values()),
           "n_failed": sum(v == "failed" for v in states.values()),
           "n_in_progress": sum(v == "in_progress" for v in states.values()),
           "n_interrupted": sum(v == "interrupted" for v in states.values()),
           "n_pending": len(pending), "next": pending[:20],
           "n_groups_group_failed": len(stopped)}
    print(json.dumps(out, indent=1))
    return 0


def cmd_run(args) -> int:
    table = load_table()
    ctx = pin_preflight(table)
    stopped = stopped_groups(table)
    if stopped:
        raise SweepError(stopped_message(stopped), 4)
    if (args.workers > 1 or args.shard is not None) and not embedding_cache_warm():
        raise SweepError("parallel workers/shards need a warm frozen-encoder cache "
                         "(embedding_cache_path of the baseline config is absent): run "
                         "`sweep.py run --max-runs 1` once first")
    sel = select(grid(table), **_filters(args))
    todo = []
    for r in sel:
        st = run_state(runs_dir() / r["run_id"])
        if st in ("absent", "interrupted"):
            todo.append(dict(r, state=st))
        if len(todo) >= args.max_runs:
            break
    if args.dry_run:
        print(json.dumps({"dry_run": True, "would_run": [r["run_id"] for r in todo],
                          "restart_interrupted": [r["run_id"] for r in todo
                                                  if r["state"] == "interrupted"]}, indent=1))
        return 0
    if check_pending_groups(table, ctx):
        return 4
    results: Dict[str, str] = {}
    stop = False
    errors: List[str] = []
    it = iter(todo)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures: Dict[Any, str] = {}

        def submit_next() -> bool:
            item = next(it, None)
            if item is None:
                return False
            # The state may have changed since selection (another shard/process).
            if run_state(runs_dir() / item["run_id"]) not in ("absent", "interrupted"):
                return True
            futures[ex.submit(execute_run, item, ctx, args.device)] = item["run_id"]
            return True

        while len(futures) < args.workers and submit_next():
            pass
        while futures:
            finished, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for f in finished:
                rid = futures.pop(f)
                try:
                    results[rid] = f.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{rid}: {type(exc).__name__}: {exc}")
                    logger.error("run %s raised %s: %s", rid, type(exc).__name__, exc)
                    stop = True
                if check_pending_groups(table, ctx):
                    stop = True
            while not stop and len(futures) < args.workers and submit_next():
                pass
    if check_pending_groups(table, ctx):
        stop = True
    n_skipped = sum(v == "skipped" for v in results.values())
    summary = {"started": len(results) - n_skipped + len(errors),
               "skipped": n_skipped,
               "done": sum(v == "done" for v in results.values()),
               "failed_train": sum(v == "failed_train" for v in results.values()),
               "failed_eval": sum(v == "failed_eval" for v in results.values()),
               "errors": errors}
    print(json.dumps(summary, indent=1))
    if errors:
        return 2
    return 4 if stop else 0


def cmd_status(args) -> int:
    table = load_table()
    ctx: Dict[str, Any] = {}
    new_failures = check_pending_groups(table, ctx)
    full = grid(table)
    sel = select(full, shard=args.shard)
    states = [run_state(runs_dir() / r["run_id"]) for r in sel]
    gstat: Dict[str, List[str]] = {}
    for cond, seed in groups_of(table):
        dirs = [runs_dir() / r for r in wc.group_run_ids(cond, seed)]
        gstat.setdefault(group_status(dirs), []).append(f"{cond},s{seed}")
    drift_runs = []
    for r in sel:
        p = runs_dir() / r["run_id"] / "go_diagnostics.jsonl"
        if p.is_file() and any(l.strip() and json.loads(l).get("drift_flag")
                               for l in p.read_text(encoding="utf-8").splitlines()):
            drift_runs.append(r["run_id"])
    cleared = [e for e in read_log() if e.get("event") == "group_cleared"]
    out = {"n_runs": int(table["n_runs"]), "n_selected": len(sel),
           "shard": None if args.shard is None else f"{args.shard[0]}/{args.shard[1]}",
           "n_done": states.count("done"), "n_failed_runs": states.count("failed"),
           "n_in_progress": states.count("in_progress"),
           "n_interrupted": states.count("interrupted"),
           "progress": f"{states.count('done')}/{len(sel)}",
           "n_groups_ok": len(gstat.get("ok", [])),
           "n_groups_unchecked": len(gstat.get("eligible", [])),
           "n_groups_failed_member": len(gstat.get("failed_member", [])),
           "n_groups_group_failed": len(gstat.get("group_failed", [])),
           "groups_group_failed": gstat.get("group_failed", []),
           "n_groups_cleared": len(cleared), "n_runs_drift_flagged": len(drift_runs),
           "drift_flagged_runs": drift_runs[:20], "new_group_failures": new_failures}
    print(json.dumps(out, indent=1))
    return 4 if new_failures else 0


def cmd_check_group(args) -> int:
    dirs = [Path(d) for d in args.run_dirs]
    for d in dirs:
        if not (d / "lc_manifest.json").is_file():
            raise SweepError(f"{d.name}: lc_manifest.json missing")
        if (_read_json(d / "lc_manifest.json") or {}).get("pilot") is True:
            raise SweepError(f"{d.name}: pilot run dirs are refused")
    failures = check_group_dirs(dirs, {}, require_epochs=None)
    print(json.dumps({"run_ids": [d.name for d in dirs], "failures": failures}, indent=1))
    return 4 if failures else 0


def _valid_date(text: str) -> bool:
    if len(text) != 10:
        return False
    try:
        _dt.date.fromisoformat(text)
    except ValueError:
        return False
    return True


def cmd_clear_group(args) -> int:
    cond, seed, aid = args.cond, int(args.seed), args.amendment
    if not re.match(wc.AMENDMENT_ID_RE, aid or ""):
        raise SweepError(f"amendment id {aid!r} does not match {wc.AMENDMENT_ID_RE}")
    try:
        rows = wc.parse_amendment_rows()
    except (OSError, ValueError) as exc:
        raise SweepError(f"protocol section 13 amendments table missing or malformed: {exc}")
    match = [r for r in rows if r["id"] == aid]
    if len(match) != 1:
        raise SweepError(f"protocol section 13 has {len(match)} amendments rows with id {aid} "
                         f"(need exactly 1)")
    row = match[0]
    if not _valid_date(row["date"]):
        raise SweepError(f"amendment {aid}: date {row['date']!r} is not YYYY-MM-DD")
    if not row["cause"].strip() or not row["fix"].strip():
        raise SweepError(f"amendment {aid}: empty cause or fix cell")
    entries = [e.strip().strip("`").strip() for e in row["groups cleared"].split(";")]
    if f"{cond},s{seed}" not in entries:
        raise SweepError(f"amendment {aid} does not list group {cond},s{seed} under "
                         f"'groups cleared' ({row['groups cleared']!r})")
    for e in read_log():
        if (e.get("event") == "group_cleared" and e.get("amendment") == aid
                and e.get("cond_id") == cond and int(e.get("seed", -1)) == seed):
            raise SweepError(f"group {cond},s{seed} was already cleared under {aid}; a "
                             f"second failure needs a new amendment")
    try:
        rids = wc.group_run_ids(cond, seed)
    except ValueError as exc:
        raise SweepError(str(exc))
    existing = [runs_dir() / r for r in rids if (runs_dir() / r).is_dir()]
    if not any((d / "GROUP_FAILED.json").is_file() for d in existing):
        raise SweepError(f"no existing run dir of group {cond},s{seed} carries GROUP_FAILED.json")
    renamed = []
    for d in existing:
        dst = next_free(d, ".groupfailed")
        os.replace(d, dst)
        renamed.append(dst.name)
    log_event({"event": "group_cleared", "cond_id": cond, "seed": seed, "renamed_to": renamed,
               "amendment": aid, "amendment_row": row, "time": _now()})
    print(json.dumps({"cleared": f"{cond},s{seed}", "renamed_to": renamed}, indent=1))
    return 0


def cmd_rescore(args) -> int:
    table = load_table()
    rid = args.run_id
    if rid not in {r["run_id"] for r in grid(table)}:
        raise SweepError(f"{rid!r} is not a run id of the grid")
    run_dir = runs_dir() / rid
    failed = _read_json(run_dir / "FAILED.json")
    if failed is None:
        raise SweepError(f"{rid}: no FAILED.json")
    if failed.get("stage") != "eval":
        raise SweepError(f"{rid}: FAILED.json stage {failed.get('stage')!r}; only eval-stage "
                         f"failures are rescored")
    if (run_dir / "DONE.json").exists():
        raise SweepError(f"{rid}: DONE.json already present")
    lc = _read_json(run_dir / "lc_manifest.json")
    if lc is None:
        raise SweepError(f"{rid}: lc_manifest.json missing")
    if not _post_checks_ok(lc) or lc.get("epochs") != wc.GRID_EPOCHS:
        raise SweepError(f"{rid}: post-training checks not all ok or epochs != "
                         f"{wc.GRID_EPOCHS}")
    ck = (lc.get("withholding") or {}).get("selected_checkpoint")
    if not ck or not (run_dir / ck).is_file():
        raise SweepError(f"{rid}: checkpoint {ck!r} missing")
    if failed.get("lc_manifest_sha256") != wc.sha256_file(run_dir / "lc_manifest.json"):
        raise SweepError(f"{rid}: lc_manifest_changed_after_failure")
    ctx = pin_preflight(table)
    stopped = stopped_groups(table)
    if stopped:
        raise SweepError(stopped_message(stopped), 4)
    attempt = 1
    if (run_dir / "eval").exists():
        dst = next_free(run_dir / "eval", ".attempt")
        attempt = int(dst.name[len("eval.attempt"):])
        os.replace(run_dir / "eval", dst)
    t0 = time.time()
    rc = score_once(run_dir, ctx["cache"], args.device)
    _attempt_line(rid, "eval", attempt, rc, t0)
    if rc == 0:
        write_done(run_dir, ctx["code_sha256"])
        os.replace(run_dir / "FAILED.json", run_dir / "FAILED.repaired.json")
        log_event({"event": "eval_repaired", "run_id": rid, "attempt": attempt, "time": _now()})
        logger.info("%s: evaluation repaired", rid)
        return 0
    failed["attempts"] = int(failed.get("attempts", 2)) + 1
    _write_atomic(run_dir / "FAILED.json", failed)
    log_event({"event": "eval_repair_failed", "run_id": rid, "attempt": attempt,
               "exit_code": rc, "time": _now()})
    logger.warning("%s: rescore failed again (exit %d)", rid, rc)
    return 3


def cmd_pin(args) -> int:
    path = code_pin_path()
    if path.exists():
        raise SweepError(f"{wc.rel_to_root(path)} exists; re-pinning is an amendment (delete it "
                         f"by hand and record why)")
    code_hash, files = wc.code_sha256()
    probe = import_probe()
    uncovered = sorted(set(probe) - set(files))
    if uncovered:
        raise SweepError(f"import probe loads repo files outside the code pin: {uncovered}")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - recorded, not required
        head = None
    pin = {"git_head": head, "code_sha256": code_hash, "files": files, "created": _now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, pin)
    wc.append_pin("G5", {"code_sha256": code_hash, "git_head": head,
                         "code_pin_sha256": wc.sha256_file(path), "n_files": len(files)})
    print(json.dumps({"code_sha256": code_hash, "git_head": head, "n_files": len(files)}))
    return 0


# ================================================================ CLI
def _max_runs(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--max-runs must be an integer, got {text!r}")
    if not 1 <= n <= MAX_RUNS_CAP:
        raise argparse.ArgumentTypeError(f"--max-runs must be in 1..{MAX_RUNS_CAP}, got {n}")
    return n


def _workers(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--workers must be an integer, got {text!r}")
    if not 1 <= n <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"--workers must be in 1..{MAX_WORKERS}, got {n}")
    return n


def _k(text: str):
    if text == "all":
        return "all"
    if text.isdigit() and int(text) in wc.KS:
        return int(text)
    raise argparse.ArgumentTypeError(f"k must be one of {list(wc.KS)} or 'all', got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def filters(p, shard_only=False):
        p.add_argument("--shard", type=parse_shard, default=None, metavar="i/N")
        if shard_only:
            return
        p.add_argument("--seeds", type=int, nargs="+", choices=list(wc.SEEDS), default=None)
        p.add_argument("--packs", nargs="+", default=None, help="pack ids, or 'all'")
        p.add_argument("--ks", type=_k, nargs="+", default=None)
        p.add_argument("--cells", nargs="+", choices=list(wc.CELLS), default=None)

    sub.add_parser("pin")
    filters(sub.add_parser("plan"))
    r = sub.add_parser("run")
    r.add_argument("--max-runs", type=_max_runs, required=True,
                   help=f"required; 1..{MAX_RUNS_CAP} runs per invocation")
    r.add_argument("--workers", type=_workers, default=1)
    r.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    r.add_argument("--dry-run", action="store_true")
    filters(r)
    filters(sub.add_parser("status"), shard_only=True)
    c = sub.add_parser("check-group")
    c.add_argument("--run-dirs", type=Path, nargs=4, required=True,
                   metavar=("BASELINE", "LOSS_ONLY", "NEG_ONLY", "BOTH"))
    cg = sub.add_parser("clear-group")
    cg.add_argument("--cond", required=True)
    cg.add_argument("--seed", type=int, required=True)
    cg.add_argument("--amendment", required=True)
    rs = sub.add_parser("rescore")
    rs.add_argument("--run-id", required=True)
    rs.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    return ap


COMMANDS = {"pin": cmd_pin, "plan": cmd_plan, "run": cmd_run, "status": cmd_status,
            "check-group": cmd_check_group, "clear-group": cmd_clear_group,
            "rescore": cmd_rescore}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return COMMANDS[args.cmd](args)
    except SweepError as exc:
        logger.error("%s", exc)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
