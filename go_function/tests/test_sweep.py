"""Sweep driver (plan item 15): shards, seed-major order, claims and stale-claim
release, resume, two failures, OOM retry, eval failures left for rescore,
device forwarding, pin refusal, and no absolute path in any written file."""
import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest

from go_function.constants import EXIT_OK, EXIT_REFUSED

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("sweep_t", ROOT / "scripts" / "go_function" / "sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load()


def _index(root: Path, aspect: str, feature: str, names_seeds):
    d = root / aspect / feature
    d.mkdir(parents=True, exist_ok=True)
    specs = [{"name": n, "block": "main", "sha256": "0", "seed": s,
              "out_dir": f"results/go_function/runs/{aspect}/{feature}/{n}"} for n, s in names_seeds]
    (d / "index.json").write_text(json.dumps({"specs": specs}))


def _grid(tmp_path, monkeypatch):
    """A fake materialised grid of 4 x 10 specs inside the repo-relative namespace."""
    root = tmp_path / "specs"
    for a in ("bp", "mf"):
        for f in ("esm2_650m", "interpro"):
            _index(root, a, f, [(f"V{i % 2}__TP__W2__p01__k0__s{s}", s)
                                for s in (13, 21, 42, 87, 123) for i in range(2)])
    monkeypatch.setattr(S, "repo_rel", lambda p: Path(p).relative_to(tmp_path).as_posix())
    return S.load_grid(root)


def test_grid_is_seed_major_and_shards_are_disjoint_and_exhaustive(tmp_path, monkeypatch):
    items = _grid(tmp_path, monkeypatch)
    assert len(items) == 40
    seeds = [r["seed"] for r in items]
    assert seeds == sorted(seeds, key=(13, 21, 42, 87, 123).index)
    assert [r["index"] for r in items] == list(range(40))
    for n in (1, 3, 7):
        parts = [S.select(items, shard=(i, n)) for i in range(1, n + 1)]
        idx = [r["index"] for p in parts for r in p]
        assert sorted(idx) == list(range(40)) and len(idx) == len(set(idx))
    # the shard is over the full grid, independent of filters
    a = {r["index"] for r in S.select(items, shard=(2, 3), aspect="mf")}
    assert a == {r["index"] for r in S.select(items, shard=(2, 3)) if r["aspect"] == "mf"}
    with pytest.raises(Exception):
        S.parse_shard("4/3")


# ------------------------------------------------------------------ fake runs
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "repo_path", lambda rel: tmp_path / rel)
    calls = []
    behaviour = {}

    def fake(cmd, log_path):
        calls.append(cmd)
        name = Path(cmd[cmd.index("--spec") + 1]).stem
        out = tmp_path / "runs" / "bp" / "interpro" / name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n")
        plan = behaviour.get(name, [])
        n = sum(1 for c in calls if Path(c[c.index("--spec") + 1]).stem == name)
        what = plan[n - 1] if n - 1 < len(plan) else "ok"
        out.mkdir(parents=True, exist_ok=True)
        if what == "ok":
            (out / "DONE.json").write_text("{}\n")
            return 0
        if what == "crash":
            return -11
        stage, err = {"train": ("train", "NonFiniteLoss"), "oom": ("train", "OutOfMemoryError"),
                      "eval": ("eval", "ValueError"), "refused": ("preflight", "Refused"),
                      "preflight": ("preflight", "FatalError")}[what]
        (out / "FAILED.json").write_text(json.dumps({"stage": stage, "error": err,
                                                     "message": "x"}))
        return EXIT_REFUSED if what == "refused" else 2

    monkeypatch.setattr(S, "run_subprocess", fake)
    return tmp_path, calls, behaviour


def _items(names):
    return [{"name": n, "aspect": "bp", "feature": "interpro", "seed": 13, "block": "smoke",
             "variant": n.split("__")[0], "spec": f"specs/{n}.json",
             "out_dir": f"runs/bp/interpro/{n}", "index": i} for i, n in enumerate(names)]


def test_resume_skips_done_runs_and_device_is_forwarded(env):
    tmp, calls, _ = env
    items = _items(["V0__a", "V3__b", "V3c__c"])
    s1 = S.run_items(items, "cuda", workers=2)
    assert s1["done"] == 3 and len(calls) == 3
    assert all(c[c.index("--device") + 1] == "cuda" for c in calls)
    assert all(c[1] == "scripts/go_function/run.py" for c in calls)
    todo = [r for r in items if S.run_state(tmp / r["out_dir"]) in S.PENDING]
    assert todo == []
    s2 = S.run_items(items, "cuda", workers=2)      # DONE -> never re-run
    assert s2["skipped"] == 3 and len(calls) == 3
    assert not list((tmp / "runs" / "bp" / "interpro").glob("*.CLAIM"))


def test_two_failures_leave_failed_json_and_one_attempt_dir(env):
    tmp, calls, beh = env
    beh["V0__x"] = ["train", "train"]
    beh["V0__y"] = ["crash", "ok"]
    s = S.run_items(_items(["V0__x", "V0__y"]), "cpu", workers=1)
    assert s["results"] == {"V0__x": "failed", "V0__y": "done"}
    d = tmp / "runs" / "bp" / "interpro"
    assert S.run_state(d / "V0__x") == "failed"
    assert (d / "V0__x.attempt1").is_dir() and (d / "V0__x" / "FAILED.json").exists()
    assert json.loads((d / "V0__y.attempt1" / "FAILED.json").read_text())["by"] == "sweep"
    assert S.run_items(_items(["V0__x"]), "cpu", 1)["skipped"] == 1      # final: not retried


def test_oom_lowers_workers_and_retries_once(env):
    tmp, calls, beh = env
    beh["V4__o"] = ["oom", "ok"]
    s = S.run_items(_items(["V4__o", "V0__p", "V0__q"]), "cuda", workers=3)
    assert s["results"]["V4__o"] == "done" and s["workers_final"] == 2
    assert (tmp / "runs" / "bp" / "interpro" / "V4__o.attempt1").is_dir()


def test_eval_failure_is_left_for_rescore(env):
    tmp, calls, beh = env
    beh["V0__e"] = ["eval"]
    s = S.run_items(_items(["V0__e"]), "cpu", workers=1)
    assert s["results"]["V0__e"] == "failed_eval" and len(calls) == 1
    assert S.run_state(tmp / "runs" / "bp" / "interpro" / "V0__e") == "failed_eval"
    assert S.runner_cmd(_items(["V0__e"])[0], "cuda", rescore=True)[-1] == "--rescore"


def test_refused_run_uses_no_attempt_stays_pending_and_stops_the_invocation(env):
    tmp, calls, beh = env
    beh["V0__r"] = ["refused"]
    s = S.run_items(_items(["V0__r", "V0__s", "V0__t"]), "cpu", workers=1)
    d = tmp / "runs" / "bp" / "interpro"
    assert s["results"] == {"V0__r": "refused"} and s["refused"] == 1 and len(calls) == 1
    assert (d / "V0__r.refused1" / "FAILED.json").exists() and not S.attempt_dirs(d / "V0__r")
    assert S.run_state(d / "V0__r") == "absent"                  # still pending
    assert S.run_items(_items(["V0__r"]), "cpu", 1)["results"] == {"V0__r": "done"}


def test_consecutive_preflight_failures_stop_the_invocation(env):
    tmp, calls, beh = env
    names = [f"V0__f{i}" for i in range(6)]
    for n in names:
        beh[n] = ["preflight", "preflight"]
    s = S.run_items(_items(names), "cpu", workers=1)
    assert s["failed"] == S.PREFLIGHT_STOP and s["started"] == S.PREFLIGHT_STOP
    assert len(calls) == 2 * S.PREFLIGHT_STOP and "stage preflight" in s["errors"][0]
    for n in names[S.PREFLIGHT_STOP:]:
        assert S.run_state(tmp / "runs" / "bp" / "interpro" / n) == "absent"


def test_claim_from_an_earlier_boot_is_stale(env, monkeypatch):
    tmp, _, _ = env
    d = tmp / "runs" / "bp" / "interpro" / "V0__b"
    d.parent.mkdir(parents=True, exist_ok=True)
    assert S.claim(d)
    assert json.loads(S.claim_path(d).read_text())["boot"] == S.boot_id()
    assert S.run_state(d) == "in_progress"                        # same boot, live pid
    S.claim_path(d).write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                           "boot": "an-earlier-boot"}))
    assert S.run_state(d) == "stale"                              # live (reused) pid, old boot


def test_claims_live_stale_and_release(env, monkeypatch, capsys):
    tmp, calls, _ = env
    d = tmp / "runs" / "bp" / "interpro" / "V0__c"
    assert S.claim(d) is True and S.claim(d) is False            # O_EXCL
    assert S.run_state(d) == "in_progress"                        # our own live pid
    assert S.run_items(_items(["V0__c"]), "cpu", 1)["skipped"] == 1
    S.release(d)
    # a stale claim: dead pid on this host, with a partial run dir
    d.mkdir(parents=True)
    (d / "epochs.jsonl").write_text("{}\n")
    S.claim_path(d).write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": socket.gethostname()}))
    assert S.run_state(d) == "stale"
    other = tmp / "runs" / "bp" / "interpro" / "V0__h"
    S.claim_path(other).write_text(json.dumps({"pid": 1, "host": "another-host"}))
    assert S.run_state(other) == "in_progress"                    # never probed
    monkeypatch.setattr(S, "_items", lambda args, require_all=False: _items(["V0__c", "V0__h"]))
    assert S.main(["status", "--release-stale"]) == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["released"] == ["V0__c"]
    assert not S.claim_path(d).exists() and (d.parent / "V0__c.interrupted1").is_dir()
    assert S.claim_path(other).exists()
    assert S.run_state(d) == "absent"


def test_run_refused_without_code_pin_and_pin_refused_when_uncommitted(env, monkeypatch):
    tmp, calls, _ = env
    monkeypatch.setattr(S, "RESULTS", tmp / "results")
    assert S.main(["run", "--max-runs", "1", "--device", "cpu"]) == EXIT_REFUSED
    monkeypatch.setattr(S, "git_committed", lambda: None)
    assert S.main(["pin"]) == EXIT_REFUSED
    assert not (tmp / "results" / "code_pin.json").exists() and calls == []


def test_pin_with_accepted_content_pin_hashes_study_code(env, monkeypatch):
    tmp, _, _ = env
    res = tmp / "results"
    monkeypatch.setattr(S, "RESULTS", res)
    monkeypatch.setattr(S, "git_committed", lambda: None)
    for a in ("bp", "mf"):
        for f in ("esm2_650m", "interpro"):
            (res / "specs" / a / f).mkdir(parents=True)
            (res / "specs" / a / f / "index.json").write_text(json.dumps({"spec_set_sha256": a + f}))
    got = {}
    monkeypatch.setattr(S, "record_pin", lambda key, entries: got.update({key: entries}) or "written")
    monkeypatch.setattr(S, "write_or_verify_file",
                        lambda p, b: (Path(p).parent.mkdir(parents=True, exist_ok=True),
                                      Path(p).write_bytes(b), "written")[-1])
    assert S.main(["pin", "--accept-content-pin"]) == EXIT_OK
    doc = json.loads((res / "code_pin.json").read_text())
    assert "go_function/train.py" in doc["files"] and "scripts/go_function/sweep.py" in doc["files"]
    assert "config/go_function/study.json" in doc["files"]
    assert not any("__pycache__" in f for f in doc["files"])
    assert doc["content_pin_accepted"] is True and got["G8"]["spec_sets"]["bp/interpro"] == "bpinterpro"


def test_no_absolute_path_in_any_written_file(env):
    tmp, _, beh = env
    beh["V0__f"] = ["train", "crash"]
    S.run_items(_items(["V0__ok", "V0__f"]), "cpu", 1)
    assert "<path>" in S.scrub(f"File {ROOT}/x.py and /Users/someone/y")
    assert str(ROOT) not in S.scrub(f"{ROOT}/go_function/train.py")
    files = [p for p in tmp.rglob("*") if p.is_file()]
    assert files
    for p in files:
        txt = p.read_text(errors="ignore")
        assert "/Users/" not in txt and str(ROOT) not in txt


def test_smoke_cells():
    g9a = S.smoke_cells("interpro")
    assert len(g9a) == 6 and {c[0] for c in g9a} == {"V0", "V3", "V3c"}
    assert {c[3] for c in g9a} == {0, 5} and {(c[1], c[2]) for c in g9a} == {("TP", "W2")}
    g9b = S.smoke_cells("esm2_650m")
    assert ("V4", "DGZ", "WDGZrel", 0) in g9b and ("V6", "TP", "W2", 5) in g9b and len(g9b) == 13
