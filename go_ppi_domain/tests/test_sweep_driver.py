"""Sweep-driver tests (design 12.3, 14, 15; plan decisions 4a-4c; user decision B).

Subprocesses are stubbed (``sweep.run_subprocess``), the study paths and pins are
monkeypatched to ``tmp_path``, and the lc_manifests are synthetic.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from go_ppi_domain import withholding as wh
from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.tests import _grid_fixtures as gf
from scripts.go_withholding import analyze_withholding as aw
from scripts.go_withholding import sweep as sw

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ALL13 = wc.group_run_ids("all", 13)


class FakeProcs:
    """Stub runner + evaluator: writes what the real ones would on success."""

    def __init__(self):
        self.calls = []
        self.cmds = []
        self.train_rc = {}
        self.eval_rc = {}
        self.sel = {}
        self.on_train = None

    def __call__(self, cmd):
        self.cmds.append(list(cmd))
        if cmd[1].endswith("run_learning_curve_point.py"):
            out = Path(cmd[cmd.index("--output-dir") + 1])
            cell = cmd[cmd.index("--study-cell") + 1]
            seed = int(cmd[cmd.index("--seed") + 1])
            cond = Path(cmd[cmd.index("--withholding-manifest") + 1]).parent.name
            self.calls.append(("train", out.name))
            if self.on_train:
                self.on_train(out.name)
            rcs = self.train_rc.get(out.name)
            rc = rcs.pop(0) if rcs else 0
            if rc == 0:
                m = gf.lc_manifest(cond, cell, seed, sel=self.sel.get(out.name))
                (out / "lc_manifest.json").write_text(json.dumps(m))
                (out / m["withholding"]["selected_checkpoint"]).write_bytes(b"ckpt")
            return rc
        run = Path(cmd[cmd.index("--run-dir") + 1])
        self.calls.append(("eval", run.name))
        rcs = self.eval_rc.get(run.name)
        rc = rcs.pop(0) if rcs else 0
        (run / "eval").mkdir(exist_ok=True)
        if rc == 0:
            (run / "eval" / "metrics.json").write_text(json.dumps({"run_id": run.name}))
        else:
            (run / "eval" / "eval_error.json").write_text("{}")
        return rc

    def trained(self):
        return [r for k, r in self.calls if k == "train"]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    table = gf.make_table(3)
    targets = tmp_path / "targets"
    gf.write_table(table, targets)
    res, conf, conds = tmp_path / "results", tmp_path / "config", tmp_path / "conditions"
    conf.mkdir()
    (tmp_path / "emb.pt").write_bytes(b"warm")
    (conf / "baseline.json").write_text(json.dumps({"embedding_cache_path": str(tmp_path / "emb.pt")}))
    monkeypatch.setattr(wc, "STUDY_RES", res)
    monkeypatch.setattr(wc, "TARGETS", targets)
    monkeypatch.setattr(wc, "CONFIG_DIR", conf)
    monkeypatch.setattr(wc, "CONDITIONS", conds)
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(tmp_path / "pins.json"))
    monkeypatch.setattr(sw, "pin_preflight",
                        lambda table: {"code_sha256": "code", "cache": tmp_path / "c.npz"})
    fake = FakeProcs()
    monkeypatch.setattr(sw, "run_subprocess", fake)
    return SimpleNamespace(tmp=tmp_path, table=table, runs=res / "runs", fake=fake, res=res)


def _dirs(env, cond="all", seed=13):
    return [env.runs / r for r in wc.group_run_ids(cond, seed)]


def _log(env):
    return sw.read_log()


def _status(capsys):
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


def _doctor(d: Path, **sel):
    """Rewrite d's lc_manifest selection hashes and refresh its DONE.json."""
    m = json.loads((d / "lc_manifest.json").read_text())
    m["negative_selection"]["selection_sha256_by_epoch"].update(sel)
    (d / "lc_manifest.json").write_text(json.dumps(m))
    done = json.loads((d / "DONE.json").read_text())
    done["lc_manifest_sha256"] = wc.sha256_file(d / "lc_manifest.json")
    (d / "DONE.json").write_text(json.dumps(done))


# ================================================================= CLI bounds
def test_run_requires_max_runs_and_caps_it(env):
    for argv in (["run"], ["run", "--max-runs", "101"], ["run", "--max-runs", "0"],
                 ["run", "--max-runs", "x"], ["run", "--max-runs", "1", "--workers", "0"],
                 ["run", "--max-runs", "1", "--shard", "4/3"]):
        with pytest.raises(SystemExit) as e:
            sw.main(argv)
        assert e.value.code == 2, argv
    assert env.fake.calls == []


def test_max_runs_bounds_the_invocation_even_for_the_full_grid(env, capsys):
    assert env.table["n_runs"] == 320
    assert sw.main(["run", "--max-runs", "100", "--dry-run"]) == 0
    out = _status(capsys)
    assert len(out["would_run"]) == 100
    assert out["would_run"][:5] == ALL13 + ["p01__k0__baseline__s13"]   # seed-major
    assert env.fake.calls == []
    assert "--all" not in sw.build_parser().format_help()


def test_subprocess_commands_are_argument_lists_and_forward_device(env):
    assert sw.main(["run", "--max-runs", "1", "--device", "cuda"]) == 0
    train, ev = env.fake.cmds
    assert train[0] == sys.executable and train[1].endswith("run_learning_curve_point.py")
    assert train[train.index("--study-cell") + 1] == "baseline"
    assert train[train.index("--seed") + 1] == "13"
    assert train[-2:] == ["--device", "cuda"] and ev[-2:] == ["--device", "cuda"]
    assert ev[2] == "score" and "--relation" in ev
    assert all(isinstance(x, str) for x in train + ev)


# ================================================================= lifecycle
def test_group_lifecycle_done_markers_and_group_ok(env):
    assert sw.main(["run", "--max-runs", "4"]) == 0
    assert env.fake.trained() == ALL13
    for d in _dirs(env):
        done = json.loads((d / "DONE.json").read_text())
        assert set(done) == {"lc_manifest_sha256", "metrics_sha256", "code_sha256", "finished"}
        assert done["lc_manifest_sha256"] == wc.sha256_file(d / "lc_manifest.json")
        assert not (d / "RUNNING.json").exists()
    assert all(wc.group_ok_valid(_dirs(env)))
    ok = json.loads((_dirs(env)[0] / "GROUP_OK.json").read_text())
    assert ok["run_ids"] == ALL13 and set(ok["selection_sha256_by_epoch"]) == {
        "baseline_loss_only", "neg_only_both"}
    lines = [e for e in _log(env) if e.get("stage") == "train"]
    assert len(lines) == 4 and all(set(e) == {"run_id", "stage", "attempt", "exit_code", "time", "wall_seconds"}
                                   for e in lines)


def test_rerunning_a_done_run_is_a_noop(env):
    assert sw.main(["run", "--max-runs", "4"]) == 0
    before = {d.name: (d / "DONE.json").read_text() for d in _dirs(env)}
    env.fake.calls.clear()
    assert sw.main(["run", "--max-runs", "4", "--packs", "all", "--seeds", "13"]) == 0
    assert env.fake.calls == []
    assert {d.name: (d / "DONE.json").read_text() for d in _dirs(env)} == before
    assert sw.main(["run", "--max-runs", "2"]) == 0
    assert env.fake.trained() == ["p01__k0__baseline__s13", "p01__k0__loss_only__s13"]


def test_eval_only_failure_rescores_without_retraining(env):
    rid = ALL13[0]
    env.fake.eval_rc[rid] = [1, 0]
    assert sw.main(["run", "--max-runs", "1"]) == 0
    assert env.fake.calls == [("train", rid), ("eval", rid), ("eval", rid)]
    assert (env.runs / rid / "eval.attempt1" / "eval_error.json").is_file()
    assert (env.runs / rid / "DONE.json").is_file()


def test_second_eval_failure_writes_failed_with_manifest_sha(env):
    rid = ALL13[0]
    env.fake.eval_rc[rid] = [1, 1]
    assert sw.main(["run", "--max-runs", "2"]) == 0
    f = json.loads((env.runs / rid / "FAILED.json").read_text())
    assert f["stage"] == "eval" and f["attempts"] == 2
    assert f["lc_manifest_sha256"] == wc.sha256_file(env.runs / rid / "lc_manifest.json")
    assert not (env.runs / rid / "DONE.json").exists()
    assert env.fake.trained() == ALL13[:2]          # the driver continued


def test_training_failure_renames_attempt1_and_retrains(env):
    rid = ALL13[0]
    env.fake.train_rc[rid] = [1, 0]
    assert sw.main(["run", "--max-runs", "1"]) == 0
    assert env.fake.trained() == [rid, rid]
    assert (env.runs / f"{rid}.attempt1").is_dir()
    assert (env.runs / rid / "DONE.json").is_file()


def test_second_training_failure_writes_failed_and_continues(env):
    rid = ALL13[0]
    env.fake.train_rc[rid] = [1, 3]
    assert sw.main(["run", "--max-runs", "2"]) == 0
    f = json.loads((env.runs / rid / "FAILED.json").read_text())
    assert f["stage"] == "train" and f["attempts"] == 2 and f["exit_code"] == 3
    assert env.fake.trained() == [rid, rid, ALL13[1]]
    assert ("eval", rid) not in env.fake.calls


def test_pilot_run_dirs_are_refused_by_check_group(env, tmp_path):
    dirs = gf.write_group(env.runs, env.table, "all", 13)
    m = json.loads((dirs[0] / "lc_manifest.json").read_text())
    m["pilot"] = True
    (dirs[0] / "lc_manifest.json").write_text(json.dumps(m))
    assert sw.main(["check-group", "--run-dirs", *map(str, dirs)]) == 2


def test_check_group_on_smoke_dirs_without_done(env, tmp_path):
    smoke = tmp_path / "smoke"
    dirs = []
    for cell in wc.CELLS:
        d = smoke / cell
        d.mkdir(parents=True)
        (d / "lc_manifest.json").write_text(json.dumps(gf.lc_manifest("p01_k0", cell, 13, epochs=2)))
        dirs.append(d)
    assert sw.main(["check-group", "--run-dirs", *map(str, dirs)]) == 0
    ok = json.loads((dirs[0] / "GROUP_OK.json").read_text())
    assert ok["run_ids"] == list(wc.CELLS)
    (dirs[1] / "GROUP_OK.json").unlink()
    _ = [p.unlink() for p in (d / "GROUP_OK.json" for d in dirs) if p.exists()]
    m = gf.lc_manifest("p01_k0", "loss_only", 13, epochs=2, sel={"0": "X", "1": "Y"})
    (dirs[1] / "lc_manifest.json").write_text(json.dumps(m))
    assert sw.main(["check-group", "--run-dirs", *map(str, dirs)]) == 4
    assert all((d / "GROUP_FAILED.json").is_file() for d in dirs)


# ================================================================= group-check scans
def test_interrupt_case_checked_before_first_training_and_by_status(env, capsys):
    gf.write_group(env.runs, env.table, "all", 13, checked=False)
    seen = []
    env.fake.on_train = lambda rid: seen.append(all(wc.group_ok_valid(_dirs(env))))
    assert sw.main(["run", "--max-runs", "1"]) == 0
    assert seen == [True]


def test_interrupt_case_status_alone(env, capsys):
    gf.write_group(env.runs, env.table, "all", 13, checked=False)
    assert sw.main(["status"]) == 0
    assert all(wc.group_ok_valid(_dirs(env)))
    assert _status(capsys)["n_groups_ok"] == 1


def test_cells_filter_group_checked_after_the_fourth(env):
    for i, cell in enumerate(wc.CELLS):
        assert sw.main(["run", "--max-runs", "1", "--cells", cell]) == 0
        present = [(d / "GROUP_OK.json").exists() for d in _dirs(env)]
        assert present == ([False] * 4 if i < 3 else [True] * 4)
    assert env.fake.trained() == ALL13


def test_mismatch_writes_group_failed_and_exits_4(env):
    env.fake.sel[ALL13[1]] = {str(e): f"Z{e}" for e in range(15)}
    assert sw.main(["run", "--max-runs", "6"]) == 4
    assert env.fake.trained() == ALL13            # stopped after the group failed
    for d in _dirs(env):
        g = json.loads((d / "GROUP_FAILED.json").read_text())
        assert "selection_sha256_by_epoch: baseline != loss_only" in g["failures"]
        assert not (d / "GROUP_OK.json").exists()


def test_mismatch_from_status(env, capsys):
    dirs = gf.write_group(env.runs, env.table, "all", 13, checked=False)
    _doctor(dirs[1], **{"0": "Z"})
    assert sw.main(["status"]) == 4
    assert all((d / "GROUP_FAILED.json").is_file() for d in dirs)


def test_stopped_grid_refuses_run_warns_plan_reports_status(env, monkeypatch, capsys, caplog):
    dirs = gf.write_group(env.runs, env.table, "all", 13, checked=False)
    _doctor(dirs[1], **{"0": "Z"})
    assert sw.main(["status"]) == 4
    capsys.readouterr()
    calls = []
    real = sw.check_pending_groups
    monkeypatch.setattr(sw, "check_pending_groups", lambda *a: calls.append(1) or real(*a))
    env.fake.calls.clear()
    assert sw.main(["run", "--max-runs", "1"]) == 4
    assert env.fake.calls == [] and calls == []
    assert "grid stopped: group all,s13" in caplog.text
    caplog.clear()
    assert sw.main(["plan"]) == 0
    assert "WARNING" in caplog.text or any(r.levelname == "WARNING" for r in caplog.records)
    capsys.readouterr()
    assert sw.main(["status"]) == 0
    st = _status(capsys)
    assert st["n_groups_group_failed"] == 1 and st["groups_group_failed"] == ["all,s13"]


def test_partial_group_ok_is_completed_and_stale_rewritten(env, capsys):
    dirs = gf.write_group(env.runs, env.table, "all", 13)
    (dirs[2] / "GROUP_OK.json").unlink()
    (dirs[3] / "GROUP_OK.json").unlink()
    assert sw.main(["status"]) == 0
    oks = [json.loads((d / "GROUP_OK.json").read_text()) for d in dirs]
    assert all(o["run_ids"] == oks[0]["run_ids"] and
               o["lc_manifest_sha256"] == oks[0]["lc_manifest_sha256"] for o in oks)
    stale = json.loads((dirs[0] / "GROUP_OK.json").read_text())
    stale["lc_manifest_sha256"][dirs[0].name] = "0" * 64
    (dirs[0] / "GROUP_OK.json").write_text(json.dumps(stale))
    assert not all(wc.group_ok_valid(dirs))
    assert sw.main(["status"]) == 0
    assert all(wc.group_ok_valid(dirs))


def test_mismatch_on_recheck_supersedes_group_ok(env):
    dirs = gf.write_group(env.runs, env.table, "all", 13)
    (dirs[2] / "GROUP_OK.json").unlink()
    (dirs[3] / "GROUP_OK.json").unlink()
    _doctor(dirs[0], **{"0": "Z"})            # GROUP_OK of dir 0 now stale -> eligible
    assert sw.main(["status"]) == 4
    for d in dirs:
        assert (d / "GROUP_FAILED.json").is_file() and not (d / "GROUP_OK.json").exists()
    assert (dirs[0] / "GROUP_OK.superseded.json").is_file()
    assert (dirs[1] / "GROUP_OK.superseded.json").is_file()


def test_manifest_drift_marks_group_failed_without_checking(env, monkeypatch):
    calls = []
    real = sw.group_checks
    monkeypatch.setattr(sw, "group_checks", lambda *a, **k: calls.append(1) or real(*a, **k))
    dirs = gf.write_group(env.runs, env.table, "all", 13, checked=False)
    m = json.loads((dirs[2] / "lc_manifest.json").read_text())
    m["train_records"] = 7
    (dirs[2] / "lc_manifest.json").write_text(json.dumps(m))
    assert sw.main(["status"]) == 4
    assert calls == []
    for d in dirs:
        g = json.loads((d / "GROUP_FAILED.json").read_text())
        assert g["failures"] == [f"lc_manifest_changed_after_done: {dirs[2].name}"]


def test_manifest_drift_supersedes_a_valid_looking_group_ok(env):
    dirs = gf.write_group(env.runs, env.table, "all", 13)
    m = json.loads((dirs[1] / "lc_manifest.json").read_text())
    m["train_records"] = 7
    (dirs[1] / "lc_manifest.json").write_text(json.dumps(m))
    assert sw.main(["status"]) == 4
    assert all((d / "GROUP_OK.superseded.json").is_file() for d in dirs)
    assert not any((d / "GROUP_OK.json").exists() for d in dirs)


def test_failed_member_group_is_never_checked(env, monkeypatch):
    calls = []
    real = sw.group_checks
    monkeypatch.setattr(sw, "group_checks", lambda *a, **k: calls.append(1) or real(*a, **k))
    dirs = gf.write_group(env.runs, env.table, "all", 13, checked=False)
    (dirs[3] / "FAILED.json").write_text(json.dumps({"stage": "train"}))
    assert sw.main(["status"]) == 0
    assert calls == [] and not any((d / "GROUP_OK.json").exists() for d in dirs)


# ================================================================= clearing / amendments
def proto(rows=(), where=13, prose="", extra_pins=True) -> str:
    head = "| id | date | gate | groups cleared | cause | fix | re-pin |\n|---|---|---|---|---|---|---|\n"
    pins = "| pin key | value | date |\n|---|---|---|\n| G1 | | |\n\n" if extra_pins else ""
    table = head + "".join(r + "\n" for r in rows)
    s12 = "## 12. Limits\n\n" + (table if where == 12 else "") + "\n"
    s13 = "## 13. Pins and amendments\n\n" + pins + (prose + "\n\n" if prose else "") + \
        (table if where == 13 else head) + "\n"
    s14 = "## 14. Changes\n\n" + (table if where == 14 else "") + "\n"
    return "# Protocol\n\n" + s12 + s13 + s14


def ROW(groups="p03_k5,s21", date="2025-11-01", cause="selector bug", fix="patched", aid="AM1"):
    return f"| {aid} | {date} | G7 | {groups} | {cause} | {fix} | no |"


def _protocol(env, monkeypatch, text):
    p = env.tmp / "protocol.md"
    p.write_text(text)
    monkeypatch.setattr(wc, "PROTOCOL", p)
    return p


def _failed_group(env, cond, seed, marked=(0, 1, 2, 3), cells=wc.CELLS):
    dirs = []
    for i, cell in enumerate(wc.CELLS):
        if cell not in cells:
            continue
        rid = wc.run_id_for(cond, cell, seed)
        d = gf.write_run(env.runs, rid, gf.lc_manifest(cond, cell, seed))
        if i in marked:
            (d / "GROUP_FAILED.json").write_text(json.dumps({"failures": ["x"]}))
        dirs.append(d)
    return dirs


def test_clear_group_renames_logs_and_next_run_retrains(env, monkeypatch, capsys):
    _protocol(env, monkeypatch, proto([ROW(groups="all,s13")]))
    _failed_group(env, "all", 13)
    (env.runs / f"{ALL13[0]}.groupfailed1").mkdir()
    assert sw.main(["clear-group", "--cond", "all", "--seed", "13", "--amendment", "AM1"]) == 0
    assert (env.runs / f"{ALL13[0]}.groupfailed2").is_dir()
    assert all((env.runs / f"{r}.groupfailed1").is_dir() for r in ALL13[1:])
    assert not any((env.runs / r).exists() for r in ALL13)
    lines = [e for e in _log(env) if e.get("event") == "group_cleared"]
    assert len(lines) == 1 and lines[0]["amendment_row"]["groups cleared"] == "all,s13"
    assert lines[0]["renamed_to"][0] == f"{ALL13[0]}.groupfailed2"
    assert aw.classify_groups(env.runs, env.table)[("all", 13)] == "incomplete"
    assert sw.main(["run", "--max-runs", "4"]) == 0
    assert env.fake.trained() == ALL13
    capsys.readouterr()
    assert sw.main(["status"]) == 0
    assert _status(capsys)["n_groups_cleared"] == 1


def test_amendment_a_frozen_v6_section_13_refuses(env, monkeypatch):
    monkeypatch.setattr(wc, "PROTOCOL", FIXTURES / "protocol_s13_v6.md")
    assert wc.parse_amendment_rows() == []
    _failed_group(env, "p03_k5", 21)
    assert sw.main(["clear-group", "--cond", "p03_k5", "--seed", "21", "--amendment", "AM1"]) == 2
    assert all((env.runs / r).is_dir() for r in wc.group_run_ids("p03_k5", 21))


def test_real_protocol_amendments_table_parses():
    rows = wc.parse_amendment_rows(wc.ROOT / ".agents/tasks/ontology-withholding-feasibility/"
                                   "protocol_go_ppi.md")
    assert [r["id"] for r in rows] == ["AM1"]
    assert all(len(r) == 7 for r in rows)
    text = (wc.ROOT / ".agents/tasks/ontology-withholding-feasibility/protocol_go_ppi.md"
            ).read_text(encoding="utf-8")
    s13 = text[text.index("## 13."):text.index("## 14.")]
    assert "a literal `|` inside a cell is written `\\|`" in s13       # plan decision 4b


def test_amendment_b_row_allows_listed_group_only(env, monkeypatch):
    _protocol(env, monkeypatch, proto([ROW()]))
    _failed_group(env, "p03_k5", 21)
    _failed_group(env, "p02_k0", 13)
    assert sw.main(["clear-group", "--cond", "p02_k0", "--seed", "13", "--amendment", "AM1"]) == 2
    assert all((env.runs / r).is_dir() for r in wc.group_run_ids("p02_k0", 13))
    assert sw.main(["clear-group", "--cond", "p03_k5", "--seed", "21", "--amendment", "AM1"]) == 0


@pytest.mark.parametrize("row", [ROW(fix=""), ROW(cause=""), ROW(date="2025-13-01"),
                                 ROW(date="01/11/2025"), ROW(date="2025-1-01")])
def test_amendment_c_bad_rows_refuse(env, monkeypatch, row):
    _protocol(env, monkeypatch, proto([row]))
    _failed_group(env, "p03_k5", 21)
    assert sw.main(["clear-group", "--cond", "p03_k5", "--seed", "21", "--amendment", "AM1"]) == 2
    assert all((env.runs / r).is_dir() for r in wc.group_run_ids("p03_k5", 21))


def test_amendment_d_prose_duplicates_and_bad_ids(env, monkeypatch):
    _failed_group(env, "p03_k5", 21)
    args = ["clear-group", "--cond", "p03_k5", "--seed", "21", "--amendment"]
    _protocol(env, monkeypatch, proto([], prose="AM1 clears p03_k5,s21 (prose only)."))
    assert sw.main(args + ["AM1"]) == 2
    _protocol(env, monkeypatch, proto([ROW(), ROW()]))
    assert sw.main(args + ["AM1"]) == 2
    _protocol(env, monkeypatch, proto([ROW(aid="AM0"), ROW(aid="AM01"), ROW(aid="A1")]))
    for bad in ("AM0", "AM01", "A1"):
        assert sw.main(args + [bad]) == 2
    assert all((env.runs / r).is_dir() for r in wc.group_run_ids("p03_k5", 21))


@pytest.mark.parametrize("where", [12, 14])
def test_amendment_e_rows_outside_section_13_are_not_found(env, monkeypatch, where):
    _protocol(env, monkeypatch, proto([ROW()], where=where))
    _failed_group(env, "p03_k5", 21)
    assert sw.main(["clear-group", "--cond", "p03_k5", "--seed", "21", "--amendment", "AM1"]) == 2


def test_amendment_f_one_row_clears_a_group_once(env, monkeypatch):
    _protocol(env, monkeypatch, proto([ROW()]))
    _failed_group(env, "p03_k5", 21)
    args = ["clear-group", "--cond", "p03_k5", "--seed", "21", "--amendment", "AM1"]
    assert sw.main(args) == 0
    _failed_group(env, "p03_k5", 21)            # retrained and failed again
    assert sw.main(args) == 2
    assert all((env.runs / r).is_dir() for r in wc.group_run_ids("p03_k5", 21))


def test_amendment_g_pins_table_ignored_backticks_and_escaped_pipe(env, monkeypatch):
    row = ROW(groups="`p03_k5,s21`; `all,s13`", cause=r"cells split on \| only when unescaped")
    p = _protocol(env, monkeypatch, proto([row]))
    rows = wc.parse_amendment_rows(p)
    assert len(rows) == 1 and rows[0]["cause"] == "cells split on | only when unescaped"
    _failed_group(env, "p03_k5", 21)
    _failed_group(env, "all", 13)
    assert sw.main(["clear-group", "--cond", "p03_k5", "--seed", "21", "--amendment", "AM1"]) == 0
    assert sw.main(["clear-group", "--cond", "all", "--seed", "13", "--amendment", "AM1"]) == 0


def test_partial_group_failed_stops_run_and_clears_all_dirs(env, monkeypatch):
    _protocol(env, monkeypatch, proto([ROW(groups="all,s13")]))
    _failed_group(env, "all", 13, marked=(0, 2))
    assert sw.main(["run", "--max-runs", "1"]) == 4
    assert env.fake.calls == []
    assert sw.main(["clear-group", "--cond", "all", "--seed", "13", "--amendment", "AM1"]) == 0
    assert all((env.runs / f"{r}.groupfailed1").is_dir() for r in ALL13)


def test_partial_group_with_three_dirs_and_unmarked_group(env, monkeypatch):
    _protocol(env, monkeypatch, proto([ROW(groups="all,s13; p01_k0,s13")]))
    _failed_group(env, "all", 13, marked=(1,), cells=wc.CELLS[:3])
    assert sw.main(["clear-group", "--cond", "all", "--seed", "13", "--amendment", "AM1"]) == 0
    assert sorted(p.name for p in env.runs.iterdir()) == sorted(f"{r}.groupfailed1"
                                                                for r in ALL13[:3])
    _failed_group(env, "p01_k0", 13, marked=())
    assert sw.main(["clear-group", "--cond", "p01_k0", "--seed", "13", "--amendment", "AM1"]) == 2


# ================================================================= rescore (review v5 finding 4, plan 4c)
def _eval_failed(env, rid=None, stage="eval"):
    dirs = gf.write_group(env.runs, env.table, "all", 13, checked=False)
    d = dirs[1] if rid is None else env.runs / rid
    (d / "DONE.json").unlink()
    (d / "eval" / "metrics.json").write_text("{}")
    (d / "FAILED.json").write_text(json.dumps({
        "stage": stage, "attempts": 2, "lc_manifest_sha256": wc.sha256_file(d / "lc_manifest.json")}))
    return dirs, d


def test_rescore_repairs_eval_stage_failure(env, capsys):
    dirs, d = _eval_failed(env)
    assert sw.main(["rescore", "--run-id", d.name]) == 0
    assert (d / "DONE.json").is_file() and (d / "FAILED.repaired.json").is_file()
    assert not (d / "FAILED.json").exists() and (d / "eval.attempt1").is_dir()
    assert [e["event"] for e in _log(env) if "event" in e] == ["eval_repaired"]
    assert env.fake.trained() == []
    assert sw.main(["status"]) == 0
    assert all(wc.group_ok_valid(dirs))


def test_rescore_refusals(env, monkeypatch):
    dirs, d = _eval_failed(env)
    args = ["rescore", "--run-id", d.name]
    f = json.loads((d / "FAILED.json").read_text())
    (d / "FAILED.json").write_text(json.dumps(dict(f, stage="train")))
    assert sw.main(args) == 2
    (d / "FAILED.json").unlink()
    assert sw.main(args) == 2
    (d / "FAILED.json").write_text(json.dumps(f))
    (d / "DONE.json").write_text("{}")
    assert sw.main(args) == 2
    (d / "DONE.json").unlink()
    ck = d / "checkpoint_epoch_14.pt"
    ck.rename(d / "moved.pt")
    assert sw.main(args) == 2
    (d / "moved.pt").rename(ck)

    def bad_pins(table):
        raise sw.SweepError("pin mismatch: G3")
    monkeypatch.setattr(sw, "pin_preflight", bad_pins)
    assert sw.main(args) == 2
    assert sw.main(["rescore", "--run-id", "nope__k0__baseline__s13"]) == 2
    assert env.fake.calls == [] and (d / "FAILED.json").is_file()


def test_rescore_refuses_after_lc_manifest_changed(env):
    dirs, d = _eval_failed(env)
    m = json.loads((d / "lc_manifest.json").read_text())
    m["train_records"] = 1
    (d / "lc_manifest.json").write_text(json.dumps(m))
    assert sw.main(["rescore", "--run-id", d.name]) == 2
    assert env.fake.calls == []


def test_rescore_exits_4_while_grid_stopped(env):
    dirs, d = _eval_failed(env)
    _failed_group(env, "p01_k0", 13)
    assert sw.main(["rescore", "--run-id", d.name]) == 4
    assert env.fake.calls == []


def test_rescore_failure_keeps_failed_and_counts(env):
    dirs, d = _eval_failed(env)
    env.fake.eval_rc[d.name] = [1]
    assert sw.main(["rescore", "--run-id", d.name]) == 3
    f = json.loads((d / "FAILED.json").read_text())
    assert f["attempts"] == 3 and not (d / "DONE.json").exists()
    ev = [e for e in _log(env) if e.get("event") == "eval_repair_failed"]
    assert len(ev) == 1 and ev[0]["exit_code"] == 1
    assert any(e.get("stage") == "eval" and e.get("attempt") == 1 for e in _log(env))


# ================================================================= analysis preflight
def _analyze(env, *extra):
    tpath = Path(wc.TARGETS) / "target_table.json"
    return aw.main(["--runs", str(env.runs), "--target-table", str(tpath), "--out",
                    str(env.tmp / "analysis"), "--B", "50", *extra])


def test_analysis_preflight_cases(env):
    v = lambda g, p, k, c, s: 0.6
    gf.write_group(env.runs, env.table, "all", 13, value=v, checked=False)
    assert _analyze(env) == 2
    assert _analyze(env, "--allow-incomplete") == 2          # unchecked is never relaxed
    assert sw.main(["status"]) == 0
    assert aw.classify_groups(env.runs, env.table)[("all", 13)] == "checked"
    assert _analyze(env, "--allow-incomplete") == 0          # present once checked
    miss = json.loads((env.tmp / "analysis" / "missing.json").read_text())
    assert miss["n_missing"] == miss["denominator"] - 3
    ok = json.loads((_dirs(env)[0] / "GROUP_OK.json").read_text())
    ok["lc_manifest_sha256"][ALL13[0]] = "0" * 64            # stale marker
    (_dirs(env)[0] / "GROUP_OK.json").write_text(json.dumps(ok))
    assert _analyze(env, "--allow-incomplete") == 2
    assert sw.main(["status"]) == 0
    d = gf.write_group(env.runs, env.table, "p01_k0", 13, value=v)
    (d[1] / "FAILED.json").write_text("{}")
    assert _analyze(env, "--allow-incomplete") == 0
    miss = json.loads((env.tmp / "analysis" / "missing.json").read_text())
    assert [m["reason"] for m in miss["missing"]
            if m["cond_id"] == "p01_k0" and m["seed"] == 13] == ["failed_member"]
    (d[2] / "GROUP_FAILED.json").write_text("{}")
    assert _analyze(env, "--allow-incomplete") == 2


# ================================================================= decision B: shards, workers, resume
def test_shards_partition_the_grid_disjointly_and_exhaustively(env):
    full = sw.grid(env.table)
    ids = [r["run_id"] for r in full]
    assert len(ids) == len(set(ids)) == env.table["n_runs"]
    for n in (1, 2, 3, 4, 7):
        parts = [sw.select(full, shard=(i, n)) for i in range(1, n + 1)]
        got = [r["run_id"] for p in parts for r in p]
        assert sorted(got) == sorted(ids)
        assert len(got) == len(set(got))
        for p in parts:
            seeds = [r["seed"] for r in p]
            assert seeds == sorted(seeds, key=list(wc.SEEDS).index)      # still seed-major
            by_group = {}
            for r in p:
                by_group.setdefault((r["cond"], r["seed"]), []).append(r["cell"])
            assert all(v == list(wc.CELLS) for v in by_group.values())  # groups stay whole
    for bad in ("0/3", "4/3", "1/0", "a/b", "3"):
        with pytest.raises(Exception):
            sw.parse_shard(bad)
    assert sw.parse_shard("2/4") == (2, 4)


def test_grid_counts_match_the_am1_singleton_table():
    for n_targets, n_cond in ((24, 121), (20, 101)):
        table = gf.make_table(n_targets)
        assert table["n_conditions"] == n_cond and table["n_runs"] == 20 * n_cond
        full = sw.grid(table)
        assert len(full) == 20 * n_cond and len(sw.groups_of(table)) == 5 * n_cond
        assert len({r["cond"] for r in full}) == n_cond
        assert full[0]["run_id"] == "all__kall__baseline__s13"
        assert full[-1]["run_id"] == f"p{n_targets:02d}__k25__both__s123"
        assert sum(len(sw.select(full, shard=(i, 4))) for i in range(1, 5)) == 20 * n_cond
    bad = dict(gf.make_table(24), n_runs=2419)
    with pytest.raises(sw.SweepError, match="n_runs"):
        sw.grid(bad)


def test_shard_filters_combine_and_status_reports_shard(env, capsys):
    assert sw.main(["run", "--max-runs", "8", "--shard", "2/3"]) == 0
    trained = env.fake.trained()
    full = {r["run_id"]: r for r in sw.grid(env.table)}
    assert len(trained) == 8 and all(full[r]["group"] % 3 == 1 for r in trained)
    capsys.readouterr()
    assert sw.main(["status", "--shard", "2/3"]) == 0
    st = _status(capsys)
    assert st["shard"] == "2/3" and st["n_done"] == 8


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _claim(d, pid):
    import socket
    d.mkdir(parents=True)
    (d / "RUNNING.json").write_text(json.dumps({"pid": pid, "host": socket.gethostname(),
                                                "started": "t"}))
    (d / "partial.log").write_text("half a run")


@pytest.mark.parametrize("marker", ["stale", "absent", "corrupt"])
def test_killed_run_restarts_cleanly(env, marker):
    rid = ALL13[0]
    d = env.runs / rid
    if marker == "stale":
        _claim(d, _dead_pid())
    else:
        d.mkdir(parents=True)
        (d / "partial.log").write_text("half")
        if marker == "corrupt":
            (d / "RUNNING.json").write_text("{not json")
    assert sw.run_state(d) == "interrupted"
    assert sw.main(["run", "--max-runs", "1"]) == 0
    assert env.fake.trained() == [rid]
    assert (env.runs / f"{rid}.interrupted1" / "partial.log").is_file()
    assert (d / "DONE.json").is_file() and not (d / "partial.log").exists()
    assert not (d / "RUNNING.json").exists()
    ev = [e for e in _log(env) if e.get("event") == "interrupted_restart"]
    assert ev[0]["renamed_to"] == f"{rid}.interrupted1"
    # idempotent: re-running the now-DONE run does nothing
    env.fake.calls.clear()
    assert sw.main(["run", "--max-runs", "1", "--packs", "all", "--cells", "baseline",
                    "--seeds", "13"]) == 0
    assert env.fake.calls == []
    assert aw.classify_groups(env.runs, env.table)[("all", 13)] == "incomplete"


def test_live_claim_is_skipped(env):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _claim(env.runs / ALL13[0], p.pid)
        assert sw.run_state(env.runs / ALL13[0]) == "in_progress"
        assert sw.main(["run", "--max-runs", "1"]) == 0
        assert env.fake.trained() == [ALL13[1]]
        assert (env.runs / ALL13[0] / "partial.log").is_file()
    finally:
        p.kill()
        p.wait()


def test_execute_run_rechecks_claim_before_rename(env):
    """Review R5: a run claimed after selection is skipped, never renamed."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        d = env.runs / ALL13[0]
        _claim(d, p.pid)
        assert sw.execute_run({"run_id": ALL13[0]}, {}, None) == "skipped"
        assert (d / "partial.log").is_file() and (d / "RUNNING.json").is_file()
        assert not list(env.runs.glob(f"{ALL13[0]}.interrupted*"))
        assert env.fake.calls == []
    finally:
        p.kill()
        p.wait()


def test_workers_respect_max_runs_and_check_groups(env):
    assert sw.main(["run", "--max-runs", "5", "--workers", "3"]) == 0
    trained = env.fake.trained()
    assert len(trained) == 5 and len(set(trained)) == 5
    assert set(trained) == set(ALL13 + ["p01__k0__baseline__s13"])
    assert all(wc.group_ok_valid(_dirs(env)))


def test_parallel_needs_a_warm_encoder_cache(env):
    (env.tmp / "emb.pt").unlink()
    assert sw.main(["run", "--max-runs", "2", "--workers", "2"]) == 2
    assert sw.main(["run", "--max-runs", "2", "--shard", "1/2"]) == 2
    assert env.fake.calls == []
    assert sw.main(["run", "--max-runs", "1"]) == 0


# ================================================================= pin preflight / pin
def make_pinned(env, monkeypatch):
    t = env.tmp
    splits, conv, val = t / "splits", t / "converted", t / "validation_targetfree"
    for d in (splits, conv, val):
        d.mkdir()
    for name in ("train.jsonl", "validation.jsonl", "negative_pools.jsonl", "test.jsonl"):
        (splits / name).write_text(name)
    (conv / "proteins.jsonl").write_text("p")
    (conv / "go_index_P.pkl").write_bytes(b"idx")
    frozen = {"train_sha256": wc.sha256_file(splits / "train.jsonl"),
              "validation_sha256": wc.sha256_file(splits / "validation.jsonl"),
              "negative_pools_sha256": wc.sha256_file(splits / "negative_pools.jsonl"),
              "test_sha256": wc.sha256_file(splits / "test.jsonl"),
              "proteins_sha256": wc.sha256_file(conv / "proteins.jsonl"),
              "go_index_P_sha256": wc.sha256_file(conv / "go_index_P.pkl")}
    (splits / "frozen_hashes.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(wc, "SPLITS", splits)
    monkeypatch.setattr(wc, "CONVERTED", conv)
    monkeypatch.setattr(wc, "VALIDATION_TF", val)
    wc.append_pin("G1", frozen)
    wc.append_pin("G1-leak", {"pass": True})
    wc.append_pin("G2-targets", {"target_table_sha256": wc.sha256_file(Path(wc.TARGETS) /
                                                                       "target_table.json")})
    (Path(wc.TARGETS) / "text_covariate.json").write_text("cov")
    wc.append_pin("G2-covariate", {"text_covariate_sha256": wc.sha256_file(
        Path(wc.TARGETS) / "text_covariate.json")})
    cache = env.res / "eval_cache"
    cache.mkdir(parents=True)
    sha16 = frozen["test_sha256"][:16]
    (cache / f"test_{sha16}.npz").write_bytes(b"npz")
    (cache / f"test_{sha16}.meta.json").write_text("{}")
    wc.append_pin("G2-cache", {"test_sha256": frozen["test_sha256"],
                               "cache_npz_sha256": wc.sha256_file(cache / f"test_{sha16}.npz"),
                               "meta_sha256": wc.sha256_file(cache / f"test_{sha16}.meta.json")})
    (val / "validation.jsonl").write_text("v")
    (val / "manifest.json").write_text("{}")
    wc.append_pin("G2-validation", {"manifest_sha256": wc.sha256_file(val / "manifest.json"),
                                    "sha256s": {"validation.jsonl": wc.sha256_file(
                                        val / "validation.jsonl")}})
    manifests = {}
    for cond in wh.condition_ids(env.table):
        d = Path(wc.CONDITIONS) / cond
        d.mkdir(parents=True)
        (d / wh.MANIFEST).write_text(cond)
        manifests[cond] = wc.sha256_file(d / wh.MANIFEST)
    wc.append_pin("G3", {"n_conditions": len(manifests), "manifests": manifests})
    lam = wc.lambda_choice_path()
    lam.parent.mkdir(parents=True)
    lam.write_text('{"lambda": 0.3}')
    wc.append_pin("G4-lambda", {"lambda_choice_sha256": wc.sha256_file(lam)})
    files = {}
    for cell in wc.CELLS:
        p = Path(wc.CONFIG_DIR) / f"{cell}.json"
        if not p.exists():
            p.write_text("{}")
        files[p.name] = wc.sha256_file(p)
    (Path(wc.CONFIG_DIR) / "configs.sha256.json").write_text(json.dumps(files))
    wc.append_pin("G4-configs", {"files": files, "configs_sha256_json": wc.sha256_file(
        Path(wc.CONFIG_DIR) / "configs.sha256.json")})
    monkeypatch.setattr(wc, "code_sha256", lambda: ("codehash", ["a.py"]))
    (env.res / "code_pin.json").write_text(json.dumps({"code_sha256": "codehash"}))
    wc.append_pin("G5", {"code_sha256": "codehash"})
    monkeypatch.setattr(sw, "pin_preflight", REAL_PREFLIGHT)


REAL_PREFLIGHT = sw.pin_preflight


def test_pin_preflight_passes_and_fails_closed(env, monkeypatch):
    make_pinned(env, monkeypatch)
    ctx = sw.pin_preflight(env.table)
    assert ctx["code_sha256"] == "codehash" and ctx["cache"].name.endswith(".npz")
    assert sw.main(["run", "--max-runs", "1"]) == 0

    def expect_fail(mutate, restore):
        mutate()
        with pytest.raises(sw.SweepError):
            sw.pin_preflight(env.table)
        assert sw.main(["run", "--max-runs", "1", "--dry-run"]) == 2
        restore()
        sw.pin_preflight(env.table)

    files = [Path(wc.SPLITS) / "train.jsonl", Path(wc.TARGETS) / "text_covariate.json",
             Path(wc.CONDITIONS) / "p02_k5" / wh.MANIFEST, Path(wc.CONFIG_DIR) / "both.json",
             wc.lambda_choice_path(), Path(wc.VALIDATION_TF) / "validation.jsonl"]
    for p in files:
        orig = p.read_bytes()
        expect_fail(lambda: p.write_bytes(orig + b"!"), lambda: p.write_bytes(orig))
    monkeypatch.setattr(wc, "code_sha256", lambda: ("other", []))
    with pytest.raises(sw.SweepError, match="code_sha256"):
        sw.pin_preflight(env.table)
    monkeypatch.setattr(wc, "code_sha256", lambda: ("codehash", ["a.py"]))
    pilot = env.runs / "x_pilot"
    pilot.mkdir(parents=True)
    (pilot / "lc_manifest.json").write_text(json.dumps({"pilot": True}))
    with pytest.raises(sw.SweepError, match="pilot"):
        sw.pin_preflight(env.table)
    shutil.rmtree(pilot)
    monkeypatch.setattr(sw, "MIN_FREE_BYTES", 10 ** 18)
    with pytest.raises(sw.SweepError, match="GB free"):
        sw.pin_preflight(env.table)


def test_pin_preflight_requires_every_gate_and_exact_config_and_g3(env, monkeypatch):
    make_pinned(env, monkeypatch)
    pins = wc.read_pins()
    path = wc.pins_path()

    def with_pins(entries):
        path.write_text(json.dumps(entries))

    with_pins([e for e in pins if e["gate"] != "G5"])
    with pytest.raises(sw.SweepError, match="G5"):
        sw.pin_preflight(env.table)
    extra = dict(pins[-2])
    extra["files"] = dict(extra["files"], **{"extra.json": "0"})
    with_pins(pins + [extra])                   # the LAST G4-configs entry is used
    with pytest.raises(sw.SweepError, match="G4-configs"):
        sw.pin_preflight(env.table)
    g3 = next(e for e in pins if e["gate"] == "G3")
    with_pins(pins + [dict(g3, n_conditions=g3["n_conditions"] - 1)])
    with pytest.raises(sw.SweepError, match="G3"):
        sw.pin_preflight(env.table)
    with_pins(pins + [{"gate": "G1-leak", "pass": False}])
    with pytest.raises(sw.SweepError, match="G1-leak"):
        sw.pin_preflight(env.table)
    with_pins(pins)
    sw.pin_preflight(env.table)


def test_pin_writes_code_pin_once_and_checks_import_coverage(env, monkeypatch):
    monkeypatch.setattr(wc, "code_sha256", lambda: ("h", ["a.py", "b.py"]))
    monkeypatch.setattr(sw, "import_probe", lambda: ["a.py", "c.py"])
    assert sw.main(["pin"]) == 2
    assert not (env.res / "code_pin.json").exists()
    monkeypatch.setattr(sw, "import_probe", lambda: ["a.py"])
    assert sw.main(["pin"]) == 0
    pin = json.loads((env.res / "code_pin.json").read_text())
    assert pin["code_sha256"] == "h" and pin["files"] == ["a.py", "b.py"]
    g5 = [e for e in wc.read_pins() if e["gate"] == "G5"]
    assert len(g5) == 1 and g5[0]["code_pin_sha256"] == wc.sha256_file(env.res / "code_pin.json")
    assert sw.main(["pin"]) == 2                 # re-pinning is an amendment
    assert len([e for e in wc.read_pins() if e["gate"] == "G5"]) == 1


def test_plan_reports_totals_from_the_table(env, capsys):
    gf.write_group(env.runs, env.table, "all", 13)
    assert sw.main(["plan", "--seeds", "13"]) == 0
    out = _status(capsys)
    assert out["n_runs"] == 320 and out["n_selected"] == 64 and out["n_done"] == 4
    assert out["next"][0] == "p01__k0__baseline__s13"
