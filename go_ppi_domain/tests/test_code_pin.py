"""Code pin, import coverage, pins file and amendment-table parser (design 1.3, 13, 14)."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import pytest

from go_ppi_domain import withholding_constants as wc

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def probe_module():
    """A throwaway untracked .py file under go_ppi_domain/, removed afterwards."""
    name = f"_codepin_probe_{uuid.uuid4().hex[:10]}"
    path = ROOT / "go_ppi_domain" / f"{name}.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        yield name, path
    finally:
        sys.modules.pop(f"go_ppi_domain.{name}", None)
        path.unlink(missing_ok=True)


def test_untracked_py_file_changes_code_sha256(probe_module):
    _name, path = probe_module
    path.unlink()
    before, files_before = wc.code_sha256()
    path.write_text("VALUE = 1\n", encoding="utf-8")
    after, files_after = wc.code_sha256()
    rel = str(path.relative_to(ROOT))
    assert rel not in files_before and rel in files_after
    assert before != after
    path.write_text("VALUE = 2\n", encoding="utf-8")
    assert wc.code_sha256()[0] != after


def test_trials_domain_and_runner_files_are_pinned():
    _sha, files = wc.code_sha256()
    assert "trials_domain/label_budget_runner.py" in files
    assert "run_phase1_embedding_evaluation.py" in files
    assert "contrastive_learning/trainer.py" in files
    assert all(f.endswith(".py") for f in files)
    assert not any(f.startswith(".venv/") for f in files)


def test_imported_repo_modules_filters(monkeypatch, probe_module):
    name, _path = probe_module
    monkeypatch.setitem(sys.modules, "fake_relative_mod",
                        types.SimpleNamespace(__file__="_classes.py"))
    # The interpreter's own torch (ROOT/.venv locally and per the remote runbook).
    venv_file = Path(importlib.util.find_spec("torch").origin)
    monkeypatch.setitem(sys.modules, "fake_venv_mod",
                        types.SimpleNamespace(__file__=str(venv_file)))
    importlib.import_module(f"go_ppi_domain.{name}")
    mods = wc.imported_repo_modules()
    assert f"go_ppi_domain/{name}.py" in mods
    assert "go_ppi_domain/withholding_constants.py" in mods
    assert "_classes.py" not in mods
    assert not any(m.startswith(".venv") for m in mods)


def test_pins_path_override_and_append(monkeypatch, tmp_path):
    target = tmp_path / "pins.json"
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(target))
    assert wc.pins_path() == target
    wc.append_pin("G1", {"a": 1})
    wc.append_pin("G1-leak", {"pass": True})
    entries = json.loads(target.read_text())
    assert [e["gate"] for e in entries] == ["G1", "G1-leak"]
    assert entries[0]["a"] == 1 and "time" in entries[0]
    monkeypatch.delenv("GO_WITHHOLDING_PINS")
    assert wc.pins_path() == wc.PINS


def test_assert_writable_target_refuses_protected(tmp_path):
    with pytest.raises(RuntimeError, match="protected"):
        wc.assert_writable_target(ROOT / "preprocess/go_ppi_splits/x.jsonl")
    with pytest.raises(RuntimeError, match="protected"):
        wc.assert_writable_target(ROOT / "dataset/go_ppi")
    wc.assert_writable_target(tmp_path / "ok.json")
    wc.assert_writable_target(ROOT / "preprocess/go_withholding/splits")


def _protocol(tmp_path: Path, s13_rows: str, extra_s12: str = "", extra_s14: str = "") -> Path:
    text = (
        "# Protocol\n\n## 12. Limits\n\n" + extra_s12 +
        "\n## 13. Pins and amendments\n\n"
        "| pin key | value | date |\n|---|---|---|\n| G1 | | |\n\n"
        "**Amendments.**\n\n"
        "| id | date | gate | groups cleared | cause | fix | re-pin |\n"
        "|---|---|---|---|---|---|---|\n" + s13_rows +
        "\nRules for rows.\n\n## 14. Changes\n\n" + extra_s14)
    path = tmp_path / "protocol.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_amendment_rows_escaped_pipe(tmp_path):
    path = _protocol(tmp_path,
                     "| AM1 | 2025-11-01 | G7 | `p03_k5,s21`; `all,s13` | a \\| b | fixed x | no |\n")
    rows = wc.parse_amendment_rows(path)
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == wc.AMENDMENT_COLUMNS
    assert row["id"] == "AM1" and row["cause"] == "a | b" and row["re-pin"] == "no"
    assert row["groups cleared"] == "`p03_k5,s21`; `all,s13`"


def test_parse_amendment_rows_ignores_rows_outside_section_13(tmp_path):
    row = "| AM1 | 2025-11-01 | G7 | p03_k5,s21 | c | f | no |\n"
    table = ("| id | date | gate | groups cleared | cause | fix | re-pin |\n"
             "|---|---|---|---|---|---|---|\n" + row)
    path = _protocol(tmp_path, "", extra_s12=table, extra_s14=table)
    assert wc.parse_amendment_rows(path) == []


def test_parse_amendment_rows_errors(tmp_path):
    with pytest.raises(ValueError, match="cells"):
        wc.parse_amendment_rows(_protocol(tmp_path, "| AM1 | 2025-11-01 | G7 | c | f | no |\n"))
    bad = tmp_path / "nosec.md"
    bad.write_text("# P\n\n## 12. x\n\n## 14. y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="13"):
        wc.parse_amendment_rows(bad)
    notable = tmp_path / "notable.md"
    notable.write_text("## 13. Pins\n\n| pin key | value |\n|---|---|\n\n## 14. C\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="amendments table"):
        wc.parse_amendment_rows(notable)


def test_real_protocol_parses_with_seven_cells_per_row():
    rows = wc.parse_amendment_rows()
    assert isinstance(rows, list)
    assert all(tuple(r) == wc.AMENDMENT_COLUMNS and len(r) == 7 for r in rows)


def test_group_ok_valid(tmp_path):
    dirs = [tmp_path / f"r{i}" for i in range(4)]
    for d in dirs:
        d.mkdir()
        (d / "lc_manifest.json").write_text(json.dumps({"d": d.name}))
    shas = {d.name: wc.sha256_file(d / "lc_manifest.json") for d in dirs}
    for d in dirs:
        (d / "DONE.json").write_text(json.dumps({"lc_manifest_sha256": shas[d.name]}))
    for d in dirs[:3]:
        (d / "GROUP_OK.json").write_text(json.dumps(
            {"run_ids": [x.name for x in dirs], "lc_manifest_sha256": shas}))
    assert wc.group_ok_valid(dirs) == [True, True, True, False]
    (dirs[1] / "lc_manifest.json").write_text("{}")
    assert wc.group_ok_valid(dirs)[1] is False
    assert wc.group_ok_valid(list(reversed(dirs))) == [False] * 4
