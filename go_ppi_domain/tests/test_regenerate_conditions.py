"""regenerate_conditions.py: deterministic, pin-verified condition regeneration on
another machine (user decision B, remote GPU execution)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from go_ppi_domain import withholding as wh
from go_ppi_domain import withholding_constants as wc
from go_ppi_domain.tests import _study_fixtures as sf
from scripts.go_withholding import make_conditions as mc
from scripts.go_withholding import regenerate_conditions as rg


@pytest.fixture()
def pinned(tmp_path, monkeypatch):
    """A layout whose G1, G2-targets, G2-validation and G3 pins were written here."""
    s = sf.build_study(tmp_path / "study", conds=())
    sf.patch_study(monkeypatch, s)
    shutil.rmtree(s["val_dir"])
    wc.append_pin("G1", s["frozen"])
    wc.append_pin("G2-targets", {"target_table_sha256": wc.sha256_file(s["table_path"])})
    common = ["--target-table", str(s["table_path"]), "--base-splits", str(s["splits"]),
              "--converted", str(s["converted"])]
    assert mc.main(["--mode", "validation", *common, "--out", str(s["val_dir"])]) == 0
    assert mc.main(["--mode", "conditions", *common, "--targetfree-validation",
                    str(s["val_dir"]), "--out", str(s["conditions"])]) == 0
    s["original"] = {p.relative_to(s["root"]).as_posix(): wc.sha256_file(p)
                     for d in (s["conditions"], s["val_dir"]) for p in sorted(d.rglob("*"))
                     if p.is_file()}
    s["n_conditions"] = s["table"]["n_conditions"]
    return s


def _wipe(s):
    shutil.rmtree(s["conditions"])
    shutil.rmtree(s["val_dir"])


def _current(s):
    return {p.relative_to(s["root"]).as_posix(): wc.sha256_file(p)
            for d in (s["conditions"], s["val_dir"]) for p in sorted(d.rglob("*"))
            if p.is_file()}


def test_regenerates_byte_identical_conditions_from_pins(pinned):
    gates_before = [p["gate"] for p in wc.read_pins()]
    _wipe(pinned)
    assert rg.main([]) == 0
    assert _current(pinned) == pinned["original"]
    assert len([d for d in pinned["conditions"].iterdir() if d.is_dir()]) == pinned["n_conditions"]
    assert [p["gate"] for p in wc.read_pins()] == gates_before          # no pin appended
    # idempotent second run: everything verified, nothing rewritten
    assert rg.main([]) == 0 and _current(pinned) == pinned["original"]


def test_emitted_files_and_pins_carry_no_machine_path(pinned):
    root = str(pinned["root"])
    for d in pinned["conditions"].iterdir():
        assert root not in (d / wh.MANIFEST).read_text()
    assert root not in (pinned["val_dir"] / "manifest.json").read_text()
    assert root not in wc.pins_path().read_text()


def test_only_subset(pinned):
    _wipe(pinned)
    assert rg.main(["--only", "all", "p02_k5"]) == 0
    assert sorted(p.name for p in pinned["conditions"].iterdir()) == ["all", "p02_k5"]
    assert rg.main(["--only", "p99_k0"]) == 2


def test_manifest_pin_mismatch_is_fatal_before_writing(pinned):
    _wipe(pinned)
    pins = json.loads(wc.pins_path().read_text())
    g3 = [p for p in pins if p["gate"] == "G3"][-1]
    g3["manifests"]["p01_k5"] = "0" * 64
    wc.pins_path().write_text(json.dumps(pins))
    assert rg.main(["--only", "p01_k5"]) == 2
    assert not (pinned["conditions"] / "p01_k5").exists()


def test_base_file_or_table_mismatch_is_fatal(pinned):
    _wipe(pinned)
    tr = pinned["splits"] / "train.jsonl"
    data = tr.read_bytes()
    tr.write_bytes(data + b"\n")
    assert rg.main([]) == 2
    tr.write_bytes(data)
    assert not pinned["conditions"].exists()
    pins = json.loads(wc.pins_path().read_text())
    for p in pins:
        if p["gate"] == "G2-targets":
            p["target_table_sha256"] = "0" * 64
    wc.pins_path().write_text(json.dumps(pins))
    assert rg.main([]) == 2


def test_validation_pin_mismatch_leaves_no_partial_dir(pinned):
    _wipe(pinned)
    pins = json.loads(wc.pins_path().read_text())
    for p in pins:
        if p["gate"] == "G2-validation":
            p["manifest_sha256"] = "0" * 64
    wc.pins_path().write_text(json.dumps(pins))
    assert rg.main([]) == 2
    assert not pinned["val_dir"].exists()
    assert not pinned["val_dir"].with_name(pinned["val_dir"].name + ".regen_tmp").exists()


def test_study_code_bakes_in_no_machine_path():
    """Remote execution: no study module, script or GO test embeds a local path."""
    root = wc.ROOT
    files = [*sorted((root / "go_ppi_domain").glob("*.py")),
             *sorted((root / "go_ppi_domain" / "tests").glob("*.py")),
             *sorted((root / "scripts" / "go_withholding").glob("*.py")),
             root / "scripts" / "run_learning_curve_point.py"]
    # Built at runtime so this file does not match its own marker.
    markers = ("/" + "Users/", str(root))
    offenders = [p.name for p in files
                 if any(m in p.read_text(encoding="utf-8") for m in markers)]
    assert offenders == []
    assert wc.ROOT == Path(wc.__file__).resolve().parents[1]
    assert wc.expand_root_token("$ROOT/a/b") == str(wc.ROOT / "a/b")
    assert wc.expand_root_token("rel/a") == "rel/a"
    assert wc.root_token(wc.ROOT / "a/b") == "$ROOT/a/b"
    assert wc.scrub_root(f"x {wc.ROOT}/a and {wc.ROOT}2/b") == f"x $ROOT/a and {wc.ROOT}2/b"


def test_missing_pin_is_fatal(tmp_path, monkeypatch):
    s = sf.build_study(tmp_path / "s", conds=())
    sf.patch_study(monkeypatch, s)
    assert rg.main([]) == 2
