"""End-to-end G1 build + leak audit on a synthetic fixture (design 1.4, 1.5).

The constants are monkeypatched (raw shas, expected counts, protected dirs) and
the pins file is redirected with ``GO_WITHHOLDING_PINS``; nothing under the
real study paths is touched.
"""
from __future__ import annotations

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go_ppi_domain import leak_pruning as lp  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402
from go_ppi_domain.tests import _leak_fixtures as fx  # noqa: E402
from scripts.go_withholding import build_pruned_features as bpf  # noqa: E402
from scripts.go_withholding import measure_leakage as ml  # noqa: E402


def _lines(path: Path) -> int:
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


@pytest.fixture()
def env(tmp_path, monkeypatch):
    bulk = tmp_path / "bulk"
    files = fx.write_bulk_dir(bulk)
    ref = fx.write_reference(tmp_path, bulk)
    monkeypatch.setattr(wc, "RAW_SHA256", {p: wc.sha256_file(p) for p in files.values()})
    counts = {n.split(".")[0]: _lines(ref["splits"] / n) for n in bpf.SPLIT_FILES}
    monkeypatch.setattr(bpf, "EXPECTED_COUNTS", counts)
    pins = tmp_path / "pins.json"
    monkeypatch.setenv("GO_WITHHOLDING_PINS", str(pins))
    return {"tmp": tmp_path, "bulk": bulk, "ref": ref, "pins": pins, "counts": counts}


def _build_args(env, conv, splits, ref_splits=None):
    return ["--bulk-dir", str(env["bulk"]),
            "--reference-converted", str(env["ref"]["converted"]),
            "--reference-splits", str(ref_splits or env["ref"]["splits"]),
            "--out-converted", str(conv), "--out-splits", str(splits)]


def _leak_args(env, conv, splits, out):
    return ["--splits", str(splits), "--converted", str(conv),
            "--reference-splits", str(env["ref"]["splits"]),
            "--reference-converted", str(env["ref"]["converted"]),
            "--bulk-dir", str(env["bulk"]), "--out", str(out)]


def test_end_to_end_build_and_leak_audit(env):
    conv, splits = env["tmp"] / "new_conv", env["tmp"] / "new_splits"
    assert bpf.main(_build_args(env, conv, splits)) == 0
    ref_conv, ref_splits = env["ref"]["converted"], env["ref"]["splits"]

    # pairs verbatim, same ids in the same order, prefix kept, partners masked
    assert (conv / "pairs.jsonl").read_bytes() == (ref_conv / "pairs.jsonl").read_bytes()
    new_rows = [json.loads(l) for l in (conv / "proteins.jsonl").read_text().splitlines()]
    ref_rows = [json.loads(l) for l in (ref_conv / "proteins.jsonl").read_text().splitlines()]
    assert [r["protein_id"] for r in new_rows] == [r["protein_id"] for r in ref_rows]
    for new, old in zip(new_rows, ref_rows):
        assert new["encoder_view"].startswith(f"{new['gene']} — Protein {new['gene']}-like factor")
        assert "[GENE]" in new["encoder_view"] and "[GENE]" not in old["encoder_view"]
        assert list(new) == list(old)

    # structural identity of every split and identical pool rows
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        assert wc.structural_sha256(splits / name) == wc.structural_sha256(ref_splits / name)
    assert bpf._pool_rows(splits / "negative_pools.jsonl") == \
        bpf._pool_rows(ref_splits / "negative_pools.jsonl")

    frozen = json.loads((splits / "frozen_hashes.json").read_text())
    assert frozen["schema_version"] == 2
    assert frozen["test_sha256"] == wc.sha256_file(splits / "test.jsonl")
    assert frozen["go_index_P_sha256"] == wc.sha256_file(conv / "go_index_P.pkl")
    assert frozen["pairs_sha256"] == wc.sha256_file(ref_conv / "pairs.jsonl")
    assert set(frozen["raw"]) == {str(p) for p in wc.RAW_SHA256}
    for p in (conv / "go_index_P.pkl", conv / "proteins.jsonl", conv / "pairs.jsonl",
              *(splits / n for n in bpf.SPLIT_FILES)):
        assert stat.S_IMODE(p.stat().st_mode) == 0o444, p
    meta = json.loads((conv / "convert_manifest.json").read_text())
    assert meta["view_masking"] is True
    assert meta["annotation_index"]["leak_pruning"]["with_from_human_protein"] is True
    assert meta["mask_manifest_sha256"] == wc.sha256_file(conv / "mask_manifest.json")
    pins = wc.read_pins()
    assert [p["gate"] for p in pins] == ["G1"]
    assert pins[0]["test_sha256"] == frozen["test_sha256"]

    # leak audit passes on the masked data; the reference leaks
    out = env["tmp"] / "leak_audit.json"
    assert ml.main(_leak_args(env, conv, splits, out)) == 0
    audit = json.loads(out.read_text())
    assert audit["pass"] is True and audit["failures"] == []
    assert audit["reference"]["test"]["2"]["a"] > 0.2
    assert audit["new"]["test"]["2"]["a"] == 0.0
    assert audit["new"]["test"]["2"]["c"] == 0.0
    assert audit["reference"]["train"]["2"]["c"] > 0.0
    assert [p["gate"] for p in wc.read_pins()] == ["G1", "G1-leak"]
    # refuses to overwrite its output
    assert ml.main(_leak_args(env, conv, splits, out)) == 2

    # refuses non-empty outputs (no --force)
    assert bpf.main(_build_args(env, conv, splits)) == 2
    assert bpf.main(_build_args(env, env["tmp"] / "c2", splits)) == 2


def test_refuses_protected_output(env, monkeypatch):
    prot = env["tmp"] / "protected"
    prot.mkdir()
    monkeypatch.setattr(wc, "PROTECTED", (prot,))
    assert bpf.main(_build_args(env, prot / "conv", env["tmp"] / "s")) == 2
    assert not (prot / "conv").exists()
    assert not env["pins"].exists()


def test_raw_sha_mismatch_exit_2(env, monkeypatch):
    bad = dict(wc.RAW_SHA256)
    bad[next(iter(bad))] = "0" * 64
    monkeypatch.setattr(wc, "RAW_SHA256", bad)
    assert bpf.main(_build_args(env, env["tmp"] / "c", env["tmp"] / "s")) == 2
    assert not env["pins"].exists()


def test_structural_mismatch_exit_2(env):
    tampered = env["tmp"] / "tampered_splits"
    shutil.copytree(env["ref"]["splits"], tampered)
    test = tampered / "test.jsonl"
    lines = test.read_text(encoding="utf-8").splitlines(keepends=True)
    test.write_text("".join(lines[:-1]), encoding="utf-8")
    assert bpf.main(_build_args(env, env["tmp"] / "c", env["tmp"] / "s", tampered)) == 2
    assert not env["pins"].exists()


def test_leak_gate_failure_exit_3(env, monkeypatch):
    monkeypatch.setattr(lp, "mask_view_text", lambda g, t, lex: (t, 0))   # masking disabled
    conv, splits = env["tmp"] / "c", env["tmp"] / "s"
    assert bpf.main(_build_args(env, conv, splits)) == 0
    out = env["tmp"] / "leak_audit.json"
    assert ml.main(_leak_args(env, conv, splits, out)) == 3
    audit = json.loads(out.read_text())
    assert audit["pass"] is False
    assert any(f.startswith("test grade-2 a rate") for f in audit["failures"])
    assert [p["gate"] for p in wc.read_pins()] == ["G1"]                  # no G1-leak pin


def test_gate_rule_boundaries():
    def new(a2, a1, b2=0.0, c2=0.0):
        return {"test": {"2": {"a": a2, "b": b2, "c": c2},
                         "1": {"a": a1, "b": 0.0, "c": 0.0}}}
    assert ml.apply_gate(new(0.005, 0.0)) == []
    assert ml.apply_gate(new(0.0051, 0.0))
    assert ml.apply_gate(new(0.004, 0.0, c2=0.006))
    assert ml.apply_gate(new(0.0, 0.0)) == []
