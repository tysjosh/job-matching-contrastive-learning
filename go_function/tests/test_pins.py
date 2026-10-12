import json

import numpy as np
import pytest
import scipy.sparse as sp

from go_function import pins
from go_function.runtime import FatalError


def test_write_or_verify_unchanged_and_mismatch(tmp_path):
    out = tmp_path / "cond"
    files = {"a.json": b"{}\n", "b.bin": b"\x00\x01"}
    assert pins.write_or_verify(out, files) == "written"
    assert pins.write_or_verify(out, files) == "unchanged"
    with pytest.raises(FatalError) as e:
        pins.write_or_verify(out, {"a.json": b"{}\n", "b.bin": b"\x00\x02"})
    assert e.value.exit_code == 2
    with pytest.raises(FatalError):
        pins.write_or_verify(out, {"a.json": b"{}\n"})


def test_write_or_verify_file(tmp_path):
    p = tmp_path / "d" / "x.npz"
    assert pins.write_or_verify_file(p, b"abc") == "written"
    assert pins.write_or_verify_file(p, b"abc") == "unchanged"
    with pytest.raises(FatalError):
        pins.write_or_verify_file(p, b"abd")


def test_protected_paths_refused():
    with pytest.raises(FatalError):
        pins.write_or_verify(pins.ROOT / "go_ppi_domain" / "x", {"a": b""})


def test_npz_bytes_deterministic_and_loadable(tmp_path):
    m = sp.csr_matrix(np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8))
    arrays = {**pins.csr_arrays("train", m), "vocab": np.asarray(["GO:1", "GO:2", "GO:3"])}
    a, b = pins.npz_bytes(arrays), pins.npz_bytes(dict(reversed(list(arrays.items()))))
    assert a == b
    p = tmp_path / "x.npz"
    p.write_bytes(a)
    with np.load(p, allow_pickle=False) as z:
        back = pins.load_csr(z, "train")
        assert list(z["vocab"]) == ["GO:1", "GO:2", "GO:3"]
    assert (back != m).nnz == 0
    assert pins.csr_sha256(back) == pins.csr_sha256(m)
    with pytest.raises(FatalError):
        pins.npz_bytes({"o": np.array([(1,), ()], dtype=object)})


def test_csr_sha256_sensitive_to_shape():
    m = sp.csr_matrix(np.eye(2, dtype=np.uint8))
    m2 = sp.csr_matrix((m.data, m.indices, m.indptr), shape=(2, 3))
    assert pins.csr_sha256(m) != pins.csr_sha256(m2)


def test_record_and_verify_pin(tmp_path):
    p = tmp_path / "protocol_pins.json"
    assert pins.record_pin("G1-labels", {"bp/labels_TP.json": "aa"}, p) == "written"
    assert pins.record_pin("G1-labels", {"bp/labels_TP.json": "aa"}, p) == "unchanged"
    assert pins.record_pin("G1-labels", {"mf/labels_TP.json": "bb"}, p) == "written"
    pins.verify_pin("G1-labels", {"bp/labels_TP.json": "aa", "mf/labels_TP.json": "bb"}, p)
    with pytest.raises(FatalError):
        pins.record_pin("G1-labels", {"bp/labels_TP.json": "cc"}, p)
    with pytest.raises(FatalError):
        pins.verify_pin("G1-labels", {"bp/labels_TP.json": "cc"}, p)
    with pytest.raises(FatalError):
        pins.verify_pin("G2", {"x": 1}, p)
    with pytest.raises(FatalError):
        pins.record_pin("G99", {"x": 1}, p)
    assert json.loads(p.read_text())["pins"]["G1-labels"]["mf/labels_TP.json"] == "bb"


def test_verify_inputs(tmp_path):
    rel = "go_function/tests/fixtures/mini.obo"
    good = {"files": {rel: {"sha256": pins.sha256_file(pins.ROOT / rel), "size": 0}}}
    hp = tmp_path / "input_hashes.json"
    hp.write_text(json.dumps(good))
    assert pins.verify_inputs([rel], hp)[rel] == good["files"][rel]["sha256"]
    hp.write_text(json.dumps({"files": {rel: {"sha256": "0" * 64, "size": 0}}}))
    with pytest.raises(FatalError):
        pins.verify_inputs([rel], hp)
    with pytest.raises(FatalError):
        pins.verify_inputs(["go_function/tests/fixtures/mini.norm"], hp)
