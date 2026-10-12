"""Hashing, canonical serialisation, write-or-verify and protocol pins.

``write_or_verify`` is copied with attribution (see its docstring). Everything
else is new. Every byte stream produced here is deterministic, so a re-run of a
generator is byte-identical and reports ``"unchanged"``.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from go_function.runtime import ROOT, FatalError, repo_path, repo_rel

PINS_PATH = ROOT / "results" / "go_function" / "protocol_pins.json"
INPUT_HASHES_PATH = ROOT / "config" / "go_function" / "input_hashes.json"

#: Never written by this study (plan ground rules).
PROTECTED = tuple(ROOT / p for p in (
    "contrastive_learning", "go_ppi_domain", "scripts/go_withholding",
    "config/go_withholding", "trials_domain", "scripts/trials_fusion_confirm",
    "scripts/run_learning_curve_point.py"))

_FORBIDDEN = (b"/Users/", str(ROOT).encode())


# ------------------------------------------------------------------ hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csr_sha256(m) -> str:
    """sha256 over (indptr, indices, data, shape) of a canonical CSR matrix
    (sorted indices, int64 index arrays, uint8 data)."""
    import scipy.sparse as sp
    m = sp.csr_matrix(m)
    m.sort_indices()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(m.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(m.indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(m.data, dtype=np.uint8).tobytes())
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ serialisation
def assert_no_abs_path(data: bytes, what: str = "artifact") -> None:
    for bad in _FORBIDDEN:
        if bad in data:
            raise FatalError(f"{what} contains an absolute path ({bad.decode()!r})")


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, 2-space indent, ``\\n``-terminated UTF-8; refuses absolute paths."""
    data = (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    assert_no_abs_path(data, "JSON")
    return data


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz`` (``np.load``-compatible): fixed member timestamps,
    sorted member order, no pickled objects."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            arr = np.asarray(arrays[name])
            if arr.dtype == object:
                raise FatalError(f"npz member {name} is an object array (not allowed)")
            member = io.BytesIO()
            np.lib.format.write_array(member, arr, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


def csr_arrays(prefix: str, m) -> Dict[str, np.ndarray]:
    """The arrays of a canonical uint8 CSR matrix, for ``npz_bytes``."""
    import scipy.sparse as sp
    m = sp.csr_matrix(m, dtype=np.uint8)
    m.sort_indices()
    return {f"{prefix}_indptr": m.indptr.astype(np.int64),
            f"{prefix}_indices": m.indices.astype(np.int32),
            f"{prefix}_data": m.data.astype(np.uint8),
            f"{prefix}_shape": np.asarray(m.shape, dtype=np.int64)}


def load_csr(z, prefix: str):
    import scipy.sparse as sp
    return sp.csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"], z[f"{prefix}_indptr"]),
                         shape=tuple(int(x) for x in z[f"{prefix}_shape"]))


# ------------------------------------------------------------------ write-or-verify
def assert_writable_target(path: Path) -> None:
    """Fatal if ``path`` is, or is inside, a protected path."""
    p = Path(path).expanduser().resolve()
    for prot in PROTECTED:
        prot = prot.resolve()
        if p == prot or p.is_relative_to(prot):
            raise FatalError(f"refusing to write inside protected path {repo_rel(prot)}")


def write_or_verify(out_dir: Path, files: Dict[str, bytes]) -> str:
    """Write ``files``; an existing non-empty dir must match exactly. Returns
    ``"written"`` or ``"unchanged"``.

    Copied with attribution from go_ppi_domain/withholding.py, function
    ``_write_or_verify`` (lines 221-241; source file sha256
    b9ed17e88525bff03204fb10237cde4809b7b9822ead64b1d4f0bd9cddb00820).
    Changes: ``wc.assert_writable_target`` and ``_fail`` are replaced by this
    module's ``assert_writable_target`` and ``FatalError`` (exit 2).
    """
    out_dir = Path(out_dir)
    assert_writable_target(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        present = sorted(p.name for p in out_dir.iterdir())
        if present != sorted(files):
            raise FatalError(f"idempotent regeneration mismatch: {out_dir.name}: files "
                             f"{present} != {sorted(files)}")
        for name, data in files.items():
            if (out_dir / name).read_bytes() != data:
                raise FatalError(f"idempotent regeneration mismatch: {name} differs")
        return "unchanged"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        tmp = out_dir / f".{name}.tmp"
        tmp.write_bytes(data)
        tmp.replace(out_dir / name)
    return "written"


def write_or_verify_file(path: Path, data: bytes) -> str:
    """Single-file form of ``write_or_verify`` for directories shared by several
    generators: an existing file must be byte-identical."""
    path = Path(path)
    assert_writable_target(path)
    if path.exists():
        if path.read_bytes() != data:
            raise FatalError(f"idempotent regeneration mismatch: {path.name} differs")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return "written"


def write_json_section(path: Path, header: Mapping[str, Any], key: str,
                       section: Mapping[str, Any]) -> str:
    """Merge ``section`` under ``doc["aspects"][key]`` of a gate JSON (created with
    ``header`` when absent). An existing different section is fatal; returns
    written/unchanged (pattern of gate_dgz_eval.py)."""
    path = Path(path)
    assert_writable_target(path)
    norm = json.loads(canonical_json(section))
    doc = json.loads(path.read_text()) if path.exists() else {**header, "aspects": {}}
    if key in doc["aspects"]:
        if doc["aspects"][key] != norm:
            raise FatalError(f"{path.name}: existing {key} section differs from this run")
        return "unchanged"
    doc["aspects"][key] = norm
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(canonical_json(doc))
    tmp.replace(path)
    return "written"


# ------------------------------------------------------------------ protocol pins
def _load_pins(pins_path: Path) -> Dict[str, Any]:
    if not pins_path.exists():
        return {"schema": 1, "pins": {}}
    return json.loads(pins_path.read_text())


def record_pin(key: str, entries: Mapping[str, Any], pins_path: Optional[Path] = None) -> str:
    """Merge ``entries`` into pin ``key``. An existing entry with a different value
    is fatal (a pin is never silently replaced). Returns written/unchanged."""
    from go_function.constants import PIN_KEYS
    if key not in PIN_KEYS:
        raise FatalError(f"unknown pin key {key!r}; allowed {PIN_KEYS}")
    pins_path = Path(pins_path) if pins_path is not None else PINS_PATH
    doc = _load_pins(pins_path)
    slot = doc["pins"].setdefault(key, {})
    changed = False
    for name, value in entries.items():
        value = json.loads(json.dumps(value))  # normalise tuples etc.
        if name in slot:
            if slot[name] != value:
                raise FatalError(f"pin {key}/{name} already recorded with a different value")
            continue
        slot[name] = value
        changed = True
    if not changed:
        return "unchanged"
    assert_writable_target(pins_path)
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pins_path.with_name(f".{pins_path.name}.tmp")
    tmp.write_bytes(canonical_json(doc))
    tmp.replace(pins_path)
    return "written"


def verify_pin(key: str, entries: Mapping[str, Any], pins_path: Optional[Path] = None) -> None:
    """Fatal unless every entry is pinned under ``key`` with exactly this value."""
    pins_path = Path(pins_path) if pins_path is not None else PINS_PATH
    slot = _load_pins(pins_path)["pins"].get(key)
    if slot is None:
        raise FatalError(f"pin {key} is missing")
    for name, value in entries.items():
        if name not in slot:
            raise FatalError(f"pin {key}/{name} is missing")
        if slot[name] != json.loads(json.dumps(value)):
            raise FatalError(f"pin {key}/{name} differs from the recorded value")


# ------------------------------------------------------------------ input hashes
def load_input_hashes(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    path = Path(path) if path is not None else INPUT_HASHES_PATH
    if not path.exists():
        raise FatalError(f"{repo_rel(path)} is missing; run scripts/go_function/pin_inputs.py")
    return json.loads(path.read_text())["files"]


def verify_inputs(rel_paths: Iterable[str], hashes_path: Optional[Path] = None) -> Dict[str, str]:
    """Hash every repo-relative path against ``input_hashes.json``; fatal on a
    missing file, an unpinned file or a mismatch. Returns {rel: sha256}."""
    pinned = load_input_hashes(hashes_path)
    out: Dict[str, str] = {}
    for rel in rel_paths:
        if rel not in pinned:
            raise FatalError(f"{rel} is not in input_hashes.json")
        p = repo_path(rel)
        if not p.exists():
            raise FatalError(f"pinned input missing: {rel}")
        got = sha256_file(p)
        if got != pinned[rel]["sha256"]:
            raise FatalError(f"sha256 mismatch for {rel}: expected {pinned[rel]['sha256']}, "
                             f"got {got}")
        out[rel] = got
    return out
