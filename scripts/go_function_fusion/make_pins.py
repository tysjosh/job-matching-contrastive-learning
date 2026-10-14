"""Pins, code hashes and shared helpers of the fusion study scripts (protocol §9, §13).

Pins live in ``<results>/protocol_pins.json`` (``<results>`` = results/go_function_fusion,
or results/go_function_fusion_dev for Mac dev runs). A pin entry is never silently
replaced. Every JSON is written with ``go_function.pins.canonical_json`` (sorted keys,
refuses absolute paths).

Usage:
  python scripts/go_function_fusion/make_pins.py code [--results R]          # dump code hashes
  python scripts/go_function_fusion/make_pins.py diamond --aspect bp \\
        --diamond-version "diamond version 2.1.x" --diamond-binary /path/to/diamond   # G-D (remote)
G-TP (Tier B bases) is pinned by ``verify_tp_bases.py`` (protocol §9, §15).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go_function.pins import canonical_json, sha256_file  # noqa: E402
from go_function_fusion import guard  # noqa: E402

RESULTS_CHOICES = ("results/go_function_fusion", "results/go_function_fusion_dev")
PIN_KEYS = ("G-0", "G-R", "G-F", "G-S2dry", "G-D", "G-TP", "G-F2", "G-T")
CODE_DIRS = ("go_function_fusion", "scripts/go_function_fusion")
CODE_FILES = ("config/go_function_fusion/study.json",)
EXIT_OK, EXIT_FATAL, EXIT_STOP, EXIT_REFUSED = 0, 2, 3, 4
WRITABLE = ("results/go_function_fusion", "results/go_function_fusion_dev", "config/go_function_fusion")


class Fatal(RuntimeError):
    exit_code = EXIT_FATAL


class Refused(RuntimeError):
    exit_code = EXIT_REFUSED


# ------------------------------------------------------------------ paths
def rel(p: Path) -> str:
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def results_dir(arg: str) -> Path:
    if arg not in RESULTS_CHOICES:
        raise Fatal(f"--results must be one of {RESULTS_CHOICES}")
    return ROOT / arg


def assert_writable(path: Path) -> None:
    r = rel(path)
    if not any(r == w or r.startswith(w + "/") for w in WRITABLE):
        raise Fatal(f"refusing to write outside {WRITABLE}: {r}")


def write_json(path: Path, obj: Any) -> str:
    """Canonical JSON (atomic); returns the sha256."""
    from go_function.pins import sha256_bytes
    path = Path(path)
    assert_writable(path)
    data = canonical_json(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return sha256_bytes(data)


def write_npz(path: Path, arrays: Mapping[str, Any]) -> str:
    from go_function.pins import npz_bytes, sha256_bytes
    path = Path(path)
    assert_writable(path)
    data = npz_bytes(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return sha256_bytes(data)


# ------------------------------------------------------------------ code hashes
def code_files() -> Iterable[Path]:
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".sh"):
                yield p
    for f in CODE_FILES:
        yield ROOT / f


def code_hashes() -> Dict[str, str]:
    return {rel(p): sha256_file(p) for p in code_files()}


def compare_code(expected: Mapping[str, str], got: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    got = dict(got if got is not None else code_hashes())
    diff = sorted(k for k in set(expected) | set(got) if expected.get(k) != got.get(k))
    return {"equal": not diff, "differs": diff}


def protocol_sha() -> str:
    from go_function_fusion import PROTOCOL_REL
    return sha256_file(ROOT / PROTOCOL_REL)


# ------------------------------------------------------------------ pins
def pins_path(results: Path) -> Path:
    return Path(results) / "protocol_pins.json"


def load_pins(results: Path) -> Dict[str, Any]:
    p = pins_path(results)
    return json.loads(p.read_text()) if p.exists() else {"schema": 1, "pins": {}}


def record_pin(results: Path, key: str, entries: Mapping[str, Any]) -> str:
    """Merge ``entries`` into pin ``key``; a different existing value is fatal."""
    if key not in PIN_KEYS:
        raise Fatal(f"unknown pin key {key!r}; allowed {PIN_KEYS}")
    doc = load_pins(results)
    slot = doc["pins"].setdefault(key, {})
    changed = False
    for name, value in entries.items():
        value = json.loads(json.dumps(value))
        if name in slot:
            if slot[name] != value:
                raise Fatal(f"pin {key}/{name} already recorded with a different value")
            continue
        slot[name] = value
        changed = True
    if not changed:
        return "unchanged"
    write_json(pins_path(results), doc)
    return "written"


def require_pin(results: Path, key: str) -> Dict[str, Any]:
    slot = load_pins(results)["pins"].get(key)
    if not slot:
        raise Refused(f"pin {key} is missing in {rel(pins_path(results))}")
    return slot


# ------------------------------------------------------------------ host checks
def is_mac() -> bool:
    return platform.system() == "Darwin"


def refuse_on_mac(what: str) -> None:
    if is_mac():
        raise Refused(f"{what} is remote only (protocol §9); refused on macOS")


def main_wrapper(fn, argv=None) -> int:
    """Run ``fn(argv)``; map guard violations to exit 2 and Refused to exit 4."""
    try:
        return int(fn(argv) or 0)
    except guard.TestAccessError as e:
        print(f"FATAL guard violation: {e}", file=sys.stderr)
        return EXIT_FATAL
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Fatal as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return EXIT_FATAL


# ------------------------------------------------------------------ CLI
def _cmd(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("code")
    c.add_argument("--results", default=RESULTS_CHOICES[0])
    c.add_argument("--write", action="store_true", help="write <results>/code_hashes.json")
    d = sub.add_parser("diamond")
    d.add_argument("--results", default=RESULTS_CHOICES[0])
    d.add_argument("--aspect", choices=("bp", "mf"), required=True)
    d.add_argument("--diamond-version", required=True)
    d.add_argument("--diamond-binary", required=True)
    a = ap.parse_args(argv)
    res = results_dir(a.results)
    if a.cmd == "code":
        h = code_hashes()
        if a.write:
            write_json(res / "code_hashes.json", {"code": h, "protocol_sha256": protocol_sha()})
        print(json.dumps(h, indent=2, sort_keys=True))
        return EXIT_OK
    if a.cmd == "diamond":
        refuse_on_mac("G-D")
        ddir = res / "diamond" / a.aspect
        files = {"train_fasta": ddir / "train_data.fa", "train_db": ddir / "train_only.dmnd",
                 "valid_res": ddir / "valid_diamond_trainonly.res",
                 "test_res": ddir / "test_diamond_trainonly.res"}
        miss = [k for k, p in files.items() if not p.exists()]
        if miss:
            raise Fatal(f"missing Diamond outputs: {miss}")
        checks = ddir / "checks.json"
        if not checks.exists():
            raise Fatal("checks.json missing: run diamond_valid.sh to completion")
        entry = {k: {"path": rel(p), "sha256": sha256_file(p)} for k, p in files.items()}
        entry["diamond_version"] = a.diamond_version
        entry["diamond_binary_sha256"] = sha256_file(Path(a.diamond_binary))
        entry["checks"] = json.loads(checks.read_text())
        print(record_pin(res, "G-D", {a.aspect: entry}))
        ih_p = ROOT / "config/go_function_fusion/input_hashes.json"
        ih = json.loads(ih_p.read_text())
        ih.setdefault("diamond_valid_by_aspect", {})[a.aspect] = entry
        ih["diamond_valid"]["status"] = "filled at G-D (see diamond_valid_by_aspect)"
        write_json(ih_p, ih)
        return EXIT_OK
    return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main_wrapper(_cmd))
