"""Write config/go_function/input_hashes.json from the dataset manifests and verify
every pinned file against it (protocol §12, plan item 4).

Sources: dataset/go_function/MANIFEST.json (download_dgz.py) and, when present,
dataset/go_function/MANIFEST_gate.json (gate downloads). Paths in the manifests
are relative to dataset/go_function/; the output uses repo-relative paths.

Write-or-verify: an existing input_hashes.json may only gain new files; a changed
hash or size for an already pinned file is fatal (exit 2). Every file is then
hashed and compared (missing or mismatching file: exit 2).

--gate requires MANIFEST_gate.json to exist (exit 2 otherwise); its files are then
added like any other.

Usage: .venv/bin/python scripts/go_function/pin_inputs.py [--gate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function.constants import EXIT_FATAL, EXIT_OK  # noqa: E402
from go_function.pins import (INPUT_HASHES_PATH, assert_writable_target, canonical_json,  # noqa: E402
                              sha256_file)
from go_function.runtime import FatalError, repo_path, repo_rel, setup_logging  # noqa: E402

DATA = ROOT / "dataset" / "go_function"
MANIFESTS = (DATA / "MANIFEST.json", DATA / "MANIFEST_gate.json")


def collect(manifests=MANIFESTS) -> dict:
    files = {}
    sources = []
    for m in manifests:
        if not m.exists():
            if m == manifests[0]:
                raise FatalError(f"{repo_rel(m)} is missing")
            continue
        doc = json.loads(m.read_text())
        sources.append({"manifest": repo_rel(m), "source": doc.get("source")})
        for e in doc["files"]:
            rel = repo_rel(m.parent / e["path"])
            entry = {"sha256": e["sha256"], "size": int(e["size"])}
            if rel in files and files[rel] != entry:
                raise FatalError(f"{rel} pinned twice with different values")
            files[rel] = entry
    return {"schema": 1, "sources": sources, "files": dict(sorted(files.items()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(INPUT_HASHES_PATH))
    ap.add_argument("--gate", action="store_true",
                    help="require dataset/go_function/MANIFEST_gate.json (gate downloads) and "
                         "add its files")
    args = ap.parse_args(argv)
    log = setup_logging("pin_inputs")
    out = Path(args.out)
    try:
        if args.gate and not MANIFESTS[1].exists():
            raise FatalError(f"{repo_rel(MANIFESTS[1])} is missing; run download_dgz_gate.py")
        doc = collect()
        status = "written"
        if out.exists():
            old = json.loads(out.read_text())
            for rel, entry in old["files"].items():
                if rel in doc["files"] and doc["files"][rel] != entry:
                    raise FatalError(f"{rel}: manifest value differs from the pinned value")
                if rel not in doc["files"]:
                    raise FatalError(f"{rel} is pinned but absent from the manifests")
            status = "unchanged" if out.read_bytes() == canonical_json(doc) else "extended"
        if status != "unchanged":
            assert_writable_target(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(canonical_json(doc))
        log.info("%s: %s (%d files)", repo_rel(out), status, len(doc["files"]))
        for rel, entry in doc["files"].items():
            p = repo_path(rel)
            if not p.exists():
                raise FatalError(f"pinned input missing: {rel}")
            got = sha256_file(p)
            if got != entry["sha256"] or p.stat().st_size != entry["size"]:
                raise FatalError(f"sha256/size mismatch for {rel}: expected {entry['sha256']}, "
                                 f"got {got}")
        log.info("verified %d files", len(doc["files"]))
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
