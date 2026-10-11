#!/usr/bin/env python3
"""Step -1 (protocol_trials_fusion.md §6.2): verify the transferred inputs on the remote.

Expected hashes come from the protocol file passed as ``--protocol`` (no embedded
constants): the §3b checkpoint table, the §7 inputs table, the §2.1 test-chunk
manifest digest, the 0-chunk validation directory and the MPNet snapshot revision.
Any mismatch or missing file aborts (exit 2). Writes ``transfer_check.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trials_domain import fusion_confirm as fc  # noqa: E402


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--code-manifest", type=Path, required=True,
                    help="Mac-side dump_code_hashes.py output; the transferred code tree must match it "
                         "(required: make_pins.py refuses a transfer check without it)")
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    out = fc.refuse_existing(args.out)

    text = fc.absp(args.protocol).read_text(encoding="utf-8")
    parsed = fc.parse_protocol(text)
    checks, failures = [], []

    def check(path: str, expected: str, kind: str) -> None:
        p = fc.absp(path)
        if not p.exists():
            failures.append({"kind": kind, "path": path, "reason": "missing"})
            return
        got = fc.file_sha256(p)
        row = {"kind": kind, "path": path, "expected": expected, "sha256": got, "ok": got == expected}
        checks.append(row)
        if got != expected:
            failures.append(row)

    for path, h in sorted(parsed["inputs"].items()):
        check(path, h, "input")
    for row in parsed["checkpoints"]:
        check(row["path"], row["sha256"], "checkpoint")

    test_dir, val_dir = fc.absp(parsed["test_dir"]), fc.absp(parsed["validation_dir"])
    chunks = fc.chunk_hashes(test_dir) if test_dir.is_dir() else {}
    digest = fc.manifest_digest_from_hashes(chunks) if chunks else None
    chunk_check = {"test_dir": parsed["test_dir"], "n_chunks": len(chunks),
                   "expected_n_chunks": parsed["n_test_chunks"], "manifest_digest": digest,
                   "expected_manifest_digest": parsed["manifest_digest"],
                   "validation_dir": parsed["validation_dir"], "validation_dir_exists": val_dir.is_dir(),
                   "validation_n_chunks": len(list(val_dir.glob("chunk_*.npz"))) if val_dir.is_dir() else None}
    if len(chunks) != parsed["n_test_chunks"] or digest != parsed["manifest_digest"]:
        failures.append({"kind": "chunks", **chunk_check})
    if not val_dir.is_dir() or chunk_check["validation_n_chunks"] != 0:
        failures.append({"kind": "validation_chunks", **chunk_check})

    # code tree vs the Mac-side dump_code_hashes.py manifest (review finding 4)
    if not fc.absp(args.code_manifest).is_file():
        raise fc.ProtocolError(f"--code-manifest {args.code_manifest} not found; run dump_code_hashes.py "
                               "on the Mac right before the rsync")
    mac = json.loads(fc.absp(args.code_manifest).read_text(encoding="utf-8"))
    code_diffs = fc.compare_code_hashes(mac["code"], fc.code_hashes())
    code_check = {"manifest": fc.rel(fc.absp(args.code_manifest)),
                  "manifest_sha256": fc.file_sha256(args.code_manifest),
                  "n_files": len(mac["code"]), "mac_git_head": (mac.get("git") or {}).get("head"),
                  "ok": not code_diffs, "differences": code_diffs}
    failures += [{"kind": "code", **d} for d in code_diffs]

    try:
        revision, _ = fc.resolve_mpnet_snapshot(parsed["mpnet_revision"])
        mpnet = {"revision": revision, "expected": parsed["mpnet_revision"], "ok": True}
    except Exception as exc:
        mpnet = {"expected": parsed["mpnet_revision"], "ok": False, "error": str(exc)}
        failures.append({"kind": "mpnet_snapshot", **mpnet})

    result = {"schema": "trials_fusion_confirm/transfer_check/v1", "created_utc": fc.utc_now(),
              "passed": not failures, "parsed_protocol_path": fc.rel(fc.absp(args.protocol)),
              "parsed_protocol_sha256": fc.file_sha256(args.protocol),
              "n_checkpoint_rows": len(parsed["checkpoints"]), "n_input_rows": len(parsed["inputs"]),
              "checks": checks, "chunks": chunk_check, "code": code_check, "mpnet": mpnet, "failures": failures,
              "run_env": fc.run_env_record()}
    out.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        # A failed check never occupies the --out leaf, so a rerun after fixing the
        # transfer can write it; the failure report gets a timestamped sibling.
        out = out.with_name(f"{out.stem}.FAILED.{fc.utc_now().replace(':', '')}{out.suffix}")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"transfer check {'PASSED' if not failures else 'FAILED'}: {len(checks)} files hashed, "
          f"{len(chunks)} test chunks, {len(failures)} failures -> {out}")
    for f in failures:
        print("  FAIL", f.get("kind"), f.get("path", ""), f.get("reason", ""))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
