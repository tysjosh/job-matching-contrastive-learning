#!/usr/bin/env python3
"""Step −1 (protocol_go_ppi_fusion.md §6.2): verify the transferred inputs on the remote. CPU.

Expected hashes come from the protocol file given as ``--protocol`` (no embedded
constants): the §3b checkpoint table (15 rows), the §7 inputs table (>= 30 rows), the two
§2.1 published prepared-cache dirs (chunk counts and manifest digests) and the MPNet
revision. Also checks the code tree against the Mac ``dump_code_hashes.py`` manifest and
runs a CPU load test of the MPNet snapshot (construct, encode one string, width 768).
Any failure exits 2 and writes ``<out>.FAILED.<utc>.json`` instead of ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402


def mpnet_check(expected: str) -> dict:
    revision, path = fc.resolve_mpnet_snapshot(expected)
    enc = fc.cpu_encoder(path)
    vec = enc.encode(["GO/PPI transfer check"], batch_size=1, convert_to_numpy=True, show_progress_bar=False)
    width = int(vec.shape[-1])
    if width != fc.EMBEDDING_DIM:
        raise fc.ProtocolError(f"MPNet load test width {width} != {fc.EMBEDDING_DIM}")
    return {"revision": revision, "expected": expected, "load_test_width": width,
            "device": str(enc.device), "ok": True}


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-manifest", required=True, type=Path,
                    help="Mac-side dump_code_hashes.py output; the transferred code tree must match it")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    out = fc.refuse_existing(args.out)
    if not fc.absp(args.code_manifest).is_file():
        raise fc.ProtocolError(f"--code-manifest {args.code_manifest} not found; run dump_code_hashes.py "
                               "on the Mac right before the transfer")

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
        cfg = str(Path(row["path"]).with_name("training_config.json"))
        if not fc.absp(cfg).exists():
            failures.append({"kind": "checkpoint_config", "path": cfg, "reason": "missing"})

    chunks = {}
    for split, spec in parsed["prepared"].items():
        res = fc.check_prepared_dir(spec)
        chunks[split] = {k: v for k, v in res.items() if k != "chunks"}
        if not res["ok"]:
            failures.append({"kind": f"prepared_{split}", **chunks[split]})

    mac = json.loads(fc.absp(args.code_manifest).read_text(encoding="utf-8"))
    diffs = fc.compare_code_hashes(mac["code"], fc.code_hashes())
    code = {"manifest": fc.rel(fc.absp(args.code_manifest)), "manifest_sha256": fc.file_sha256(args.code_manifest),
            "n_files": len(mac["code"]), "mac_git_head": (mac.get("git") or {}).get("head"),
            "ok": not diffs, "differences": diffs}
    failures += [{"kind": "code", **d} for d in diffs]

    try:
        mpnet = mpnet_check(parsed["mpnet_revision"])
    except Exception as exc:
        mpnet = {"expected": parsed["mpnet_revision"], "ok": False, "error": str(exc)}
        failures.append({"kind": "mpnet_snapshot", **mpnet})

    result = {"schema": "go_ppi_fusion_confirm/transfer_check/v1", "created_utc": fc.utc_now(),
              "passed": not failures, "parsed_protocol_path": fc.rel(fc.absp(args.protocol)),
              "parsed_protocol_sha256": fc.file_sha256(args.protocol),
              "n_checkpoint_rows": len(parsed["checkpoints"]), "n_input_rows": len(parsed["inputs"]),
              "checks": checks, "prepared": chunks, "code": code, "mpnet": mpnet, "failures": failures,
              "run_env": fc.run_env_record()}
    if failures:
        out = out.with_name(f"{out.stem}.FAILED.{fc.utc_now().replace(':', '')}{out.suffix}")
    fc.write_json(out, result)
    print(f"transfer check {'PASSED' if not failures else 'FAILED'}: {len(checks)} files hashed, "
          f"{len(failures)} failures -> {fc.rel(out)}")
    for f in failures:
        print("  FAIL", f.get("kind"), f.get("path", ""), f.get("reason", ""))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
