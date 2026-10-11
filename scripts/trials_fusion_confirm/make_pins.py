#!/usr/bin/env python3
"""Write the study pins (protocol_trials_fusion.md §7).

Revision 1 (``pins.json``) is written ON THE REMOTE GPU HOST after the step -1 transfer
check and gate 1, before training. It also writes the derived training configs
``<results>/configs/*.json`` (once). ``--revise`` writes ``pins_r{n}.json``: it recomputes
every field, never writes configs (it re-hashes them and requires equality), always
recomputes ``git``/``environment`` (previous values go to ``amendment.previous``) and
requires the changed non-whitelisted fields to equal ``--changed-fields`` exactly, with
``--confirmatory-output-seen false`` for any declared change.

``--dev-local`` writes a NON-CONFIRMATORY ``dev_pins.json`` under
``results/trials_fusion_confirm/dev_local/`` for the Mac CPU smoke only (no GPU, no
transfer check); confirmatory scripts refuse it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trials_domain import fusion_confirm as fc  # noqa: E402

NEW_2022 = [(25, 87), (25, 123), (100, 87), (100, 123)]
SEEDS_2021 = [13, 21, 42]


def derived_configs(results_dir: str, cv_dir: str, n_folds: int) -> Dict[str, Dict[str, Any]]:
    """§3b derived configs: base JSON with only embedding_cache_path (and trials_split_dir)."""
    base = json.loads(fc.absp(fc.BASE_CONFIG_REL).read_text(encoding="utf-8"))
    cache = f"{results_dir}/cache/trials_text_embeddings.pt"
    out = {}
    cfg = dict(base, embedding_cache_path=cache)
    out[f"{results_dir}/configs/lc_trials_baseline_cache.json"] = {
        "content": cfg, "diff_vs_base": {"embedding_cache_path": [base["embedding_cache_path"], cache]},
        "expected_unexpected_fields": ["embedding_cache_path"]}
    for k in range(n_folds):
        split = f"{cv_dir}/fold_{k}"
        cfg = dict(base, embedding_cache_path=cache, trials_split_dir=split)
        out[f"{results_dir}/configs/lc_trials_baseline_cv2021_fold{k}.json"] = {
            "content": cfg,
            "diff_vs_base": {"embedding_cache_path": [base["embedding_cache_path"], cache],
                             "trials_split_dir": [base["trials_split_dir"], split]},
            "expected_unexpected_fields": ["embedding_cache_path", "trials_split_dir"]}
    return out


def check_configs_unchanged(prev_configs: Dict[str, Any]) -> Dict[str, Any]:
    """``--revise`` never writes configs: re-hash them and require equality (§7)."""
    bad = fc.verify_pinned_hashes({"configs": prev_configs}, ("configs",))
    if bad:
        raise fc.ProtocolError(f"--revise: config hashes differ from the previous revision: {bad}")
    return prev_configs


def build_body(args, results_dir: str, cv_dir: str, write_configs: bool, prev_configs=None) -> Dict[str, Any]:
    protocol_rel = fc.rel(fc.absp(args.protocol))
    parsed = fc.parse_protocol(fc.absp(args.protocol).read_text(encoding="utf-8"))
    cv = json.loads((fc.absp(cv_dir) / "cv2021_manifest.json").read_text(encoding="utf-8"))
    if set(fc.DEV_SMOKE_EXPOSED_2021) & set(cv.get("exposed_topics_closed") or []):
        raise fc.ProtocolError("§9 A0 dev-smoke topics overlap exposed_topics_closed; the fold build changed")

    # environment first, so a failed GPU probe aborts before any config is written
    revision_expected, _ = fc.resolve_mpnet_snapshot(parsed["mpnet_revision"])
    env = fc.environment_record(args.host_label, revision_expected, probe_gpu=not args.dev_local)
    if not args.dev_local and not env.get("gpu_name"):
        raise fc.ProtocolError("no CUDA GPU found by the probe; pins.json is written on the remote GPU host")

    # cache copy starting state (§3b): sha256(copy) == sha256(original) == protocol value
    copy_rel = f"{results_dir}/cache/trials_text_embeddings.pt"
    expected_cache = parsed["inputs"][fc.ORIGINAL_CACHE_REL]
    orig_sha = fc.file_sha256(fc.ORIGINAL_CACHE_REL)
    copy_sha = fc.file_sha256(copy_rel) if fc.absp(copy_rel).exists() else None
    if not (orig_sha == expected_cache and (copy_sha == orig_sha or args.revise)):
        raise fc.ProtocolError(f"cache copy {copy_rel} sha {copy_sha} / original {orig_sha} != {expected_cache}")

    inputs = {p: fc.file_sha256(p) for p in sorted(parsed["inputs"])}
    for p, h in parsed["inputs"].items():
        if inputs[p] != h:
            raise fc.ProtocolError(f"input {p} sha256 {inputs[p]} != protocol {h}")
    for name, h in sorted(cv["fold_files"].items()):
        p = f"{cv_dir}/{name}"
        inputs[p] = fc.file_sha256(p)
        if inputs[p] != h:
            raise fc.ProtocolError(f"fold file {p} changed since build_cv2021.py")
    inputs[f"{cv_dir}/cv2021_manifest.json"] = fc.file_sha256(f"{cv_dir}/cv2021_manifest.json")
    inputs[f"{copy_rel}@pin_time"] = expected_cache

    chunks = fc.chunk_hashes(parsed["test_dir"])
    if fc.manifest_digest_from_hashes(chunks) != parsed["manifest_digest"] or len(chunks) != parsed["n_test_chunks"]:
        raise fc.ProtocolError("published test chunks do not match the protocol manifest digest")
    for row in parsed["checkpoints"]:
        if fc.file_sha256(row["path"]) != row["sha256"]:
            raise fc.ProtocolError(f"checkpoint {row['path']} sha256 mismatch")

    # configs (written once, in step 0)
    if write_configs:
        configs = {"base": {"path": fc.BASE_CONFIG_REL, "sha256": fc.file_sha256(fc.BASE_CONFIG_REL)}, "derived": {}}
        for path, d in derived_configs(results_dir, cv_dir, cv["n_folds"]).items():
            target = fc.refuse_existing(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(d["content"], indent=2) + "\n", encoding="utf-8")
            configs["derived"][path] = {"sha256": fc.file_sha256(target), "diff_vs_base": d["diff_vs_base"],
                                        "expected_unexpected_fields": d["expected_unexpected_fields"]}
    else:
        configs = check_configs_unchanged(prev_configs)

    tc_path = f"{results_dir}/transfer_check.json"
    if args.dev_local:
        transfer = {"path": None, "sha256": None, "passed": None, "parsed_protocol_sha256": None,
                    "note": "dev-local smoke: no transfer check"}
    else:
        tc = json.loads(fc.absp(tc_path).read_text(encoding="utf-8"))
        transfer = {"path": tc_path, "sha256": fc.file_sha256(tc_path), "passed": bool(tc.get("passed")),
                    "parsed_protocol_sha256": tc.get("parsed_protocol_sha256")}
    fold_k = {f"fold_{k}": f"{results_dir}/cohort2021/fold_{k}" for k in range(cv["n_folds"])}
    body = {
        "schema": "trials_fusion_confirm/pins/v7",
        "confirmatory": not args.dev_local,
        "created_utc": fc.utc_now(),
        "protocol": {"path": protocol_rel, "sha256": fc.file_sha256(args.protocol)},
        "git": fc.git_record(),
        "environment": env,
        "transfer_check": transfer,
        "results_dir": results_dir,
        "inputs": inputs,
        "published_prepared_cache": {
            "test_dir": parsed["test_dir"], "validation_dir": parsed["validation_dir"], "chunks": chunks,
            "n_chunks": {"test": len(chunks), "validation": len(list(fc.absp(parsed["validation_dir"]).glob("chunk_*.npz")))},
            "manifest_digest": parsed["manifest_digest"]},
        "code": fc.code_hashes(),
        "configs": configs,
        "cohort2022": {
            "existing_checkpoints": [{"fraction": r["fraction"], "seed": r["seed"], "path": r["path"],
                                      "sha256": r["sha256"], "training_device": None} for r in parsed["checkpoints"]],
            "new_runs": [{"fraction": F, "seed": S, "device": "cuda",
                          "output_dir": f"{results_dir}/cohort2022/checkpoints/trials_baseline_f{F}_s{S}"}
                         for F, S in NEW_2022],
            "validation": fc.VAL2022_REL, "test": fc.TEST2022_REL},
        "cohort2021": {**{k: v for k, v in cv.items() if k != "run_env"},
                       "run_dirs": fold_k,
                       # §9 A0: closed-unexposed topics scored under fusion by the Mac dev smoke
                       "dev_smoke_exposed_topics": list(fc.DEV_SMOKE_EXPOSED_2021),
                       "dev_smoke_exposed_source": "protocol §9 A0"},
        "analysis": fc.ANALYSIS_PARAMS,
    }
    return body


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--host-label", default=None, help="human label for pins.environment.host")
    ap.add_argument("--cv-dir", default=fc.CV2021_REL)
    ap.add_argument("--revise", action="store_true")
    ap.add_argument("--previous", type=Path, default=None)
    ap.add_argument("--amendment", default=None, help="§9 entry id, e.g. A1")
    ap.add_argument("--reason", default="")
    ap.add_argument("--changed-fields", default="")
    ap.add_argument("--confirmatory-output-seen", choices=["false", "true"], default=None)
    ap.add_argument("--dev-local", action="store_true", help="non-confirmatory smoke pins (Mac)")
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    out = fc.refuse_existing(args.out)

    if args.dev_local:
        fc.require_dev_local(out)
        if out.name != "dev_pins.json" or args.revise:
            raise fc.ProtocolError("--dev-local writes only <dev_local>/dev_pins.json (revision 1)")
        results_dir = fc.rel(out.parent)
        body = build_body(args, results_dir, args.cv_dir, write_configs=True)
        pins = {"schema": body.pop("schema"), "revision": 1, "supersedes": None, "amendment": None, **body}
    elif not args.revise:
        if out.name != "pins.json":
            raise fc.ProtocolError("revision 1 must be written as pins.json")
        results_dir = fc.rel(out.parent)
        tc = json.loads(fc.absp(f"{results_dir}/transfer_check.json").read_text(encoding="utf-8"))
        if not tc.get("passed") or tc.get("parsed_protocol_sha256") != fc.file_sha256(args.protocol):
            raise fc.ProtocolError("transfer_check.json missing/failed or parsed a different protocol file")
        # §9 A0: the code tree must have been checked against the Mac manifest, and pins.code
        # must still equal it (nothing edited between step -1 and here)
        code = tc.get("code") or {}
        if code.get("ok") is not True:
            raise fc.ProtocolError("transfer_check.json has no passing code-tree check (run verify_transfer.py "
                                   "with --code-manifest)")
        if fc.file_sha256(code["manifest"]) != code["manifest_sha256"]:
            raise fc.ProtocolError(f"code manifest {code['manifest']} changed since the transfer check")
        mac = json.loads(fc.absp(code["manifest"]).read_text(encoding="utf-8"))["code"]
        diffs = fc.compare_code_hashes(mac, fc.code_hashes())
        if diffs:
            raise fc.ProtocolError(f"code tree differs from the Mac manifest: {diffs[:5]}")
        body = build_body(args, results_dir, args.cv_dir, write_configs=True)
        pins = {"schema": body.pop("schema"), "revision": 1, "supersedes": None, "amendment": None, **body}
    else:
        if args.previous is None or args.amendment is None or args.confirmatory_output_seen is None:
            raise fc.ProtocolError("--revise needs --previous, --amendment and --confirmatory-output-seen")
        prev = fc.load_verified_pins(args.previous)
        n = prev["revision"] + 1
        if out.name != f"pins_r{n}.json" or out.parent != fc.absp(args.previous).parent:
            raise fc.ProtocolError(f"revision {n} must be written as pins_r{n}.json next to the previous revision")
        body = build_body(args, prev["results_dir"], args.cv_dir, write_configs=False, prev_configs=prev["configs"])
        body["transfer_check"] = prev["transfer_check"]
        changed = [f for f in args.changed_fields.split(",") if f.strip()]
        pins = {"schema": body.pop("schema"), "revision": n,
                "supersedes": {"path": fc.rel(fc.absp(args.previous)), "sha256": fc.file_sha256(args.previous)},
                "amendment": {"id": args.amendment, "date": fc.utc_now()[:10], "reason": args.reason,
                              "confirmatory_output_seen": args.confirmatory_output_seen == "true",
                              "changed_fields": sorted(changed),
                              "previous": {"git": prev["git"], "environment": prev["environment"]}},
                **body}
        fc.verify_revision(prev, pins, pins["supersedes"]["path"], pins["supersedes"]["sha256"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {fc.rel(out)} (revision {pins['revision']}, confirmatory={pins['confirmatory']}, "
          f"gpu={pins['environment'].get('gpu_name')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
