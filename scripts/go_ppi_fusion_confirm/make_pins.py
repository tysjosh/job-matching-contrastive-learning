#!/usr/bin/env python3
"""Write the study pins (protocol_go_ppi_fusion.md §7). Remote host only.

Revision 1 (``pins.json``) is written ON THE REMOTE L40S HOST after the step −1 transfer
check, gate 1, ``build_inputs.py``, ``encode.py`` and the cache copy, before gate 2. This is
the one step-0 script run WITHOUT ``CUDA_VISIBLE_DEVICES=""``: it reads the GPU name
(``torch.cuda.get_device_name(0)`` must contain "L40S", exactly one device visible) and
touches no data on the GPU.

It requires a passing ``transfer_check.json`` that parsed this protocol file and a code tree
equal to the Mac manifest; asserts sha256(cache copy) == sha256(original) == the protocol
value; re-hashes every §7 input, published checkpoint and prepared-cache chunk; hashes every
``build_inputs.py`` output (from ``build_manifest.json``) and the masked-encode directory;
and records the derived configs written by ``build_inputs.py`` (never rewritten).

``--revise`` writes ``pins_r{n}.json``: everything recomputed, configs re-hashed and required
unchanged, ``git``/``environment`` recomputed (previous values in ``amendment.previous``), and
the changed non-whitelisted fields must equal ``--changed-fields`` exactly, with
``--confirmatory-output-seen false`` for any declared change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402

DESCRIPTIVE_INPUTS = ("results/ontology_ceiling/go_independent_signal.json",)


def derived_block(build_manifest_path: str) -> Dict[str, Any]:
    bm = json.loads(fc.absp(build_manifest_path).read_text(encoding="utf-8"))
    bad = [p for p, h in bm["files"].items() if not fc.absp(p).exists() or fc.file_sha256(p) != h]
    if bad:
        raise fc.ProtocolError(f"build_inputs outputs changed since build_manifest.json: {bad[:5]}")
    cv_path = next(p for p in bm["files"] if p.endswith("cv/cv_manifest.json"))
    cv = json.loads(fc.absp(cv_path).read_text(encoding="utf-8"))
    hu_path = next(p for p in bm["files"] if p.endswith("huri/huri_manifest.json"))
    hu = json.loads(fc.absp(hu_path).read_text(encoding="utf-8"))
    files = dict(bm["files"])
    files[fc.rel(fc.absp(build_manifest_path))] = fc.file_sha256(build_manifest_path)
    data_dir = str(Path(cv_path).parent.parent.as_posix())
    return {
        "files": files, "build_manifest": fc.rel(fc.absp(build_manifest_path)), "data_dir": data_dir,
        "phi_dir": str(Path(next(p for p in bm["files"] if p.endswith("n_pmid.json"))).parent.as_posix()),
        "cv": {"manifest": cv_path, "n_folds": cv["n_folds"], "fold_seed": cv["fold_seed"],
               "fold_dir": f"{data_dir}/cv", "folds": cv["folds"], "view_groups": cv["groups"],
               "group_counts": cv["group_counts"], "two_build_identical": cv["two_build_identical"],
               "exposure": cv["exposure"],
               "fold_records": {f"fold_{m['k']}": m["records_by_grade"] for m in cv["fold_manifests"]},
               "dropped_pairs": {f"fold_{m['k']}": m["pairs_dropped_cross_partition"] for m in cv["fold_manifests"]}},
        "bclean": {"anchors": cv["bclean"]["anchors"], "sha256": cv["bclean"]["sha256"],
                   "n_anchors": cv["bclean"]["n_anchors"], "hard_defined": cv["bclean"]["hard_defined"],
                   "all_B_hard_defined": cv["bclean"]["all_B_hard_defined"]},
        "bclean_records_with_exposed_partner": cv["bclean"]["bclean_records_with_exposed_partner"],
        "huri": {"manifest": hu_path, "counts": hu["counts"], "grade1_counts": hu["counts"]["grade1_counts"],
                 "hard_defined_anchors": hu["counts"]["hard_defined_anchors"],
                 "anchors_sha256": hu["anchors_sha256"]},
        "shuffle": bm["shuffle"], "typed_check": bm["typed_check"], "isa_index": bm["isa_index"],
        "n_pmid": bm["n_pmid"], "ux_to_x_changed_records_A_test": bm["ux_to_x_changed_records_A_test"],
        "mention_flags": bm["mention_flags"], "alignment": bm["alignment"], "cohortA_inputs": bm["cohortA"],
        "configs": bm["configs"],
    }


def check_configs(configs: Dict[str, Any]) -> Dict[str, Any]:
    bad = fc.verify_pinned_hashes({"configs": configs}, ("configs",))
    if bad:
        raise fc.ProtocolError(f"derived config hashes differ: {bad}")
    for p, d in configs["derived"].items():
        if d["expected_unexpected_fields"] != fc.EXPECTED_UNEXPECTED_FIELDS:
            raise fc.ProtocolError(f"{p}: audit set {d['expected_unexpected_fields']} != §3b")
    return configs


def build_body(args, results_dir: str, prev_configs=None) -> Dict[str, Any]:
    parsed = fc.parse_protocol(fc.absp(args.protocol).read_text(encoding="utf-8"))
    revision, _ = fc.resolve_mpnet_snapshot(parsed["mpnet_revision"])
    gpu = fc.gpu_record(fc.ANALYSIS_PARAMS["gpu_name_must_contain"])
    env = fc.environment_record(gpu, args.host_label, revision)

    expected_cache = parsed["inputs"][fc.MASKED_CACHE_REL]
    orig = fc.file_sha256(fc.MASKED_CACHE_REL)
    copy = fc.file_sha256(fc.CACHE_COPY_REL) if fc.absp(fc.CACHE_COPY_REL).exists() else None
    if orig != expected_cache or (copy != orig and not args.revise):
        raise fc.ProtocolError(f"cache copy {fc.CACHE_COPY_REL} sha {copy} / original {orig} != protocol "
                               f"{expected_cache} (cp -n the original first, §3b)")
    inputs = {p: fc.file_sha256(p) for p in sorted(parsed["inputs"])}
    for p, h in parsed["inputs"].items():
        if inputs[p] != h:
            raise fc.ProtocolError(f"input {p} sha256 {inputs[p]} != protocol {h}")
    for p in DESCRIPTIVE_INPUTS:
        if fc.absp(p).exists():
            inputs[p] = fc.file_sha256(p)
    inputs[f"{fc.CACHE_COPY_REL}@pin_time"] = expected_cache
    for row in parsed["checkpoints"]:
        if fc.file_sha256(row["path"]) != row["sha256"]:
            raise fc.ProtocolError(f"published checkpoint {row['path']} sha256 mismatch")
    prepared = {}
    for split, spec in parsed["prepared"].items():
        res = fc.check_prepared_dir(spec)
        if not res["ok"]:
            raise fc.ProtocolError(f"published prepared cache {split} does not match §2.1: {res}")
        prepared[split] = {"dir": spec["dir"], "chunks": res["chunks"], "n_chunks": res["n_chunks"],
                           "manifest_digest": res["digest"]}
    md = fc.absp(args.masked_encodes)
    if not md.is_dir():
        raise fc.ProtocolError(f"masked encodes {fc.rel(md)} missing; run encode.py first")
    em = json.loads((md / "encode_manifest.json").read_text(encoding="utf-8"))
    masked = {"dir": fc.rel(md), "digest": fc.manifest_digest(md), "n_chunks": len(fc.chunk_hashes(md)),
              "n_texts": em["n_texts"], "encode_manifest_sha256": fc.file_sha256(md / "encode_manifest.json"),
              "device": em["device"]}
    if masked["digest"] != em["manifest_digest"]:
        raise fc.ProtocolError("masked-encode chunks changed since encode.py")

    derived = derived_block(args.build_manifest)
    configs = check_configs(derived.pop("configs")) if prev_configs is None else check_configs(prev_configs)
    pub = {r: json.loads(fc.absp(p).read_text(encoding="utf-8")) for r, p in fc.PUBLISHED_FUSION_REL.items()}
    ties = fc.near_tie_cells(pub)
    if [c[:3] for c in ties] != sorted([list(c) for c in fc.GATE2_NEAR_TIE_CELLS]):
        raise fc.ProtocolError(f"published near-tie cells {ties} != §5.2 2a {fc.GATE2_NEAR_TIE_CELLS}")
    analysis = dict(fc.ANALYSIS_PARAMS)
    analysis["gate2_near_tie_cells"] = ties
    analysis["T_computed"] = bool(derived["typed_check"]["passed"])
    data_dir = derived["data_dir"]
    cv_dir = derived["cv"]["fold_dir"]
    body = {
        "schema": "go_ppi_fusion_confirm/pins/v1", "confirmatory": True, "created_utc": fc.utc_now(),
        "protocol": {"path": fc.rel(fc.absp(args.protocol)), "sha256": fc.file_sha256(args.protocol)},
        "git": fc.git_record(), "environment": env, "results_dir": results_dir, "data_dir": data_dir,
        "inputs": inputs, "published_prepared_cache": prepared, "masked_encodes": masked,
        "derived": derived, "code": fc.code_hashes(), "configs": configs,
        "published_checkpoints": [{"fraction": r["fraction"], "seed": r["seed"], "path": r["path"],
                                   "sha256": r["sha256"]} for r in parsed["checkpoints"]],
        "cohortA": {"validation": f"{fc.MASKED_SPLIT_DIR}/validation.jsonl", "test": f"{fc.MASKED_SPLIT_DIR}/test.jsonl",
                    "aligned_published": {"validation": f"{fc.PUB_SPLIT_DIR}/validation.jsonl",
                                          "test": f"{fc.PUB_SPLIT_DIR}/test.jsonl"},
                    "new_runs": [{"run": f"A:f{F}_s{S}", "fraction": F, "seed": S, "device": "cuda",
                                  "config": f"{results_dir}/configs/lc_go_ppi_baseline_masked.json",
                                  "train_file": f"{data_dir}/cohortA/frac_{F}/train.jsonl",
                                  "validation_file": f"{data_dir}/cohortA/validation_positive.jsonl",
                                  "fraction_arg": fc.FRACTION_ARG[F],
                                  "output_dir": f"{results_dir}/cohortA/checkpoints/go_ppi_baseline_f{F}_s{S}"}
                                 for F, S in fc.COHORT_A_RUNS]},
        "cohortB": {"n_folds": derived["cv"]["n_folds"], "seeds": list(fc.COHORT_B_SEEDS), "fold_dir": cv_dir,
                    "runs": [{"run": f"B:k{k}_s{S}", "k": k, "seed": S, "device": "cuda",
                              "config": f"{results_dir}/configs/lc_go_ppi_baseline_cv_fold{k}.json",
                              "train_file": f"{cv_dir}/fold_{k}/train.jsonl",
                              "validation_file": f"{cv_dir}/fold_{k}/validation_positive.jsonl",
                              "fraction_arg": "1.0",
                              "output_dir": f"{results_dir}/cohortB/fold_{k}/go_ppi_baseline_full_s{S}"}
                             for k in range(derived["cv"]["n_folds"]) for S in fc.COHORT_B_SEEDS]},
        "huri": {"test": f"{data_dir}/huri/test.jsonl",
                 "test_grade1_unrestricted": f"{data_dir}/huri/test_grade1_unrestricted.jsonl"},
        "analysis": analysis,
    }
    return body


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--build-manifest", default=f"{fc.RESULTS_REL}/build_manifest.json")
    ap.add_argument("--masked-encodes", default=fc.MASKED_PREP_REL)
    ap.add_argument("--host-label", default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--revise", action="store_true")
    ap.add_argument("--previous", type=Path, default=None)
    ap.add_argument("--amendment", default=None, help="§9 entry id, e.g. A1")
    ap.add_argument("--reason", default="")
    ap.add_argument("--changed-fields", default="")
    ap.add_argument("--confirmatory-output-seen", choices=["false", "true"], default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    out = fc.refuse_existing(args.out)
    results_dir = fc.rel(out.parent)
    if not args.revise:
        if out.name != "pins.json":
            raise fc.ProtocolError("revision 1 must be written as pins.json")
        tc = json.loads(fc.absp(f"{results_dir}/transfer_check.json").read_text(encoding="utf-8"))
        if not tc.get("passed") or tc.get("parsed_protocol_sha256") != fc.file_sha256(args.protocol):
            raise fc.ProtocolError("transfer_check.json missing/failed or parsed a different protocol file")
        code = tc.get("code") or {}
        if code.get("ok") is not True or fc.file_sha256(code["manifest"]) != code["manifest_sha256"]:
            raise fc.ProtocolError("transfer_check.json has no passing code-tree check, or the manifest changed")
        mac = json.loads(fc.absp(code["manifest"]).read_text(encoding="utf-8"))["code"]
        diffs = fc.compare_code_hashes(mac, fc.code_hashes())
        if diffs:
            raise fc.ProtocolError(f"code tree differs from the Mac manifest: {diffs[:5]}")
        body = build_body(args, results_dir)
        body["transfer_check"] = {"path": f"{results_dir}/transfer_check.json",
                                  "sha256": fc.file_sha256(f"{results_dir}/transfer_check.json"),
                                  "passed": True, "parsed_protocol_sha256": tc["parsed_protocol_sha256"]}
        pins = {"schema": body.pop("schema"), "revision": 1, "supersedes": None, "amendment": None, **body}
    else:
        if args.previous is None or args.amendment is None or args.confirmatory_output_seen is None:
            raise fc.ProtocolError("--revise needs --previous, --amendment and --confirmatory-output-seen")
        prev = fc.load_verified_pins(args.previous)
        n = prev["revision"] + 1
        if out.name != f"pins_r{n}.json" or out.parent != fc.absp(args.previous).parent:
            raise fc.ProtocolError(f"revision {n} must be written as pins_r{n}.json next to the previous revision")
        body = build_body(args, prev["results_dir"], prev_configs=prev["configs"])
        body["derived"].pop("configs", None)
        body["transfer_check"] = prev["transfer_check"]
        changed = [f for f in args.changed_fields.split(",") if f.strip()]
        pins = {"schema": body.pop("schema"), "revision": n,
                "supersedes": {"path": fc.rel(fc.absp(args.previous)), "sha256": fc.file_sha256(args.previous)},
                "amendment": {"id": args.amendment, "date": fc.utc_now()[:10], "reason": args.reason,
                              "confirmatory_output_seen": args.confirmatory_output_seen == "true",
                              "changed_fields": sorted(changed),
                              "previous": {"git": prev["git"], "environment": prev["environment"]}},
                **body}
        pins = json.loads(json.dumps(fc.json_safe(pins)))       # compare in JSON form
        fc.verify_revision(prev, pins, pins["supersedes"]["path"], pins["supersedes"]["sha256"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc.json_safe(pins), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {fc.rel(out)} (revision {pins['revision']}, gpu={pins['environment'].get('gpu_name')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
