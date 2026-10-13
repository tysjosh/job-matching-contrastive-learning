#!/usr/bin/env python3
"""Step 0 label-free builds (protocol_go_ppi_fusion.md §2.2-§3d, §4.5, §6.2). CPU only.

Writes, all label-free with respect to every fusion outcome (no text score, no fused score):

* ``<data>/cohortA/frac_{F}/train.jsonl`` and ``validation_positive.jsonl`` (§3b);
* ``<data>/cv/fold_{k}/`` protein-disjoint CV files and ``cv/cv_manifest.json`` with the view
  groups, fold membership, exposure sets and ``Bclean`` (§3c);
* ``<data>/huri/test.jsonl``, ``test_grade1_unrestricted.jsonl`` and ``huri_manifest.json`` (§3d);
* ``<phi>/shuffle_maps/{file_id}.json`` for every scored file (§2.3), ``<phi>/n_pmid.json``
  (§2.4), ``<phi>/go_index_P_isa_only.pkl`` if the typed-parser check passes (§2.2.1),
  ``<phi>/mention_flags_A_test.json`` (§4.5);
* ``<configs>/lc_go_ppi_baseline_masked.json`` and ``lc_go_ppi_baseline_cv_fold{k}.json`` with
  their ``report_factor`` audit (§3b, §3c);
* ``build_manifest.json`` (sha256 of every output, counts, checks; pinned by make_pins.py).

Any failed build assert stops the script (resolved by a pre-pins amendment).

``--structure-only --out <dev path>`` (Mac dev smoke, §6.5) builds the cohort-B fold
structure in memory and writes only its counts; no record file is written, nothing is scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402

EXPECTED_GROUPS = {"groups": 14, "proteins": 31, "max_size": 4}


def load_proteins(path: str) -> Dict[str, Dict[str, Any]]:
    return {p["protein_id"]: p for p in fc.read_jsonl(path)}


def check_groups(groups: List[List[str]]) -> Dict[str, int]:
    got = {"groups": len(groups), "proteins": sum(len(g) for g in groups),
           "max_size": max((len(g) for g in groups), default=0)}
    if got != EXPECTED_GROUPS:
        raise fc.ProtocolError(f"identical-view groups {got} != expected {EXPECTED_GROUPS} (§3c); "
                               "stop and amend before pins")
    return got


def build_cohort_b(proteins, pairs, n_folds: int, seed: int, out_dir=None) -> Dict[str, Any]:
    """§3c: view groups, fold assignment and fold files, each built twice independently
    (the two builds must give identical sha256 for every file), exposure sets and Bclean.
    ``out_dir=None`` = structure only (nothing written)."""
    groups = fc.view_groups(proteins)
    group_counts = check_groups(groups)
    folds = fc.assign_protein_folds(proteins, groups, n_folds, seed)
    if folds != fc.assign_protein_folds(proteins, groups, n_folds, seed):
        raise fc.ProtocolError("fold assignment is not deterministic")
    manifests = []
    for k in range(n_folds):
        files, m = fc.build_fold(proteins, pairs, folds, k, seed, groups=groups)
        _, m2 = fc.build_fold(proteins, pairs, folds, k, seed, groups=groups)
        if m["rows_sha256"] != m2["rows_sha256"]:
            raise fc.ProtocolError(f"fold {k}: two independent builds differ (§3c)")
        if out_dir is not None:
            public = fc.write_fold(files, m, Path(out_dir) / f"fold_{k}")
            if public["sha256"] != m["rows_sha256"]:
                raise fc.ProtocolError(f"fold {k}: written files differ from the in-memory build")
        del files
        manifests.append(m)
    exposure = fc.exposure_sets([f"{fc.PUB_SPLIT_DIR}/validation.jsonl", f"{fc.PUB_SPLIT_DIR}/test.jsonl"], groups)
    bclean = fc.bclean_summary(manifests, set(exposure["exposed"]), set(exposure["exposed_closed"]))
    if bclean["hard_defined"] < fc.ANALYSIS_PARAMS["bclean_min_hard_defined"]:
        raise fc.ProtocolError(f"hard-defined Bclean = {bclean['hard_defined']} < "
                               f"{fc.ANALYSIS_PARAMS['bclean_min_hard_defined']} (§3c); stop and amend before pins")
    public = [{k: v for k, v in m.items() if not k.startswith("_")} for m in manifests]
    return {"groups": groups, "group_counts": group_counts, "folds": folds, "fold_manifests": public,
            "two_build_identical": True, "exposure": exposure, "bclean": bclean}


def shuffle_for(path: str, simgic, seed: int) -> Dict[str, Any]:
    recs = fc.read_jsonl(path)
    q = [r["metadata"]["resume_id"] for r in recs]
    a = [r["resume"]["protein_id"] for r in recs]
    b = [r["job"]["partner_id"] for r in recs]
    ua = [list(r["resume"].get("skill_uris") or []) for r in recs]
    ub = [list(r["job"].get("skill_uris") or []) for r in recs]
    out = fc.shuffle_map(q, a, b, ua, ub, simgic, seed)
    out.update({"file": path, "file_sha256": fc.file_sha256(path)})
    return out


def write_config(path: Path, content: Dict[str, Any], files: Dict[str, str]) -> str:
    fc.refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    files[fc.rel(path)] = fc.file_sha256(path)
    return fc.rel(path)


def derived_configs(out_configs: Path, data_rel: str, n_folds: int) -> Dict[str, Dict[str, Any]]:
    base = json.loads(fc.absp(fc.BASE_CONFIG_REL).read_text(encoding="utf-8"))
    masked = dict(base, go_ppi_split_dir=fc.MASKED_SPLIT_DIR, go_ppi_converted_dir=fc.MASKED_CONVERTED_DIR,
                  embedding_cache_path=fc.CACHE_COPY_REL)
    out = {fc.rel(out_configs / "lc_go_ppi_baseline_masked.json"): masked}
    for k in range(n_folds):
        out[fc.rel(out_configs / f"lc_go_ppi_baseline_cv_fold{k}.json")] = dict(
            masked, go_ppi_split_dir=f"{data_rel}/cv/fold_{k}")
    return out


def main(argv=None) -> int:
    fc.require_hashseed()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--protocol", type=Path, default=Path(fc.PROTOCOL_REL))
    ap.add_argument("--fold-seed", type=int, default=fc.FOLD_SEED)
    ap.add_argument("--n-folds", type=int, default=fc.N_FOLDS)
    ap.add_argument("--shuffle-seed", type=int, default=fc.SHUFFLE_SEED)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out-data", type=Path, default=Path(fc.DATA_REL))
    ap.add_argument("--out-phi", type=Path, default=Path(f"{fc.RESULTS_REL}/phi"))
    ap.add_argument("--out-configs", type=Path, default=Path(f"{fc.RESULTS_REL}/configs"))
    ap.add_argument("--out-manifest", type=Path, default=None,
                    help="default: <parent of --out-phi>/build_manifest.json")
    ap.add_argument("--structure-only", action="store_true",
                    help="Mac dev smoke: cohort-B fold structure counts only (writes --out)")
    ap.add_argument("--out", type=Path, default=None, help="--structure-only output JSON")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    fc.require_cpu(args.device, "data builds")
    if (args.fold_seed, args.n_folds, args.shuffle_seed) != (fc.FOLD_SEED, fc.N_FOLDS, fc.SHUFFLE_SEED):
        raise fc.ProtocolError("fold seed 2027, 5 folds and shuffle seed 11 are fixed by the protocol")
    t0 = time.time()
    proteins = load_proteins(fc.MASKED_PROTEINS_REL)
    pairs = fc.read_jsonl(fc.PAIRS_REL)

    if args.structure_only:
        if args.out is None:
            raise fc.ProtocolError("--structure-only needs --out under results/go_ppi_fusion_confirm_dev/")
        fc.require_dev(args.out)
        fc.refuse_existing(args.out)
        cb = build_cohort_b(proteins, pairs, args.n_folds, args.fold_seed, None)
        summary = {"schema": "go_ppi_fusion_confirm/fold_structure_dev/v1", "created_utc": fc.utc_now(),
                   "confirmatory": False, "note": "NON-CONFIRMATORY dev smoke (§6.5): fold structure only, "
                   "no record file written, nothing scored",
                   "group_counts": cb["group_counts"], "two_build_identical": cb["two_build_identical"],
                   "fold_sizes": [len(f) for f in cb["folds"]],
                   "fold_manifests": cb["fold_manifests"],
                   "exposure_counts": {k: len(v) for k, v in cb["exposure"].items()},
                   "bclean": {k: v for k, v in cb["bclean"].items() if k != "anchors"},
                   "seconds": round(time.time() - t0, 1), "run_env": fc.run_env_record()}
        fc.write_json(args.out, summary)
        b = cb["bclean"]
        print(f"fold structure: sizes {summary['fold_sizes']}, Bclean {b['n_anchors']} anchors "
              f"({b['hard_defined']} hard-defined; all-B {b['all_B_hard_defined']}) -> {fc.show(args.out)}")
        return 0

    out_data, out_phi, out_cfg = fc.absp(args.out_data), fc.absp(args.out_phi), fc.absp(args.out_configs)
    manifest_path = fc.absp(args.out_manifest or (Path(args.out_phi).parent / "build_manifest.json"))
    for p in (out_data, out_phi, out_cfg, manifest_path):
        fc.refuse_existing(p)
    files: Dict[str, str] = {}
    man: Dict[str, Any] = {"schema": "go_ppi_fusion_confirm/build_manifest/v1", "created_utc": fc.utc_now(),
                           "protocol": {"path": fc.rel(fc.absp(args.protocol)), "sha256": fc.file_sha256(args.protocol)},
                           "label_free": "no text score or fused score is computed by this script"}
    pruned = fc.load_index(fc.PRUNED_INDEX_REL)
    unpruned = fc.load_index(fc.UNPRUNED_INDEX_REL)
    simgic = fc.make_matcher(pruned, "simgic")

    # ---- 2b alignment (label-free) and cohort A inputs
    align = {}
    for split in ("train", "validation", "test"):
        pub = fc.read_jsonl(f"{fc.PUB_SPLIT_DIR}/{split}.jsonl")
        msk = fc.read_jsonl(f"{fc.MASKED_SPLIT_DIR}/{split}.jsonl")
        fc.align_records(pub, msk, f"{split}: published vs masked")
        align[split] = len(pub)
    uri_check = fc.check_protein_uris(proteins.values(), pruned)
    if not uri_check["ok"]:
        raise fc.ProtocolError(f"pruned proteins.jsonl go_uris differ from the pruned index direct: {uri_check}")
    man["alignment"] = {"records": align, "protein_uris": uri_check}
    ca = fc.build_cohort_a(out_data / "cohortA")
    files.update(ca["files"])
    man["cohortA"] = {k: v for k, v in ca.items() if k != "files"}
    print(f"cohort A inputs written ({time.time() - t0:.0f}s)", flush=True)

    # ---- cohort B
    cb = build_cohort_b(proteins, pairs, args.n_folds, args.fold_seed, out_data / "cv")
    for k, m in enumerate(cb["fold_manifests"]):
        for name, h in m["rows_sha256"].items():
            files[fc.rel(out_data / "cv" / f"fold_{k}" / name)] = h
        files[fc.rel(out_data / "cv" / f"fold_{k}" / "fold_manifest.json")] = fc.file_sha256(
            out_data / "cv" / f"fold_{k}" / "fold_manifest.json")
    cv_manifest = {"fold_seed": args.fold_seed, "n_folds": args.n_folds, "groups": cb["groups"],
                   "group_counts": cb["group_counts"], "folds": cb["folds"],
                   "two_build_identical": cb["two_build_identical"], "exposure": cb["exposure"],
                   "bclean": cb["bclean"], "fold_manifests": cb["fold_manifests"]}
    p = fc.write_json(out_data / "cv" / "cv_manifest.json", cv_manifest)
    files[fc.rel(p)] = fc.file_sha256(p)
    man["cohortB"] = {"group_counts": cb["group_counts"], "fold_sizes": [len(f) for f in cb["folds"]],
                      "two_build_identical": cb["two_build_identical"],
                      "exposure_counts": {k: len(v) for k, v in cb["exposure"].items()},
                      "bclean": {k: v for k, v in cb["bclean"].items() if k != "anchors"}}
    print(f"cohort B folds written; Bclean hard-defined {cb['bclean']['hard_defined']} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- HuRI cohort
    test_partition = set(fc.exposure_sets([f"{fc.PUB_SPLIT_DIR}/test.jsonl"], [])["exposed"])
    gene_terms = {pid: pruned.closure.get(pid, frozenset()) for pid in proteins}
    ens = fc.ensembl_gene_map(fc.ALIASES_REL, fc.INFO_REL, gene_terms)
    edges = fc.huri_edges(fc.HURI_REL, ens)
    nbrs = fc.string_neighbours(fc.LINKS_REL, fc.INFO_REL, test_partition)
    hu = fc.build_huri_rows(proteins, pairs, test_partition, edges, nbrs,
                            {pid: len(v) for pid, v in gene_terms.items()}, args.fold_seed)
    exp_n, tol = fc.ANALYSIS_PARAMS["huri_expected_anchors"]
    if abs(hu["counts"]["anchors"] - exp_n) > tol:
        raise fc.ProtocolError(f"HuRI anchors {hu['counts']['anchors']} not within {exp_n} ± {tol} (§3d); "
                               "stop and amend before pins")
    hdir = out_data / "huri"
    files[fc.rel(hdir / "test.jsonl")] = fc.write_jsonl(hdir / "test.jsonl", hu["rows"])
    files[fc.rel(hdir / "test_grade1_unrestricted.jsonl")] = fc.write_jsonl(
        hdir / "test_grade1_unrestricted.jsonl", hu["rows_grade1_unrestricted"])
    hman = {"counts": hu["counts"], "anchors": hu["anchors"], "anchors_sha256": fc.list_sha256(hu["anchors"]),
            "string_grade2_positives": hu["string_grade2_positives"], "mapped_edges": [list(e) for e in edges],
            "n_mapped_edges": len(edges), "ensembl_mapped_genes": len(ens),
            "partition": "entities of preprocess/go_ppi_splits/test.jsonl", "n_partition": len(test_partition),
            "shuffle_seeds": {"test": f"{args.fold_seed}|huri|test",
                              "test_grade1_unrestricted": f"{args.fold_seed}|huri|test_grade1_unrestricted"}}
    p = fc.write_json(hdir / "huri_manifest.json", hman)
    files[fc.rel(p)] = fc.file_sha256(p)
    man["huri"] = hu["counts"]
    print(f"HuRI cohort written: {hu['counts']} ({time.time() - t0:.0f}s)", flush=True)

    # ---- shuffle maps for every scored file (§2.3)
    scored = {"A_validation": f"{fc.MASKED_SPLIT_DIR}/validation.jsonl", "A_test": f"{fc.MASKED_SPLIT_DIR}/test.jsonl",
              "huri_test": fc.rel(hdir / "test.jsonl"),
              "huri_test_grade1_unrestricted": fc.rel(hdir / "test_grade1_unrestricted.jsonl")}
    for k in range(args.n_folds):
        for split in ("validation", "test"):
            scored[f"B_fold{k}_{split}"] = fc.rel(out_data / "cv" / f"fold_{k}" / f"{split}.jsonl")
    man["shuffle"] = {}
    for fid, path in scored.items():
        sm = shuffle_for(path, simgic, args.shuffle_seed)
        p = fc.write_json(out_phi / "shuffle_maps" / f"{fid}.json", sm)
        files[fc.rel(p)] = fc.file_sha256(p)
        man["shuffle"][fid] = {k: v for k, v in sm.items() if k != "mapping"}
        print(f"  shuffle {fid}: draws {[s['draws'] for s in sm['strata']]}", flush=True)

    # ---- n_PMID (§2.4)
    n_pmid_all = fc.parse_n_pmid(fc.GAF_REL)
    n_pmid = {pid: int(n_pmid_all.get(pid, 0)) for pid in sorted(proteins)}
    p = fc.write_json(out_phi / "n_pmid.json", {"rule": "distinct PMID: tokens in GAF col 6, all aspects and "
                                                "evidence, NOT lines excluded", "gaf": fc.GAF_REL,
                                                "gaf_sha256": fc.file_sha256(fc.GAF_REL), "n_pmid": n_pmid})
    files[fc.rel(p)] = fc.file_sha256(p)
    man["n_pmid"] = {"proteins": len(n_pmid), "zero": sum(1 for v in n_pmid.values() if v == 0)}

    # ---- typed-parser check (gate 2c) and the is_a-only index (§2.2.1)
    tc = fc.typed_check(pruned, fc.OBO_REL)
    man["typed_check"] = tc
    if tc["passed"]:
        isa = fc.build_typed_index(pruned, fc.OBO_REL, ("is_a",))
        target = fc.refuse_existing(out_phi / fc.ISA_INDEX_NAME)
        isa.save(target)
        files[fc.rel(target)] = fc.file_sha256(target)
        man["isa_index"] = {"path": fc.rel(target), "genes_annotated": len(isa.closure), "terms": len(isa.ancestors)}
    else:
        man["isa_index"] = None
        print("typed-parser check FAILED: T is not computed; C1t reported as not computed", flush=True)

    # ---- mention flags on cohort-A test (§4.5) and the UX -> X change count (§2.2)
    from go_ppi_domain import leak_pruning as lp
    lexicon = lp.lexicon_from_json(json.loads(fc.absp(fc.MASK_LEXICON_REL).read_text(encoding="utf-8")))
    unmasked = load_proteins(fc.UNPRUNED_PROTEINS_REL)
    a_test = fc.read_jsonl(f"{fc.MASKED_SPLIT_DIR}/test.jsonl")
    flags = fc.mention_flags(a_test, unmasked, lexicon)
    by_grade = {str(g): int(sum(1 for r, f in zip(a_test, flags) if f and fc.record_grade(r) == g)) for g in (2, 1, 0)}
    p = fc.write_json(out_phi / "mention_flags_A_test.json",
                      {"file": f"{fc.MASKED_SPLIT_DIR}/test.jsonl",
                       "file_sha256": fc.file_sha256(f"{fc.MASKED_SPLIT_DIR}/test.jsonl"),
                       "rule": "lexicon entry owned by the partner symbol in the unmasked anchor view body, "
                               "or owned by the anchor symbol in the partner view body",
                       "flagged_record_indices": [int(i) for i in flags.nonzero()[0]], "flagged_by_grade": by_grade})
    files[fc.rel(p)] = fc.file_sha256(p)
    pub_test = fc.read_jsonl(f"{fc.PUB_SPLIT_DIR}/test.jsonl")
    X = fc.phi_array(fc.make_matcher(pruned, "exact"), [r["resume"]["skill_uris"] for r in a_test],
                     [r["job"]["skill_uris"] for r in a_test])
    UX = fc.phi_array(fc.make_matcher(unpruned, "exact"), [r["resume"]["skill_uris"] for r in pub_test],
                      [r["job"]["skill_uris"] for r in pub_test])
    man["ux_to_x_changed_records_A_test"] = int((X != UX).sum())
    man["mention_flags"] = by_grade

    # ---- derived configs and their single-factor audit (§3b)
    rlcp = fc.load_script_module("scripts/run_learning_curve_point.py", "run_learning_curve_point_audit")
    configs = {}
    for path, content in derived_configs(out_cfg, fc.rel(out_data), args.n_folds).items():
        write_config(fc.absp(path), content, files)
        audit = rlcp.report_factor("go_ppi", fc.absp(fc.BASE_CONFIG_REL), fc.absp(path))
        if audit["unexpected_fields"] != fc.EXPECTED_UNEXPECTED_FIELDS:
            raise fc.ProtocolError(f"{path}: unexpected_fields {audit['unexpected_fields']} != "
                                   f"{fc.EXPECTED_UNEXPECTED_FIELDS} (§3b); stop and amend before pins")
        configs[path] = {"sha256": files[path], "diff_vs_base": audit["diff"],
                         "expected_unexpected_fields": audit["unexpected_fields"]}
    man["configs"] = {"base": {"path": fc.BASE_CONFIG_REL, "sha256": fc.file_sha256(fc.BASE_CONFIG_REL)},
                      "derived": configs}

    man["files"] = dict(sorted(files.items()))
    man["seconds"] = round(time.time() - t0, 1)
    man["run_env"] = fc.run_env_record()
    fc.write_json(manifest_path, man)
    print(f"wrote {len(files)} files + {fc.rel(manifest_path)} ({man['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
