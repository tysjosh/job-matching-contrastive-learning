"""Build the leak-pruned GO/PPI features and the frozen study splits (design 1.4, gate G1).

The converter is NOT re-run (its grade-0 sampling depends on PYTHONHASHSEED,
design 0.1-1). Instead:

1. refuse protected or non-empty output dirs (no ``--force``);
2. verify the raw sha256s (``withholding_constants.RAW_SHA256``);
3. build the pruned BP index in memory (``GoIndex.build(..., leak_pruning=...)``,
   never ``build_or_load``) and save it as ``go_index_P.pkl``;
4. build the alias table and mask lexicon from the reference texts; write
   ``mask_lexicon.json`` and ``mask_manifest.json``;
5. copy ``pairs.jsonl`` byte-for-byte; rewrite ``proteins.jsonl`` (same ids and
   order, masked ``encoder_view``, pruned ``go_uris``/``coarse_uris``/``n_terms``);
   write ``convert_manifest.json``;
6. assert the pairs sha and the protein-id order;
7. re-run the deterministic splitter (seed 42, val cap 20, test cap 0) and assert
   structural hashes, identical pool rows and the record counts;
8. write ``frozen_hashes.json``;
9. make the frozen files read-only (0o444) and ``append_pin("G1", ...)``.

Any failure: message, exit 2. Partial output is left in place but is never
reused, because step 1 refuses non-empty dirs.

Usage (protocol section 10, G1)::

    "$PY" "$R/scripts/go_withholding/build_pruned_features.py" --bulk-dir "$R/dataset/go_ppi/bulk" \\
      --reference-converted "$R/preprocess/go_ppi" --reference-splits "$R/preprocess/go_ppi_splits" \\
      --out-converted "$R/preprocess/go_withholding/converted" \\
      --out-splits "$R/preprocess/go_withholding/splits"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import leak_pruning as lp  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("build_pruned_features")

#: Pruning applied to the study's BP index (design 1.2 / 1.4 step 3).
LEAK_PRUNING = {"with_from_human_protein": True, "excluded_term_roots": ["GO:0005515"]}
#: Reference splitter parameters (the review reproduced the reference files with them).
SPLIT_PARAMS = {"seed": 42, "val_no_interaction_cap": 20, "test_no_interaction_cap": 0}
#: Expected record / pool-row counts (protocol section 2). Read at call time.
EXPECTED_COUNTS = {"train": 15708, "validation": 5587, "test": 22341,
                   "negative_pools": 7041}
SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "negative_pools.jsonl")
BULK_FILES = {"obo": "go-basic.obo", "gaf": "goa_human.gaf.gz",
              "aliases": "9606.protein.aliases.v12.0.txt.gz",
              "info": "9606.protein.info.v12.0.txt.gz"}


class BuildError(RuntimeError):
    """Any G1 build failure (exit 2)."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise BuildError(msg)


def _refuse_output(path: Path) -> None:
    wc.assert_writable_target(path)
    if path.exists():
        _check(path.is_dir(), f"output {path} exists and is not a directory")
        _check(not any(path.iterdir()), f"output dir {path} is not empty (no --force)")


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _protein_ids(path: Path) -> List[str]:
    return [r["protein_id"] for r in _read_jsonl(path)]


def _pool_rows(path: Path) -> List[tuple]:
    return [(r["split"], r["anchor_id"], tuple(r.get("weak_evidence", [])),
             tuple(r.get("no_interaction", []))) for r in _read_jsonl(path)]


def _line_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def verify_raw(bulk_dir: Path) -> Dict[str, str]:
    """Every ``RAW_SHA256`` file matches, and every bulk file used is one of them."""
    raw = {Path(p): s for p, s in wc.RAW_SHA256.items()}
    for name in BULK_FILES.values():
        p = (bulk_dir / name).resolve()
        _check(p in {q.resolve() for q in raw},
               f"bulk file {p} is not covered by RAW_SHA256")
    out = {}
    for p, expected in sorted(raw.items(), key=lambda kv: str(kv[0])):
        _check(p.is_file(), f"raw file missing: {p}")
        got = wc.sha256_file(p)
        _check(got == expected, f"raw sha256 mismatch for {p}: {got} != {expected}")
        out[wc.rel_to_root(p)] = got
    return out


def build(bulk_dir: Path, reference_converted: Path, reference_splits: Path,
          out_converted: Path, out_splits: Path) -> Dict[str, Any]:
    from go_ppi_domain.data_splitter import build_splits
    from go_ppi_domain.go_ontology import GoIndex

    t0 = time.time()
    # 1. outputs
    _refuse_output(out_converted)
    _refuse_output(out_splits)
    _check(out_converted.resolve() != out_splits.resolve(),
           "--out-converted and --out-splits must differ")

    # 2. raw inputs
    raw = verify_raw(bulk_dir)
    bulk = {k: bulk_dir / v for k, v in BULK_FILES.items()}
    out_converted.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)

    # 3. pruned BP index, in memory, then saved
    acc_map = lp.human_accession_map(bulk["gaf"], bulk["aliases"], bulk["info"])
    pruned = GoIndex.build(bulk["obo"], bulk["gaf"], "P", leak_pruning=dict(LEAK_PRUNING),
                           with_from_symbol_map=acc_map)
    # No machine-specific absolute paths in emitted artifacts (user decision B).
    pruned.meta["obo_path"] = wc.rel_to_root(Path(pruned.meta["obo_path"]))
    pruned.meta["gaf_path"] = wc.rel_to_root(Path(pruned.meta["gaf_path"]))
    index_path = out_converted / "go_index_P.pkl"
    pruned.save(index_path)
    logger.info("pruned index: %d genes, gaf_stats=%s", len(pruned.closure),
                pruned.meta["leak_pruning"]["gaf_stats"])

    # 4. mask lexicon
    ref_proteins = reference_converted / "proteins.jsonl"
    texts = lp.read_protein_texts(ref_proteins)
    aliases = lp.load_symbol_aliases(bulk["aliases"], bulk["info"], set(texts))
    lexicon = lp.build_mask_lexicon(texts, aliases)
    lexicon_path = out_converted / "mask_lexicon.json"
    lexicon_path.write_text(json.dumps(lp.lexicon_to_json(lexicon), ensure_ascii=False,
                                       sort_keys=True) + "\n", encoding="utf-8")
    mask_manifest = {
        "lexicon_size": len(lexicon.owner),
        "lexicon_alnum": len(lexicon.alnum),
        "lexicon_special": len(lexicon.special),
        "symbols_unmaskable": lexicon.params["symbols_unmaskable"],
        "aliases_dropped_docfreq": lexicon.params["aliases_dropped_docfreq"],
        "alias_sources": sorted(lp.ALIAS_SOURCES),
        "mask_token": lp.MASK_TOKEN,
        "params": {k: v for k, v in lexicon.params.items()
                   if k not in ("symbols_unmaskable", "aliases_dropped_docfreq")},
        "lexicon_sha256": wc.sha256_file(lexicon_path),
    }
    mask_manifest_path = out_converted / "mask_manifest.json"
    mask_manifest_path.write_text(json.dumps(mask_manifest, indent=2, ensure_ascii=False,
                                             sort_keys=True) + "\n", encoding="utf-8")
    mask_manifest_sha = wc.sha256_file(mask_manifest_path)

    # 5. pairs copied verbatim; proteins rewritten
    ref_pairs = reference_converted / "pairs.jsonl"
    out_pairs = out_converted / "pairs.jsonl"
    shutil.copyfile(ref_pairs, out_pairs)
    out_proteins = out_converted / "proteins.jsonl"
    rewrite_stats = lp.rewrite_proteins(
        ref_proteins, out_proteins, pruned,
        lambda g, t: lp.mask_view_text(g, t, lexicon))
    ref_manifest = json.loads((reference_converted / "convert_manifest.json")
                              .read_text(encoding="utf-8"))
    pairs_sha = wc.sha256_file(out_pairs)
    convert_manifest = dict(ref_manifest)
    convert_manifest.update({
        "annotation_index": pruned.meta,
        "view_masking": True,
        "mask_manifest_sha256": mask_manifest_sha,
        "pairs_copied_from": wc.rel_to_root(ref_pairs),
        "pairs_sha256": pairs_sha,
        "rewrite_stats": rewrite_stats,
        "go_index_cache": wc.rel_to_root(index_path),
    })
    (out_converted / "convert_manifest.json").write_text(
        json.dumps(convert_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 6. pairs sha and id order
    ref_pairs_sha = wc.sha256_file(ref_pairs)
    _check(pairs_sha == ref_pairs_sha,
           f"pairs sha256 {pairs_sha} != reference {ref_pairs_sha}")
    _check(_protein_ids(out_proteins) == _protein_ids(ref_proteins),
           "protein-id list (in order) differs from the reference")

    # 7. deterministic re-split, structural asserts
    manifest = build_splits(out_converted, out_splits, **SPLIT_PARAMS)
    overlap = manifest["entity_overlap_between_splits"]
    _check(all(v == 0 for v in overlap.values()), f"entity overlap {overlap}")
    structural = {}
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        new_s = wc.structural_sha256(out_splits / name)
        ref_s = wc.structural_sha256(reference_splits / name)
        _check(new_s == ref_s, f"structural sha256 mismatch for {name}: {new_s} != {ref_s}")
        structural[name] = new_s
    _check(_pool_rows(out_splits / "negative_pools.jsonl")
           == _pool_rows(reference_splits / "negative_pools.jsonl"),
           "negative_pools.jsonl rows differ from the reference")
    counts = {name.split(".")[0]: _line_count(out_splits / name) for name in SPLIT_FILES}
    expected = dict(EXPECTED_COUNTS)
    _check(counts == expected, f"record counts {counts} != expected {expected}")

    # 8. frozen hashes
    if reference_splits == Path(wc.REFERENCE_SPLITS).resolve():
        ref_test = wc.sha256_file(reference_splits / "test.jsonl")
        _check(ref_test == wc.REFERENCE_TEST_SHA256,
               f"reference test.jsonl sha256 {ref_test} != {wc.REFERENCE_TEST_SHA256}")
    frozen = {
        "schema_version": 2,
        "test_sha256": wc.sha256_file(out_splits / "test.jsonl"),
        "test_structural_sha256": structural["test.jsonl"],
        "reference_test_sha256": wc.sha256_file(reference_splits / "test.jsonl"),
        "train_sha256": wc.sha256_file(out_splits / "train.jsonl"),
        "validation_sha256": wc.sha256_file(out_splits / "validation.jsonl"),
        "negative_pools_sha256": wc.sha256_file(out_splits / "negative_pools.jsonl"),
        "pairs_sha256": pairs_sha,
        "proteins_sha256": wc.sha256_file(out_proteins),
        "go_index_P_sha256": wc.sha256_file(index_path),
        "mask_manifest_sha256": mask_manifest_sha,
        "raw": raw,
        "code_sha256": wc.code_sha256()[0],
    }
    (out_splits / "frozen_hashes.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # 9. read-only + pin
    for p in (index_path, out_proteins, out_pairs, *(out_splits / n for n in SPLIT_FILES)):
        os.chmod(p, 0o444)
    wc.append_pin("G1", frozen)

    gaf_stats = pruned.meta["leak_pruning"]["gaf_stats"]
    return {"frozen_hashes": frozen, "counts": counts,
            "gaf_kept": gaf_stats.get("kept", 0),
            "gaf_drop_with_from_human_protein": gaf_stats.get("drop_with_from_human_protein", 0),
            "gaf_drop_excluded_root_descendant": gaf_stats.get("drop_excluded_root_descendant", 0),
            "empty_go_uris": rewrite_stats["empty_go_uris"],
            "token_mask_rate": rewrite_stats["token_mask_rate"],
            "lexicon_size": mask_manifest["lexicon_size"],
            "aliases_dropped_docfreq": len(mask_manifest["aliases_dropped_docfreq"]),
            "wall_seconds": round(time.time() - t0, 1)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bulk-dir", type=Path, default=ROOT / "dataset/go_ppi/bulk")
    ap.add_argument("--reference-converted", type=Path, default=None)
    ap.add_argument("--reference-splits", type=Path, default=None)
    ap.add_argument("--out-converted", type=Path, default=None)
    ap.add_argument("--out-splits", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = build(
            Path(args.bulk_dir).resolve(),
            Path(args.reference_converted or wc.REFERENCE_CONVERTED).resolve(),
            Path(args.reference_splits or wc.REFERENCE_SPLITS).resolve(),
            Path(args.out_converted or wc.CONVERTED).resolve(),
            Path(args.out_splits or wc.SPLITS).resolve())
    except Exception as exc:  # noqa: BLE001 - every failure is fatal, exit 2
        logger.error("build_pruned_features failed: %s: %s", type(exc).__name__, exc)
        return 2
    print(json.dumps({k: v for k, v in summary.items() if k != "frozen_hashes"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
