"""Build the target table of one aspect (design §4, protocol §4; gate G6, gap 7).

Reads the pinned go.obo and the G1 label files (TP, DGZ, and TPnoIPI for MF),
selects 48 targets in 8 packs and writes
preprocess/go_function/<aspect>/targets/target_table.json (canonical JSON,
schema 1). A second run must be byte-identical ("unchanged"), else exit 2.
Records pin G6 entry "<aspect>/targets/target_table.json". Logs counts only:
target ids and names are never printed.

Exit codes: 0 ok; 2 fatal (input hash or pin mismatch, differing existing table);
3 protocol stop (slot shortfall after borrowing, or no feasible packing).

Usage:
  .venv/bin/python scripts/go_function/build_targets.py --aspect bp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function import labels as L  # noqa: E402
from go_function import targets as T  # noqa: E402
from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_STOP  # noqa: E402
from go_function.ontology import DgzRule, parse_obo  # noqa: E402
from go_function.pins import (canonical_json, record_pin, sha256_bytes, sha256_file,  # noqa: E402
                              verify_inputs, verify_pin, write_or_verify_file)
from go_function.runtime import (FatalError, ProtocolStop, repo_path, repo_rel,  # noqa: E402
                                 setup_logging)

DGZ = "dataset/go_function/dgz"
PRE = "preprocess/go_function"


def load_regime(aspect: str, regime: str):
    """Labels of one regime, checked against the G1-labels pin and the npz hash."""
    man_rel = f"{PRE}/{aspect}/labels_{regime}.json"
    man_bytes = repo_path(man_rel).read_bytes()
    verify_pin("G1-labels", {f"{aspect}/labels_{regime}.json": sha256_bytes(man_bytes)})
    man = json.loads(man_bytes)
    npz = repo_path(man["npz"])
    if sha256_file(npz) != man["npz_sha256"]:
        raise FatalError(f"{man['npz']} differs from its manifest hash")
    return L.load_labels(npz), man_rel, sha256_bytes(man_bytes)


def load_inputs(aspect: str) -> T.TargetInputs:
    hashes = verify_inputs([f"{DGZ}/go.obo"])
    ont = parse_obo(repo_path(f"{DGZ}/go.obo"))
    dgz = DgzRule(repo_path(f"{DGZ}/go.obo"))
    tp, tp_rel, tp_sha = load_regime(aspect, "TP")
    dg, dg_rel, dg_sha = load_regime(aspect, "DGZ")
    inputs = dict(hashes)
    inputs[tp_rel] = tp_sha
    inputs[dg_rel] = dg_sha
    if tp["vocab"] != dg["vocab"] or tp["proteins_train"] != dg["proteins_train"]:
        raise FatalError("TP and DGZ label files disagree on vocabulary or train proteins")
    vocab = L.build_vocab(tp["vocab"], ont)
    raw = set()
    for split in L.SPLITS:
        d = dg[f"direct_{split}"]
        raw.update(vocab.ids[c] for c in d.indices)
        for extra in dg[f"direct_extra_{split}"]:
            raw.update(extra)
    noipi = {}
    if aspect == "mf":
        ni, ni_rel, ni_sha = load_regime(aspect, "TPnoIPI")
        inputs[ni_rel] = ni_sha
        if ni["vocab"] != tp["vocab"] or ni["proteins_train"] != tp["proteins_train"]:
            raise FatalError("TPnoIPI and TP label files disagree on vocabulary or proteins")
        noipi = {"noipi_train": ni["train"], "noipi_test": ni["test"],
                 "noipi_eval_test": ni["eval_test"]}
    return T.TargetInputs(aspect=aspect, ont=ont, dgz=dgz, vocab=vocab, tp_train=tp["train"],
                          tp_test=tp["test"], eval_test=tp["eval_test"],
                          tp_direct_train=tp["direct_train"],
                          tp_extra_train=tp["direct_extra_train"], dgz_train=dg["train"],
                          dgz_raw_ids=frozenset(raw), proteins_train=tp["proteins_train"],
                          inputs=inputs, **noipi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    args = ap.parse_args(argv)
    out_dir = ROOT / PRE / args.aspect / "targets"
    log = setup_logging("build_targets", out_dir)
    t0 = time.time()
    try:
        inp = load_inputs(args.aspect)
        table = T.build_table(inp)
        data = canonical_json(table)
        path = out_dir / "target_table.json"
        status = write_or_verify_file(path, data)
        sha = sha256_bytes(data)
        pin = record_pin("G6", {f"{args.aspect}/targets/target_table.json": sha})
        summary = T.summary_counts(table)
        log.info("%s: %s (%s); pin G6: %s", repo_rel(path), status, sha[:16], pin)
        print(json.dumps({"aspect": args.aspect, "status": status, "sha256": sha,
                          "pin": pin, "counts": summary,
                          "seconds": round(time.time() - t0, 1)}, sort_keys=True))
    except ProtocolStop as e:
        log.error("protocol stop: %s", e)
        return EXIT_STOP
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
