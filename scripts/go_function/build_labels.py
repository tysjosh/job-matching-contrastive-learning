"""Build the label matrices of one aspect and run every G1 check (design §2, gate G1).

Writes preprocess/go_function/<aspect>/labels_<regime>.npz and labels_<regime>.json
(write-or-verify: a re-run must be byte-identical and reports "unchanged") and
records the pins G1-labels (manifest hashes) and G1-norm (go.norm parse counts,
relation index and ladder counts) in results/go_function/protocol_pins.json.
With --verify-pins the pins must already exist and match instead.

Exit codes: 0 ok; 2 fatal (input hash mismatch, any design §2.4 check, go.norm
parse check, a differing existing output or pin).

Usage:
  .venv/bin/python scripts/go_function/build_labels.py --aspect mf --regime all
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function import labels as L  # noqa: E402
from go_function.constants import EXIT_FATAL, EXIT_OK  # noqa: E402
from go_function.normforms import assert_ladder, load_normal_forms  # noqa: E402
from go_function.ontology import DgzRule, parse_obo  # noqa: E402
from go_function.pins import (canonical_json, npz_bytes, record_pin, sha256_bytes,  # noqa: E402
                              verify_inputs, verify_pin, write_or_verify_file)
from go_function.runtime import FatalError, repo_path, repo_rel, setup_logging  # noqa: E402

DGZ = "dataset/go_function/dgz"
OUT = ROOT / "preprocess" / "go_function"


def regimes_for(aspect: str, regime: str):
    if regime == "all":
        return ("TP", "DGZ", "TPnoIPI") if aspect == "mf" else ("TP", "DGZ")
    if regime == "TPnoIPI" and aspect != "mf":
        raise FatalError("TPnoIPI applies to MF only")
    return (regime,)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--regime", choices=("TP", "DGZ", "TPnoIPI", "all"), default="all")
    ap.add_argument("--verify-pins", action="store_true",
                    help="verify G1-labels/G1-norm against existing pins instead of recording")
    args = ap.parse_args(argv)
    out_dir = OUT / args.aspect
    log = setup_logging("build_labels", out_dir)
    t0 = time.time()
    try:
        regimes = regimes_for(args.aspect, args.regime)
        inputs = [f"{DGZ}/go.obo", f"{DGZ}/go.norm", f"{DGZ}/{args.aspect}/terms.pkl"] + \
                 [f"{DGZ}/{args.aspect}/{s}_data.pkl" for s in L.SPLITS]
        hashes = verify_inputs(inputs)
        log.info("verified %d input hashes", len(hashes))

        ont = parse_obo(repo_path(f"{DGZ}/go.obo"))
        dgz = DgzRule(repo_path(f"{DGZ}/go.obo"))
        vocab_ids = L.read_terms(repo_path(f"{DGZ}/{args.aspect}/terms.pkl"))
        ctx = L.Context.build(args.aspect, ont, dgz, vocab_ids)
        log.info("ontology %s; vocab %d columns, %d alt, %d duplicate", ont.report,
                 len(ctx.vocab), len(ctx.vocab.alt_cols), len(ctx.vocab.duplicate_cols))

        nf = load_normal_forms(repo_path(f"{DGZ}/go.norm"), ctx.vocab.index)
        ladder = assert_ladder(nf)
        c = nf.counts()
        norm_pin = {
            "go.norm": {"sha256": hashes[f"{DGZ}/go.norm"], "nf1": c["nf1"], "nf2": c["nf2"],
                        "nf3": c["nf3"], "nf4": c["nf4"], "unparsed": c["unparsed"],
                        "nothing_lines": c["nothing_lines"], "rel_index": nf.rel_index},
            f"{args.aspect}_ladder": ladder,
        }
        log.info("go.norm: %s", c)

        results = {}
        for split in L.SPLITS:
            cols = L.load_split_columns(repo_path(f"{DGZ}/{args.aspect}/{split}_data.pkl"))
            results[split] = L.build_split(ctx, split, cols["proteins"], cols["exp_annotations"],
                                           cols["annotations"], cols["prop_annotations"],
                                           regimes)
            del cols
            gc.collect()
            log.info("%s: %d proteins, checks %s", split, len(results[split].proteins),
                     results[split].check_counts)
        L.raise_on_failures(results)
        log.info("every design §2.4 check passed on every protein of every split")

        statuses = {}
        label_pin = {}
        summary = {}
        for r in regimes:
            npz_path = out_dir / f"labels_{r}.npz"
            js_path = out_dir / f"labels_{r}.json"
            blob = npz_bytes(L.regime_arrays(ctx.vocab, results, r))
            man = L.regime_manifest(ctx, results, r, repo_rel(npz_path), sha256_bytes(blob), hashes)
            jb = canonical_json(man)
            statuses[repo_rel(npz_path)] = write_or_verify_file(npz_path, blob)
            statuses[repo_rel(js_path)] = write_or_verify_file(js_path, jb)
            label_pin[f"{args.aspect}/labels_{r}.json"] = sha256_bytes(jb)
            summary[r] = {s: {k: man["splits"][s][k] for k in
                              ("n_labels", "n_zero_positive_columns")} for s in L.SPLITS}
        for p, s in statuses.items():
            log.info("%s: %s", p, s)
        if args.verify_pins:
            verify_pin("G1-labels", label_pin)
            verify_pin("G1-norm", norm_pin)
            log.info("pins G1-labels and G1-norm verified")
        else:
            log.info("pin G1-labels: %s; pin G1-norm: %s", record_pin("G1-labels", label_pin),
                     record_pin("G1-norm", norm_pin))
        print(json.dumps({"aspect": args.aspect, "outputs": statuses,
                          "check_counts": {s: results[s].check_counts for s in L.SPLITS},
                          "labels": summary, "seconds": round(time.time() - t0, 1)},
                         sort_keys=True))
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
