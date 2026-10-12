"""Generate the closed-world withholding conditions of one aspect (design §5 v4; gate G6).

Blocks (protocol §6.2; one directory per condition under
preprocess/go_function/<aspect>/conditions/<regime>/<rule>/<pack>_k<k>[_d2]/):

  base         the 'all' condition (no withholding) per regime: <regime>/all/
  TP_W2        TP/W2,   8 packs x k in {0,1,5,10,25}
  DGZ_W2       DGZ/W2,  8 packs x k in {0,1,5,10,25}
  TP_W1        TP/W1,   8 packs x k in {0,1,5}
  DGZ_WDGZrel  DGZ/WDGZrel, 8 packs x k0
  DGZ_W2drop   DGZ/W2drop,  8 packs x k0
  TP_W2_d2     TP/W2 retention draw 2, 8 packs x k in {1,5}
  TPnoIPI_W2   MF only: TPnoIPI/W2, 8 packs x k in {0,1,5} (+ TPnoIPI 'all' in base)
  validation   target-free validation column masks per regime: <regime>/validation/

Every condition passes design §5.5 assertions 1-10 before it is written; files are
write-or-verify (a re-run is byte-identical and reports "unchanged"; a differing
existing directory is exit 2). With --block all, conditions/index.json (every
manifest hash) is written and pinned under G6 together with the validation
manifests. Logs carry counts and condition names (regime/rule/pack/k) only.

Exit codes: 0 ok; 2 fatal (pin/hash mismatch, any assertion, differing output);
3 protocol stop.

Usage:
  .venv/bin/python scripts/go_function/make_conditions.py --aspect bp --block all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from go_function import labels as L  # noqa: E402
from go_function import withholding as WH  # noqa: E402
from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_STOP  # noqa: E402
from go_function.ontology import DgzRule, ancestors, parse_obo  # noqa: E402
from go_function.pins import (canonical_json, csr_sha256, record_pin, sha256_bytes,  # noqa: E402
                              sha256_file, verify_inputs, verify_pin, write_or_verify,
                              write_or_verify_file)
from go_function.runtime import (FatalError, ProtocolStop, repo_path, repo_rel,  # noqa: E402
                                 setup_logging)

DGZ = "dataset/go_function/dgz"
PRE = "preprocess/go_function"
BLOCKS = ("base", "TP_W2", "DGZ_W2", "TP_W1", "DGZ_WDGZrel", "DGZ_W2drop", "TP_W2_d2",
          "TPnoIPI_W2", "validation")


def regimes_for(aspect: str):
    return ("TP", "DGZ", "TPnoIPI") if aspect == "mf" else ("TP", "DGZ")


def block_specs(aspect: str, block: str, packs):
    S = WH.Spec
    if block == "base":
        return [S(aspect, r, "all") for r in regimes_for(aspect)]
    if block == "TP_W2":
        return [S(aspect, "TP", "W2", p, k) for p in packs for k in WH.NUMERIC_KS]
    if block == "DGZ_W2":
        return [S(aspect, "DGZ", "W2", p, k) for p in packs for k in WH.NUMERIC_KS]
    if block == "TP_W1":
        return [S(aspect, "TP", "W1", p, k) for p in packs for k in (0, 1, 5)]
    if block == "DGZ_WDGZrel":
        return [S(aspect, "DGZ", "WDGZrel", p, 0) for p in packs]
    if block == "DGZ_W2drop":
        return [S(aspect, "DGZ", "W2drop", p, 0) for p in packs]
    if block == "TP_W2_d2":
        return [S(aspect, "TP", "W2", p, k, draw=2) for p in packs for k in (1, 5)]
    if block == "TPnoIPI_W2":
        if aspect != "mf":
            return []
        return [S(aspect, "TPnoIPI", "W2", p, k) for p in packs for k in (0, 1, 5)]
    if block == "validation":
        return []
    raise FatalError(f"unknown block {block!r}")


def load_regime(aspect: str, regime: str, ont):
    man_rel = f"{PRE}/{aspect}/labels_{regime}.json"
    man_bytes = repo_path(man_rel).read_bytes()
    man_sha = sha256_bytes(man_bytes)
    verify_pin("G1-labels", {f"{aspect}/labels_{regime}.json": man_sha})
    man = json.loads(man_bytes)
    if sha256_file(repo_path(man["npz"])) != man["npz_sha256"]:
        raise FatalError(f"{man['npz']} differs from its manifest hash")
    lab = L.load_labels(repo_path(man["npz"]))
    for split in L.SPLITS:
        if csr_sha256(lab[split]) != man["splits"][split]["labels_sha256"]:
            raise FatalError(f"labels_{regime} {split} hash differs from its manifest")
    vocab = L.build_vocab(lab["vocab"], ont)
    rd = WH.RegimeData(regime=regime, vocab=vocab, train=lab["train"],
                       direct=lab["direct_train"], extra=lab["direct_extra_train"],
                       proteins=lab["proteins_train"],
                       valid_sha256=man["splits"]["valid"]["labels_sha256"],
                       test_sha256=man["splits"]["test"]["labels_sha256"],
                       base_manifest_sha256=man_sha)
    return rd, man


def zero_disagreement(aspect: str, table, rd_dgz, log):
    """Per target column: train proteins where prop_annotations (= base DGZ labels)
    and zero_annotations disagree (descriptive, design §5.2)."""
    import pandas as pd
    rel = f"{DGZ}/{aspect}/train_data.pkl"
    verify_inputs([rel])
    zero = pd.read_pickle(repo_path(rel))["zero_annotations"].tolist()
    if len(zero) != rd_dgz.train.shape[0]:
        raise FatalError("zero_annotations row count differs from the DGZ train labels")
    zsets = [set(z) for z in zero]
    out = {}
    for t in sorted(table["targets"]):
        c = table["targets"][t]["column"]
        prop = set(WH.column_positive_rows(rd_dgz.train, c).tolist())
        zr = {i for i, z in enumerate(zsets) if t in z}
        out[t] = len(prop ^ zr)
    del zero
    log.info("zero_annotations vs prop_annotations: %d of %d targets disagree on >= 1 protein",
             sum(1 for v in out.values() if v), len(out))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--block", choices=BLOCKS + ("all",), default="all")
    args = ap.parse_args(argv)
    aspect = args.aspect
    cond_root = ROOT / PRE / aspect / "conditions"
    log = setup_logging("make_conditions", cond_root)
    t0 = time.time()
    try:
        tt_rel = f"{PRE}/{aspect}/targets/target_table.json"
        tt_bytes = repo_path(tt_rel).read_bytes()
        tt_sha = sha256_bytes(tt_bytes)
        verify_pin("G6", {f"{aspect}/targets/target_table.json": tt_sha})
        table = json.loads(tt_bytes)
        packs = sorted(table["packs"])
        hashes = verify_inputs([f"{DGZ}/go.obo"])
        ont = parse_obo(repo_path(f"{DGZ}/go.obo"))
        dgz = DgzRule(repo_path(f"{DGZ}/go.obo"))
        anc = ancestors(ont)
        rds, mans, props = {}, {}, {}
        for r in regimes_for(aspect):
            rds[r], mans[r] = load_regime(aspect, r, ont)
            props[r] = WH.Propagator(r, rds[r].vocab, aspect, anc,
                                     dgz if r == "DGZ" else None)
        if rds["TP"].vocab.ids != rds["DGZ"].vocab.ids:
            raise FatalError("TP and DGZ vocabularies differ")
        all_t = WH.all_target_ids(table)
        cols = {t: table["targets"][t]["column"] for t in all_t}
        P_tp = {t: WH.column_positive_rows(rds["TP"].train, cols[t]) for t in all_t}
        P_dgz = {t: WH.column_positive_rows(rds["DGZ"].train, cols[t]) for t in all_t}
        log.info("inputs verified: %d hashes, target table %s, %d targets, %d packs",
                 len(hashes) + len(rds), tt_sha[:16], len(all_t), len(packs))

        # target-free validation masks (always computed; their hash enters every manifest)
        val_sha, val_status, val_pin = {}, {}, {}
        for r in regimes_for(aspect):
            union = sorted({int(c) for t in all_t for c in props[r].dstar_cols(t)})
            files = WH.validation_files(r, aspect, union, tt_sha, rds[r].valid_sha256)
            val_sha[r] = sha256_bytes(files["validation_mask.npz"])
            vdir = cond_root / r / "validation"
            if args.block in ("all", "validation"):
                val_status[r] = write_or_verify(vdir, files)
                val_pin[f"{aspect}/conditions/{r}/validation/validation_manifest.json"] = \
                    sha256_bytes(files["validation_manifest.json"])
                log.info("validation %s: %d columns masked, %s", r, len(union), val_status[r])

        blocks = [b for b in BLOCKS if b != "validation"] if args.block == "all" \
            else [args.block]
        zero_dis = None
        w2_sha = {}
        index, summary = {}, {}
        for block in blocks:
            specs = block_specs(aspect, block, packs)
            st = Counter()
            asserted = Counter()
            for spec in specs:
                infos = WH.pack_target_infos(table, spec.pack) if spec.pack else []
                cond = WH.make_condition(spec, rds[spec.regime], props[spec.regime], infos,
                                         P_tp=P_tp, P_dgz=P_dgz)
                extra = {}
                w2 = None
                if spec.rule == "W2drop":
                    if spec.pack not in w2_sha:
                        ref = WH.make_condition(WH.Spec(aspect, "DGZ", "W2", spec.pack, 0),
                                                rds["DGZ"], props["DGZ"], infos,
                                                P_tp=P_tp, P_dgz=P_dgz)
                        w2_sha[spec.pack] = csr_sha256(ref.labels)
                    w2 = w2_sha[spec.pack]
                    extra["w2_reference"] = f"DGZ/W2/{spec.pack}_k0"
                if spec.rule == "WDGZrel":
                    if zero_dis is None:
                        zero_dis = zero_disagreement(aspect, table, rds["DGZ"], log)
                    extra["zero_vs_prop_disagreement"] = {t: zero_dis[t] for t in cond.check.targets}
                if spec.regime == "TPnoIPI" and spec.rule != "all":
                    pk = [i for i in infos]
                    extra["noipi_targets_dropped"] = sum(1 for i in pk if not i.noipi_kept)
                    extra["noipi_targets_kept"] = sum(1 for i in pk if i.noipi_kept)
                if spec.rule == "W2" and spec.regime == "DGZ" and spec.k == 0:
                    w2_sha[spec.pack] = csr_sha256(cond.labels)
                base_sha = csr_sha256(rds[spec.regime].train)
                done = WH.run_assertions(cond.check, rds[spec.regime].vocab,
                                         base_labels_sha=base_sha, w2_labels_sha=w2)
                files = WH.condition_files(cond, target_table_sha=tt_sha,
                                           base=rds[spec.regime],
                                           validation_sha=val_sha[spec.regime],
                                           assertions=done + [4], extra=extra)
                # assertion 4 on the manifest as written (base split hashes)
                WH.assertion_4(json.loads(files["withholding_manifest.json"]), mans[spec.regime])
                d = cond_root / spec.name
                status = write_or_verify(d, files)
                st[status] += 1
                asserted.update(sorted(set(done + [4])))
                index[spec.name] = sha256_bytes(files["withholding_manifest.json"])
                log.info("%s: %s (assertions %s; W rows %s; mask %d)", spec.name, status,
                         sorted(set(done + [4])), cond.stats.get("n_withheld_rows", 0),
                         cond.stats["mask_count"])
            summary[block] = {"n_conditions": len(specs), "status": dict(st),
                              "assertion_counts": dict(sorted(asserted.items()))}
        out = {"aspect": aspect, "block": args.block, "blocks": summary,
               "validation": val_status}
        if args.block == "all":
            idx = {"schema": 1, "aspect": aspect, "target_table_sha256": tt_sha,
                   "conditions": dict(sorted(index.items())),
                   "validation": {r: val_sha[r] for r in sorted(val_sha)}}
            ib = canonical_json(idx)
            ipath = cond_root / "index.json"
            out["index"] = write_or_verify_file(ipath, ib)
            entries = dict(val_pin)
            entries[f"{aspect}/conditions/index.json"] = sha256_bytes(ib)
            out["pin"] = record_pin("G6", entries)
            out["n_conditions"] = len(index)
            log.info("%s: %s; pin G6: %s", repo_rel(ipath), out["index"], out["pin"])
        out["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(out, sort_keys=True))
    except ProtocolStop as e:
        log.error("protocol stop: %s", e)
        return EXIT_STOP
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
