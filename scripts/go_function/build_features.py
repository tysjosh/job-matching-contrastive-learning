"""Build the InterPro feature matrices of one aspect (design §3.1, gate G5 local part).

Writes preprocess/go_function/<aspect>/features_interpro.npz (CSR uint8 per split
over the 26,406-entry interpros.pkl vocabulary, rows in pickle order, plus
proteins_<split> and vocab) and features_interpro.json (counts, hashes). Both are
write-or-verify: a re-run must be byte-identical and reports "unchanged". The
protein order of every split must equal the label manifest's (labels_TP.json).
Records pin G5 entry "<aspect>/features_interpro.json" in
results/go_function/protocol_pins.json.

Exit codes: 0 ok; 2 fatal (input hash mismatch, protein order differs from the
labels, a differing existing output or pin).

Usage:
  .venv/bin/python scripts/go_function/build_features.py --aspect bp
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

from go_function import features as F  # noqa: E402
from go_function.constants import EXIT_FATAL, EXIT_OK  # noqa: E402
from go_function.labels import SPLITS  # noqa: E402
from go_function.pins import (canonical_json, csr_arrays, csr_sha256, npz_bytes,  # noqa: E402
                              record_pin, sha256_bytes, verify_inputs, write_or_verify_file)
from go_function.runtime import FatalError, repo_path, repo_rel, setup_logging  # noqa: E402

DGZ = "dataset/go_function/dgz"
OUT = ROOT / "preprocess" / "go_function"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    args = ap.parse_args(argv)
    out_dir = OUT / args.aspect
    log = setup_logging("build_features", out_dir)
    t0 = time.time()
    try:
        import numpy as np
        import pandas as pd
        inputs = [f"{DGZ}/{args.aspect}/interpros.pkl"] + \
                 [f"{DGZ}/{args.aspect}/{s}_data.pkl" for s in SPLITS]
        hashes = verify_inputs(inputs)
        vocab = F.read_interpro_vocab(repo_path(f"{DGZ}/{args.aspect}/interpros.pkl"))
        if len(vocab) != F.INTERPRO_VOCAB_SIZE:
            raise FatalError(f"interpros.pkl has {len(vocab)} ids, expected "
                             f"{F.INTERPRO_VOCAB_SIZE}")
        index = {g: i for i, g in enumerate(vocab)}
        label_man_path = out_dir / "labels_TP.json"
        if not label_man_path.exists():
            raise FatalError(f"{repo_rel(label_man_path)} is missing; run build_labels.py first")
        label_man = json.loads(label_man_path.read_text())

        arrays = {"vocab": np.asarray(vocab, dtype=str)}
        splits = {}
        for split in SPLITS:
            df = pd.read_pickle(repo_path(f"{DGZ}/{args.aspect}/{split}_data.pkl"))
            proteins = [str(x) for x in df["proteins"].tolist()]
            rows = [list(x) for x in df["interpros"].tolist()]
            del df
            gc.collect()
            psha = sha256_bytes("\n".join(proteins).encode())
            if psha != label_man["splits"][split]["proteins_sha256"]:
                raise FatalError(f"{split}: protein order differs from {repo_rel(label_man_path)}")
            m, unknown = F.interpro_csr(rows, index)
            n_zero = int(np.sum(np.diff(m.indptr) == 0))
            arrays.update(csr_arrays(split, m))
            arrays[f"proteins_{split}"] = np.asarray(proteins, dtype=str)
            splits[split] = {"n_rows": int(m.shape[0]), "nnz": int(m.nnz),
                             "n_zero_rows": n_zero,
                             "zero_row_share": round(n_zero / m.shape[0], 6),
                             "unknown_dropped": int(unknown),
                             "features_sha256": csr_sha256(m), "proteins_sha256": psha}
            log.info("%s: %d rows, %d nnz, %d zero rows (%.2f%%), %d unknown ids dropped",
                     split, m.shape[0], m.nnz, n_zero, 100 * n_zero / m.shape[0], unknown)

        npz_path = out_dir / "features_interpro.npz"
        js_path = out_dir / "features_interpro.json"
        blob = npz_bytes(arrays)
        man = {"schema": 1, "aspect": args.aspect, "feature": "interpro",
               "npz": repo_rel(npz_path), "npz_sha256": sha256_bytes(blob),
               "inputs": dict(sorted(hashes.items())), "n_vocab": len(vocab),
               "vocab_sha256": sha256_bytes("\n".join(vocab).encode()), "splits": splits}
        jb = canonical_json(man)
        statuses = {repo_rel(npz_path): write_or_verify_file(npz_path, blob),
                    repo_rel(js_path): write_or_verify_file(js_path, jb)}
        for p, s in statuses.items():
            log.info("%s: %s", p, s)
        log.info("pin G5: %s", record_pin("G5", {f"{args.aspect}/features_interpro.json":
                                                 sha256_bytes(jb)}))
        print(json.dumps({"aspect": args.aspect, "outputs": statuses,
                          "zero_row_share": {s: splits[s]["zero_row_share"] for s in SPLITS},
                          "unknown_dropped": {s: splits[s]["unknown_dropped"] for s in SPLITS},
                          "seconds": round(time.time() - t0, 1)}, sort_keys=True))
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
