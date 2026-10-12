"""Download and hash the DeepGOZero gate files (G2, G2b, G3; design §16, plan item 7).

Source: https://bio2vec.net/data/deepgo/deepgozero/data/<aspect>/ (DGZ public
release). Per aspect bp/mf:
  checkpoints: mlp.th, deepgozero.th, deepgozero_zero.th
  predictions: predictions_{mlp,deepgozero,deepgozero_zero}.pkl
  zero:        zero_gos.pkl, zero_term_auc{,_all,_train}.pkl and
               zero_<GO>_auc{,_all,_train}.pkl for the withheld classes of DGZ
               evaluate_terms.py (eval_terms: 7 BP, 5 MF classes).
Files go to dataset/go_function/dgz/<aspect>/. Same rules as download_dgz.py: an
existing file is never overwritten (it is hashed and recorded instead); a
download whose size differs from Content-Length is fatal. Before downloading,
the Content-Length of every missing file is summed; if free disk after the
download would drop below 8 GB (8e9 bytes), nothing is downloaded (exit 4).
Writes dataset/go_function/MANIFEST_gate.json (entries of earlier --only runs are
kept); then run pin_inputs.py --gate.

Exit codes: 0 ok; 2 fatal (network error, size mismatch, a changed file); 4 refused
(not enough free disk).

Usage:
  .venv/bin/python scripts/go_function/download_dgz_gate.py --only all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://bio2vec.net/data/deepgo/deepgozero/data/"
DATA = ROOT / "dataset" / "go_function"
OUT = DATA / "dgz"
MANIFEST = DATA / "MANIFEST_gate.json"
MIN_FREE_BYTES = 8 * 10**9
EXIT_OK, EXIT_FATAL, EXIT_REFUSED = 0, 2, 4

#: DGZ evaluate_terms.py ``eval_terms`` (the classes withheld for zero-shot runs).
ZERO_CLASSES = {
    "bp": ["GO:0000381", "GO:0032729", "GO:0032755", "GO:0032760", "GO:0046330",
           "GO:0051897", "GO:0120162"],
    "mf": ["GO:0001227", "GO:0001228", "GO:0003735", "GO:0004867", "GO:0005096"],
}
GROUPS = {
    "checkpoints": lambda a: ["mlp.th", "deepgozero.th", "deepgozero_zero.th"],
    "predictions": lambda a: [f"predictions_{m}.pkl" for m in ("mlp", "deepgozero",
                                                               "deepgozero_zero")],
    "zero": lambda a: (["zero_gos.pkl"] + [f"zero_term_auc{s}.pkl" for s in ("", "_all", "_train")]
                       + [f"zero_{g}_auc{s}.pkl" for g in ZERO_CLASSES[a]
                          for s in ("", "_all", "_train")]),
}


def files_for(only: str):
    groups = list(GROUPS) if only == "all" else [only]
    return [f"{a}/{f}" for a in ("bp", "mf") for g in groups for f in GROUPS[g](a)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        n = r.headers.get("Content-Length")
    if n is None:
        raise RuntimeError(f"no Content-Length for {url}")
    return int(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=("checkpoints", "predictions", "zero", "all"), default="all")
    args = ap.parse_args(argv)
    rels = files_for(args.only)
    try:
        missing = [r for r in rels if not (OUT / r).exists()]
        need = sum(content_length(BASE + r) for r in missing)
        free = shutil.disk_usage(DATA).free
        print(f"{len(missing)} of {len(rels)} files to download, {need / 1e9:.2f} GB; "
              f"free {free / 1e9:.2f} GB", flush=True)
        if free - need < MIN_FREE_BYTES:
            print(f"refused: free disk after download would be {(free - need) / 1e9:.2f} GB "
                  f"< {MIN_FREE_BYTES / 1e9:.0f} GB", file=sys.stderr)
            return EXIT_REFUSED

        old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
        entries = {e["path"]: e for e in old["files"]}
        for rel in rels:
            url = BASE + rel
            dest = OUT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                tmp = dest.with_suffix(dest.suffix + ".part")
                with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
                    expected = int(r.headers.get("Content-Length", -1))
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                if expected >= 0 and tmp.stat().st_size != expected:
                    print(f"size mismatch for {url}", file=sys.stderr)
                    return EXIT_FATAL
                tmp.rename(dest)
            key = str(dest.relative_to(DATA))
            entry = {"path": key, "url": url, "size": dest.stat().st_size, "sha256": sha256(dest)}
            if key in entries and (entries[key]["sha256"], entries[key]["size"]) != \
                    (entry["sha256"], entry["size"]):
                print(f"{key} changed since it was recorded in {MANIFEST.name}", file=sys.stderr)
                return EXIT_FATAL
            entries[key] = entry
            print(entry["size"], entry["sha256"], rel, flush=True)
        manifest = {"source": BASE, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "files": [entries[k] for k in sorted(entries)]}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"free disk now {shutil.disk_usage(DATA).free / 1e9:.2f} GB", flush=True)
    except (OSError, RuntimeError) as e:
        print(f"fatal: {e}", file=sys.stderr)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
