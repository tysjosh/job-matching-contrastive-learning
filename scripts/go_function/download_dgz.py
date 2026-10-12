"""Download and pin the DeepGOZero data release files (gap item 1).

Source: https://bio2vec.net/data/deepgo/deepgozero/data/ (DGZ public release).
Writes dataset/go_function/dgz/<relpath> and dataset/go_function/MANIFEST.json
with url, size and sha256 per file. Never overwrites an existing file; an
existing file is hashed and recorded instead.
"""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://bio2vec.net/data/deepgo/deepgozero/data/"
ROOT = Path(__file__).resolve().parents[2] / "dataset" / "go_function"
OUT = ROOT / "dgz"

TOP = ["go.obo", "go.owl", "go.norm", "definitions_go.txt", "swissprot_exp.fa"]
PER_ONT = ["train_data.pkl", "valid_data.pkl", "test_data.pkl", "interpros.pkl",
           "terms.pkl", "terms_zero.pkl", "terms_zero_10.pkl", "test_diamond.res"]
FILES = TOP + [f"{o}/{f}" for o in ("bp", "mf", "cc") for f in PER_ONT]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = []
    for rel in FILES:
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
                sys.exit(f"size mismatch for {url}")
            tmp.rename(dest)
        entries.append({"path": str(dest.relative_to(ROOT)), "url": url,
                        "size": dest.stat().st_size, "sha256": sha256(dest)})
        print(entries[-1]["size"], entries[-1]["sha256"], rel, flush=True)
    manifest = {"source": BASE, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "files": entries}
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
