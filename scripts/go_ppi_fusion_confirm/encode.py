#!/usr/bin/env python3
"""Step 0 fresh encode of every distinct masked text (protocol_go_ppi_fusion.md §2.1). CPU only.

Collects the encoder texts of every scored masked file (``validation.jsonl`` and
``test*.jsonl`` under each ``--inputs`` dir: cohort A masked splits, cohort-B folds, the HuRI
cohort) and encodes the distinct ones with ``SentenceTransformer(<pinned local snapshot>,
device="cpu")``, ``batch_size=64``, ``convert_to_numpy=True``, no normalisation, into epf
chunk files keyed by sha256(text). Label-free. Scoring reads masked texts only from this
directory (a missing text aborts). The manifest digest is pinned by ``make_pins.py``.

Work goes to ``<out>.partial/`` (resumable after an interruption) and is renamed to
``--out`` when complete; ``--out`` must not exist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402


def scored_files(inputs: List[Path]) -> List[str]:
    """Scored masked files under the inputs: ``validation.jsonl`` and ``test*.jsonl``."""
    out = set()
    for item in inputs:
        p = fc.absp(item)
        if p.is_file():
            out.add(fc.rel(p))
            continue
        if not p.is_dir():
            raise fc.ProtocolError(f"--inputs {item} does not exist")
        for f in p.rglob("*.jsonl"):
            if f.name == "validation.jsonl" or f.name.startswith("test"):
                out.add(fc.rel(f))
    return sorted(out)


def main(argv=None) -> int:
    fc.require_hashseed()
    fc.require_hf_offline()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(fc.MASKED_PREP_REL))
    ap.add_argument("--protocol", type=Path, default=Path(fc.PROTOCOL_REL))
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    fc.require_cpu(args.device, "fresh masked encodes")
    out = fc.refuse_existing(args.out)
    partial = out.with_name(out.name + ".partial")
    t0 = time.time()
    files = scored_files(args.inputs)
    if not files:
        raise fc.ProtocolError("no scored masked files found under --inputs")
    texts = fc.collect_masked_texts(files)
    revision = fc.parse_protocol(fc.absp(args.protocol).read_text(encoding="utf-8"))["mpnet_revision"]
    revision, snap = fc.resolve_mpnet_snapshot(revision)
    print(f"{len(files)} files, {len(texts)} distinct texts; MPNet {revision} on CPU", flush=True)
    encoder = fc.cpu_encoder(snap)
    if str(encoder.device) != "cpu":
        raise fc.ProtocolError(f"encoder on {encoder.device}, not cpu")
    counts = fc.encode_into(texts, partial, encoder)
    have = fc.load_chunks(partial)
    missing = [t for t in texts if fc.text_sha256(t) not in have]
    if missing:
        raise fc.ProtocolError(f"{len(missing)} texts missing after encoding")
    fc.check_embedding_dim(have.values())
    manifest = {"schema": "go_ppi_fusion_confirm/masked_encodes/v1", "created_utc": fc.utc_now(),
                "files": {f: fc.file_sha256(f) for f in files}, "n_texts": len(texts), "counts": counts,
                "mpnet_revision": revision, "device": "cpu", "batch_size": 64, "normalize": False,
                "manifest_digest": fc.manifest_digest(partial), "n_chunks": len(fc.chunk_hashes(partial)),
                "seconds": round(time.time() - t0, 1), "run_env": fc.run_env_record()}
    fc.write_json(partial / "encode_manifest.json", manifest)
    os.replace(partial, out)
    print(f"wrote {fc.rel(out)}: {manifest['n_chunks']} chunks, digest {manifest['manifest_digest']} "
          f"({manifest['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
