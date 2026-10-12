"""MPNet term-text embeddings for the text term towers V5/V6 (design §3.3, gate G5;
REMOTE in this plan).

Classes: the BP and MF vocabularies (terms.pkl) united with every class of go.norm
(both aspects' class indices). Text per class: "<name>. <definition>" from the
pinned 2021 go.obo (alt ids take their primary's text). The go.norm classes that
are obsolete in that go.obo (728) take their own stanza's text, with one leading
"obsolete " removed from the name and the definition as written (protocol
amendment A3). owl:Nothing is skipped; a GO class with no stanza is fatal (exit 2,
ids listed). Encoder:
frozen sentence-transformers/all-mpnet-base-v2 at revision
e8c3b32edf5434bc2275fc9bab85f82640a19130, normalize_embeddings=True, batch 256.

Outputs preprocess/go_function/term_text_mpnet.npy (float32, rows = term_ids.txt),
term_ids.txt and term_text_mpnet.json. Existing outputs are never overwritten
(exit 4). Records pin G5 entries "term_text_mpnet.json" and "mpnet_revision".

Exit codes: 0 ok; 2 fatal; 4 refused (existing output).

Usage (remote):
  .venv/bin/python scripts/go_function/embed_terms.py --device cuda
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function import features as F  # noqa: E402
from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_REFUSED  # noqa: E402
from go_function.labels import read_terms  # noqa: E402
from go_function.normforms import load_normal_forms  # noqa: E402
from go_function.ontology import parse_obo  # noqa: E402
from go_function.pins import (assert_writable_target, canonical_json, record_pin,  # noqa: E402
                              sha256_bytes, verify_inputs)
from go_function.runtime import (DEVICES, FatalError, repo_path, repo_rel,  # noqa: E402
                                 resolve_device, setup_logging)

DGZ = "dataset/go_function/dgz"
OUT = ROOT / "preprocess" / "go_function"
NAMES = ("term_text_mpnet.npy", "term_ids.txt", "term_text_mpnet.json")


def class_ids(obo_rel: str = f"{DGZ}/go.obo", norm_rel: str = f"{DGZ}/go.norm"):
    """(ontology, every class of a term tower: V_bp ∪ V_mf ∪ go.norm classes)."""
    ont = parse_obo(repo_path(obo_rel))
    ids = set()
    for aspect in ("bp", "mf"):
        vocab = read_terms(repo_path(f"{DGZ}/{aspect}/terms.pkl"))
        nf = load_normal_forms(repo_path(norm_rel), {g: i for i, g in enumerate(vocab)})
        ids.update(vocab)
        ids.update(nf.class_ids())
    return ont, ids


def main(argv=None, *, encoder=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=DEVICES, default="auto")
    ap.add_argument("--revision", default=F.MPNET_REVISION)
    args = ap.parse_args(argv)
    log = setup_logging("embed_terms")
    t0 = time.time()
    try:
        try:
            device = resolve_device(args.device)
        except ValueError as e:
            raise FatalError(str(e)) from None
        if args.revision != F.MPNET_REVISION:
            raise FatalError(f"MPNet revision is pinned to {F.MPNET_REVISION}")
        paths = [OUT / n for n in NAMES]
        for p in paths:
            assert_writable_target(p)
        if any(p.exists() for p in paths):
            log.error("term-text outputs exist in %s; refusing to overwrite", repo_rel(OUT))
            return EXIT_REFUSED
        inputs = [f"{DGZ}/go.obo", f"{DGZ}/go.norm", f"{DGZ}/bp/terms.pkl", f"{DGZ}/mf/terms.pkl"]
        hashes = verify_inputs(inputs)
        ont, ids = class_ids()
        tstats: dict = {}
        term_ids, texts, skipped = F.term_text_table(ont, ids, stats=tstats)
        log.info("%d classes with text (%d obsolete, own stanza text), %d owl:Nothing skipped",
                 len(term_ids), tstats["n_obsolete_text"], skipped)

        import numpy as np
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(F.MPNET_MODEL, revision=args.revision, device=device)
        vec = F.encode_texts(encoder, texts)
        buf = io.BytesIO()
        np.save(buf, vec, allow_pickle=False)
        vb = buf.getvalue()
        ib = ("\n".join(term_ids) + "\n").encode()
        man = {"schema": 1, "model": F.MPNET_MODEL, "revision": args.revision, "device": device,
               "normalize_embeddings": True, "batch_size": F.MPNET_BATCH,
               "n_terms": len(term_ids), "dim": int(vec.shape[1]),
               "n_owl_nothing_skipped": skipped, "text": "<name>. <definition>",
               "n_obsolete_text": tstats["n_obsolete_text"],
               "obsolete_text_rule": "own go.obo stanza; one leading 'obsolete ' stripped "
                                     "from the name; definition as written (amendment A3)",
               "embeddings_sha256": sha256_bytes(vb), "term_ids_sha256": sha256_bytes(ib),
               "inputs": dict(sorted(hashes.items())),
               "wall_seconds": round(time.time() - t0, 1)}
        jb = canonical_json(man)
        OUT.mkdir(parents=True, exist_ok=True)
        for p, data in zip(paths, (vb, ib, jb)):
            tmp = p.with_name(f".{p.name}.tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
        log.info("wrote %s (%d x %d)", repo_rel(paths[0]), vec.shape[0], vec.shape[1])
        log.info("pin G5: %s", record_pin("G5", {"term_text_mpnet.json": sha256_bytes(jb),
                                                 "mpnet_revision": args.revision}))
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
