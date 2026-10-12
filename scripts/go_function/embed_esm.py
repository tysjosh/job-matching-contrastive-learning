"""ESM-2 650M protein embeddings (design §3.2, gate G5; REMOTE, L40S).

Union of the BP and MF train/valid/test sequences, deduplicated by protein name
(a differing sequence is fatal), sorted by name, truncated to 1,022 residues;
HF EsmModel facebook/esm2_t33_650M_UR50D (fp16 on CUDA, eval, no_grad); mean of
the last hidden layer over residues (BOS/EOS/pad excluded); length-sorted batches
of 16,384 tokens, halved on CUDA OOM down to 2,048; then the 64-protein fp32
batch-1 cross-check (cosine >= 0.999, fatal otherwise).

Outputs preprocess/go_function/esm2_650m/{embeddings.npy, proteins.txt,
manifest.json} (with --limit N: esm2_650m_limit<N>/, a test run that is never
pinned). An existing non-empty output directory is never overwritten (exit 4).
Records pin G5 entries "esm2_650m/manifest.json" and "esm2_650m_revision".

--device cpu (or auto resolving to cpu) without --limit is refused (exit 4): a
full CPU run is not planned anywhere. MPS is never used.

Exit codes: 0 ok; 2 fatal (input hash, sequence conflict, model load after 3
retries, OOM at the minimum budget, cross-check); 4 refused (cpu without --limit,
existing output).

Usage (remote):
  .venv/bin/python scripts/go_function/embed_esm.py --device cuda --revision <sha>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function import features as F  # noqa: E402
from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_REFUSED  # noqa: E402
from go_function.labels import SPLITS  # noqa: E402
from go_function.pins import (assert_writable_target, canonical_json, record_pin,  # noqa: E402
                              sha256_bytes, verify_inputs)
from go_function.runtime import (DEVICES, FatalError, repo_path, repo_rel,  # noqa: E402
                                 resolve_device, seed_everything, setup_logging)

DGZ = "dataset/go_function/dgz"
OUT = ROOT / "preprocess" / "go_function"
RETRIES = 3
BACKOFF_S = 30


def load_model(revision, device, log):
    """EsmModel + EsmTokenizer at ``revision``; network errors retried x3 with a
    30 s backoff. Returns (model, tokenizer, resolved commit hash)."""
    from transformers import EsmModel, EsmTokenizer
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            tok = EsmTokenizer.from_pretrained(F.ESM_MODEL, revision=revision)
            model = EsmModel.from_pretrained(F.ESM_MODEL, revision=revision,
                                             add_pooling_layer=False)
            resolved = getattr(model.config, "_commit_hash", None) or revision
            if revision is not None and resolved != revision:
                raise FatalError(f"ESM revision mismatch: requested {revision}, got {resolved}")
            return model, tok, resolved
        except FatalError:
            raise
        except Exception as e:  # noqa: BLE001 - network/hub errors are retried
            last = e
            log.warning("ESM load attempt %d/%d failed: %s", attempt, RETRIES, e)
            if attempt < RETRIES:
                time.sleep(BACKOFF_S)
    raise FatalError(f"ESM model could not be loaded after {RETRIES} attempts: {last}")


def main(argv=None, *, load=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=DEVICES, default="auto")
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only the first N proteins by name (test run, never pinned)")
    ap.add_argument("--revision", default=None, help="model commit sha to pin")
    ap.add_argument("--token-budget", type=int, default=F.TOKEN_BUDGET)
    args = ap.parse_args(argv)
    log = setup_logging("embed_esm")
    t0 = time.time()
    try:
        try:
            device = resolve_device(args.device)
        except ValueError as e:
            raise FatalError(str(e)) from None
        if device == "cpu" and args.limit is None:
            log.error("refusing a full ESM run on cpu; pass --limit N for a test run "
                      "(the full run is remote, --device cuda)")
            return EXIT_REFUSED
        if args.limit is not None and args.limit < 1:
            raise FatalError("--limit must be >= 1")
        name = "esm2_650m" if args.limit is None else f"esm2_650m_limit{args.limit}"
        out_dir = OUT / name
        assert_writable_target(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            log.error("%s exists and is not empty; refusing to overwrite", repo_rel(out_dir))
            return EXIT_REFUSED
        seed_everything(0)

        import numpy as np
        import pandas as pd
        import torch
        inputs = [f"{DGZ}/{a}/{s}_data.pkl" for a in ("bp", "mf") for s in SPLITS]
        hashes = verify_inputs(inputs)
        frames = []
        for rel in inputs:
            df = pd.read_pickle(repo_path(rel))
            frames.append((df["proteins"].tolist(), df["sequences"].tolist()))
            del df
        names, seqs = F.collect_sequences(frames)
        del frames
        n_all = len(names)
        if args.limit is not None:
            names, seqs = names[:args.limit], seqs[:args.limit]
        seqs, n_trunc = F.truncate(seqs)
        log.info("%d unique proteins (%d embedded), %d truncated to %d residues", n_all,
                 len(names), n_trunc, F.MAX_RESIDUES)

        model, tok, resolved = (load or load_model)(args.revision, device, log)
        dtype = "float16" if device == "cuda" else "float32"
        model = model.to(device=device, dtype=getattr(torch, dtype)).eval()
        t1 = time.time()
        emb, batching = F.embed_all(model, tok, seqs, device, budget=args.token_budget)
        t_embed = time.time() - t1
        model = model.float()
        check = F.cross_check(model, tok, seqs, names, emb, device)
        log.info("cross-check passed: %s", check)

        buf = io.BytesIO()
        np.save(buf, emb.astype(np.float32), allow_pickle=False)
        emb_bytes = buf.getvalue()
        prot_bytes = ("\n".join(names) + "\n").encode()
        man = {"schema": 1, "model": F.ESM_MODEL, "revision": resolved,
               "revision_requested": args.revision, "device": device, "dtype": dtype,
               "limit": args.limit, "n_proteins": len(names), "n_unique_proteins": n_all,
               "dim": int(emb.shape[1]), "max_residues": F.MAX_RESIDUES,
               "n_truncated": n_trunc, "token_budget": args.token_budget, **batching,
               "pooling": "mean over residues, BOS/EOS/pad excluded, last hidden layer",
               "cross_check": check, "embeddings_sha256": sha256_bytes(emb_bytes),
               "proteins_sha256": sha256_bytes(prot_bytes), "inputs": dict(sorted(hashes.items())),
               "wall_seconds": {"embed": round(t_embed, 1), "total": round(time.time() - t0, 1)},
               "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}}
        jb = canonical_json(man)
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname, data in (("embeddings.npy", emb_bytes), ("proteins.txt", prot_bytes),
                            ("manifest.json", jb)):
            tmp = out_dir / f".{fname}.tmp"
            tmp.write_bytes(data)
            tmp.replace(out_dir / fname)
        log.info("wrote %s (%d x %d)", repo_rel(out_dir), emb.shape[0], emb.shape[1])
        if args.limit is None:
            log.info("pin G5: %s", record_pin("G5", {"esm2_650m/manifest.json": sha256_bytes(jb),
                                                     "esm2_650m_revision": resolved}))
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
