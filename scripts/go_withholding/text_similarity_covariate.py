"""Frozen-MPNet text-similarity covariate per target (design 10, gate G2).

Per selected target t (embeddings L2-normalised, cos = dot product):

* ``tsim_excess(t)`` (primary): mean over i in P_test(t) of
  [mean_j cos(e_i, e_j), j in NP_train_sib(t)] minus the same with j from a
  fixed reference of 2,000 train-partition proteins outside
  T_train(t) | NP_train_sib(t), drawn with ``random.Random(0)`` from the sorted
  eligible list.
* ``tsim_train(t)`` (secondary): mean pairwise cos between P_train(t) and
  NP_train_sib(t).
* ``tsim_excess_2`` / ``tsim_train_2``: the same with the N_2 neighbourhood
  (sensitivity only).

Test-protein embeddings come from the eval cache npz (``build-cache``). Train
texts (``content_to_text`` of the protein's ``encoder_view``) are looked up in
the trainer's frozen-encoder cache (``--train-embeddings``, normally absent at
G2) and otherwise encoded with the same frozen MPNet (``max_seq_length`` 512);
``n_from_cache`` and ``n_encoded`` are recorded. Output sha pinned as
``G2-covariate``. Refuses to overwrite. Any failure: exit 2.

Usage (protocol section 10, G2)::

    "$PY" "$S/text_similarity_covariate.py" --target-table "$R/preprocess/go_withholding/targets/target_table.json" \\
      --cache "$CACHE" --train-embeddings "$R/embedding_cache/go_withholding_text_embeddings.pt" \\
      --out "$R/preprocess/go_withholding/targets/text_covariate.json"
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("text_similarity_covariate")

N_REFERENCE = 2000
REFERENCE_SEED = 0
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_encoder(model_name: str = MODEL_NAME, device: Optional[str] = None):
    """The frozen MPNet (tests replace this with a stub)."""
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model_name, device=device)
    enc.max_seq_length = 512
    return enc


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def _mean_cos(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not len(a) or not len(b):
        return None
    return float(np.mean(a @ b.T))


def excess(test_vecs: np.ndarray, nbr_vecs: np.ndarray, ref_vecs: np.ndarray) -> Optional[float]:
    """mean_i [mean_j cos(i, nbr_j)] - mean_i [mean_j cos(i, ref_j)] (unit vectors)."""
    if not len(test_vecs) or not len(nbr_vecs) or not len(ref_vecs):
        return None
    return float(np.mean(np.mean(test_vecs @ nbr_vecs.T, axis=1)
                         - np.mean(test_vecs @ ref_vecs.T, axis=1)))


def reference_sample(partition: Iterable[str], exclude: Iterable[str],
                     n: Optional[int] = None, seed: int = REFERENCE_SEED) -> List[str]:
    """``min(n, |pool|)`` proteins of the sorted eligible pool, ``random.Random(seed)``.

    ``n`` defaults to the module's ``N_REFERENCE`` (2,000), read at call time.
    """
    n = N_REFERENCE if n is None else int(n)
    pool = sorted(set(partition) - set(exclude))
    return random.Random(seed).sample(pool, min(n, len(pool)))


def train_partition(splits: Path) -> List[str]:
    from go_ppi_domain.withholding_targets import partition_roles

    def rows(path: Path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    recs = list(rows(Path(splits) / "train.jsonl"))
    pools = [r for r in rows(Path(splits) / "negative_pools.jsonl") if r.get("split") == "train"]
    return sorted(partition_roles(recs, pools)["any"])


def load_views(converted: Path) -> Dict[str, str]:
    out = {}
    with open(Path(converted) / "proteins.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[str(row["protein_id"])] = row["encoder_view"]
    return out


def train_embeddings(needed: Sequence[str], views: Dict[str, str],
                     cache_file: Optional[Path], encode: Callable[[List[str]], np.ndarray]
                     ) -> Dict[str, Any]:
    """Vectors for ``needed`` train proteins: trainer cache first, else ``encode``."""
    from run_phase1_embedding_evaluation import content_to_text

    store: Dict[str, Any] = {}
    keyer = None
    if cache_file is not None and Path(cache_file).is_file():
        import torch
        from contrastive_learning.embedding_cache import EmbeddingCache

        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        store = payload.get("cache", {}) or {}
        keyer = EmbeddingCache(max_cache_size=0, device=torch.device("cpu"), enable_stats=False)
    vecs: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for p in needed:
        if p not in views:
            raise RuntimeError(f"train protein {p} has no encoder_view")
        hit = None
        if keyer is not None:
            # Resume-slot and job-slot keys (``title`` is the only other hashed field).
            for content in ({"encoder_view": views[p]},
                            {"encoder_view": views[p], "title": p}):
                v = store.get(keyer.get_content_key(content))
                if v is not None:
                    hit = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                                     dtype=np.float32).reshape(-1)
                    break
        if hit is None:
            missing.append(p)
        else:
            vecs[p] = hit
    n_from_cache = len(vecs)
    if missing:
        texts = [content_to_text({"encoder_view": views[p]}, "resume") for p in missing]
        emb = np.asarray(encode(texts), dtype=np.float32)
        if emb.shape[0] != len(missing):
            raise RuntimeError(f"encoder returned {emb.shape[0]} rows for {len(missing)} texts")
        for p, v in zip(missing, emb):
            vecs[p] = v
    return {"vectors": vecs, "n_from_cache": n_from_cache, "n_encoded": len(missing)}


def compute_covariate(table: Dict[str, Any], test_ids: Sequence[str], test_emb: np.ndarray,
                      partition: Sequence[str], train_vecs: Dict[str, np.ndarray]
                      ) -> Dict[str, Dict[str, Any]]:
    """The per-target covariate values (pure; unit-normalises every vector)."""
    tpos = {p: i for i, p in enumerate(test_ids)}
    tunit = _unit(test_emb)

    def tr(ps: Iterable[str]) -> np.ndarray:
        ps = list(ps)
        return _unit(np.stack([train_vecs[p] for p in ps])) if ps else np.zeros((0, 1))

    out: Dict[str, Dict[str, Any]] = {}
    for t in table["targets"]:
        P_test = [p for p in t["test_anchors"] if p in tpos]
        tv = tunit[[tpos[p] for p in P_test]] if P_test else np.zeros((0, tunit.shape[1]))
        T_train = set(t["train_annotated_proteins"])
        row: Dict[str, Any] = {"n_P_test": len(P_test), "n_P_train": len(t["train_proteins"])}
        for suffix, key in (("", "nbr_train_proteins_sib"), ("_2", "nbr_train_proteins_2")):
            NP = sorted(t.get(key, []))
            ref = reference_sample(partition, T_train | set(NP))
            nv, rv, pv = tr(NP), tr(ref), tr(sorted(t["train_proteins"]))
            row[f"tsim_excess{suffix}"] = excess(tv, nv, rv) if len(nv) and len(rv) else None
            row[f"tsim_train{suffix}"] = _mean_cos(pv, nv) if len(nv) else None
            row[f"n_NP_train{suffix}"] = len(NP)
            row[f"n_reference{suffix}"] = len(ref)
        out[t["go_id"]] = row
    return out


def needed_train_proteins(table: Dict[str, Any], partition: Sequence[str]) -> List[str]:
    need = set()
    for t in table["targets"]:
        T_train = set(t["train_annotated_proteins"])
        need |= set(t["train_proteins"])
        for key in ("nbr_train_proteins_sib", "nbr_train_proteins_2"):
            NP = set(t.get(key, []))
            need |= NP
            need |= set(reference_sample(partition, T_train | NP))
    return sorted(need)


def run(table_path: Path, cache_path: Path, train_embeddings_path: Optional[Path], out: Path,
        splits: Path, converted: Path, device: Optional[str] = None) -> Dict[str, Any]:
    wc.assert_writable_target(out)
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {wc.rel_to_root(out)}")
    table = json.loads(Path(table_path).read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as z:
        test_ids = [str(p) for p in z["protein_ids"]]
        test_emb = np.asarray(z["emb"], dtype=np.float32)
    partition = train_partition(splits)
    views = load_views(converted)
    need = needed_train_proteins(table, partition)
    encoder_box: Dict[str, Any] = {}

    def encode(texts: List[str]) -> np.ndarray:
        if "enc" not in encoder_box:
            encoder_box["enc"] = load_encoder(MODEL_NAME, device)
        enc = encoder_box["enc"]
        return np.asarray(enc.encode(texts, batch_size=64, convert_to_numpy=True,
                                     show_progress_bar=False), dtype=np.float32)

    te = train_embeddings(need, views, train_embeddings_path, encode)
    per = compute_covariate(table, test_ids, test_emb, partition, te["vectors"])
    doc = {"schema_version": 1, "model": MODEL_NAME, "n_reference": N_REFERENCE,
           "reference_seed": REFERENCE_SEED, "n_targets": len(per),
           "n_train_needed": len(need), "n_encoded": te["n_encoded"],
           "n_from_cache": te["n_from_cache"],
           "inputs": {"target_table_sha256": wc.sha256_file(table_path),
                      "cache_npz_sha256": wc.sha256_file(cache_path)},
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "targets": per}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    wc.append_pin("G2-covariate", {"text_covariate_sha256": wc.sha256_file(out),
                                   "n_targets": len(per), "n_encoded": te["n_encoded"],
                                   "n_from_cache": te["n_from_cache"]})
    return {k: doc[k] for k in ("n_targets", "n_train_needed", "n_encoded", "n_from_cache")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target-table", type=Path, default=None)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--train-embeddings", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--converted", type=Path, default=None)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    device = None if args.device in (None, "auto") else args.device
    try:
        summary = run(Path(args.target_table or (Path(wc.TARGETS) / "target_table.json")),
                      args.cache, args.train_embeddings,
                      Path(args.out or (Path(wc.TARGETS) / "text_covariate.json")),
                      Path(args.splits or wc.SPLITS), Path(args.converted or wc.CONVERTED),
                      device)
    except Exception as exc:  # noqa: BLE001
        logger.error("text_similarity_covariate failed: %s: %s", type(exc).__name__, exc)
        return 2
    summary["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
