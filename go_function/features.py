"""Protein and term features (design §3).

- InterPro (§3.1): binary CSR over the 26,406-entry ``interpros.pkl`` vocabulary,
  rows in pickle order; unknown ids dropped and counted; proteins without an
  InterPro entry get an all-zero row (as in DGZ ``deepgozero.get_data``).
- ESM-2 650M (§3.2, remote): sequence collection, truncation, length-sorted token
  batches with OOM halving, mean pooling over residues (BOS/EOS/pad excluded),
  the fp32 batch-1 cross-check, and per-aspect views by protein name.
- Term text (§3.3): ``"<name>. <definition>"`` from the pinned go.obo for every GO
  class of a term tower; ``owl:Nothing`` is skipped, a GO class without a live
  term is fatal.

No model is downloaded here; ``scripts/go_function/embed_esm.py`` and
``embed_terms.py`` load the models and pass them in.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from go_function.constants import OWL_NOTHING
from go_function.runtime import FatalError

logger = logging.getLogger("go_function.features")

INTERPRO_VOCAB_SIZE = 26406
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM_DIM = 1280
MAX_RESIDUES = 1022
TOKEN_BUDGET = 16384
MIN_TOKEN_BUDGET = 2048
CROSS_CHECK_N = 64
CROSS_CHECK_MIN_COS = 0.999
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
MPNET_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
MPNET_BATCH = 256


# ------------------------------------------------------------------ InterPro (§3.1)
def read_interpro_vocab(path) -> List[str]:
    import pandas as pd
    ids = [str(x) for x in pd.read_pickle(path)["interpros"].tolist()]
    if len(set(ids)) != len(ids):
        raise FatalError("interpros.pkl has duplicate ids")
    return ids


def interpro_csr(rows: Sequence[Iterable[str]], index: Mapping[str, int]) -> Tuple[sp.csr_matrix, int]:
    """Binary CSR (uint8) of the InterPro ids per protein; returns (matrix,
    number of unknown ids dropped)."""
    indptr = [0]
    indices: List[int] = []
    unknown = 0
    for ids in rows:
        cols = set()
        for ipr in ids:
            j = index.get(ipr)
            if j is None:
                unknown += 1
            else:
                cols.add(j)
        indices.extend(sorted(cols))
        indptr.append(len(indices))
    m = sp.csr_matrix((np.ones(len(indices), dtype=np.uint8), np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int64)), shape=(len(indptr) - 1, len(index)))
    return m, unknown


# ------------------------------------------------------------------ ESM-2 (§3.2)
def collect_sequences(frames: Iterable[Tuple[Sequence[str], Sequence[str]]]) -> Tuple[List[str], List[str]]:
    """Union of (proteins, sequences) over pickles, deduplicated by name and sorted
    by name. The same name with a different sequence is fatal."""
    seen: Dict[str, str] = {}
    bad: List[str] = []
    for names, seqs in frames:
        for n, s in zip(names, seqs):
            n, s = str(n), str(s)
            old = seen.setdefault(n, s)
            if old != s:
                bad.append(n)
    if bad:
        raise FatalError(f"{len(bad)} proteins have differing sequences across pickles; "
                         f"first: {sorted(set(bad))[:10]}")
    names = sorted(seen)
    return names, [seen[n] for n in names]


def truncate(seqs: Sequence[str], max_residues: int = MAX_RESIDUES) -> Tuple[List[str], int]:
    out = [s[:max_residues] for s in seqs]
    return out, sum(1 for s in seqs if len(s) > max_residues)


def length_batches(lengths: Sequence[int], budget: int) -> List[List[int]]:
    """Index batches sorted by length (stable), each with n * (max_len + 2) <= budget
    tokens (BOS/EOS included)."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: List[List[int]] = []
    cur: List[int] = []
    for i in order:
        width = lengths[i] + 2
        if width > budget:
            raise FatalError(f"sequence of {lengths[i]} residues exceeds the token budget {budget}")
        if cur and (len(cur) + 1) * width > budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def residue_mask(attention_mask):
    """Attention mask with BOS (position 0) and EOS (last attended position) removed."""
    import torch
    m = attention_mask.clone()
    lengths = attention_mask.sum(dim=1)
    m[:, 0] = 0
    m[torch.arange(m.shape[0], device=m.device), lengths - 1] = 0
    return m


def mean_pool(hidden, attention_mask):
    """Mean of the last hidden layer over residue positions (float32)."""
    hidden = hidden.float()
    m = residue_mask(attention_mask).to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def embed_batch(model, tokenizer, seqs: Sequence[str], device: str) -> np.ndarray:
    import torch
    enc = tokenizer(list(seqs), padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state
    return mean_pool(hidden, enc["attention_mask"]).cpu().numpy().astype(np.float32)


def _is_oom(e: BaseException) -> bool:
    import torch
    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(e, oom)) or "out of memory" in str(e).lower()


def embed_all(model, tokenizer, seqs: Sequence[str], device: str, budget: int = TOKEN_BUDGET,
              min_budget: int = MIN_TOKEN_BUDGET, embed: Optional[Callable] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Embed every sequence in length-sorted token batches. On OOM the budget is
    halved (down to ``min_budget``, then fatal) and the remaining sequences are
    re-batched; finished batches are kept."""
    embed = embed or embed_batch
    out: Optional[np.ndarray] = None
    done = np.zeros(len(seqs), dtype=bool)
    halvings = 0
    while not done.all():
        todo = np.flatnonzero(~done)
        batches = length_batches([len(seqs[i]) for i in todo], budget)
        try:
            for b in batches:
                idx = todo[b]
                vec = embed(model, tokenizer, [seqs[i] for i in idx], device)
                if out is None:
                    out = np.zeros((len(seqs), vec.shape[1]), dtype=np.float32)
                out[idx] = vec
                done[idx] = True
        except Exception as e:  # noqa: BLE001 - only OOM is recoverable
            if not _is_oom(e):
                raise
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
            if budget // 2 < min_budget:
                raise FatalError(f"ESM embedding: out of memory at the minimum token budget "
                                 f"{budget}") from e
            budget //= 2
            halvings += 1
            logger.warning("ESM embedding OOM; token budget halved to %d", budget)
    if out is None:
        out = np.zeros((0, 0), dtype=np.float32)
    return out, {"final_token_budget": budget, "n_oom_halvings": halvings}


def cross_check_ids(names: Sequence[str], n: int = CROSS_CHECK_N) -> List[int]:
    """Row indices of the first ``n`` proteins by sha256(name) (design §3.2)."""
    key = sorted(range(len(names)), key=lambda i: hashlib.sha256(names[i].encode()).hexdigest())
    return sorted(key[:n])


def cross_check(model_fp32, tokenizer, seqs: Sequence[str], names: Sequence[str],
                stored: np.ndarray, device: str, n: int = CROSS_CHECK_N,
                min_cos: float = CROSS_CHECK_MIN_COS) -> Dict[str, object]:
    """Recompute ``n`` fixed proteins in fp32 with batch size 1; every cosine with
    the stored vector must be >= ``min_cos`` (fatal otherwise)."""
    rows = cross_check_ids(names, n)
    cos = []
    for i in rows:
        v = embed_batch(model_fp32, tokenizer, [seqs[i]], device)[0].astype(np.float64)
        s = stored[i].astype(np.float64)
        cos.append(float(v @ s / (np.linalg.norm(v) * np.linalg.norm(s))))
    bad = [names[i] for i, c in zip(rows, cos) if not c >= min_cos]
    if bad:
        raise FatalError(f"ESM cross-check failed (cosine < {min_cos}) for {bad}")
    return {"n": len(rows), "min_cosine": min(cos) if cos else None}


def esm_view(emb: np.ndarray, emb_proteins: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Rows of ``emb`` for ``names`` (per-aspect feature matrix); a missing protein
    is fatal."""
    index = {p: i for i, p in enumerate(emb_proteins)}
    missing = [n for n in names if n not in index]
    if missing:
        raise FatalError(f"{len(missing)} proteins missing from the ESM embeddings; "
                         f"first: {missing[:10]}")
    return emb[np.asarray([index[n] for n in names], dtype=np.int64)]


# ------------------------------------------------------------------ term text (§3.3)
OBSOLETE_PREFIX = "obsolete "


def obsolete_term_text(ont, go_id: str) -> Optional[str]:
    """Text of an obsolete class from its own go.obo stanza (amendment A3): the name
    with one leading "obsolete " removed, then the definition as written. None if
    ``go_id`` is not an obsolete id of ``ont``."""
    entry = getattr(ont, "obsolete_text", {}).get(go_id)
    if entry is None:
        return None
    name, definition = entry
    if name.startswith(OBSOLETE_PREFIX):
        name = name[len(OBSOLETE_PREFIX):]
    return f"{name}. {definition}".rstrip()


def term_text_table(ont, class_ids: Iterable[str], stats: Optional[Dict[str, int]] = None
                    ) -> Tuple[List[str], List[str], int]:
    """(ids, texts, n_owl_nothing_skipped) for the sorted unique GO classes of
    ``class_ids``. Alt ids take their primary's text; an obsolete class (go.norm
    names 728 in the pinned release) takes its own stanza's text
    (``obsolete_term_text``, amendment A3). An id that is neither live nor obsolete,
    or whose text is empty, is fatal (exit 2) with the list of ids. ``stats``, if
    given, receives ``n_obsolete_text``."""
    ids = sorted(set(class_ids))
    skipped = 0
    n_obsolete = 0
    out_ids: List[str] = []
    texts: List[str] = []
    missing: List[str] = []
    for g in ids:
        if g == OWL_NOTHING:
            skipped += 1
            continue
        p = ont.primary(g)
        if p is not None:
            t = ont.terms[p]
            text = f"{t.name}. {t.definition}".rstrip()
        else:
            text = obsolete_term_text(ont, g)
            n_obsolete += text is not None
        if not text or text == ".":
            missing.append(g)
            continue
        out_ids.append(g)
        texts.append(text)
    if missing:
        raise FatalError(f"{len(missing)} GO classes have no term text in go.obo: {missing}")
    if stats is not None:
        stats["n_obsolete_text"] = n_obsolete
    return out_ids, texts, skipped


def encode_texts(encoder, texts: Sequence[str], batch_size: int = MPNET_BATCH) -> np.ndarray:
    """Frozen sentence encoder, ``normalize_embeddings=True``, float32."""
    vec = encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[0] != len(texts):
        raise FatalError(f"encoder returned {vec.shape[0]} vectors for {len(texts)} texts")
    return vec
