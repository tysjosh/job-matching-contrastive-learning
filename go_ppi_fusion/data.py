"""Read-only port of ``evaluate_publication_fusion.prepare_split`` (no encoder, no mkdir).

The published ``prepare_split`` constructs a ``SentenceTransformer`` without a
device (MPS on the Mac) and creates the prepared-embedding directory. This port
resolves every text vector from the same two read-only sources, in the same
order (frozen cache by content key, then prepared chunks by sha256(text)), and
raises if a vector is missing instead of encoding. ``unique_texts`` order and
the ``base`` stacking order are identical, so ``epf.text_scores`` on the result
is bit-identical to the published run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from go_ppi_fusion import guard

ROOT = guard.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLITS = {
    "validation": ROOT / "preprocess" / "go_ppi_splits" / "validation.jsonl",
    "test": ROOT / "preprocess" / "go_ppi_splits" / "test.jsonl",
}
PRUNED_SPLITS = {
    "validation": ROOT / "preprocess" / "go_withholding" / "splits" / "validation.jsonl",
    "test": ROOT / "preprocess" / "go_withholding" / "splits" / "test.jsonl",
}
PREPARED_DIR = ROOT / "results" / "publication_phase1_fixedval" / "prepared_embeddings" / "go_ppi"
CONFIG_PATH = ROOT / "config" / "lc_go_ppi_baseline.json"
CHECKPOINT_DIR = ROOT / "results" / "publication_phase1_fixedval" / "checkpoints"
FRACTIONS = (10, 25, 50, 75, 100)
SEEDS = (42, 13, 21)
EMBEDDING_DIM = 768


def checkpoint_grid() -> List[Dict[str, Any]]:
    """The 15 published checkpoints in the published JSON order."""
    return [{"fraction": f, "seed": s,
             "checkpoint": CHECKPOINT_DIR / f"go_ppi_baseline_f{f}_s{s}" / "best_checkpoint.pt"}
            for f in FRACTIONS for s in SEEDS]


def file_sha256(path: Path) -> str:
    guard.check_path(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, split: Optional[str] = None) -> List[Dict[str, Any]]:
    guard.check_path(path, split)
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _grade(record: Dict[str, Any]) -> Optional[int]:
    grade = (record.get("job") or {}).get("grade")
    if grade is None:
        label = (record.get("metadata") or {}).get("original_label")
        grade = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}.get(label)
    return grade


@dataclass
class Split:
    """Everything the scorer needs for one split, in record order."""

    name: str
    grades: np.ndarray
    queries: np.ndarray            # anchor protein id per record (``metadata.resume_id``)
    anchors: List[str]
    partners: List[str]
    uris_a: List[List[str]]        # unpruned record skill_uris (direct BP terms)
    uris_b: List[List[str]]
    base: Optional[np.ndarray] = None
    left_index: Optional[np.ndarray] = None
    right_index: Optional[np.ndarray] = None
    pruned_uris_a: Optional[List[List[str]]] = None
    pruned_uris_b: Optional[List[List[str]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.grades)

    def prepared(self) -> Dict[str, Any]:
        """The dict shape ``epf.text_scores`` expects."""
        return {"base": self.base, "left_index": self.left_index,
                "right_index": self.right_index, "embedding_dim": EMBEDDING_DIM,
                "grades": self.grades, "queries": self.queries}


def assert_aligned(records: Sequence[Dict[str, Any]],
                   pruned: Sequence[Dict[str, Any]]) -> None:
    """Pruned records must match the published records 1:1 (order, anchor, partner, grade)."""
    if len(records) != len(pruned):
        raise AssertionError(f"pruned/unpruned length mismatch {len(records)} != {len(pruned)}")
    for i, (r, p) in enumerate(zip(records, pruned)):
        key_r = ((r.get("metadata") or {}).get("resume_id"), r["resume"].get("protein_id"),
                 r["job"].get("partner_id"), _grade(r))
        key_p = ((p.get("metadata") or {}).get("resume_id"), p["resume"].get("protein_id"),
                 p["job"].get("partner_id"), _grade(p))
        if key_r != key_p:
            raise AssertionError(f"pruned/unpruned misaligned at record {i}: {key_r} != {key_p}")


def load_frozen_cache(config) -> Dict[str, Any]:
    import torch
    path = Path(config.embedding_cache_path)
    if not path.is_absolute():
        path = ROOT / path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("cache", {})


def _prepared_chunks(split_path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    prefix = f"{split_path.stem}_{file_sha256(split_path)[:16]}_"
    dirs = [d for d in PREPARED_DIR.glob(prefix + "*") if d.is_dir()]
    for d in dirs:
        guard.check_path(d)
        for chunk_path in sorted(d.glob("chunk_*.npz")):
            with np.load(chunk_path, allow_pickle=False) as c:
                for key, vec in zip(c["keys"], c["embeddings"]):
                    out[str(key)] = np.asarray(vec, dtype=np.float32)
    return out


def load_split(name: str, config=None, frozen_cache: Optional[Dict[str, Any]] = None,
               with_embeddings: bool = True, with_pruned: bool = True) -> Split:
    """Load one published split read-only, optionally with aligned pruned URIs."""
    if name not in SPLITS:
        raise ValueError(name)
    guard.check_split(name)
    path = SPLITS[name]
    records = read_jsonl(path, name)
    kept = [r for r in records
            if isinstance(r.get("resume"), dict) and isinstance(r.get("job"), dict)
            and _grade(r) in (0, 1, 2)]
    if len(kept) != len(records):
        raise AssertionError("published GO/PPI splits have no unscorable records")
    split = Split(
        name=name,
        grades=np.asarray([int(_grade(r)) for r in records], dtype=int),
        queries=np.asarray([str(r["metadata"]["resume_id"]) for r in records], dtype=object),
        anchors=[r["resume"]["protein_id"] for r in records],
        partners=[r["job"]["partner_id"] for r in records],
        uris_a=[list(r["resume"].get("skill_uris") or []) for r in records],
        uris_b=[list(r["job"].get("skill_uris") or []) for r in records],
        meta={"path": str(path), "sha256": file_sha256(path)},
    )
    if with_pruned:
        pruned = read_jsonl(PRUNED_SPLITS[name], name)
        assert_aligned(records, pruned)
        split.pruned_uris_a = [list(p["resume"].get("skill_uris") or []) for p in pruned]
        split.pruned_uris_b = [list(p["job"].get("skill_uris") or []) for p in pruned]
        split.meta["pruned_sha256"] = file_sha256(PRUNED_SPLITS[name])
    if with_embeddings:
        _attach_embeddings(split, records, path, config, frozen_cache)
    return split


def _attach_embeddings(split: Split, records, path: Path, config, frozen_cache) -> None:
    import torch
    from contrastive_learning.embedding_cache import EmbeddingCache
    from run_phase1_embedding_evaluation import content_to_text

    keyer = EmbeddingCache(max_cache_size=0, device=torch.device("cpu"), enable_stats=False)
    by_hash = _prepared_chunks(path)
    texts_a: List[str] = []
    texts_b: List[str] = []
    by_text: Dict[str, np.ndarray] = {}
    for r in records:
        left, right = r["resume"], r["job"]
        ta, tb = content_to_text(left, "resume"), content_to_text(right, "job")
        if not ta.strip() or not tb.strip():
            raise ValueError("blank encoder text")
        texts_a.append(ta)
        texts_b.append(tb)
        if frozen_cache:
            for content, tv in ((left, ta), (right, tb)):
                cached = frozen_cache.get(keyer.get_content_key(content))
                if cached is not None and tv not in by_text:
                    if isinstance(cached, torch.Tensor):
                        cached = cached.detach().cpu().numpy()
                    by_text[tv] = np.asarray(cached, dtype=np.float32)
    unique = list(dict.fromkeys(texts_a + texts_b))
    hashes = {t: hashlib.sha256(t.encode("utf-8")).hexdigest() for t in unique}
    n_frozen = n_chunk = 0
    vecs = []
    for t in unique:
        if t in by_text:
            vecs.append(by_text[t]); n_frozen += 1
        elif hashes[t] in by_hash:
            vecs.append(by_hash[hashes[t]]); n_chunk += 1
        else:
            raise RuntimeError(f"no cached embedding for a {split.name} text; refusing to encode")
    index = {t: i for i, t in enumerate(unique)}
    split.base = np.stack(vecs)
    split.left_index = np.asarray([index[t] for t in texts_a], dtype=int)
    split.right_index = np.asarray([index[t] for t in texts_b], dtype=int)
    split.meta.update({"unique_texts": len(unique), "from_frozen_cache": n_frozen,
                       "from_prepared_chunks": n_chunk})


def load_config():
    from contrastive_learning.data_structures import TrainingConfig
    return TrainingConfig.from_json(str(CONFIG_PATH))


def text_scores(split: Split, checkpoint: Path, config) -> np.ndarray:
    """Published ``epf.text_scores`` on the read-only prepared split (CPU)."""
    from scripts import evaluate_publication_fusion as epf
    return epf.text_scores(split.prepared(), Path(checkpoint), config)
