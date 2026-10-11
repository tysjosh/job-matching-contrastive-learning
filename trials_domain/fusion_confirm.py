"""TREC-CT MeSH score-fusion confirmation study: shared library.

Implements the preregistered protocol
``.agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md`` (v7).
Section numbers (``§``) below refer to that protocol; where this module and the
protocol differ, the protocol wins.

Contents (all pure functions unless noted):

* run-environment guards (``PYTHONHASHSEED=0``, ``HF_HUB_OFFLINE=1``) and repo-root handling;
* hashing, the cwd-independent chunk-manifest digest (§2.1);
* the read-only embedding loader (port of ``epf.prepare_split``; never writes into the
  published prepared cache, never constructs an encoder for 2022 files) (§2.1);
* φ variants: hierarchy, exact, group-aware shuffled derangement (§2.2-2.3);
* z-scoring, fusion and validation-nDCG weight selection with the smaller-w tie rule (§2.4-2.5);
* per-patient metrics (§2.6), the two-level bootstrap and p-values (§4.1);
* Holm with H1-first tie ordering, S4 equivalence, C2 states, the ordered final-label
  rule and headline (§1, §4.3);
* paraphrase grouping, exposure closure, fold assignment and fold-file builder (§3);
* the gate-2 leaf classifier/comparator (§5.2, design review round 7 finding 1);
* the gate-3 device check (§5.3), the pins-chain verifier (§7), the protocol-table
  parser (§6.2 step -1) and the append-only run ledger (§6.3).

``scripts/evaluate_publication_fusion.py`` (``epf``) is imported by path and reused
for ``rank_auc``, ``ranking_metrics``, ``z_stats``, ``z_apply``, ``text_scores`` and
``evaluate_score``; it is never edited.
"""

from __future__ import annotations

import collections
import copy
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------- root
#: Repo root. Override with ``TRIALS_FUSION_ROOT`` (or a script's ``--root``); every
#: path written to configs, manifests, pins or outputs is relative to it.
ROOT = Path(os.environ.get("TRIALS_FUSION_ROOT") or Path(__file__).resolve().parents[1]).resolve()

PROTOCOL_REL = ".agents/tasks/ontology-withholding-feasibility/protocol_trials_fusion.md"
RESULTS_REL = "results/trials_fusion_confirm"
DEV_LOCAL_REL = "results/trials_fusion_confirm/dev_local"
CV2021_REL = "preprocess/trials_fusion_confirm/cv2021"
BASE_CONFIG_REL = "config/lc_trials_baseline.json"
ORIGINAL_CACHE_REL = "embedding_cache/trials_text_embeddings.pt"
CACHE_COPY_REL = "results/trials_fusion_confirm/cache/trials_text_embeddings.pt"
PUBLISHED_FUSION_REL = "results/publication_phase1_fixedval/trials_fusion_learning_curve_ndcg_selected.json"
PUBLISHED_CKPT_PATTERN = "results/publication_phase1_fixedval/checkpoints/trials_baseline_f{F}_s{S}/best_checkpoint.pt"
PUBLISHED_PREP_DIR_REL = "results/publication_phase1_fixedval/prepared_embeddings/trials"
FRESH_PREP_DIR_REL = "results/trials_fusion_confirm/prepared_embeddings/trials"
VAL2022_REL = "preprocess/trec_ct_splits/validation.jsonl"
TEST2022_REL = "preprocess/trec_ct_splits/test.jsonl"
MPNET_REPO = "sentence-transformers/all-mpnet-base-v2"

ORIGINAL_CACHE_SHA256 = "fa83e10a95a37a378c3d54e110678a64f702a850aef2442b06148d2589e31d4e"
EMBEDDING_DIM = 768

#: §9 A0. Closed-unexposed 2021 topics that the Mac dev smoke scored under fusion (the
#: whole fold-1 inner-validation file of outer fold 0 was scored; 2021_27 and 2021_64,
#: also in that file, were already in ``exposed_topics_closed``). Exposure is by topic id,
#: so this list does not depend on the remote fold build.
DEV_SMOKE_EXPOSED_2021 = ("2021_7", "2021_9", "2021_12", "2021_14", "2021_29", "2021_31", "2021_32",
                          "2021_35", "2021_43", "2021_46", "2021_47", "2021_54", "2021_67")

#: §9 A0 disclosures, printed in every ``report.md``.
DEV_SMOKE_DISCLOSURES = (
    "§9 A0 (2022 C2 preview): before pins revision 1, the Mac dev smoke ran the full C1-C3 and C2-state "
    "path, with the group-aware derangement, on the 15 published cohort-2022 checkpoints (15 of the 19 "
    "C2_2022 inputs; dev settings B=200). C2_2022 is therefore not blind to the estimator's behaviour "
    "on those checkpoints.",
    "§9 A0 (2021 exposure): the Mac dev smoke scored the fold-1 inner-validation file of outer fold 0 "
    "under fusion (zero-shot and a 2-epoch smoke model; validation nDCG@10 curves for hierarchy, exact "
    "and shuffled φ). This exposed 13 of the 52 closed-unexposed patients: "
    + ", ".join(DEV_SMOKE_EXPOSED_2021) + ". The preregistered `unexposed_closed` (52) is unchanged; the "
    "descriptive `unexposed_closed_minus_dev_smoke` (39) excludes them.",
)


def rel(path: os.PathLike | str) -> str:
    """Repo-relative POSIX string (absolute paths outside ROOT are refused)."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    for base in (ROOT, ROOT.resolve()):
        for cand in (p, p.resolve()):
            try:
                return cand.relative_to(base).as_posix()
            except ValueError:
                continue
    raise ValueError(f"path {p} is outside the repo root; refusing to record it")


def absp(path: os.PathLike | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_root(root: Optional[os.PathLike | str]) -> Path:
    """Re-point ROOT (used by scripts' ``--root``) and chdir there."""
    global ROOT
    if root is not None:
        ROOT = Path(root).resolve()
    os.chdir(ROOT)
    return ROOT


# ---------------------------------------------------------------------------- guards
class ProtocolError(RuntimeError):
    """A protocol rule was violated; the script must stop."""


def require_hashseed() -> str:
    """§6: every new script asserts ``PYTHONHASHSEED == "0"`` at start-up."""
    value = os.environ.get("PYTHONHASHSEED")
    if value != "0":
        raise ProtocolError(f"PYTHONHASHSEED must be '0' (got {value!r}); "
                            "run with `export PYTHONHASHSEED=0`")
    return value


def require_hf_offline() -> str:
    value = os.environ.get("HF_HUB_OFFLINE")
    if value != "1":
        raise ProtocolError(f"HF_HUB_OFFLINE must be '1' (got {value!r}); "
                            "run with `export HF_HUB_OFFLINE=1`")
    return value


def run_env_record() -> Dict[str, Any]:
    return {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def require_cpu_scoring(device: str) -> None:
    """§6 device policy: scoring, gate 2, fresh encodes and analysis are CPU only."""
    if device != "cpu":
        raise ProtocolError(f"--device {device!r}: scoring/analysis are forced to CPU by the protocol")
    import torch
    if torch.cuda.is_available():
        raise ProtocolError("CUDA is visible; run scoring/analysis with CUDA_VISIBLE_DEVICES=\"\"")


def refuse_existing(path: os.PathLike | str) -> Path:
    """§6.1 no-overwrite rule: the named leaf must not exist (parents may)."""
    p = absp(path)
    if p.exists():
        raise ProtocolError(f"{rel(p)} already exists; no-overwrite rule (§6.1), no --force")
    return p


def require_dev_local(path: os.PathLike | str) -> Path:
    """Non-confirmatory outputs must live under dev_local/ or /tmp."""
    p = absp(path).resolve()
    dev = (ROOT / DEV_LOCAL_REL).resolve()
    tmp_roots = [Path("/tmp").resolve(), Path(os.environ.get("TMPDIR", "/tmp")).resolve()]
    if not (str(p).startswith(str(dev) + os.sep) or p == dev
            or any(str(p).startswith(str(t) + os.sep) for t in tmp_roots)):
        raise ProtocolError(f"non-confirmatory output {p} must be under {DEV_LOCAL_REL}/ or /tmp")
    return p


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------- epf
_EPF = None


def epf():
    """``scripts/evaluate_publication_fusion.py`` imported by path (§6.1)."""
    global _EPF
    if _EPF is None:
        path = ROOT / "scripts" / "evaluate_publication_fusion.py"
        for extra in (str(ROOT), str(ROOT / "scripts")):
            if extra not in sys.path:
                sys.path.insert(0, extra)
        spec = importlib.util.spec_from_file_location("evaluate_publication_fusion", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _EPF = module
    return _EPF


# ---------------------------------------------------------------------------- hashing
def file_sha256(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(absp(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_digest_from_hashes(hashes: Mapping[str, str]) -> str:
    """§2.1: sha256 of ``"<sha256>  <basename>\\n"`` lines sorted by basename."""
    lines = "".join(f"{hashes[name]}  {Path(name).name}\n"
                    for name in sorted(hashes, key=lambda n: Path(n).name))
    return text_sha256(lines)


def chunk_hashes(chunk_dir: os.PathLike | str) -> Dict[str, str]:
    """``{basename: sha256}`` for every ``chunk_*.npz`` in ``chunk_dir``."""
    d = absp(chunk_dir)
    return {p.name: file_sha256(p) for p in sorted(d.glob("chunk_*.npz"))}


def manifest_digest(chunk_dir: os.PathLike | str) -> str:
    return manifest_digest_from_hashes(chunk_hashes(chunk_dir))


# ---------------------------------------------------------------------------- topics
def topic_key(topic_id: str) -> Tuple[str, int]:
    """§3 sort key ``(year, int(topic_id.split('_')[1]))``."""
    year, number = str(topic_id).split("_", 1)
    return year, int(number)


def sort_topics(topics: Iterable[str]) -> List[str]:
    return sorted(set(topics), key=topic_key)


#: §3 list (1), from findings_trials.md §A (prefix 2021_).
FIXED_PARAPHRASE_EDGES: Tuple[Tuple[str, str], ...] = tuple(
    (f"2021_{a}", f"2021_{b}") for a, b in (
        (40, 74), (27, 64), (25, 61), (33, 69), (57, 63), (23, 59), (39, 73), (41, 75),
        (36, 71), (34, 70), (38, 72), (21, 57), (24, 60), (30, 66), (31, 67), (22, 58)))
EXPECTED_ADDED_EDGES = frozenset(frozenset(e) for e in (
    ("2021_26", "2021_63"), ("2021_26", "2021_57"), ("2021_15", "2021_21")))
EXPECTED_LARGEST_COMPONENT = ("2021_15", "2021_21", "2021_26", "2021_57", "2021_63")
EXPOSED_TOPICS: Tuple[str, ...] = tuple(f"2021_{n}" for n in (
    6, 16, 17, 23, 26, 37, 42, 45, 49, 51, 61, 63, 64, 69, 74))
EXPOSED_TOPICS_CLOSED: Tuple[str, ...] = tuple(f"2021_{n}" for n in (
    6, 15, 16, 17, 21, 23, 25, 26, 27, 33, 37, 40, 42, 45, 49, 51, 57, 59, 61, 63, 64, 69, 74))
#: §2.6: topics whose MeSH sets have no condition descriptor.
CONDITION_SIGNAL_ABSENT = ("2021_53", "2021_67", "2022_1", "2022_10", "2022_15", "2022_17", "2022_31")


def cosine_edges(vectors: Mapping[str, np.ndarray], threshold: float,
                 audit_low: float = 0.70, near_tol: float = 1e-6) -> Dict[str, Any]:
    """§3 list (2): pairs with float64 L2-normalised cosine >= threshold.

    Returns ``edges`` (``[a, b, cos]``), ``audit`` pairs in [audit_low, threshold) and
    ``near`` pairs within ``near_tol`` of the threshold (build must stop on any).
    """
    topics = sort_topics(vectors)
    mat = np.stack([np.asarray(vectors[t], dtype=np.float64) for t in topics])
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    cos = mat @ mat.T
    edges, audit, near = [], [], []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            c = float(cos[i, j])
            row = [topics[i], topics[j], round(c, 6)]
            if abs(c - threshold) <= near_tol:
                near.append(row)
            if c >= threshold:
                edges.append(row)
            elif c >= audit_low:
                audit.append(row)
    return {"edges": edges, "audit": audit, "near": near}


def connected_components(nodes: Iterable[str], edges: Iterable[Sequence[str]]) -> List[List[str]]:
    """Components (each sorted by topic key), ordered by their smallest topic key."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    for n in parent:
        groups[find(n)].append(n)
    comps = [sort_topics(g) for g in groups.values()]
    return sorted(comps, key=lambda c: topic_key(c[0]))


def exposure_closure(exposed: Iterable[str], components: Iterable[Sequence[str]]) -> List[str]:
    """§0 ``exposed_closed``: union of components containing an exposed topic."""
    exposed = set(exposed)
    out: Set[str] = set()
    for comp in components:
        if exposed & set(comp):
            out.update(comp)
    return sort_topics(out | exposed)


def group_lookup(components: Iterable[Sequence[str]]) -> Dict[str, int]:
    """Topic -> group id for multi-topic components (others are singletons)."""
    out: Dict[str, int] = {}
    for gid, comp in enumerate(components):
        if len(comp) > 1:
            for t in comp:
                out[t] = gid
    return out


def assign_folds(components: Sequence[Sequence[str]], n_folds: int = 5, seed: int = 2021) -> List[List[str]]:
    """§3 fold assignment: groups ordered by smallest topic key, ``random.Random(seed)
    .shuffle``, then greedy to the fold with fewest topics (ties -> lowest index)."""
    groups = sorted((list(c) for c in components), key=lambda c: topic_key(min(c, key=topic_key)))
    random.Random(seed).shuffle(groups)
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for g in groups:
        k = min(range(n_folds), key=lambda i: (len(folds[i]), i))
        folds[k].extend(g)
    return [sort_topics(f) for f in folds]


def outer_split(k: int, n_folds: int = 5) -> Dict[str, Any]:
    val = (k + 1) % n_folds
    return {"k": k, "test": k, "validation": val,
            "train": [i for i in range(n_folds) if i not in (k, val)]}


# ---------------------------------------------------------------------------- fold files
def _shuffled(rows: List[Any], seed: str) -> List[Any]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def build_fold_files(topics: Mapping[str, Dict[str, Any]], trials: Mapping[str, Dict[str, Any]],
                     graded: Mapping[str, Mapping[int, Sequence[str]]], folds: Sequence[Sequence[str]],
                     k: int, out_dir: os.PathLike | str, components: Sequence[Sequence[str]] = (),
                     val_cap: int = 60) -> Dict[str, Any]:
    """Write ``fold_{k}`` files (§3 "Fold files") and return its manifest.

    ``graded[topic][grade]`` = qrel nct ids. Records come from
    ``trials_domain.data_splitter._build_record`` / ``_ontology_slot``.
    """
    from trials_domain.data_splitter import _build_record, _ontology_slot

    split = outer_split(k, len(folds))
    test_topics = sort_topics(folds[split["test"]])
    val_topics = sort_topics(folds[split["validation"]])
    train_topics = sort_topics(t for i in split["train"] for t in folds[i])
    sets = {"train": set(train_topics), "validation": set(val_topics), "test": set(test_topics)}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sets[a] & sets[b]:
            raise ProtocolError(f"fold {k}: {a}/{b} topic overlap {sorted(sets[a] & sets[b])}")
    fold_of = {t: i for i, f in enumerate(folds) for t in f}
    for comp in components:
        if len({fold_of[t] for t in comp if t in fold_of}) > 1:
            raise ProtocolError(f"paraphrase group {list(comp)} split across folds")

    skipped: Dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    def rows_for(split_name: str, split_topics: Sequence[str], grades: Sequence[int], cap: int) -> List[Dict[str, Any]]:
        rows = []
        for tid in split_topics:
            topic = topics[tid]
            anchor = _ontology_slot(topic, "topic_id", tid)
            for grade in grades:
                ncts = sorted(graded.get(tid, {}).get(grade, []))
                if cap and grade == 0 and len(ncts) > cap:
                    ncts = sorted(random.Random(f"2021|validation|{tid}|0").sample(ncts, cap))
                for nct in ncts:
                    trial = trials.get(nct)
                    if trial is None:
                        skipped[split_name] += 1
                        continue
                    rows.append(_build_record(tid, topic, anchor, nct, trial, grade))
        return rows

    out = Path(absp(out_dir))
    out.mkdir(parents=True, exist_ok=False)
    files: Dict[str, List[Any]] = {}
    files["train"] = _shuffled(rows_for("train", train_topics, (2,), 0), f"2021|fold{k}|train")
    files["validation"] = _shuffled(rows_for("validation", val_topics, (2, 1, 0), val_cap), f"2021|fold{k}|validation")
    files["validation_positive"] = [r for r in files["validation"] if r["job"]["grade"] == 2]
    files["test"] = _shuffled(rows_for("test", test_topics, (2, 1, 0), 0), f"2021|fold{k}|test")
    pools = [{"topic_id": tid, "split": sp,
              "ineligible": sorted(graded.get(tid, {}).get(1, [])),
              "not_relevant": sorted(graded.get(tid, {}).get(0, []))}
             for sp, tl in (("train", train_topics), ("validation", val_topics)) for tid in tl]
    files["negative_pools"] = _shuffled(pools, f"2021|fold{k}|negative_pools")
    tr_pool = {p["topic_id"] for p in pools if p["split"] == "train"}
    va_pool = {p["topic_id"] for p in pools if p["split"] == "validation"}
    if tr_pool & va_pool:
        raise ProtocolError(f"fold {k}: train/validation pool overlap")
    hashes = {}
    for name, rows in files.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[f"{name}.jsonl"] = file_sha256(path)
    manifest = {"k": k, "outer": split,
                "topics": {"train": train_topics, "validation": val_topics, "test": test_topics},
                "records": {n: len(r) for n, r in files.items()},
                "skipped_missing_trials": skipped, "val_not_relevant_cap": val_cap,
                "sha256": hashes}
    (out / "fold_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------- loader
def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    with open(absp(path), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_embedding_dim(vectors: Iterable[np.ndarray], dim: int = EMBEDDING_DIM) -> None:
    for v in vectors:
        if np.asarray(v).shape != (dim,):
            raise ProtocolError(f"embedding width {np.asarray(v).shape} != ({dim},); hard abort (§2.1)")


def load_frozen_cache(path: os.PathLike | str = ORIGINAL_CACHE_REL) -> Dict[str, Any]:
    """Read-only load of the frozen text cache (``{"cache": {key: vec}}``)."""
    import torch
    payload = torch.load(absp(path), map_location="cpu", weights_only=False)
    return payload.get("cache", {})


def load_chunks(chunk_dir: os.PathLike | str) -> Dict[str, np.ndarray]:
    """``{text_sha256: vector}`` from epf-format ``chunk_*.npz`` files (read-only)."""
    out: Dict[str, np.ndarray] = {}
    d = absp(chunk_dir)
    if not d.is_dir():
        return out
    for path in sorted(d.glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            keys, vecs = data["keys"], data["embeddings"]
            if len(keys) != len(vecs):
                raise ProtocolError(f"malformed chunk {rel(path)}")
            for key, vec in zip(keys, vecs):
                out[str(key)] = np.asarray(vec, dtype=np.float32)
    return out


def resolve_mpnet_snapshot(expected: Optional[str]) -> Tuple[str, str]:
    """§2.1 snapshot check -> ``(revision, local_path)``; aborts on mismatch."""
    require_hf_offline()
    from huggingface_hub import snapshot_download
    path = snapshot_download(MPNET_REPO, local_files_only=True)
    revision = Path(path).name
    if expected is not None and revision != expected:
        raise ProtocolError(f"MPNet snapshot {revision} != pinned {expected}")
    return revision, path


def prepared_dir_name(path: os.PathLike | str, model: str = MPNET_REPO) -> str:
    p = absp(path)
    return f"{p.stem}_{file_sha256(p)[:16]}_{model.replace('/', '_')}"


def load_split(path: os.PathLike | str, frozen_cache: Mapping[str, Any], *,
               published_chunk_dir: Optional[os.PathLike | str] = None,
               fresh_encode_root: Optional[os.PathLike | str] = None,
               encoder_factory: Optional[Callable[[], Any]] = None,
               keyer: Optional[Any] = None,
               content_to_text: Optional[Callable[[Dict[str, Any], str], str]] = None,
               log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Read-only port of ``epf.prepare_split`` (§2.1), returning an ``epf``-compatible
    ``prepared`` dict plus per-record ``nct_ids``, ``topic_uris``, ``trial_uris`` and
    ``condition_signal`` arrays.

    Resolution order per text: frozen cache by content key; then (2022 files)
    ``published_chunk_dir`` by text sha256; then (2021 fold files) a fresh CPU encode
    via ``encoder_factory`` written in epf chunk format under ``fresh_encode_root``.
    Exactly one of ``published_chunk_dir`` / ``fresh_encode_root`` may be set; with
    the former a missing text is a hard abort and no encoder is constructed.
    """
    if published_chunk_dir is not None and fresh_encode_root is not None:
        raise ValueError("a split is either a 2022 (published chunks) or a 2021 (fresh encode) file")
    if keyer is None:
        import torch
        from contrastive_learning.embedding_cache import EmbeddingCache
        keyer = EmbeddingCache(max_cache_size=0, device=torch.device("cpu"), enable_stats=False)
    if content_to_text is None:
        epf()
        from run_phase1_embedding_evaluation import content_to_text as _ctt
        content_to_text = _ctt

    records = read_jsonl(path)
    texts_a, texts_b, grades, queries = [], [], [], []
    nct_ids, topic_uris, trial_uris, cond = [], [], [], []
    cached_by_text: Dict[str, np.ndarray] = {}
    for record in records:
        left, right = record.get("resume"), record.get("job")
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        meta = record.get("metadata") or {}
        grade = right.get("grade")
        if grade is None:
            grade = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}.get(meta.get("original_label"))
        if grade not in (0, 1, 2):
            continue
        query = meta.get("resume_id")
        if query is None:
            raise ProtocolError(f"record in {rel(absp(path))} has no metadata.resume_id")
        text_a, text_b = content_to_text(left, "resume"), content_to_text(right, "job")
        if not text_a.strip() or not text_b.strip():
            raise ProtocolError(f"blank encoder text in {rel(absp(path))}")
        for content, text_value in ((left, text_a), (right, text_b)):
            cached = frozen_cache.get(keyer.get_content_key(content))
            if cached is not None and text_value not in cached_by_text:
                if hasattr(cached, "detach"):
                    cached = cached.detach().cpu().numpy()
                cached_by_text[text_value] = np.asarray(cached, dtype=np.float32)
        texts_a.append(text_a)
        texts_b.append(text_b)
        grades.append(int(grade))
        queries.append(str(query))
        nct_ids.append(str(meta.get("nct_id") or right.get("nct_id")))
        topic_uris.append(list(left.get("skill_uris") or []))
        trial_uris.append(list(right.get("skill_uris") or []))
        cond.append(bool(meta.get("condition_signal_present", True)))

    unique = list(dict.fromkeys(texts_a + texts_b))
    hashes = {t: text_sha256(t) for t in unique}
    by_hash: Dict[str, np.ndarray] = {}
    fresh_dir = None
    if published_chunk_dir is not None:
        by_hash = load_chunks(published_chunk_dir)
    elif fresh_encode_root is not None:
        fresh_dir = absp(fresh_encode_root) / prepared_dir_name(path)
        by_hash = load_chunks(fresh_dir)  # resumable: chunks from an interrupted run
    missing = [t for t in unique if t not in cached_by_text and hashes[t] not in by_hash]
    log(f"{rel(absp(path))}: {len(unique)} unique texts, {len(unique) - len(missing)} resolved, "
        f"{len(missing)} missing")
    fresh: Dict[str, np.ndarray] = {}
    if missing:
        if fresh_dir is None or encoder_factory is None:
            raise ProtocolError(f"{len(missing)} texts of {rel(absp(path))} missing from the frozen "
                                "cache and the published chunks; 2022 texts are never re-encoded (§2.1)")
        encoder = encoder_factory()
        fresh_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing), 640):
            chunk = missing[start:start + 640]
            enc = np.asarray(encoder.encode(chunk, batch_size=64, convert_to_numpy=True,
                                            show_progress_bar=False), dtype=np.float32)
            keys = np.asarray([hashes[t] for t in chunk])
            cid = hashlib.sha256("|".join(str(k) for k in keys).encode("ascii")).hexdigest()[:16]
            np.savez(fresh_dir / f"chunk_{cid}.npz", keys=keys, embeddings=enc)
            fresh.update(zip(chunk, enc))
            log(f"  encoded {start + len(chunk)}/{len(missing)}")

    def vec(t: str) -> np.ndarray:
        if t in cached_by_text:
            return cached_by_text[t]
        if hashes[t] in by_hash:
            return by_hash[hashes[t]]
        return fresh[t]

    vectors = [vec(t) for t in unique]
    check_embedding_dim(vectors)
    index = {t: i for i, t in enumerate(unique)}
    sources = {"frozen_cache": sum(t in cached_by_text for t in unique),
               "chunks": sum(t not in cached_by_text and hashes[t] in by_hash for t in unique),
               "fresh_encoded": len(fresh)}
    return {
        "base": np.stack(vectors) if vectors else np.zeros((0, EMBEDDING_DIM), np.float32),
        "left_index": np.asarray([index[t] for t in texts_a], dtype=int),
        "right_index": np.asarray([index[t] for t in texts_b], dtype=int),
        "grades": np.asarray(grades, dtype=int),
        "queries": np.asarray(queries, dtype=object),
        "embedding_dim": EMBEDDING_DIM,
        "records_scored": len(grades),
        "source_records": len(records),
        "nct_ids": np.asarray(nct_ids, dtype=object),
        "topic_uris": topic_uris,
        "trial_uris": trial_uris,
        "condition_signal": np.asarray(cond, dtype=bool),
        "path": rel(absp(path)),
        "sha256": file_sha256(path),
        "embedding_sources": sources,
        "fresh_dir": rel(fresh_dir) if fresh_dir is not None else None,
    }


def zero_shot_scores(prepared: Mapping[str, Any]) -> np.ndarray:
    """Frozen-MPNet cosine (§3a; ``raw_cos`` in the exploration)."""
    b = np.asarray(prepared["base"], dtype=np.float32)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(bn[prepared["left_index"]] * bn[prepared["right_index"]], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------- phi
def build_matchers(config) -> Dict[str, Any]:
    """Hierarchy (production, §2.2) and exact (§2.3) matchers on the pinned index."""
    from trials_domain.mesh_ontology import MeshIndex, MeshMatcher
    from trials_domain.run_config import build_mesh_matcher
    hier = build_mesh_matcher(config)
    if hier.similarity_mode != "hierarchy":
        raise ProtocolError("config must leave mesh_similarity_mode at the 'hierarchy' default")
    index = MeshIndex.build_or_load(config.mesh_descriptor_path, config.mesh_index_cache)
    exact = MeshMatcher(index, alpha=getattr(config, "mesh_alpha", 0.5),
                        max_hops=getattr(config, "mesh_max_hops", 12), similarity_mode="exact")
    return {"hierarchy": hier, "exact": exact}


def phi_scores(matcher, topic_uris: Sequence[Sequence[str]], trial_uris: Sequence[Sequence[str]]) -> np.ndarray:
    out = np.empty(len(topic_uris), dtype=np.float64)
    for i, (a, b) in enumerate(zip(topic_uris, trial_uris)):
        try:
            out[i] = float(matcher.ontology_set_similarity(a, b)) if a and b else 0.0
        except Exception:
            out[i] = 0.0
    return out


def group_derangement(topics: Sequence[str], group_of: Mapping[str, Any], seed: int = 11,
                      max_draws: int = 10000) -> Tuple[Dict[str, str], int]:
    """§2.3 group-aware derangement for one file.

    ``topics`` are the file's distinct topics (sorted here by topic key). Returns
    ``({receiver: source}, draws)``: topic ``order[i]`` receives the topic-side URIs
    of topic ``order[(i+1) % n]``.
    """
    ts = sort_topics(topics)
    n = len(ts)
    groups = [group_of.get(t, ("singleton", t)) for t in ts]
    rng = np.random.default_rng(seed)
    for draw in range(1, max_draws + 1):
        order = rng.permutation(n)
        if all(groups[order[i]] != groups[order[(i + 1) % n]] for i in range(n)):
            return {ts[order[i]]: ts[order[(i + 1) % n]] for i in range(n)}, draw
    raise ProtocolError(f"group-aware derangement infeasible after {max_draws} draws; amend §9")


def shuffled_topic_uris(queries: Sequence[str], topic_uris: Sequence[Sequence[str]],
                        mapping: Mapping[str, str]) -> List[List[str]]:
    first: Dict[str, List[str]] = {}
    for q, u in zip(queries, topic_uris):
        first.setdefault(str(q), list(u))
    return [first[mapping[str(q)]] for q in queries]


# ---------------------------------------------------------------------------- fusion
WEIGHT_GRID = tuple(i / 10 for i in range(6))
GATE_WEIGHT_GRID = tuple(i / 10 for i in range(11))


def zstats(values: np.ndarray) -> Tuple[float, float]:
    return epf().z_stats(np.asarray(values))


def zapply(values: np.ndarray, stats: Tuple[float, float]) -> np.ndarray:
    return epf().z_apply(np.asarray(values), stats)


def within_patient_z(values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """§2.4 sensitivity: z within each patient's pool, SD 0 -> z = 0."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    queries = np.asarray(queries)
    for q in np.unique(queries):
        m = queries == q
        v = values[m]
        sd = float(np.std(v))
        out[m] = (v - float(np.mean(v))) / sd if sd > 0 else 0.0
    return out


def fuse(text_z: np.ndarray, phi_z: np.ndarray, w: float) -> np.ndarray:
    return (1 - w) * np.asarray(text_z) + w * np.asarray(phi_z)


def validation_ndcg(scores: np.ndarray, grades: np.ndarray, queries: np.ndarray) -> float:
    return float(epf().ranking_metrics(np.asarray(scores), np.asarray(grades), np.asarray(queries))["macro"]["ndcg@10"])


def select_weight(val_text_z: np.ndarray, val_phi_z: np.ndarray, grades: np.ndarray, queries: np.ndarray,
                  grid: Sequence[float] = WEIGHT_GRID) -> Tuple[float, Dict[str, float]]:
    """§2.5: macro nDCG@10 on validation; exact-float ties -> smaller w
    (``max`` over the ascending grid returns the first maximiser, as in ``epf.main``)."""
    grid = sorted(grid)
    curve = {str(w): validation_ndcg(fuse(val_text_z, val_phi_z, w), grades, queries) for w in grid}
    return max(grid, key=lambda w: curve[str(w)]), curve


# ---------------------------------------------------------------------------- metrics
METRICS = ("pooled", "hard", "g1g0", "ap", "ndcg10", "p10")
METRIC_LABEL = {"pooled": "H1 pooled AUC (2 vs 0,1)", "ap": "H2 AP (grade 2)", "ndcg10": "S1 nDCG@10",
                "p10": "S2 P@10", "g1g0": "S3 AUC 1 vs 0", "hard": "S4 hard AUC (2 vs 1)"}


def per_patient_metrics(scores: np.ndarray, grades: np.ndarray, queries: np.ndarray,
                        patients: Sequence[str]) -> Dict[str, np.ndarray]:
    """§2.6 per patient; NaN = undefined for that patient (same set for every score)."""
    E = epf()
    scores, grades, queries = np.asarray(scores, float), np.asarray(grades), np.asarray(queries).astype(str)
    rank = E.ranking_metrics(scores, grades, queries)
    rows = {r["query"]: r for r in rank["per_query"]}
    out = {m: np.full(len(patients), np.nan) for m in METRICS}
    for i, p in enumerate(patients):
        m = queries == p
        s, g = scores[m], grades[m]
        out["pooled"][i] = E.rank_auc(s[g == 2], s[g != 2])
        out["hard"][i] = E.rank_auc(s[g == 2], s[g == 1])
        out["g1g0"][i] = E.rank_auc(s[g == 1], s[g == 0])
        row = rows.get(p)
        if row is not None:
            out["ap"][i] = row["average_precision"]
            out["ndcg10"][i] = row["ndcg@10"]
            out["p10"][i] = row["precision@10"]
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ---------------------------------------------------------------------------- bootstrap
def draw_indices(rng: np.random.Generator, n_models: int, n_patients: int, B: int) -> Tuple[np.ndarray, np.ndarray]:
    """§4.1 one replicate = c* (n_models with replacement) and p* (n_patients), independently.
    Drawn once per cohort and reused for every metric/variant/contrast (common random numbers)."""
    C = rng.integers(0, n_models, size=(B, n_models))
    P = rng.integers(0, n_patients, size=(B, n_patients))
    return C, P


def draw_stratified_models(rng: np.random.Generator, strata: Sequence[Any], B: int) -> np.ndarray:
    """Fraction-stratified c* (resample seeds within each fraction)."""
    strata = list(strata)
    C = np.empty((B, len(strata)), dtype=int)
    groups = collections.OrderedDict()
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    for b in range(B):
        col = []
        for members in groups.values():
            col.extend(np.asarray(members)[rng.integers(0, len(members), len(members))])
        C[b] = col
    return C


def point_stat(D: np.ndarray) -> float:
    """Mean over defined patients of the mean over models of ``D[c, p]``."""
    D = np.asarray(D, float)
    with np.errstate(invalid="ignore"):
        per_p = D.mean(axis=0)
    per_p = per_p[np.isfinite(per_p)]
    return float(per_p.mean()) if len(per_p) else float("nan")


def boot_stats(D: np.ndarray, C: np.ndarray, P: np.ndarray) -> np.ndarray:
    D = np.asarray(D, float)
    M = D[C].mean(axis=1)                        # (B, n_p): mean over c* per patient
    T = np.take_along_axis(M, P, axis=1)        # (B, n_p*)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(T, axis=1)


def boot_p(T: np.ndarray) -> float:
    """§4.1 two-sided bootstrap p."""
    T = np.asarray(T)[np.isfinite(T)]
    Bn = len(T)
    lo = (1 + np.sum(T <= 0)) / (Bn + 1)
    hi = (1 + np.sum(T >= 0)) / (Bn + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def ci(T: np.ndarray, level: float) -> Tuple[float, float]:
    T = np.asarray(T)[np.isfinite(T)]
    a = (1 - level) / 2 * 100
    return float(np.percentile(T, a)), float(np.percentile(T, 100 - a))


def summarize(D: np.ndarray, C: np.ndarray, P: np.ndarray) -> Dict[str, Any]:
    T = boot_stats(D, C, P)
    with np.errstate(invalid="ignore"):
        defined = int(np.isfinite(np.asarray(D, float).mean(axis=0)).sum())
    return {"point": point_stat(D), "p": boot_p(T), "ci95": ci(T, 0.95), "ci975": ci(T, 0.975),
            "ci90": ci(T, 0.90), "sd": float(np.nanstd(T)), "n_defined": defined, "_T": T}


# ---------------------------------------------------------------------------- decisions
def holm(p1: float, p2: float, point1: float, point2: float, alpha: float = 0.05,
         names: Tuple[str, str] = ("H1", "H2")) -> Dict[str, Any]:
    """§4.3 Holm on bootstrap p; ties in p -> names[0] first; positive iff rejected and point > 0."""
    ps = {names[0]: p1, names[1]: p2}
    pts = {names[0]: point1, names[1]: point2}
    order = sorted(names, key=lambda n: (ps[n], names.index(n)))
    first, second = order
    rej = {first: ps[first] <= alpha / 2, second: False}
    if rej[first]:
        rej[second] = ps[second] <= alpha
    companion = {first: 0.975, second: 0.95 if rej[first] else 0.975}
    out = {"order": order, "step3_reached": bool(rej[first]), "companion_level": companion}
    for n in names:
        out[n] = {"p": ps[n], "point": pts[n], "rejected": bool(rej[n]),
                  "positive": bool(rej[n] and pts[n] > 0), "harm": bool(rej[n] and pts[n] < 0),
                  "companion_level": companion[n]}
    return out


def equivalence_state(lo90: float, hi90: float, bound: float = 0.01) -> str:
    """§1 S4 with the 90% two-level CI."""
    if lo90 > -bound and hi90 < bound:
        return "equivalent"
    if lo90 >= bound or hi90 <= -bound:
        return "not equivalent"
    return "inconclusive"


def c2_state(c2_lb95: float, c2_ub95: float, diff_lb95: float, hier_point: float, c2_point: float,
             pass_ub: float = 0.005) -> str:
    """§1 C2 states, evaluated in order (fail, pass, inconclusive)."""
    if c2_lb95 > 0 or (diff_lb95 <= 0 and hier_point <= c2_point):
        return "fail"
    if c2_ub95 < pass_ub and diff_lb95 > 0:
        return "pass"
    return "inconclusive"


def final_label(positive_2022: bool, harm_2022: bool, positive_2021: bool, harm_2021: bool,
                c2_2022: str, c2_2021: str) -> Dict[str, Any]:
    """§4.3 ordered per-Hi final label (first match wins) with the inconclusive suffix."""
    raw_conf = bool(positive_2022)
    raw_rep = raw_conf and bool(positive_2021)
    if not raw_conf:
        return {"label_id": 1, "label": "significant harm" if harm_2022 else "not confirmed"}
    if c2_2022 == "fail":
        lid, label = 2, "not attributable to patient-specific MeSH information (2022)"
    elif not raw_rep:
        lid, label = 3, "confirmed, not replicated" + ("; significant harm in 2021" if harm_2021 else "")
    elif c2_2021 == "fail":
        lid, label = 4, "confirmed; 2021 replication not attributable to patient-specific MeSH information"
    else:
        lid, label = 5, "replicated"
    reached = ["2022"] + (["2021"] if lid in (4, 5) else [])
    states = {"2022": c2_2022, "2021": c2_2021}
    for cohort in reached:
        if states[cohort] == "inconclusive":
            label += f" (shuffled control inconclusive: {cohort})"
    return {"label_id": lid, "label": label}


def headline(labels: Mapping[str, Mapping[str, Any]]) -> str:
    """§4.3 headline from ``{"H1": label, "H2": label}``."""
    l1, l2 = labels["H1"], labels["H2"]
    if l1["label_id"] == 5 and l2["label_id"] == 5:
        return "MeSH fusion improves within-patient ranking, replicated: " \
               f"H1: {l1['label']}; H2: {l2['label']}"
    if (l1["label_id"] == 5) != (l2["label_id"] == 5):
        return ("MeSH fusion improves within-patient ranking, partially replicated: "
                f"H1: {l1['label']}; H2: {l2['label']}")
    return f"H1: {l1['label']}; H2: {l2['label']}"


# ---------------------------------------------------------------------------- gate 2
GATE2_TOLERANCE_ABS = {"continuous": 1e-6, "rank": 1e-4}
_SKIP_KEYS = {"binary_classification", "ordinal_classification"}
_RANK_KEYS = {"validation_selection_curve", "ranking", "ranking_macro", "per_query", "global_auc", "macro_query_auc"}
GATE2_LEAF_CLASSES = {
    "path_semantics": ("leaf paths are key tuples (never dot-joined), relative to one runs[i]; "
                       "per_query rows are paired by their 'query' field"),
    "order": ["skipped", "exact", "rank", "continuous"],
    "skipped": "binary_classification or ordinal_classification in P",
    "exact": ("P[-1] in {selected_weight, n_queries} or queries_with_both_classes in P or "
              "confusion_matrix in P or (P[-1] == n and tier_stats in P)"),
    "rank": ("any of {validation_selection_curve, ranking, ranking_macro, per_query, global_auc, "
             "macro_query_auc} in P, or P[-1] in {kendall_tau_b, spearman_rho}"),
    "continuous": "everything else (validation_stats, tier_stats mean/sd/median, cohens_d)",
    "identifier": "fraction, seed, checkpoint (row identity, compared exactly)",
}


def leaf_class(P: Tuple[str, ...]) -> str:
    """Gate-2 class of a leaf with key tuple ``P`` (design review round 7, finding 1).

    ``skipped`` is tested first: the threshold metrics are not computed by the scorer,
    and their ``confusion_matrix`` counts would otherwise match ``exact``.
    """
    P = tuple(str(x) for x in P)
    if _SKIP_KEYS & set(P):
        return "skipped"
    if len(P) == 1 and P[0] in ("fraction", "seed", "checkpoint"):
        return "identifier"
    if (P[-1] in ("selected_weight", "n_queries") or "queries_with_both_classes" in P
            or "confusion_matrix" in P or (P[-1] == "n" and "tier_stats" in P)):
        return "exact"
    if _RANK_KEYS & set(P) or P[-1] in ("kendall_tau_b", "spearman_rho"):
        return "rank"
    return "continuous"


def iter_leaves(obj: Any, prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Any]]:
    """Leaves as key tuples; ``per_query`` lists are keyed by each row's ``query``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, prefix + (str(k),))
    elif isinstance(obj, list):
        if prefix and prefix[-1] == "per_query":
            seen = set()
            for row in obj:
                q = str(row["query"])
                if q in seen:
                    raise ProtocolError(f"duplicate per_query row {q} at {prefix}")
                seen.add(q)
                yield from iter_leaves({k: v for k, v in row.items() if k != "query"}, prefix + (q,))
        else:
            for i, v in enumerate(obj):
                yield from iter_leaves(v, prefix + (f"[{i}]",))
    else:
        yield prefix, obj


def compare_gate2_run(published: Mapping[str, Any], computed: Mapping[str, Any],
                      tol: Mapping[str, float] = GATE2_TOLERANCE_ABS) -> Dict[str, Any]:
    """Compare one published ``runs[i]`` with the scorer's equivalent (§5.2)."""
    comp = dict(iter_leaves(computed))
    failures, skipped, missing = [], [], []
    max_by_class: Dict[str, float] = collections.defaultdict(float)
    max_by_group: Dict[str, float] = collections.defaultdict(float)
    n_compared = collections.Counter()
    for P, pv in iter_leaves(published):
        cls = leaf_class(P)
        if cls == "skipped":
            skipped.append(P)
            continue
        if P not in comp:
            missing.append(P)
            failures.append({"path": list(P), "class": cls, "reason": "not computed"})
            continue
        cv = comp[P]
        n_compared[cls] += 1
        group = ".".join(x for x in P[:3] if x not in ("macro",))
        if isinstance(pv, str) or isinstance(cv, str) or cls == "identifier":
            if pv != cv:
                failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv})
            continue
        if pv is None or cv is None:
            if not (pv is None and cv is None):
                failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv})
            continue
        diff = abs(float(pv) - float(cv))
        max_by_class[cls] = max(max_by_class[cls], diff)
        max_by_group[f"{cls}:{group}"] = max(max_by_group[f"{cls}:{group}"], diff)
        limit = 0.0 if cls == "exact" else tol[cls]
        if (cls == "exact" and float(pv) != float(cv)) or diff > limit:
            failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv, "abs_diff": diff})
    if computed.get("selected_weight") != published.get("selected_weight"):
        if not any(f["path"] == ["selected_weight"] for f in failures):
            failures.append({"path": ["selected_weight"], "class": "exact",
                             "published": published.get("selected_weight"),
                             "computed": computed.get("selected_weight")})
    return {"passed": not failures, "failures": failures, "missing": [list(p) for p in missing],
            "skipped": len(skipped), "n_compared": dict(n_compared),
            "max_abs_by_class": dict(max_by_class), "max_abs_by_group": dict(max_by_group)}


def json_safe(value):
    return epf().json_safe(value)


# ---------------------------------------------------------------------------- gate 3
def check_device_block(device: Optional[Mapping[str, Any]], gpu_name: Optional[str],
                       require: str = "cuda") -> List[str]:
    """§5.3 device check on ``lc_manifest.json["device"]``; returns failures."""
    if not device:
        return ["lc_manifest.json has no device block"]
    fails = []
    if device.get("requested") != require:
        fails.append(f"device.requested={device.get('requested')!r} != {require!r}")
    if device.get("resolved") != require:
        fails.append(f"device.resolved={device.get('resolved')!r} != {require!r}")
    if require == "cuda" and device.get("cuda_device_name") != gpu_name:
        fails.append(f"cuda_device_name={device.get('cuda_device_name')!r} != pinned {gpu_name!r}")
    return fails


def gate3_failures(out_dir: os.PathLike | str, exit_code: int, expected_unexpected: Sequence[str],
                   gpu_name: Optional[str], original_cache_ok: bool, require_device: str = "cuda") -> List[str]:
    out = absp(out_dir)
    fails = []
    if exit_code != 0:
        fails.append(f"exit code {exit_code}")
    manifest_path, ckpt = out / "lc_manifest.json", out / "best_checkpoint.pt"
    if not manifest_path.exists():
        fails.append("lc_manifest.json missing")
    if not ckpt.exists():
        fails.append("best_checkpoint.pt missing")
    if (out / "run_error.json").exists():
        fails.append("run_error.json present")
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        unexpected = ((m.get("single_factor_audit") or {}).get("unexpected_fields"))
        if unexpected != list(expected_unexpected):
            fails.append(f"single_factor_audit.unexpected_fields={unexpected} != {list(expected_unexpected)}")
        cc = (m.get("attach") or {}).get("cache_check") or {}
        if not (cc.get("expected_text_dim") == EMBEDDING_DIM and list(cc.get("dims", {})) == [f"({EMBEDDING_DIM},)"]):
            fails.append(f"attach.cache_check not passing: {cc}")
        fails.extend(check_device_block(m.get("device"), gpu_name, require_device))
    if not original_cache_ok:
        fails.append("embedding_cache/trials_text_embeddings.pt sha256 changed")
    return fails


# ---------------------------------------------------------------------------- protocol parser
CKPT_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9a-f]{64})\s*\|\s*$")
INPUT_ROW = re.compile(r"^\|\s*(\S+)[^|]*\|\s*([0-9a-f]{64})\s*\|\s*$")


def parse_protocol(text: str, min_ckpt: int = 15, min_inputs: int = 18) -> Dict[str, Any]:
    """§6.2 step -1 parser. Whole-line, anchored rows: a checkpoint row has exactly four
    cells whose last is 64-hex; an inputs row has exactly two cells (design review
    round 7, finding 2). Also extracts the manifest digest, the prepared-cache dirs and
    the MPNet revision."""
    ckpts, inputs = [], {}
    for line in text.splitlines():
        m = CKPT_ROW.match(line)
        if m:
            F, S, n, h = m.groups()
            ckpts.append({"fraction": int(F), "seed": int(S), "records": int(n), "sha256": h,
                          "path": PUBLISHED_CKPT_PATTERN.format(F=F, S=S)})
            continue
        m = INPUT_ROW.match(line)
        if m:
            inputs[m.group(1)] = m.group(2)
    if len(ckpts) != min_ckpt:
        raise ProtocolError(f"protocol parse found {len(ckpts)} checkpoint rows (need exactly {min_ckpt})")
    if len(inputs) < min_inputs:
        raise ProtocolError(f"protocol parse found {len(inputs)} input rows (need >= {min_inputs})")
    digest = re.search(r"manifest digest `([0-9a-f]{64})`", text)
    test_dir = re.search(r'"test_dir": "([^"]+)"', text)
    val_dir = re.search(r'"validation_dir": "([^"]+)"', text)
    rev = re.search(r"revision `([0-9a-f]{40})`", text)
    n_test = re.search(r"`test_[0-9a-f]{16}_…` directory has (\d+) chunks", text)
    if not (digest and test_dir and val_dir and rev and n_test):
        raise ProtocolError("protocol parse: manifest digest / prepared dirs / MPNet revision not found")
    return {"checkpoints": ckpts, "inputs": inputs, "manifest_digest": digest.group(1),
            "test_dir": test_dir.group(1), "validation_dir": val_dir.group(1),
            "n_test_chunks": int(n_test.group(1)), "mpnet_revision": rev.group(1)}


# ---------------------------------------------------------------------------- pins chain
PINS_WHITELIST_TOP = {"code", "git", "environment", "amendment", "supersedes", "revision", "created_utc"}


def _changed_paths(a: Any, b: Any, prefix: str, depth: int) -> Set[str]:
    if a == b:
        return set()
    if depth > 1 and isinstance(a, dict) and isinstance(b, dict):
        out: Set[str] = set()
        for k in set(a) | set(b):
            out |= _changed_paths(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{prefix}.{k}", depth - 1)
        return out
    return {prefix}


def changed_fields(prev: Mapping[str, Any], new: Mapping[str, Any]) -> Set[str]:
    """Changed non-whitelisted fields as dotted paths at depth 2 (``analysis.B``,
    ``cohort2022.new_runs``, ``inputs.<relpath>``); ``protocol.sha256`` is whitelisted."""
    out: Set[str] = set()
    for k in set(prev) | set(new):
        if k in PINS_WHITELIST_TOP or k == "configs":
            continue
        a, b = prev.get(k, "<absent>"), new.get(k, "<absent>")
        if k == "protocol" and isinstance(a, dict) and isinstance(b, dict):
            a = {x: y for x, y in a.items() if x != "sha256"}
            b = {x: y for x, y in b.items() if x != "sha256"}
        out |= _changed_paths(a, b, k, 2)
    return out


def verify_revision(prev: Mapping[str, Any], new: Mapping[str, Any], prev_path: str, prev_sha: str) -> None:
    """§7 rule for one link of the supersedes chain; raises ProtocolError."""
    if new.get("revision") != prev.get("revision", 1) + 1:
        raise ProtocolError(f"revision {new.get('revision')} does not follow {prev.get('revision')}")
    sup = new.get("supersedes") or {}
    if sup.get("path") != prev_path or sup.get("sha256") != prev_sha:
        raise ProtocolError(f"broken supersedes link: {sup} vs {prev_path} {prev_sha}")
    if new.get("configs") != prev.get("configs"):
        raise ProtocolError("configs changed between revisions; impossible by revision (§7)")
    am = new.get("amendment") or {}
    previous = am.get("previous") or {}
    if previous.get("git") != prev.get("git") or previous.get("environment") != prev.get("environment"):
        raise ProtocolError("amendment.previous must hold the previous git and environment")
    changed = changed_fields(prev, new)
    declared = set(am.get("changed_fields") or [])
    if changed != declared:
        raise ProtocolError(f"changed fields {sorted(changed)} != declared {sorted(declared)}")
    if changed and am.get("confirmatory_output_seen") is not False:
        raise ProtocolError("declared changes require confirmatory_output_seen == false")


def pins_revision_files(results_dir: os.PathLike | str) -> Dict[int, Path]:
    d = absp(results_dir)
    out = {}
    if (d / "pins.json").exists():
        out[1] = d / "pins.json"
    for p in d.glob("pins_r*.json"):
        m = re.fullmatch(r"pins_r(\d+)\.json", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def load_verified_pins(pins_path: os.PathLike | str, results_dir: Optional[os.PathLike | str] = None,
                       allow_non_latest: bool = False) -> Dict[str, Any]:
    """Load ``--pins``, verify the whole supersedes chain and that it is the latest revision."""
    path = absp(pins_path)
    results_dir = absp(results_dir) if results_dir is not None else path.parent
    pins = json.loads(path.read_text(encoding="utf-8"))
    revs = pins_revision_files(results_dir)
    if not allow_non_latest and revs and max(revs) != pins.get("revision"):
        raise ProtocolError(f"{rel(path)} is revision {pins.get('revision')}, latest present is {max(revs)}")
    cur, cur_path = pins, path
    while cur.get("supersedes"):
        prev_path = absp(cur["supersedes"]["path"])
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        verify_revision(prev, cur, rel(prev_path), file_sha256(prev_path))
        cur, cur_path = prev, prev_path
    # A dev_pins.json root is accepted only for a wholly non-confirmatory chain.
    dev_root = (cur_path.name == "dev_pins.json" and not pins.get("confirmatory")
                and not cur.get("confirmatory"))
    if cur.get("revision") != 1 or not (cur_path.name == "pins.json" or dev_root):
        raise ProtocolError(f"chain does not end at revision 1 pins.json ({rel(cur_path)})")
    return pins


def verify_pinned_hashes(pins: Mapping[str, Any], sections: Sequence[str] = ("inputs", "code", "chunks", "configs")) -> List[str]:
    """Re-hash every pinned file; returns mismatches (scripts refuse to run on any)."""
    bad = []
    if "inputs" in sections:
        for p, h in (pins.get("inputs") or {}).items():
            if p.endswith("@pin_time"):
                continue
            if not absp(p).exists() or file_sha256(p) != h:
                bad.append(f"inputs:{p}")
    if "code" in sections:
        for p, h in (pins.get("code") or {}).items():
            if not absp(p).exists() or file_sha256(p) != h:
                bad.append(f"code:{p}")
    if "chunks" in sections:
        pc = pins.get("published_prepared_cache") or {}
        for name, h in (pc.get("chunks") or {}).items():
            p = absp(pc["test_dir"]) / name
            if not p.exists() or file_sha256(p) != h:
                bad.append(f"chunk:{name}")
    if "configs" in sections:
        cfg = pins.get("configs") or {}
        base = cfg.get("base") or {}
        if base and file_sha256(base["path"]) != base["sha256"]:
            bad.append(f"configs:{base['path']}")
        for p, d in (cfg.get("derived") or {}).items():
            if not absp(p).exists() or file_sha256(p) != d["sha256"]:
                bad.append(f"configs:{p}")
    return bad


# ---------------------------------------------------------------------------- ledger
class Ledger:
    """Append-only ``run_ledger.json`` (§6.3).

    Serialisation: ``[\\n`` + one ``json.dumps(entry, sort_keys=True)`` per line joined by
    ``,\\n`` + ``\\n]\\n``. Each append rewrites atomically (temp + ``os.replace``) and
    asserts the previous entry lines are byte-identical in the new file.
    """

    def __init__(self, path: os.PathLike | str):
        self.path = absp(path)

    def entries(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _render(entries: Sequence[Mapping[str, Any]]) -> str:
        if not entries:
            return "[\n]\n"
        return "[\n" + ",\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n]\n"

    def append(self, entry: Mapping[str, Any]) -> None:
        old = self.entries()
        old_lines = [json.dumps(e, sort_keys=True) for e in old]
        text = self._render(list(old) + [dict(entry)])
        new_lines = text.splitlines()[1:-1]
        new_lines = [ln[:-1] if ln.endswith(",") else ln for ln in new_lines]
        if new_lines[:len(old_lines)] != old_lines:
            raise ProtocolError("ledger append would alter an existing entry")
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)

    def status(self, run_id: str) -> Dict[str, Any]:
        es = [e for e in self.entries() if e.get("run") == run_id]
        return {"passed": any(e.get("status") == "passed" for e in es),
                "excluded": any(e.get("status") == "excluded" for e in es),
                "failed": sum(e.get("status") == "failed" for e in es),
                "entries": es}


def run_with_resume(ledger: Ledger, run: Mapping[str, Any], attempt: Callable[[Mapping[str, Any], int], Dict[str, Any]],
                    max_failures: int = 2) -> str:
    """§6.3 resume semantics for one run (fixed-order walk calls this per run).

    ``attempt(run, attempt_no)`` trains once and returns a ledger entry dict with
    ``status`` "passed" or "failed" (it must not touch other entries). Returns the
    run's final status: "skipped-passed", "skipped-excluded", "passed" or "excluded".
    """
    rid, out = run["run"], absp(run["output_dir"])
    st = ledger.status(rid)
    if st["passed"]:
        return "skipped-passed"
    if st["excluded"]:
        return "skipped-excluded"
    failures = st["failed"]

    def rename_out() -> Optional[str]:
        """Rename a failed attempt's output dir to ``<dir>.failed{n}`` (n = failures)."""
        if not out.exists():
            return None
        target = out.with_name(f"{out.name}.failed{failures}")
        if target.exists():
            raise ProtocolError(f"{rel(target)} already exists")
        os.replace(out, target)
        return rel(target)

    if out.exists():
        failures += 1
        ledger.append({"run": rid, "order_index": run.get("order_index"), "status": "failed",
                       "reason": "output dir present without a passing ledger entry (interrupted attempt)",
                       "renamed_to": rename_out(), "failed_attempts": failures, "time_utc": utc_now()})
    while failures < max_failures:
        entry = dict(attempt(run, failures + 1))
        if entry.get("status") == "passed":
            entry["failed_attempts"] = failures
            ledger.append(entry)
            return "passed"
        failures += 1
        entry.update({"status": "failed", "failed_attempts": failures, "renamed_to": rename_out()})
        ledger.append(entry)
    ledger.append({"run": rid, "order_index": run.get("order_index"), "status": "excluded",
                   "failed_attempts": failures, "time_utc": utc_now()})
    return "excluded"


# ---------------------------------------------------------------------------- environment
def _gpu_probe() -> Dict[str, Any]:
    """Query the GPU in a child process with CUDA visible (pins are written with
    ``CUDA_VISIBLE_DEVICES=""``, which hides the device from this process)."""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    code = ("import json,torch;a=torch.cuda.is_available();"
            "print(json.dumps({'available':a,'gpu_name':torch.cuda.get_device_name(0) if a else None,"
            "'gpu_count':torch.cuda.device_count() if a else 0,'cuda_runtime':torch.version.cuda,"
            "'cudnn':(torch.backends.cudnn.version() if a else None)}))")
    try:
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        info = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception as exc:  # pragma: no cover - host specific
        info = {"available": False, "gpu_name": None, "gpu_count": 0, "error": repr(exc)}
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
        info["nvidia_driver"] = drv.stdout.strip().splitlines()[0] if drv.returncode == 0 and drv.stdout.strip() else None
    except Exception:
        info["nvidia_driver"] = None
    return info


def blas_info() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        parts = [f"{k}:{(deps.get(k) or {}).get('name')} {(deps.get(k) or {}).get('version')}" for k in ("blas", "lapack")]
        return "; ".join(parts)
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc!r})"


def environment_record(host_label: Optional[str] = None, mpnet_snapshot: Optional[str] = None,
                       probe_gpu: bool = True) -> Dict[str, Any]:
    import scipy
    import sentence_transformers
    import torch
    gpu = _gpu_probe() if probe_gpu else {"gpu_name": None, "gpu_count": 0}
    return {"host": host_label or socket.gethostname(), "hostname": socket.gethostname(),
            "platform": platform.platform(), "machine": platform.machine(),
            "gpu_name": gpu.get("gpu_name"), "gpu_count": gpu.get("gpu_count"),
            "nvidia_driver": gpu.get("nvidia_driver"), "cuda_runtime": gpu.get("cuda_runtime"),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda, "cudnn": gpu.get("cudnn"),
            "python": platform.python_version(), "sentence_transformers": sentence_transformers.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "blas": blas_info(),
            "torch_blas": "; ".join(ln.strip() for ln in torch.__config__.show().splitlines()
                                    if "BLAS" in ln or "LAPACK" in ln)[:500],
            "torch_mkl": bool(torch.backends.mkl.is_available()),
            "mpnet_snapshot": mpnet_snapshot,
            "device_policy": {"training": "cuda", "scoring": "cpu", "fresh_encode": "cpu", "analysis": "cpu"},
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}


def git_record() -> Dict[str, Any]:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain") or ""
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_paths": sorted(line[3:] for line in status.splitlines() if line.strip())}


#: Code pinned in ``pins.code`` (new study files plus every reused module).
CODE_GLOBS = ("trials_domain/fusion_confirm.py", "trials_domain/tests/test_fusion_confirm*.py",
              "scripts/trials_fusion_confirm/*.py", "scripts/trials_fusion_confirm/requirements-remote.txt",
              "scripts/evaluate_publication_fusion.py", "scripts/run_learning_curve_point.py",
              "run_phase1_embedding_evaluation.py", "trials_domain/*.py", "contrastive_learning/*.py")


def code_hashes() -> Dict[str, str]:
    files = sorted({p for g in CODE_GLOBS for p in ROOT.glob(g) if p.is_file()})
    return {rel(p): file_sha256(p) for p in files}


def compare_code_hashes(expected: Mapping[str, str], got: Mapping[str, str]) -> List[Dict[str, Any]]:
    """Step -1 code-tree check: Mac-side ``code_hashes()`` dump vs the transferred tree.
    Any missing, extra or differing file is a failure."""
    out = []
    for p in sorted(set(expected) | set(got)):
        if p not in got:
            out.append({"path": p, "reason": "missing on remote"})
        elif p not in expected:
            out.append({"path": p, "reason": "not in the Mac manifest"})
        elif expected[p] != got[p]:
            out.append({"path": p, "reason": "sha256 differs", "expected": expected[p], "sha256": got[p]})
    return out


# ---------------------------------------------------------------------------- analysis params
ANALYSIS_PARAMS: Dict[str, Any] = {
    "weight_grid": list(WEIGHT_GRID), "gate_weight_grid": "[i/10 for i in range(11)]",
    "select_on": "ndcg@10", "tie_break": "smaller_w",
    "normalisation": "validation_file_global_z", "sensitivity_normalisation": "within_patient_z_sd0_to_0",
    "phi_variants": ["hierarchy", "exact", "shuffled"], "per_variant_w_selection": True,
    "c3_fixed_w": 0.2, "B": 2000, "rng": {"2022": [31415, 2022], "2021": [31415, 2021]},
    "rng_fraction_stratified_2022": [31415, 2022, 1],
    "shuffle_seed": 11, "shuffle": "group_aware_cyclic_shift_rejection_per_file", "shuffle_max_draws": 10000,
    "embedding_dim": EMBEDDING_DIM, "manifest_digest_rule": "sha256 of sorted '<sha256>  <basename>\\n' lines",
    "training_order": (["2022:f25_s87", "2022:f25_s123", "2022:f100_s87", "2022:f100_s123"]
                       + [f"2021:k{k}_s{s}" for k in range(5) for s in (13, 21, 42)]),
    "sensitivity_unexposed_closed": True, "descriptive_unexposed60": True,
    "descriptive_unexposed_closed_minus_dev_smoke": True,
    "equivalence_bound": 0.01, "equivalence_ci": 0.90, "alpha": 0.05,
    "holm_family": ["H1", "H2"], "decision": "holm_p_and_positive_point",
    "c2_pass_ub": 0.005, "min_checkpoints_2022": 17, "min_complete_folds_2021": 4,
    "holm_tie_order": ["H1", "H2"], "final_label_rule": "ordered_5_label_v1",
    "gate2_tolerance_abs": dict(GATE2_TOLERANCE_ABS), "gate2_selected_weight": "exact",
    "gate2_leaf_classes": GATE2_LEAF_CLASSES,
    "gate3_gpu_name": "pins.environment.gpu_name (exact match)",
}
