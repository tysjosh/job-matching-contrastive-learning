"""GO/PPI inference-time fusion confirmation study: shared library.

Implements the preregistered protocol
``.agents/tasks/ontology-withholding-feasibility/protocol_go_ppi_fusion.md`` (DRAFT v4).
Section numbers (``§``) refer to that protocol; where this module and the protocol
differ, the protocol wins.

Primary fusion AZ-X (§2.6): within-anchor z of the text score and of φ = X (exact-term
Jaccard of pruned direct BP terms), w selected on validation by global pooled AUC over
W = {0.0, ..., 1.0}, ties to the smaller w. Secondary fusion GZ-P: validation-file global
z of text and φ = P (pruned simGIC), w by validation macro nDCG@10. Every control uses
the form of the fusion it controls (§1.5).

Contents:

* run-environment guards, repo-root handling, hashing and the chunk-manifest digest (§2.1);
* the protocol-table parser used by step −1 (§6.2);
* the read-only embedding loader (port of ``epf.prepare_split``: no encoder, no mkdir);
* φ variants X, P, U, UX, N, T, S, S_P, Bz (§2.2-2.4), the typed OBO parser and the
  is_a-only index (§2.2.1), n_PMID (§2.4) and the stratified derangement (§2.3, through
  ``go_ppi_fusion.final.derange_anchors``);
* AZ / GZ normalisers, fusion, 1-D and 2-D weight selection under both rules (§2.5-2.6);
* per-anchor metrics (§2.7), the two-level bootstrap, DE slope, Holm(3), TOST, the
  four-state label, the C2 / C5 states and the ordered final labels and headline (§4);
* cohort-A masked fractions, protein-disjoint CV folds, exposure sets and ``Bclean``,
  the HuRI cohort and the mention flags (§3, §4.5);
* the gate-2 leaf classifier / comparator, gate-3 device check, pins chain, run ledger,
  environment record and code hashes (§5-§7).

Reused without editing: ``go_ppi_fusion.fusion.within_query_z``,
``go_ppi_fusion.metrics.global_auc`` / ``QueryIndex``, ``go_ppi_fusion.final.derange_anchors``;
``scripts/evaluate_publication_fusion.py`` (``epf``, imported by path) for ``rank_auc``,
``ranking_metrics``, ``z_stats``, ``z_apply``, ``text_scores``; ``go_ppi_domain.go_ontology``,
``data_splitter`` and ``leak_pruning``. Generic helpers are copied from
``trials_domain/fusion_confirm.py`` (not edited, not imported, so this study's ROOT is the
only root in play).
"""

from __future__ import annotations

import collections
import datetime as _dt
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------- root and paths
#: Repo root. Override with ``GO_PPI_FUSION_ROOT`` (or a script's ``--root``); every path
#: written to configs, manifests, pins or outputs is relative to it.
ROOT = Path(os.environ.get("GO_PPI_FUSION_ROOT") or Path(__file__).resolve().parents[1]).resolve()

PROTOCOL_REL = ".agents/tasks/ontology-withholding-feasibility/protocol_go_ppi_fusion.md"
CODE_MANIFEST_REL = ".agents/tasks/ontology-withholding-feasibility/code_hashes_go_ppi_fusion.mac.json"
RESULTS_REL = "results/go_ppi_fusion_confirm"
DEV_REL = "results/go_ppi_fusion_confirm_dev"
DATA_REL = "preprocess/go_ppi_fusion_confirm"
BASE_CONFIG_REL = "config/lc_go_ppi_baseline.json"

PUB_SPLIT_DIR = "preprocess/go_ppi_splits"
MASKED_SPLIT_DIR = "preprocess/go_withholding/splits"
MASKED_CONVERTED_DIR = "preprocess/go_withholding/converted"
UNPRUNED_CONVERTED_DIR = "preprocess/go_ppi"
LC_DIR = "preprocess/go_ppi_lc"
PRUNED_INDEX_REL = f"{MASKED_CONVERTED_DIR}/go_index_P.pkl"
UNPRUNED_INDEX_REL = f"{UNPRUNED_CONVERTED_DIR}/go_index_P.pkl"
MASKED_PROTEINS_REL = f"{MASKED_CONVERTED_DIR}/proteins.jsonl"
UNPRUNED_PROTEINS_REL = f"{UNPRUNED_CONVERTED_DIR}/proteins.jsonl"
PAIRS_REL = f"{MASKED_CONVERTED_DIR}/pairs.jsonl"
MASK_LEXICON_REL = f"{MASKED_CONVERTED_DIR}/mask_lexicon.json"
BULK_DIR = "dataset/go_ppi/bulk"
OBO_REL = f"{BULK_DIR}/go-basic.obo"
GAF_REL = f"{BULK_DIR}/goa_human.gaf.gz"
ALIASES_REL = f"{BULK_DIR}/9606.protein.aliases.v12.0.txt.gz"
INFO_REL = f"{BULK_DIR}/9606.protein.info.v12.0.txt.gz"
LINKS_REL = f"{BULK_DIR}/9606.protein.links.detailed.v12.0.txt.gz"
HURI_REL = "dataset/go_ppi/independent/HuRI.tsv"

ORIGINAL_CACHE_REL = "embedding_cache/go_ppi_text_embeddings.pt"
MASKED_CACHE_REL = "embedding_cache/go_withholding_text_embeddings.pt"
CACHE_COPY_REL = f"{RESULTS_REL}/cache/go_withholding_text_embeddings.pt"
PUBLISHED_FUSION_REL = {
    "pooled_auc": "results/publication_phase1_fixedval/go_ppi_fusion_learning_curve.json",
    "ndcg@10": "results/publication_phase1_fixedval/go_ppi_fusion_learning_curve_ndcg_selected.json",
}
BEST_TEST_RESULTS_REL = "results/go_ppi_fusion_best/test_results.json"
PUBLISHED_CKPT_PATTERN = ("results/publication_phase1_fixedval/checkpoints/"
                          "go_ppi_baseline_f{F}_s{S}/best_checkpoint.pt")
PUBLISHED_PREP_PARENT = "results/publication_phase1_fixedval/prepared_embeddings/go_ppi"
MASKED_PREP_REL = f"{RESULTS_REL}/prepared_embeddings/go_ppi_masked"
ISA_INDEX_NAME = "go_index_P_isa_only.pkl"
MPNET_REPO = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
VIEW_SEP = " — "

#: Best-fusion variant labels reproduced by gate 2f (§5.2): primary and secondary fusion.
GATE2F_VARIANTS = {"AZ-X": "F3_P", "GZ-P": "PUB_ndcg_P"}
GATE2F_METRICS = {"hard": "auc_hard", "pooled": "auc_pooled", "easy": "auc_easy",
                  "ap": "average_precision", "ndcg10": "ndcg@10"}
#: §5.2 2a near-tie cells (published validation top-two gap < 1e-4, F§6 R7).
GATE2_NEAR_TIE_CELLS = (("ndcg@10", 10, 42), ("pooled_auc", 10, 42), ("pooled_auc", 75, 21))
GATE2_NEAR_TIE_GAP = 1e-4
#: §5.2 2a zero-shot published-file values (3 dp, F§5).
GATE2_ZERO_SHOT = {"hard": 0.786, "pooled": 0.838, "ndcg10": 0.772}


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


def show(path: os.PathLike | str) -> str:
    """Display form of a path for log lines (repo-relative when possible; never recorded)."""
    try:
        return rel(absp(path))
    except ValueError:
        return str(path)


def absp(path: os.PathLike | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_root(root: Optional[os.PathLike | str]) -> Path:
    """Re-point ROOT (scripts' ``--root``) and chdir there."""
    global ROOT
    if root is not None:
        ROOT = Path(root).resolve()
    os.chdir(ROOT)
    return ROOT


# ---------------------------------------------------------------------------- guards
class ProtocolError(RuntimeError):
    """A protocol rule was violated; the script must stop."""


def require_hashseed() -> str:
    """§6: every new script asserts ``PYTHONHASHSEED == "0"`` at start-up (simGIC sums
    iterate frozensets, whose order depends on string hashing)."""
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


def require_cpu(device: str, what: str = "scoring/analysis") -> None:
    """§6 device policy: everything except training runs on the CPU with CUDA hidden."""
    if device != "cpu":
        raise ProtocolError(f"--device {device!r}: {what} runs on the CPU by the protocol "
                            "device policy (training is the only CUDA step)")
    import torch
    if torch.cuda.is_available():
        raise ProtocolError(f'CUDA is visible; run {what} with CUDA_VISIBLE_DEVICES=""')


def refuse_existing(path: os.PathLike | str) -> Path:
    """§6.1 no-overwrite rule: the named leaf must not exist (parents may)."""
    p = absp(path)
    if p.exists():
        raise ProtocolError(f"{rel(p)} already exists; no-overwrite rule (§6.1), no --force")
    return p


def require_dev(path: os.PathLike | str) -> Path:
    """Non-confirmatory (Mac dev smoke, §6.5) outputs live under the dev dir or /tmp."""
    p = absp(path).resolve()
    dev = (ROOT / DEV_REL).resolve()
    tmp_roots = [Path("/tmp").resolve(), Path(os.environ.get("TMPDIR", "/tmp")).resolve()]
    if not (p == dev or str(p).startswith(str(dev) + os.sep)
            or any(str(p).startswith(str(t) + os.sep) for t in tmp_roots)):
        raise ProtocolError(f"non-confirmatory output {p} must be under {DEV_REL}/ or /tmp (§6.5)")
    return p


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------- reused modules
_EPF = None


def epf():
    """``scripts/evaluate_publication_fusion.py`` imported by path (§6.1, never edited)."""
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


def load_script_module(rel_path: str, name: str):
    """Import a protected script by path (``run_learning_curve_point``, ...)."""
    path = ROOT / rel_path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def json_safe(value):
    """Numpy scalars / arrays and non-finite floats to strict JSON values (as ``epf``)."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def write_json(path: os.PathLike | str, obj: Any, refuse: bool = True) -> Path:
    p = refuse_existing(path) if refuse else absp(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(obj), indent=2) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------- hashing
def file_sha256(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(absp(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def list_sha256(items: Iterable[str]) -> str:
    """sha256 of ``json.dumps(sorted(items))`` (used for pinned id lists such as Bclean)."""
    return text_sha256(json.dumps(sorted(items)))


def manifest_digest_from_hashes(hashes: Mapping[str, str]) -> str:
    """§2.1: sha256 of the UTF-8 text of sorted ``"<sha256>  <basename>\\n"`` lines."""
    lines = "".join(f"{hashes[name]}  {Path(name).name}\n"
                    for name in sorted(hashes, key=lambda n: Path(n).name))
    return text_sha256(lines)


def chunk_hashes(chunk_dir: os.PathLike | str) -> Dict[str, str]:
    d = absp(chunk_dir)
    return {p.name: file_sha256(p) for p in sorted(d.glob("chunk_*.npz"))}


def manifest_digest(chunk_dir: os.PathLike | str) -> str:
    return manifest_digest_from_hashes(chunk_hashes(chunk_dir))


# ---------------------------------------------------------------------------- jsonl
def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    with open(absp(path), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: os.PathLike | str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write rows as ``data_splitter._write_jsonl`` does; refuses an existing leaf."""
    p = refuse_existing(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return file_sha256(p)


def record_grade(record: Mapping[str, Any]) -> Optional[int]:
    grade = (record.get("job") or {}).get("grade")
    if grade is None:
        label = (record.get("metadata") or {}).get("original_label")
        grade = {"good_fit": 2, "potential_fit": 1, "no_fit": 0}.get(label)
    return grade


def record_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    m = record.get("metadata") or {}
    return str(m["resume_id"]), str(m["partner_id"])


# ---------------------------------------------------------------------------- protocol parser
CKPT_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9a-f]{64})\s*\|\s*$")
INPUT_ROW = re.compile(r"^\|\s*(\S+)[^|]*\|\s*([0-9a-f]{64})\s*\|\s*$")
_PREP_TEST = re.compile(r"`(" + re.escape(PUBLISHED_PREP_PARENT) + r"/test_[0-9a-f]{16}_[^`/]+)/` "
                        r"\((\d+) chunks, manifest digest `([0-9a-f]{64})`\)")
_PREP_VAL = re.compile(r"`…/(validation_[0-9a-f]{16}_[^`/]+)/` \((\d+) chunks, digest `([0-9a-f]{64})`\)")
_REVISION = re.compile(r"[Rr]evision `([0-9a-f]{40})`")


def parse_protocol(text: str, n_ckpt: int = 15, min_inputs: int = 30) -> Dict[str, Any]:
    """§6.2 step −1 parser: whole-line anchored rows. A checkpoint row is
    ``| F | S | <64-hex> |`` (§3b); an inputs row is ``| <path> | <64-hex> |`` with the
    path = first whitespace token of the first cell (§7). Also extracts the two published
    prepared-cache dirs with chunk counts and digests (§2.1) and the MPNet revision."""
    ckpts, inputs = [], {}
    for line in text.splitlines():
        m = CKPT_ROW.match(line)
        if m:
            F, S, h = m.groups()
            ckpts.append({"fraction": int(F), "seed": int(S), "sha256": h,
                          "path": PUBLISHED_CKPT_PATTERN.format(F=F, S=S)})
            continue
        m = INPUT_ROW.match(line)
        if m:
            inputs[m.group(1)] = m.group(2)
    if len(ckpts) != n_ckpt:
        raise ProtocolError(f"protocol parse found {len(ckpts)} checkpoint rows (need exactly {n_ckpt})")
    if len(inputs) < min_inputs:
        raise ProtocolError(f"protocol parse found {len(inputs)} input rows (need >= {min_inputs})")
    t, v, r = _PREP_TEST.search(text), _PREP_VAL.search(text), _REVISION.search(text)
    if not (t and v and r):
        raise ProtocolError("protocol parse: prepared-cache dirs / digests / MPNet revision not found")
    return {"checkpoints": ckpts, "inputs": inputs, "mpnet_revision": r.group(1),
            "prepared": {"test": {"dir": t.group(1), "n_chunks": int(t.group(2)), "digest": t.group(3)},
                         "validation": {"dir": f"{PUBLISHED_PREP_PARENT}/{v.group(1)}",
                                        "n_chunks": int(v.group(2)), "digest": v.group(3)}}}


def check_prepared_dir(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Chunk count and manifest digest of one published prepared-cache dir (§2.1)."""
    d = absp(spec["dir"])
    hashes = chunk_hashes(d) if d.is_dir() else {}
    digest = manifest_digest_from_hashes(hashes) if hashes else None
    return {"dir": spec["dir"], "n_chunks": len(hashes), "expected_n_chunks": spec["n_chunks"],
            "digest": digest, "expected_digest": spec["digest"], "chunks": hashes,
            "ok": len(hashes) == spec["n_chunks"] and digest == spec["digest"]}


# ---------------------------------------------------------------------------- loader
def check_embedding_dim(vectors: Iterable[np.ndarray], dim: int = EMBEDDING_DIM) -> None:
    for v in vectors:
        if np.asarray(v).shape != (dim,):
            raise ProtocolError(f"embedding width {np.asarray(v).shape} != ({dim},); hard abort (§2.1)")


def load_frozen_cache(path: os.PathLike | str = ORIGINAL_CACHE_REL) -> Dict[str, Any]:
    """Read-only load of a frozen text cache (``{"cache": {key: vec}}``)."""
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


def _content_to_text():
    epf()
    from run_phase1_embedding_evaluation import content_to_text
    return content_to_text


def load_prepared(path: os.PathLike | str, *, chunk_dir: os.PathLike | str,
                  frozen_cache: Optional[Mapping[str, Any]] = None, keyer: Any = None,
                  content_to_text: Optional[Callable[[Dict[str, Any], str], str]] = None,
                  log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Read-only port of ``epf.prepare_split`` (§2.1): an ``epf``-compatible prepared dict
    plus per-record anchors, partners and record URIs. Never constructs an encoder and
    never creates a directory.

    Resolution per text: the frozen cache by ``EmbeddingCache.get_content_key(slot)``
    (only when ``frozen_cache`` is given: published files), then ``chunk_dir`` by
    sha256(text). A text in neither source is a hard abort.
    """
    if content_to_text is None:
        content_to_text = _content_to_text()
    if frozen_cache and keyer is None:
        import torch
        from contrastive_learning.embedding_cache import EmbeddingCache
        keyer = EmbeddingCache(max_cache_size=0, device=torch.device("cpu"), enable_stats=False)
    records = read_jsonl(path)
    texts_a, texts_b, grades, queries, anchors, partners, ua, ub = [], [], [], [], [], [], [], []
    cached_by_text: Dict[str, np.ndarray] = {}
    for record in records:
        left, right = record.get("resume"), record.get("job")
        grade = record_grade(record)
        if not isinstance(left, dict) or not isinstance(right, dict) or grade not in (0, 1, 2):
            raise ProtocolError(f"unscorable record in {rel(absp(path))}; GO/PPI files have none")
        query = (record.get("metadata") or {}).get("resume_id")
        if query is None:
            raise ProtocolError(f"record in {rel(absp(path))} has no metadata.resume_id")
        ta, tb = content_to_text(left, "resume"), content_to_text(right, "job")
        if not ta.strip() or not tb.strip():
            raise ProtocolError(f"blank encoder text in {rel(absp(path))}")
        if frozen_cache:
            for content, tv in ((left, ta), (right, tb)):
                cached = frozen_cache.get(keyer.get_content_key(content))
                if cached is not None and tv not in cached_by_text:
                    if hasattr(cached, "detach"):
                        cached = cached.detach().cpu().numpy()
                    cached_by_text[tv] = np.asarray(cached, dtype=np.float32)
        texts_a.append(ta)
        texts_b.append(tb)
        grades.append(int(grade))
        queries.append(str(query))
        anchors.append(str(left.get("protein_id")))
        partners.append(str(right.get("partner_id")))
        ua.append(list(left.get("skill_uris") or []))
        ub.append(list(right.get("skill_uris") or []))
    unique = list(dict.fromkeys(texts_a + texts_b))
    hashes = {t: text_sha256(t) for t in unique}
    by_hash = load_chunks(chunk_dir)
    missing = [t for t in unique if t not in cached_by_text and hashes[t] not in by_hash]
    if missing:
        raise ProtocolError(f"{len(missing)} texts of {rel(absp(path))} are in neither the frozen cache "
                            f"nor {rel(absp(chunk_dir))}; scoring never encodes (§2.1)")
    vectors = [cached_by_text[t] if t in cached_by_text else by_hash[hashes[t]] for t in unique]
    check_embedding_dim(vectors)
    index = {t: i for i, t in enumerate(unique)}
    sources = {"frozen_cache": sum(t in cached_by_text for t in unique),
               "chunks": sum(t not in cached_by_text for t in unique)}
    log(f"{rel(absp(path))}: {len(records)} records, {len(unique)} unique texts {sources}")
    return {
        "base": np.stack(vectors) if vectors else np.zeros((0, EMBEDDING_DIM), np.float32),
        "left_index": np.asarray([index[t] for t in texts_a], dtype=int),
        "right_index": np.asarray([index[t] for t in texts_b], dtype=int),
        "grades": np.asarray(grades, dtype=int),
        "queries": np.asarray(queries, dtype=object),
        "embedding_dim": EMBEDDING_DIM,
        "records_scored": len(grades),
        "anchors": anchors, "partners": partners, "uris_a": ua, "uris_b": ub,
        "path": rel(absp(path)), "sha256": file_sha256(path), "embedding_sources": sources,
        "chunk_dir": rel(absp(chunk_dir)),
    }


def zero_shot_scores(prepared: Mapping[str, Any]) -> np.ndarray:
    """Frozen-MPNet cosine (§2.1 zero-shot text score)."""
    b = np.asarray(prepared["base"], dtype=np.float32)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(bn[prepared["left_index"]] * bn[prepared["right_index"]], axis=1).astype(np.float64)


def text_scores(prepared: Mapping[str, Any], checkpoint: os.PathLike | str, config) -> np.ndarray:
    """``epf.text_scores`` (CPU) as float64."""
    return np.asarray(epf().text_scores(prepared, absp(checkpoint), config), dtype=np.float64)


def load_base_config():
    from contrastive_learning.data_structures import TrainingConfig
    return TrainingConfig.from_json(str(absp(BASE_CONFIG_REL)))


def align_records(reference: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]],
                  what: str = "") -> None:
    """§5.2 2b: record-for-record alignment (same order, anchor, partner, grade)."""
    if len(reference) != len(other):
        raise ProtocolError(f"{what}: length mismatch {len(reference)} != {len(other)}")
    for i, (r, o) in enumerate(zip(reference, other)):
        kr = ((r.get("metadata") or {}).get("resume_id"), r["resume"].get("protein_id"),
              r["job"].get("partner_id"), record_grade(r))
        ko = ((o.get("metadata") or {}).get("resume_id"), o["resume"].get("protein_id"),
              o["job"].get("partner_id"), record_grade(o))
        if kr != ko:
            raise ProtocolError(f"{what}: misaligned at record {i}: {kr} != {ko}")


# ---------------------------------------------------------------------------- MPNet + encoding
def resolve_mpnet_snapshot(expected: Optional[str]) -> Tuple[str, str]:
    """§2.1 snapshot check -> ``(revision, local_path)``; the basename must be the revision."""
    require_hf_offline()
    from huggingface_hub import snapshot_download
    path = snapshot_download(MPNET_REPO, local_files_only=True)
    revision = Path(path).name
    if expected is not None and revision != expected:
        raise ProtocolError(f"MPNet snapshot {revision} != pinned {expected}")
    return revision, path


def cpu_encoder(snapshot_path: str):
    """``SentenceTransformer(<snapshot>, device="cpu")``: the device is never left to the library."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(snapshot_path, device="cpu")


def collect_masked_texts(paths: Sequence[os.PathLike | str],
                         content_to_text: Optional[Callable] = None) -> List[str]:
    """Distinct encoder texts of every record in ``paths`` (both slots), first-seen order."""
    if content_to_text is None:
        content_to_text = _content_to_text()
    out: Dict[str, None] = {}
    for path in paths:
        for r in read_jsonl(path):
            for slot, kind in ((r["resume"], "resume"), (r["job"], "job")):
                t = content_to_text(slot, kind)
                if not t.strip():
                    raise ProtocolError(f"blank encoder text in {rel(absp(path))}")
                out.setdefault(t, None)
    return list(out)


def encode_into(texts: Sequence[str], chunk_dir: os.PathLike | str, encoder, chunk: int = 640,
                log: Callable[[str], None] = print) -> Dict[str, int]:
    """Encode every text not yet in ``chunk_dir`` and write epf-format chunks keyed by
    sha256(text) (``batch_size=64``, ``convert_to_numpy=True``, no normalisation, §2.1).
    Resumable: texts already present are skipped."""
    d = absp(chunk_dir)
    d.mkdir(parents=True, exist_ok=True)
    have = load_chunks(d)
    todo = [t for t in dict.fromkeys(texts) if text_sha256(t) not in have]
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        enc = np.asarray(encoder.encode(list(part), batch_size=64, convert_to_numpy=True,
                                        show_progress_bar=False), dtype=np.float32)
        check_embedding_dim(enc)
        keys = np.asarray([text_sha256(t) for t in part])
        cid = hashlib.sha256("|".join(str(k) for k in keys).encode("ascii")).hexdigest()[:16]
        np.savez(d / f"chunk_{cid}.npz", keys=keys, embeddings=enc)
        log(f"  encoded {start + len(part)}/{len(todo)}")
    return {"texts": len(set(texts)), "already_present": len(set(texts)) - len(todo), "encoded": len(todo)}


# ---------------------------------------------------------------------------- φ: indexes and matchers
#: φ variants (§2.2): index, record URIs, GoMatcher mode. "pruned"/"unpruned" are never mixed.
PHI_VARIANTS = {
    "X": ("pruned", "pruned", "exact"),
    "P": ("pruned", "pruned", "simgic"),
    "U": ("unpruned", "unpruned", "simgic"),
    "UX": ("unpruned", "unpruned", "exact"),
    "N": ("pruned", "pruned", "ancestor"),
    "T": ("isa", "pruned", "simgic"),
    "S": ("pruned", "shuffled", "exact"),
    "S_P": ("pruned", "shuffled", "simgic"),
}
PHI_ORDER = ("X", "P", "U", "UX", "N", "T", "S", "S_P", "Bz")


def load_index(path: os.PathLike | str):
    """``GoIndex.load`` only (never ``build_or_load``, which can rebuild and write, F§1.1)."""
    from go_ppi_domain.go_ontology import GoIndex
    return GoIndex.load(absp(path))


def make_matcher(index, mode: str):
    from go_ppi_domain.go_ontology import GoMatcher
    return GoMatcher(index, similarity_mode=mode)


def phi_array(matcher, A: Sequence[Sequence[str]], B: Sequence[Sequence[str]]) -> np.ndarray:
    """φ per record exactly as the published scorer: empty side -> 0.0; errors -> 0.0."""
    out = np.empty(len(A), dtype=np.float64)
    for i, (a, b) in enumerate(zip(A, B)):
        try:
            out[i] = float(matcher.ontology_set_similarity(a, b)) if a and b else 0.0
        except Exception:
            out[i] = 0.0
    return out


def build_matchers(pruned_index, unpruned_index, isa_index=None) -> Dict[str, Any]:
    """One matcher per φ variant (S / S_P share X / P's index and mode)."""
    out = {"X": make_matcher(pruned_index, "exact"), "P": make_matcher(pruned_index, "simgic"),
           "N": make_matcher(pruned_index, "ancestor"),
           "U": make_matcher(unpruned_index, "simgic"), "UX": make_matcher(unpruned_index, "exact")}
    out["S"], out["S_P"] = out["X"], out["P"]
    if isa_index is not None:
        out["T"] = make_matcher(isa_index, "simgic")
    return out


def compute_phis(f: Mapping[str, Any], matchers: Mapping[str, Any], n_pmid: Mapping[str, int],
                 variants: Sequence[str] = PHI_ORDER) -> Dict[str, np.ndarray]:
    """All φ arrays for one scored file. ``f`` carries ``uris_a``/``uris_b`` (pruned),
    ``uris_a_u``/``uris_b_u`` (unpruned), ``uris_a_shuf`` (pruned, deranged anchor side)
    and ``partners``. Variants without a matcher (T when gate 2c failed) are skipped."""
    out: Dict[str, np.ndarray] = {}
    for v in variants:
        if v == "Bz":
            out["Bz"] = bz_scores(f["partners"], n_pmid)
            continue
        if v not in matchers:
            continue
        _, uris, _ = PHI_VARIANTS[v]
        A = {"pruned": f["uris_a"], "unpruned": f["uris_a_u"], "shuffled": f.get("uris_a_shuf")}[uris]
        B = f["uris_b_u"] if uris == "unpruned" else f["uris_b"]
        if A is None:
            continue
        out[v] = phi_array(matchers[v], A, B)
    return out


# ---------------------------------------------------------------------------- typed parser (§2.2.1)
TYPED_RELATIONS = (("is_a",), ("is_a", "part_of"))


def parse_obo_typed(path: os.PathLike | str, relations: Sequence[str]):
    """``go_ontology.parse_obo`` line rules with a ``relations`` switch.

    ``("is_a", "part_of")`` reproduces ``parse_obo`` exactly; ``("is_a",)`` drops the
    ``relationship: part_of`` edges. Returns (parents, namespace, alt_to_main, name).
    """
    relations = tuple(relations)
    if relations not in TYPED_RELATIONS:
        raise ValueError(f"relations must be one of {TYPED_RELATIONS}, got {relations}")
    use_part_of = "part_of" in relations
    parents: Dict[str, Set[str]] = collections.defaultdict(set)
    namespace: Dict[str, str] = {}
    alt_to_main: Dict[str, str] = {}
    name: Dict[str, str] = {}
    state = {"cur": None, "ns": None, "name": None, "parents": set(), "alts": set(), "obsolete": False}

    def flush() -> None:
        cur = state["cur"]
        if cur and not state["obsolete"]:
            parents[cur] |= state["parents"]
            if state["ns"]:
                namespace[cur] = state["ns"]
            if state["name"]:
                name[cur] = state["name"]
            for a in state["alts"]:
                alt_to_main[a] = cur

    in_term = False
    with open(absp(path), "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("["):
                flush()
                in_term = line == "[Term]"
                state.update(cur=None, ns=None, name=None, parents=set(), alts=set(), obsolete=False)
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: GO:"):
                state["cur"] = line[4:].strip()
            elif line.startswith("namespace: "):
                state["ns"] = line[11:].strip()
            elif line.startswith("name: "):
                state["name"] = line[6:].strip()
            elif line.startswith("alt_id: GO:"):
                state["alts"].add(line[8:].strip())
            elif line.startswith("is_a: GO:"):
                state["parents"].add(line[6:].split("!")[0].strip())
            elif use_part_of and line.startswith("relationship: part_of GO:"):
                state["parents"].add(line[22:].split("!")[0].strip())
            elif line.startswith("is_obsolete: true"):
                state["obsolete"] = True
    flush()
    return parents, namespace, alt_to_main, name


def build_typed_index(base_index, obo_path: os.PathLike | str, relations: Sequence[str], aspect: str = "P"):
    """In-memory ``GoIndex`` over the typed closure: ``direct`` from ``base_index``
    (unchanged), ancestors / depth by ``GoIndex.build``'s rules, IC recomputed over genes
    with a non-empty closure (§2.2.1)."""
    from go_ppi_domain.go_ontology import ASPECT_NAMESPACE, GoIndex
    parents, namespace, _alt, _name = parse_obo_typed(obo_path, relations)
    ns = ASPECT_NAMESPACE[aspect]
    keep = {t for t, v in namespace.items() if v == ns}
    ancestors: Dict[str, frozenset] = {}
    for t in keep:
        acc, stack = {t}, [t]
        while stack:
            n = stack.pop()
            for p in parents.get(n, ()):
                if p in keep and p not in acc:
                    acc.add(p)
                    stack.append(p)
        ancestors[t] = frozenset(acc)
    roots = {t for t, a in ancestors.items() if len(a) == 1}
    children: Dict[str, Set[str]] = collections.defaultdict(set)
    for t in keep:
        for p in parents.get(t, ()):
            if p in keep:
                children[p].add(t)
    depth: Dict[str, int] = {r: 0 for r in roots}
    frontier = list(roots)
    while frontier:
        nxt = []
        for n in frontier:
            for ch in children.get(n, ()):
                if ch not in depth:
                    depth[ch] = depth[n] + 1
                    nxt.append(ch)
        frontier = nxt
    coarse_max = int(base_index.meta.get("coarse_max_depth", 3))
    direct, closure, coarse = {}, {}, {}
    for gene, terms in base_index.direct.items():
        acc: Set[str] = set()
        for t in terms:
            acc |= ancestors.get(t, frozenset())
        if not acc:
            continue
        direct[gene] = frozenset(terms)
        closure[gene] = frozenset(acc)
        coarse[gene] = frozenset(x for x in acc if depth.get(x, 99) <= coarse_max)
    freq: Dict[str, int] = collections.defaultdict(int)
    for acc in closure.values():
        for t in acc:
            freq[t] += 1
    total = len(closure)
    ic = {t: -math.log(n / total) for t, n in freq.items() if n > 0}
    meta = dict(base_index.meta, relations=list(relations),
                typed_from="go_ppi_domain.fusion_confirm.build_typed_index", genes_annotated=total)
    return GoIndex(ancestors, depth, direct, closure, coarse, ic, meta)


def typed_check(base_index, obo_path: os.PathLike | str) -> Dict[str, Any]:
    """Gate 2c: with ``("is_a", "part_of")`` the typed code must reproduce the base index's
    ``ancestors`` for every BP term and ``ic`` for every term exactly (float equality)."""
    full = build_typed_index(base_index, obo_path, ("is_a", "part_of"))
    anc_bad = sorted(t for t in set(base_index.ancestors) | set(full.ancestors)
                     if base_index.ancestors.get(t) != full.ancestors.get(t))
    ic_bad = sorted(t for t in set(base_index.ic) | set(full.ic) if base_index.ic.get(t) != full.ic.get(t))
    return {"passed": not anc_bad and not ic_bad, "n_terms": len(base_index.ancestors),
            "n_ic_terms": len(base_index.ic), "ancestor_mismatches": len(anc_bad),
            "ic_mismatches": len(ic_bad), "examples": (anc_bad + ic_bad)[:10]}


# ---------------------------------------------------------------------------- n_PMID, Bz (§2.4)
def parse_n_pmid(gaf_path: os.PathLike | str) -> Dict[str, int]:
    """Distinct ``PMID:`` tokens in column 6 (``|``-separated) per column-3 symbol over all
    lines, all aspects and evidence codes, excluding lines whose qualifier contains NOT."""
    refs: Dict[str, Set[str]] = collections.defaultdict(set)
    p = absp(gaf_path)
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            if "NOT" in c[3]:
                continue
            for tok in c[5].split("|"):
                tok = tok.strip()
                if tok.startswith("PMID:"):
                    refs[c[2]].add(tok)
            refs.setdefault(c[2], set())
    return {g: len(v) for g, v in refs.items()}


def bz_scores(partners: Sequence[str], n_pmid: Mapping[str, int]) -> np.ndarray:
    """Bz(a, b) = log(1 + n_PMID(b)); a protein with no GAF line gets 0."""
    return np.log1p(np.asarray([float(n_pmid.get(p, 0)) for p in partners], dtype=np.float64))


# ---------------------------------------------------------------------------- shuffle (§2.3)
def shuffle_map(queries: Sequence[str], anchors: Sequence[str], partners: Sequence[str],
                uris_a: Sequence[Sequence[str]], uris_b: Sequence[Sequence[str]], simgic_matcher,
                seed: int = 11) -> Dict[str, Any]:
    """§2.3 decile-stratified, candidate-excluding anchor-side derangement for one file.

    Delegates to ``go_ppi_fusion.final.derange_anchors`` (units = anchors with a non-empty
    pruned set; strata = 10 contiguous closure-size deciles via ``np.array_split``; rules
    (a)-(e); one ``default_rng(seed)`` per file consumed stratum by stratum; 10,000-draw cap
    per stratum). Returns ``{"mapping": {receiver: donor}, "strata": [...], ...}``.
    """
    from go_ppi_fusion.data import Split
    from go_ppi_fusion.final import derange_anchors
    n = len(queries)
    split = Split(name="scored", grades=np.zeros(n, dtype=int),
                  queries=np.asarray(list(queries), dtype=object), anchors=list(anchors),
                  partners=list(partners), uris_a=[list(u) for u in uris_a],
                  uris_b=[list(u) for u in uris_b], pruned_uris_a=[list(u) for u in uris_a],
                  pruned_uris_b=[list(u) for u in uris_b])
    try:
        return derange_anchors(split, simgic_matcher, seed=seed)
    except RuntimeError as exc:
        raise ProtocolError(f"{exc}; resolve by a pre-pins amendment of §2.3") from None


def apply_shuffle(queries: Sequence[str], uris_a: Sequence[Sequence[str]],
                  mapping: Mapping[str, str]) -> List[List[str]]:
    """Every record of a receiver gets the donor anchor's pruned anchor-side set; anchors
    outside the map (empty-set anchors) keep theirs. Partner sides are untouched."""
    own: Dict[str, List[str]] = {}
    for q, u in zip(queries, uris_a):
        own.setdefault(str(q), sorted(u))
    out = []
    for q, u in zip(queries, uris_a):
        d = mapping.get(str(q))
        out.append(list(own[d]) if d is not None else list(u))
    return out


# ---------------------------------------------------------------------------- fusion (§2.5-2.6)
WEIGHT_GRID = tuple(i / 10 for i in range(11))
C5_GRID = tuple((wg, wb) for wg in WEIGHT_GRID for wb in WEIGHT_GRID if wg + wb <= 1 + 1e-12)
RULES = ("global_pooled_auc", "ndcg@10")


class AnchorIndex:
    """Per-anchor record indices (anchors in ``np.unique`` order, records in file order)
    plus the ``go_ppi_fusion`` ``QueryIndex`` used by ``within_query_z``."""

    def __init__(self, queries: Sequence[str], grades: Sequence[int]):
        self.queries = np.asarray(list(queries), dtype=object)
        self.grades = np.asarray(grades, dtype=int)
        self.anchors, inverse = np.unique(self.queries, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        bounds = np.flatnonzero(np.r_[True, inverse[order][1:] != inverse[order][:-1], True])
        self.idx = [order[s:e] for s, e in zip(bounds[:-1], bounds[1:])]
        self.inverse = inverse
        self.n = len(self.anchors)
        self._qi = None

    @property
    def qi(self):
        if self._qi is None:
            from go_ppi_fusion.metrics import QueryIndex
            self._qi = QueryIndex(self.queries, np.zeros(len(self.queries), dtype=int))
        return self._qi


def anchor_z(values: np.ndarray, ai: AnchorIndex) -> np.ndarray:
    """§2.5 AZ: per-anchor mean / population SD over the scored file, SD 0 -> 1
    (constant -> 0). This is ``go_ppi_fusion.fusion.within_query_z``."""
    from go_ppi_fusion.fusion import within_query_z
    return within_query_z(np.asarray(values, dtype=np.float64), ai.qi)


def gz_stats(values: np.ndarray) -> Tuple[float, float]:
    """§2.5 GZ: validation-file mean / population SD, SD 0 -> 1 (``epf.z_stats``)."""
    return epf().z_stats(np.asarray(values, dtype=np.float64))


def gz_apply(values: np.ndarray, stats: Sequence[float]) -> np.ndarray:
    return epf().z_apply(np.asarray(values, dtype=np.float64), tuple(stats))


def fuse(text_z: np.ndarray, phi_z: np.ndarray, w: float) -> np.ndarray:
    return (1 - w) * np.asarray(text_z) + w * np.asarray(phi_z)


def fuse2(text_z: np.ndarray, go_z: np.ndarray, bz_z: np.ndarray, wg: float, wb: float) -> np.ndarray:
    return (1 - wg - wb) * np.asarray(text_z) + wg * np.asarray(go_z) + wb * np.asarray(bz_z)


def global_pooled_auc(scores: np.ndarray, grades: np.ndarray) -> float:
    """Tie-aware rank AUC of grade 2 vs {0, 1} over all records (``go_ppi_fusion.metrics.global_auc``)."""
    from go_ppi_fusion.metrics import global_auc
    return float(global_auc(np.asarray(scores, dtype=np.float64), np.asarray(grades)))


def selection_value(scores: np.ndarray, grades: np.ndarray, queries: np.ndarray, rule: str) -> float:
    if rule == "global_pooled_auc":
        return global_pooled_auc(scores, grades)
    if rule == "ndcg@10":
        return float(epf().ranking_metrics(np.asarray(scores, dtype=np.float64), np.asarray(grades),
                                           np.asarray(queries, dtype=object))["macro"]["ndcg@10"])
    raise ValueError(f"unknown selection rule {rule!r}")


def top_two_gap(values: Iterable[float]) -> float:
    v = sorted((float(x) for x in values), reverse=True)
    return float(v[0] - v[1]) if len(v) > 1 else float("inf")


def select_1d(val_text_z: np.ndarray, val_phi_z: np.ndarray, grades, queries, rule: str,
              grid: Sequence[float] = WEIGHT_GRID) -> Dict[str, Any]:
    """§2.6: validation curve over ascending W; exact-float ties -> smaller w (first maximiser,
    as ``epf.main``'s ``max``)."""
    grid = sorted(grid)
    curve = {str(w): selection_value(fuse(val_text_z, val_phi_z, w), grades, queries, rule) for w in grid}
    w = max(grid, key=lambda x: curve[str(x)])
    return {"w": float(w), "curve": curve, "top2_gap": top_two_gap(curve.values()), "rule": rule}


def select_2d(val_text_z, val_go_z, val_bz_z, grades, queries, rule: str = "global_pooled_auc",
              grid: Sequence[Tuple[float, float]] = C5_GRID) -> Dict[str, Any]:
    """§2.6 C5 grid (66 points, w_g + w_b <= 1); ties -> smaller w_g, then smaller w_b."""
    pts = sorted(grid)
    curve = {f"{wg}|{wb}": selection_value(fuse2(val_text_z, val_go_z, val_bz_z, wg, wb), grades, queries, rule)
             for wg, wb in pts}
    best = max(pts, key=lambda p: curve[f"{p[0]}|{p[1]}"])
    return {"w": [float(best[0]), float(best[1])], "curve": curve, "top2_gap": top_two_gap(curve.values()),
            "rule": rule}


#: Fused arms (§1.5, §1.2b, §2.6, §4.5). ``phi`` names a φ variant (or two for the 2-D C5
#: arm); ``norm`` is "az" or "gz"; ``rule`` the validation selection rule; ``fixed_w`` a
#: fixed weight instead of selection.
ARMS: Dict[str, Dict[str, Any]] = {
    "AZ-X": {"phi": "X", "norm": "az", "rule": "global_pooled_auc", "role": "primary fusion"},
    "AZ-P": {"phi": "P", "norm": "az", "rule": "global_pooled_auc", "role": "C1 (P - X)"},
    "AZ-N": {"phi": "N", "norm": "az", "rule": "global_pooled_auc", "role": "C1a ladder"},
    "AZ-T": {"phi": "T", "norm": "az", "rule": "global_pooled_auc", "role": "C1t (P - T)"},
    "AZ-S": {"phi": "S", "norm": "az", "rule": "global_pooled_auc", "role": "C2 (shuffled)"},
    "AZ-UX": {"phi": "UX", "norm": "az", "rule": "global_pooled_auc", "role": "C3 (UX - X)"},
    "AZ-Bz": {"phi": "Bz", "norm": "az", "rule": "global_pooled_auc", "role": "C4; C5 comparator"},
    "AZ-X+Bz": {"phi": ("X", "Bz"), "norm": "az", "rule": "global_pooled_auc", "role": "C5 (2-D)"},
    "AZ-X@0.4": {"phi": "X", "norm": "az", "fixed_w": 0.4, "role": "C1 fixed-w (ii)"},
    "AZ-P@0.4": {"phi": "P", "norm": "az", "fixed_w": 0.4, "role": "C1 fixed-w (ii)"},
    "AZ-X|ndcg": {"phi": "X", "norm": "az", "rule": "ndcg@10", "role": "weight-rule sensitivity"},
    "GZ-P": {"phi": "P", "norm": "gz", "rule": "ndcg@10", "role": "secondary fusion"},
    "GZ-U": {"phi": "U", "norm": "gz", "rule": "ndcg@10", "role": "GZ-P C3 (U - P); published form"},
    "GZ-S_P": {"phi": "S_P", "norm": "gz", "rule": "ndcg@10", "role": "GZ-P C2 (descriptive)"},
    "GZ-P|pooled": {"phi": "P", "norm": "gz", "rule": "global_pooled_auc", "role": "weight-rule sensitivity"},
    "GZ-X": {"phi": "X", "norm": "gz", "rule": "global_pooled_auc", "role": "normaliser sensitivity"},
    "GZ-U|pooled": {"phi": "U", "norm": "gz", "rule": "global_pooled_auc", "role": "A-pub published pooled form"},
}
#: Paired contrasts as linear combinations of per-anchor Δ (each Δ is fused − the same text).
CONTRASTS: Dict[str, Dict[str, int]] = {
    "X-S": {"AZ-X": 1, "AZ-S": -1},                       # C2 pair-specific component
    "C1:P-X": {"AZ-P": 1, "AZ-X": -1},
    "C1a:P-N": {"AZ-P": 1, "AZ-N": -1},
    "C1a:N-X": {"AZ-N": 1, "AZ-X": -1},
    "C1t:P-T": {"AZ-P": 1, "AZ-T": -1},
    "C1ii:P-X@0.4": {"AZ-P@0.4": 1, "AZ-X@0.4": -1},
    "C3:UX-X": {"AZ-UX": 1, "AZ-X": -1},
    "C5": {"AZ-X+Bz": 1, "AZ-Bz": -1},                     # Δ_C5 = (text+X+Bz) − (text+Bz)
    "X-C5": {"AZ-X": 1, "AZ-X+Bz": -1, "AZ-Bz": 1},         # Δ_X − Δ_C5 (absorbed share)
    "AZX-GZP": {"AZ-X": 1, "GZ-P": -1},                    # head-to-head
    "GZ:P-S_P": {"GZ-P": 1, "GZ-S_P": -1},
    "GZ:U-P": {"GZ-U": 1, "GZ-P": -1},
}
COHORT_ARMS = tuple(a for a in ARMS if a != "GZ-U|pooled")
APUB_ARMS = ("AZ-X", "AZ-UX", "GZ-P", "GZ-U", "GZ-U|pooled")


def arm_phis(arm: str) -> Tuple[str, ...]:
    p = ARMS[arm]["phi"]
    return tuple(p) if isinstance(p, (tuple, list)) else (p,)


def arm_available(arm: str, phis: Mapping[str, Any]) -> bool:
    return all(p in phis for p in arm_phis(arm))


def fit_arm(arm: str, val: Mapping[str, Any]) -> Dict[str, Any]:
    """Select one arm's weight on validation. ``val`` = {"text", "phi": {...}, "grades",
    "queries", "ai"}. GZ arms record their validation z-stats (text and φ)."""
    spec = ARMS[arm]
    names = arm_phis(arm)
    state: Dict[str, Any] = {"arm": arm, "norm": spec["norm"]}
    if spec["norm"] == "az":
        tz = anchor_z(val["text"], val["ai"])
        pz = [anchor_z(val["phi"][n], val["ai"]) for n in names]
    else:
        ts = gz_stats(val["text"])
        ps = [gz_stats(val["phi"][n]) for n in names]
        state["text_stats"], state["phi_stats"] = list(ts), [list(s) for s in ps]
        tz = gz_apply(val["text"], ts)
        pz = [gz_apply(val["phi"][n], s) for n, s in zip(names, ps)]
    if "fixed_w" in spec:
        state.update(w=float(spec["fixed_w"]), curve=None, top2_gap=None, rule="fixed")
    elif len(names) == 2:
        state.update(select_2d(tz, pz[0], pz[1], val["grades"], val["queries"], spec["rule"]))
    else:
        state.update(select_1d(tz, pz[0], val["grades"], val["queries"], spec["rule"]))
    return state


def apply_arm(arm: str, state: Mapping[str, Any], tst: Mapping[str, Any]) -> np.ndarray:
    """Fused scores on a scored file with the validation-selected weight (and, for GZ,
    the validation z-stats). AZ needs no stats: z is recomputed within each anchor."""
    names = arm_phis(arm)
    if ARMS[arm]["norm"] == "az":
        tz = anchor_z(tst["text"], tst["ai"])
        pz = [anchor_z(tst["phi"][n], tst["ai"]) for n in names]
    else:
        tz = gz_apply(tst["text"], state["text_stats"])
        pz = [gz_apply(tst["phi"][n], s) for n, s in zip(names, state["phi_stats"])]
    if len(names) == 2:
        wg, wb = state["w"]
        return fuse2(tz, pz[0], pz[1], wg, wb)
    return fuse(tz, pz[0], state["w"])


# ---------------------------------------------------------------------------- per-anchor metrics (§2.7)
METRICS = ("pooled", "hard", "easy", "g1g0", "ap", "rr", "ndcg10", "p10")
HYP_METRIC = {"H1": "pooled", "H2": "ap", "H3": "hard"}
SECONDARY_METRIC = {"S1": "ndcg10", "S2": "easy", "S3": "g1g0", "S4": "rr", "S5": "p10"}
METRIC_LABEL = {"pooled": "pooled AUC (2 vs 0,1)", "ap": "AP (grade 2)", "hard": "hard AUC (2 vs 1)",
                "ndcg10": "nDCG@10", "easy": "easy AUC (2 vs 0)", "g1g0": "AUC 1 vs 0", "rr": "MRR (grade 2)",
                "p10": "P@10 (descriptive)"}


def anchor_metrics(scores: np.ndarray, ai: AnchorIndex) -> Dict[str, np.ndarray]:
    """Per anchor over all its records: tie-aware AUCs by ``epf.rank_auc``; AP, RR, nDCG@10
    and P@10 by ``epf.ranking_metrics`` on the anchor's records (file order). NaN = undefined
    (the undefined set depends only on grades, so pairing holds)."""
    E = epf()
    s_all = np.asarray(scores, dtype=np.float64)
    out = {m: np.full(ai.n, np.nan) for m in METRICS}
    for i, idx in enumerate(ai.idx):
        s, g = s_all[idx], ai.grades[idx]
        out["pooled"][i] = E.rank_auc(s[g == 2], s[g != 2])
        out["hard"][i] = E.rank_auc(s[g == 2], s[g == 1])
        out["easy"][i] = E.rank_auc(s[g == 2], s[g == 0])
        out["g1g0"][i] = E.rank_auc(s[g == 1], s[g == 0])
        row = E.ranking_metrics(s, g, np.full(len(idx), ai.anchors[i], dtype=object))["per_query"][0]
        out["ap"][i] = row["average_precision"]
        out["rr"][i] = row["reciprocal_rank"]
        out["ndcg10"][i] = row["ndcg@10"]
        out["p10"][i] = row["precision@10"]
    return out


# ---------------------------------------------------------------------------- bootstrap (§4.1)
B_DEFAULT = 2000
RNG = {"A": [31415, 1], "Bclean": [31415, 2], "huri": [31415, 3], "all_B": [31415, 4]}
HOLM_LEVELS = (1 - 0.05 / 3, 1 - 0.05 / 2, 1 - 0.05)


def draw_indices(rng: np.random.Generator, n_models: int, n_anchors: int, B: int) -> Tuple[np.ndarray, np.ndarray]:
    """One replicate = c* (n_models with replacement) and a* (n_anchors), independently."""
    C = rng.integers(0, n_models, size=(B, n_models))
    P = rng.integers(0, n_anchors, size=(B, n_anchors))
    return C, P


def draw_stratified_models(rng: np.random.Generator, strata: Sequence[Any], B: int) -> np.ndarray:
    """Fraction-stratified c* (seeds resampled within each fraction)."""
    strata = list(strata)
    groups: Dict[Any, List[int]] = collections.OrderedDict()
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    C = np.empty((B, len(strata)), dtype=int)
    for b in range(B):
        col: List[int] = []
        for members in groups.values():
            col.extend(np.asarray(members)[rng.integers(0, len(members), len(members))])
        C[b] = col
    return C


def _counts(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n), dtype=np.float64)
    for b in range(idx.shape[0]):
        out[b] = np.bincount(idx[b], minlength=n)
    return out


class Draws:
    """Common random numbers for one cohort (§4.1): drawn once and reused for every
    metric, φ variant, contrast and model type.

    Stream order of ``default_rng(seed)``: c* (B × n_models), a* (B × n_anchors), then, when
    ``strata`` is given, the fraction-stratified c*, then ``extra_models`` (B × n) for the
    descriptive A-pub arm. Zero-shot uses ``zero_shot()`` (same a*, one model)."""

    def __init__(self, seed: Sequence[int], n_models: int, n_anchors: int, B: int = B_DEFAULT,
                 strata: Optional[Sequence[Any]] = None, extra_models: Optional[int] = None):
        rng = np.random.default_rng(list(seed))
        self.seed, self.B, self.n_models, self.n_anchors = list(seed), B, n_models, n_anchors
        self.C, self.P = draw_indices(rng, max(n_models, 1), n_anchors, B)
        self.Cs = draw_stratified_models(rng, strata, B) if strata is not None else None
        self.C_extra = rng.integers(0, extra_models, size=(B, extra_models)) if extra_models else None
        self.anchor_counts = _counts(self.P, n_anchors)
        self.model_weights = _counts(self.C, max(n_models, 1)) / max(n_models, 1)

    def zero_shot(self) -> "Draws":
        z = object.__new__(Draws)
        z.__dict__.update(self.__dict__)
        z.n_models, z.C = 1, np.zeros((self.B, 1), dtype=int)
        z.model_weights = np.ones((self.B, 1))
        z.Cs = z.C_extra = None
        return z

    def with_models(self, C: np.ndarray, n_total: Optional[int] = None) -> "Draws":
        """Same a*, other c* draws. ``C`` holds indices into a model axis of length
        ``n_total`` (default ``C.shape[1]``); each replicate averages its ``C.shape[1]`` draws."""
        n_total = C.shape[1] if n_total is None else n_total
        z = object.__new__(Draws)
        z.__dict__.update(self.__dict__)
        z.C, z.n_models = C, n_total
        z.model_weights = _counts(C, n_total) / C.shape[1]
        return z

    def stratified(self) -> "Draws":
        if self.Cs is None:
            raise ValueError("no stratified draws")
        return self.with_models(self.Cs, self.n_models)


def _defined(D: np.ndarray) -> np.ndarray:
    fin = np.isfinite(D)
    defined = fin.all(axis=0)
    if not np.array_equal(defined, fin.any(axis=0)):
        raise ProtocolError("undefined-anchor set differs across models (pairing broken, §2.7)")
    return defined


def boot_p(T: np.ndarray) -> float:
    """§4.1 two-sided p = min(1, 2 min((1+#{T*<=0})/(B+1), (1+#{T*>=0})/(B+1)))."""
    T = np.asarray(T)[np.isfinite(T)]
    Bn = len(T)
    lo = (1 + np.sum(T <= 0)) / (Bn + 1)
    hi = (1 + np.sum(T >= 0)) / (Bn + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def ci(T: np.ndarray, level: float) -> Tuple[float, float]:
    T = np.asarray(T)[np.isfinite(T)]
    if not len(T):
        return float("nan"), float("nan")
    a = (1 - level) / 2 * 100
    return float(np.percentile(T, a)), float(np.percentile(T, 100 - a))


def summarize(D: np.ndarray, draws: Draws, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Two-level bootstrap of the mean over defined anchors of the mean over models of D[c, a].

    ``D`` (n_models, n_anchors), NaN = undefined. ``mask`` restricts the anchor population
    (subset sensitivities reuse the cohort's a* draws, so contrasts stay paired)."""
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    if D.shape != (draws.model_weights.shape[1], draws.n_anchors):
        raise ValueError(f"D shape {D.shape} != draws ({draws.model_weights.shape[1]}, {draws.n_anchors})")
    keep = _defined(D)
    if mask is not None:
        keep = keep & np.asarray(mask, dtype=bool)
    Dz = np.where(keep[None, :], D, 0.0)
    M = draws.model_weights @ Dz                      # (B, n_a) mean over c*
    cnt = draws.anchor_counts * keep[None, :]          # a* multiplicities, defined only
    with np.errstate(invalid="ignore", divide="ignore"):
        T = (cnt * M).sum(axis=1) / cnt.sum(axis=1)
    point = float(D[:, keep].mean(axis=0).mean()) if keep.any() else float("nan")
    out = {"point": point, "p": boot_p(T), "ci95": ci(T, 0.95), "ci90": ci(T, 0.90),
           "ci975": ci(T, 0.975), "ci9833": ci(T, HOLM_LEVELS[0]),
           "sd": float(np.nanstd(T)), "n_defined": int(keep.sum()), "_T": T}
    with np.errstate(invalid="ignore"):
        per_model = np.nanmean(np.where(keep[None, :], D, np.nan), axis=1) if keep.any() else np.full(D.shape[0], np.nan)
    out["per_model"] = [float(x) for x in per_model]
    out["n_models_positive"] = int(np.sum(per_model > 0))
    return out


def summarize_T(T: np.ndarray, point: float, n: Any = None) -> Dict[str, Any]:
    """Summary of an already-combined replicate vector (pooled / heterogeneity contrasts)."""
    return {"point": float(point), "p": boot_p(T), "ci95": ci(T, 0.95), "ci90": ci(T, 0.90),
            "sd": float(np.nanstd(T)), "n_defined": n, "_T": np.asarray(T)}


def de_slope(D: np.ndarray, fractions: Sequence[int], draws: Draws,
             mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """§1.3 β_DE: OLS slope of the per-checkpoint means over a* of D[c, a] on
    log10(F/100), with c* drawn stratified by fraction within each replicate."""
    if draws.Cs is None:
        raise ValueError("DE slope needs fraction-stratified c* draws")
    D = np.asarray(D, dtype=np.float64)
    keep = _defined(D)
    if mask is not None:
        keep = keep & np.asarray(mask, dtype=bool)
    x = np.log10(np.asarray(fractions, dtype=np.float64) / 100.0)
    Dz = np.where(keep[None, :], D, 0.0)
    cnt = draws.anchor_counts * keep[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        Mc = (cnt @ Dz.T) / cnt.sum(axis=1, keepdims=True)          # (B, n_c)
    Y = np.take_along_axis(Mc, draws.Cs, axis=1)
    X = x[draws.Cs]
    Xc = X - X.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = (Xc * (Y - Y.mean(axis=1, keepdims=True))).sum(axis=1) / (Xc ** 2).sum(axis=1)
    y0 = D[:, keep].mean(axis=1)
    xc = x - x.mean()
    point = float((xc * (y0 - y0.mean())).sum() / (xc ** 2).sum())
    return {"point": point, "p": boot_p(T), "ci95": ci(T, 0.95), "ci90": ci(T, 0.90),
            "sd": float(np.nanstd(T)), "n_defined": int(keep.sum()), "_T": T,
            "per_checkpoint_mean": [float(v) for v in y0], "log10_fraction": [float(v) for v in x]}


# ---------------------------------------------------------------------------- decisions (§4.2-4.4)
def holm3(ps: Mapping[str, float], points: Mapping[str, float], alpha: float = 0.05,
          names: Sequence[str] = ("H1", "H2", "H3")) -> Dict[str, Any]:
    """§4.3 Holm over three hypotheses on bootstrap p; ties in p ordered as ``names``.
    Positive iff rejected and point > 0; a rejected negative point is "significant harm".
    Companion levels 98.33% / 97.5% / 95% by Holm step reached; a step not reached gives
    the remaining hypotheses the level of the last reached step."""
    names = list(names)
    order = sorted(names, key=lambda n: (ps[n], names.index(n)))
    thresholds = (alpha / 3, alpha / 2, alpha)
    levels = (1 - alpha / 3, 1 - alpha / 2, 1 - alpha)
    rejected = {n: False for n in names}
    reached = 1
    for i, n in enumerate(order):
        if ps[n] <= thresholds[i]:
            rejected[n] = True
            reached = min(3, i + 2)
        else:
            break
    out: Dict[str, Any] = {"order": order, "steps_reached": reached}
    for i, n in enumerate(order):
        lvl = levels[i] if i < reached else levels[reached - 1]
        out[n] = {"p": float(ps[n]), "point": float(points[n]), "rejected": rejected[n],
                  "positive": bool(rejected[n] and points[n] > 0),
                  "harm": bool(rejected[n] and points[n] < 0), "step": i + 1, "companion_level": lvl}
    return out


def tost_state(lo90: float, hi90: float, bound: float = 0.01) -> str:
    """§4.2 with the 90% two-level CI: equivalent / not equivalent / inconclusive."""
    if lo90 > -bound and hi90 < bound:
        return "equivalent"
    if lo90 >= bound or hi90 <= -bound:
        return "not equivalent"
    return "inconclusive"


def four_state(lo95: float, hi95: float, lo90: float, hi90: float, bound: float = 0.01) -> str:
    """§4.2 four-state label, first match wins: positive, negative, equivalent, inconclusive."""
    if lo95 > 0:
        return "positive"
    if hi95 < 0:
        return "negative"
    if tost_state(lo90, hi90, bound) == "equivalent":
        return "equivalent"
    return "inconclusive"


def four_state_of(s: Mapping[str, Any], bound: float = 0.01) -> str:
    return four_state(s["ci95"][0], s["ci95"][1], s["ci90"][0], s["ci90"][1], bound)


def c2_state(dx_point: float, ds_point: float, ds_lo95: float, ds_hi95: float,
             diff_lo95: float, diff_hi95: float, pass_ub: float = 0.005,
             fail_pairspecific_ub: float = 0.005) -> str:
    """§4.3 ``go_pair_specific_v4`` (diff = Δ_X − Δ_S), first match wins:
    fail (branch 1: LB(X−S) ≤ 0 and [pt(Δ_X) ≤ pt(Δ_S) or LB(Δ_S) > 0]; branch 2:
    UB(X−S) < 0.005 and LB(Δ_S) > 0), pass (UB(Δ_S) < 0.005 and LB(X−S) > 0),
    partial (LB(Δ_S) > 0 and LB(X−S) > 0), inconclusive."""
    if (diff_lo95 <= 0 and (dx_point <= ds_point or ds_lo95 > 0)) or \
            (diff_hi95 < fail_pairspecific_ub and ds_lo95 > 0):
        return "fail"
    if ds_hi95 < pass_ub and diff_lo95 > 0:
        return "pass"
    if ds_lo95 > 0 and diff_lo95 > 0:
        return "partial"
    return "inconclusive"


def c2_trials_state(dx_point: float, ds_point: float, ds_lo95: float, ds_hi95: float,
                    diff_lo95: float, pass_ub: float = 0.005) -> str:
    """The trials §4.3 C2 rule verbatim (reported as ``c2_trials_state``, never in a label)."""
    if ds_lo95 > 0 or (diff_lo95 <= 0 and dx_point <= ds_point):
        return "fail"
    if ds_hi95 < pass_ub and diff_lo95 > 0:
        return "pass"
    return "inconclusive"


def c5_state(c5_point: float, c5_lo95: float, c5_hi95: float, absorbed_lo95: float,
             fail_ub: float = 0.005) -> str:
    """§4.3 ``absorbed_share_v3`` (absorbed = Δ_X − Δ_C5), first match wins:
    fail (branch 1: LB(Δ_C5) ≤ 0 and [pt(Δ_C5) ≤ 0 or LB(absorbed) > 0]; branch 2:
    UB(Δ_C5) < 0.005 and LB(absorbed) > 0), pass (LB(Δ_C5) > 0), inconclusive."""
    if (c5_lo95 <= 0 and (c5_point <= 0 or absorbed_lo95 > 0)) or \
            (c5_hi95 < fail_ub and absorbed_lo95 > 0):
        return "fail"
    if c5_lo95 > 0:
        return "pass"
    return "inconclusive"


def c2_states_of(dx: Mapping[str, Any], ds: Mapping[str, Any], diff: Mapping[str, Any]) -> Dict[str, str]:
    return {"state": c2_state(dx["point"], ds["point"], ds["ci95"][0], ds["ci95"][1],
                              diff["ci95"][0], diff["ci95"][1]),
            "trials_state": c2_trials_state(dx["point"], ds["point"], ds["ci95"][0], ds["ci95"][1],
                                            diff["ci95"][0])}


def c5_state_of(c5: Mapping[str, Any], absorbed: Mapping[str, Any]) -> str:
    return c5_state(c5["point"], c5["ci95"][0], c5["ci95"][1], absorbed["ci95"][0])


def de_state(s: Mapping[str, Any], alpha: float = 0.05, bound: float = 0.005) -> str:
    """§4.4: supported / reversed / flat / inconclusive."""
    if s["p"] <= alpha and s["point"] < 0:
        return "supported"
    if s["p"] <= alpha and s["point"] > 0:
        return "reversed"
    if tost_state(s["ci90"][0], s["ci90"][1], bound) == "equivalent":
        return "flat"
    return "inconclusive"


def _ci_text(t: Sequence[float]) -> str:
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def final_label(A: Mapping[str, Any], B: Mapping[str, Any], huri_state: str) -> Dict[str, Any]:
    """§4.3 ordered 7-label rule (``ordered_7_label_v4``), first match wins, with suffixes.

    ``A`` / ``B`` (B = Bclean): ``positive``, ``harm``, ``tost`` (TOST label of the
    co-primary), ``c2`` and ``c5`` states, ``ds`` = (point, lo, hi) of Δ_S and ``xs`` =
    (point, lo, hi) of X − S (95% CIs, for suffix a2). ``huri_state`` = four-state label of Hi-HuRI.
    """
    raw_conf = bool(A["positive"])
    raw_rep = raw_conf and bool(B["positive"])
    if not raw_conf:
        return {"label_id": 1,
                "label": "significant harm (A)" if A["harm"] else f"not confirmed ({A['tost']})"}
    if A["c2"] == "fail":
        lid, label = 2, "not attributable to anchor-specific GO annotations (A)"
    elif A["c5"] == "fail":
        lid, label = 3, "not separable from study bias (A)"
    elif not raw_rep:
        lid, label = 4, f"confirmed, not replicated ({B['tost']})" + ("; significant harm in B" if B["harm"] else "")
    elif B["c2"] == "fail":
        lid, label = 5, "confirmed; replication not attributable to anchor-specific GO annotations"
    elif B["c5"] == "fail":
        lid, label = 6, "confirmed; replication not separable from study bias"
    else:
        lid, label = 7, "replicated"
    reached = [("A", A)] if lid >= 3 else []
    if lid >= 6:
        reached.append(("B", B))
    inc = [c for c, s in reached if s["c2"] == "inconclusive"]
    if inc:
        label += f" (shuffled control inconclusive: {', '.join(inc)})"
    for c, s in reached:
        if s["c2"] == "partial":
            label += (f" (partly carried by annotation density: Δ_S = {_ci_text(s['ds'])}; "
                      f"X − S = {_ci_text(s['xs'])}; {c})")
    inc5 = [c for c, s in reached if s["c5"] == "inconclusive"]
    if inc5:
        label += f" (study-bias control inconclusive: {', '.join(inc5)})"
    label += f" [HuRI: {huri_state}]"
    if huri_state in ("equivalent", "negative"):
        label += " (not supported on literature-independent HuRI labels)"
    return {"label_id": lid, "label": label}


HEADLINE_BASE = "GO fusion (AZ-X, exact pruned BP terms, per-anchor z) improves within-anchor ranking"


def headline(labels: Mapping[str, Mapping[str, Any]], names: Sequence[str] = ("H1", "H2", "H3")) -> str:
    """§4.3 headline (first line of report.md); labels (with suffixes) are always printed."""
    rep = [h for h in names if labels[h]["label_id"] == 7]
    others = [h for h in names if h not in rep]
    if len(rep) == len(names):
        return f"{HEADLINE_BASE}, replicated: " + "; ".join(f"{h}: {labels[h]['label']}" for h in names)
    if rep:
        return (f"{HEADLINE_BASE}, partially replicated ({', '.join(rep)}): "
                + "; ".join(f"{h}: {labels[h]['label']}" for h in others + rep))
    return "; ".join(f"{h}: {labels[h]['label']}" for h in names)


def exposure_warnings(labels: Mapping[str, Mapping[str, Any]], bclean_points: Mapping[str, float],
                      all_b_points: Mapping[str, float]) -> List[str]:
    """§4.5 mandatory warning: label 5-7 and all_B point sign differs from the Bclean point."""
    out = []
    for h, lab in labels.items():
        if lab["label_id"] in (5, 6, 7) and np.sign(all_b_points[h]) != np.sign(bclean_points[h]):
            out.append(f"exposed anchors disagree with the clean replication ({h})")
    return out


# ---------------------------------------------------------------------------- cohort A inputs (§3b)
FRACTIONS = (10, 25, 50, 75, 100)
LC_ROW_COUNTS = {10: 1571, 25: 3927, 50: 7854, 75: 11781, 100: 15708}
VALIDATION_POSITIVE_COUNT = 321
COHORT_A_SEEDS = {10: (42, 13, 21), 25: (42, 13, 21, 87, 123), 50: (42, 13, 21),
                  75: (42, 13, 21), 100: (42, 13, 21, 87, 123)}
COHORT_A_RUNS = tuple((F, S) for F in FRACTIONS for S in COHORT_A_SEEDS[F])
FRACTION_ARG = {10: "0.1", 25: "0.25", 50: "0.5", 75: "0.75", 100: "1.0"}
PUBLISHED_SEEDS = (42, 13, 21)
COHORT_B_SEEDS = (13, 21, 42)
N_FOLDS = 5
FOLD_SEED = 2027
SHUFFLE_SEED = 11
TRAINING_ORDER = ([f"A:f{F}_s{S}" for F, S in COHORT_A_RUNS]
                  + [f"B:k{k}_s{S}" for k in range(N_FOLDS) for S in COHORT_B_SEEDS])
EXPECTED_UNEXPECTED_FIELDS = ["embedding_cache_path", "go_ppi_converted_dir", "go_ppi_split_dir"]


def build_cohort_a(out_dir: os.PathLike | str, lc_dir: str = LC_DIR, masked_dir: str = MASKED_SPLIT_DIR,
                   row_counts: Mapping[int, int] = LC_ROW_COUNTS,
                   valpos_count: int = VALIDATION_POSITIVE_COUNT) -> Dict[str, Any]:
    """§3b: masked fraction files in the published row order, and masked validation
    positives in the published order, matched by (resume_id, partner_id)."""
    out = absp(out_dir)
    masked_train = read_jsonl(f"{masked_dir}/train.jsonl")
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in masked_train:
        k = record_key(r)
        if k in by_key:
            raise ProtocolError(f"duplicate (resume_id, partner_id) {k} in masked train")
        by_key[k] = r
    files: Dict[str, str] = {}
    for F in sorted(row_counts):
        pub = read_jsonl(f"{lc_dir}/frac_{F}/train.jsonl")
        rows = []
        for r in pub:
            m = by_key.get(record_key(r))
            if m is None or record_grade(m) != record_grade(r):
                raise ProtocolError(f"frac_{F}: published row {record_key(r)} has no masked counterpart")
            rows.append(m)
        if len(rows) != row_counts[F]:
            raise ProtocolError(f"frac_{F}: {len(rows)} rows != {row_counts[F]}")
        files[rel(out / f"frac_{F}" / "train.jsonl")] = write_jsonl(out / f"frac_{F}" / "train.jsonl", rows)
    masked_val = {}
    for r in read_jsonl(f"{masked_dir}/validation.jsonl"):
        if record_grade(r) == 2:
            k = record_key(r)
            if k in masked_val:
                raise ProtocolError(f"duplicate key {k} in masked validation")
            masked_val[k] = r
    pub_vp = read_jsonl(f"{lc_dir}/validation_positive.jsonl")
    vp = []
    for r in pub_vp:
        m = masked_val.get(record_key(r))
        if m is None:
            raise ProtocolError(f"validation_positive row {record_key(r)} has no masked grade-2 counterpart")
        vp.append(m)
    if len(vp) != valpos_count:
        raise ProtocolError(f"validation_positive: {len(vp)} rows != {valpos_count}")
    files[rel(out / "validation_positive.jsonl")] = write_jsonl(out / "validation_positive.jsonl", vp)
    return {"files": files, "row_counts": {str(F): row_counts[F] for F in row_counts},
            "validation_positive": len(vp)}


def check_protein_uris(proteins: Iterable[Mapping[str, Any]], index) -> Dict[str, Any]:
    """Pruned record URIs (``proteins.jsonl`` ``go_uris``) equal the pruned index ``direct``."""
    bad, n = [], 0
    for p in proteins:
        n += 1
        if sorted(p.get("go_uris") or []) != sorted(index.direct.get(p["protein_id"], ())):
            bad.append(p["protein_id"])
    return {"n_proteins": n, "mismatches": len(bad), "examples": bad[:10], "ok": not bad}


# ---------------------------------------------------------------------------- cohort B folds (§3c)
def view_body(encoder_view: str) -> str:
    """Masked view body: text after the first ``" — "`` (the gene prefix is dropped)."""
    head, sep, tail = encoder_view.partition(VIEW_SEP)
    return tail if sep else encoder_view


def view_groups(proteins: Mapping[str, Mapping[str, Any]]) -> List[List[str]]:
    """Groups (size >= 2) of proteins with identical masked view bodies, each sorted,
    ordered by their smallest protein_id."""
    by_body: Dict[str, List[str]] = collections.defaultdict(list)
    for pid in sorted(proteins):
        by_body[view_body(proteins[pid]["encoder_view"])].append(pid)
    return sorted((sorted(g) for g in by_body.values() if len(g) > 1), key=lambda g: g[0])


def assign_protein_folds(proteins: Iterable[str], multi_groups: Sequence[Sequence[str]],
                         n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> List[List[str]]:
    """§3c: groups (view groups + singletons) ordered by smallest protein_id,
    ``random.Random(seed).shuffle``, then each to the fold with the fewest proteins so far
    (ties -> lowest index)."""
    in_group = {p for g in multi_groups for p in g}
    groups = [sorted(g) for g in multi_groups] + [[p] for p in sorted(set(proteins)) if p not in in_group]
    groups.sort(key=lambda g: g[0])
    random.Random(seed).shuffle(groups)
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for g in groups:
        k = min(range(n_folds), key=lambda i: (len(folds[i]), i))
        folds[k].extend(g)
    return [sorted(f) for f in folds]


def outer_split(k: int, n_folds: int = N_FOLDS) -> Dict[str, Any]:
    val = (k + 1) % n_folds
    return {"k": k, "test": k, "validation": val, "train": [i for i in range(n_folds) if i not in (k, val)]}


def rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """sha256 of exactly the bytes ``write_jsonl`` writes for ``rows``."""
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return h.hexdigest()


def write_fold(files: Mapping[str, Sequence[Mapping[str, Any]]], manifest: Dict[str, Any],
               out_dir: os.PathLike | str) -> Dict[str, Any]:
    """Write one fold's files (leaf dir must not exist) and its ``fold_manifest.json``."""
    out = absp(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    public = {k: v for k, v in manifest.items() if not k.startswith("_")}
    public["sha256"] = {}
    for name, rows in files.items():
        public["sha256"][f"{name}.jsonl"] = write_jsonl(out / f"{name}.jsonl", rows)
    (out / "fold_manifest.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return public


def build_fold(proteins: Mapping[str, Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
               folds: Sequence[Sequence[str]], k: int, seed: int = FOLD_SEED, val_cap: int = 20,
               groups: Sequence[Sequence[str]] = ()) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """§3c fold files for outer fold k, in memory (records via ``data_splitter._slot`` /
    ``_build_record``). Returns ``(files, manifest)``; ``manifest["rows_sha256"]`` holds the
    sha256 each file will have, ``manifest["_test_anchor_info"]`` the per-anchor test grades
    and partners (for Bclean). ``write_fold`` writes them."""
    from go_ppi_domain.data_splitter import _build_record, _slot
    sp = outer_split(k, len(folds))
    part: Dict[str, str] = {}
    for name, ids in (("test", folds[sp["test"]]), ("validation", folds[sp["validation"]]),
                      ("train", [p for i in sp["train"] for p in folds[i]])):
        for p in ids:
            part[p] = name
    fold_of = {p: i for i, f in enumerate(folds) for p in f}
    for g in groups:
        if len({fold_of.get(p) for p in g}) > 1:
            raise ProtocolError(f"view group {list(g)} split across folds")
    kept: Dict[str, Dict[str, Dict[int, Set[str]]]] = {s: collections.defaultdict(lambda: collections.defaultdict(set))
                                                      for s in ("train", "validation", "test")}
    dropped = 0
    for pr in pairs:
        a, b = pr["anchor_id"], pr["partner_id"]
        pa, pb = part.get(a), part.get(b)
        if pa is None or pb is None or pa != pb:
            dropped += 1
            continue
        kept[pa][a][int(pr["grade"])].add(b)
    plan = {"train": (2,), "validation": (2, 1, 0), "test": (2, 1, 0)}
    files: Dict[str, List[Any]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    anchor_info: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        rows = []
        per_grade: collections.Counter = collections.Counter()
        for anchor in sorted(kept[split]):
            slot = _slot(proteins[anchor], "protein_id")
            for grade in plan[split]:
                ids = sorted(kept[split][anchor].get(grade, ()))
                if split == "validation" and grade == 0 and val_cap and len(ids) > val_cap:
                    ids = sorted(random.Random(f"{seed}|fold{k}|validation|{anchor}|0").sample(ids, val_cap))
                for pid in ids:
                    rows.append(_build_record(anchor, slot, proteins[pid], grade))
                    per_grade[grade] += 1
                    if split == "test":
                        info = anchor_info.setdefault(anchor, {"grades": set(), "partners": []})
                        info["grades"].add(grade)
                        info["partners"].append(pid)
        random.Random(f"{seed}|fold{k}|{split}").shuffle(rows)
        files[split] = rows
        counts[split] = {str(g): per_grade[g] for g in (2, 1, 0)}
    files["validation_positive"] = [r for r in files["validation"] if r["job"]["grade"] == 2]
    pools = [{"anchor_id": a, "split": s,
              "weak_evidence": sorted(kept[s][a].get(1, ())), "no_interaction": sorted(kept[s][a].get(0, ()))}
             for s in ("train", "validation") for a in sorted(kept[s])]
    files["negative_pools"] = pools
    ents = {s: {r["resume"]["protein_id"] for r in files[s]} | {r["job"]["partner_id"] for r in files[s]}
            for s in ("train", "validation", "test")}
    pool_ents = {s: {p["anchor_id"] for p in pools if p["split"] == s} for s in ("train", "validation")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ents[a] & ents[b]:
            raise ProtocolError(f"fold {k}: protein leak {a}/{b}: {sorted(ents[a] & ents[b])[:5]}")
    if pool_ents["train"] & pool_ents["validation"]:
        raise ProtocolError(f"fold {k}: train/validation pool overlap")

    def hard_defined(split):
        g = collections.defaultdict(set)
        for r in files[split]:
            g[r["resume"]["protein_id"]].add(r["job"]["grade"])
        return sum(1 for s in g.values() if 2 in s and 1 in s)

    manifest = {"k": k, "outer": sp, "seed": seed, "val_no_interaction_cap": val_cap,
                "proteins": {s: sum(1 for v in part.values() if v == s) for s in ("train", "validation", "test")},
                "anchors": {s: len({r["resume"]["protein_id"] for r in files[s]}) for s in ("train", "validation", "test")},
                "hard_defined_anchors": {s: hard_defined(s) for s in ("validation", "test")},
                "records_by_grade": counts, "records": {n: len(r) for n, r in files.items()},
                "pairs_dropped_cross_partition": dropped,
                "rows_sha256": {f"{n}.jsonl": rows_sha256(r) for n, r in files.items()},
                "_test_anchor_info": anchor_info}
    return files, manifest


def exposure_sets(pub_paths: Sequence[os.PathLike | str], groups: Sequence[Sequence[str]]) -> Dict[str, List[str]]:
    """§3c: ``exposed`` = every protein in any role of the published validation/test files;
    ``exposed_closed`` = union of the view groups containing an exposed protein."""
    exposed: Set[str] = set()
    for path in pub_paths:
        for r in read_jsonl(path):
            exposed.add(r["resume"]["protein_id"])
            exposed.add(r["job"]["partner_id"])
    closed = set(exposed)
    for g in groups:
        if closed & set(g):
            closed |= set(g)
    return {"exposed": sorted(exposed), "exposed_closed": sorted(closed)}


def bclean_summary(fold_manifests: Sequence[Mapping[str, Any]], exposed: Set[str],
                   exposed_closed: Set[str]) -> Dict[str, Any]:
    """§3c ``Bclean`` = cohort-B test anchors (all folds) not in ``exposed_closed``; the
    hard-defined count, the anchor list + sha256 and the exposed-partner record counts."""
    anchors, hard, rec_exp, anc_exp = [], 0, 0, set()
    all_test, all_hard = [], 0
    for m in fold_manifests:
        for a, info in m["_test_anchor_info"].items():
            all_test.append(a)
            hd = 2 in info["grades"] and 1 in info["grades"]
            all_hard += hd
            if a in exposed_closed:
                continue
            anchors.append(a)
            hard += hd
            n = sum(1 for p in info["partners"] if p in exposed)
            rec_exp += n
            if n:
                anc_exp.add(a)
    anchors = sorted(anchors)
    return {"anchors": anchors, "sha256": list_sha256(anchors), "n_anchors": len(anchors),
            "hard_defined": hard, "all_B_test_anchors": len(all_test), "all_B_hard_defined": all_hard,
            "bclean_records_with_exposed_partner": rec_exp,
            "bclean_anchors_with_exposed_partner": len(anc_exp)}


# ---------------------------------------------------------------------------- HuRI cohort (§3d)
def decile_bins(genes: Sequence[str], values: Sequence[float]) -> Dict[str, int]:
    """``scripts/go_independent_signal_diagnosis.decile_bins`` (same rule, copied verbatim)."""
    cuts = np.quantile(np.asarray(values), np.linspace(0.0, 1.0, 11))
    return {gene: int(min(9, np.searchsorted(cuts[1:-1], value, side="right")))
            for gene, value in zip(genes, values)}


def ensembl_gene_map(aliases_gz: os.PathLike | str, info_gz: os.PathLike | str,
                     gene_terms: Mapping[str, Any]) -> Dict[str, str]:
    """``go_independent_signal_diagnosis.load_ensembl_gene_map`` with explicit paths:
    Ensembl gene -> STRING preferred symbol present in ``gene_terms``; ties -> the richest
    closure, then the symbol."""
    protein_to_symbol = {}
    with gzip.open(absp(info_gz), "rt") as handle:
        next(handle)
        for line in handle:
            c = line.rstrip("\n").split("\t")
            protein_to_symbol[c[0]] = c[1]
    ens: Dict[str, Set[str]] = collections.defaultdict(set)
    with gzip.open(absp(aliases_gz), "rt") as handle:
        next(handle)
        for line in handle:
            c = line.rstrip("\n").split("\t")
            if len(c) < 3 or c[2] != "Ensembl_gene":
                continue
            symbol = protein_to_symbol.get(c[0])
            if c[1].startswith("ENSG") and symbol in gene_terms:
                ens[c[1]].add(symbol)
    return {g: sorted(s, key=lambda x: (-len(gene_terms[x]), x))[0] for g, s in ens.items()}


def huri_edges(huri_tsv: os.PathLike | str, ensembl_to_symbol: Mapping[str, str]) -> List[Tuple[str, str]]:
    """Mapped, undirected, self-free HuRI edges (sorted endpoint pairs)."""
    out = set()
    with open(absp(huri_tsv)) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            a, b = ensembl_to_symbol.get(cols[0]), ensembl_to_symbol.get(cols[1])
            if a and b and a != b:
                out.add(tuple(sorted((a, b))))
    return sorted(out)


def string_neighbours(links_gz: os.PathLike | str, info_gz: os.PathLike | str,
                      anchors: Set[str]) -> Dict[str, Set[str]]:
    """Proteins with a row of any channel to each anchor in ``protein.links.detailed``."""
    ensp = {}
    with gzip.open(absp(info_gz), "rt") as handle:
        next(handle)
        for line in handle:
            c = line.rstrip("\n").split("\t")
            ensp[c[0]] = c[1]
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    with gzip.open(absp(links_gz), "rt") as handle:
        next(handle)
        for line in handle:
            c = line.split(" ", 2)
            a, b = ensp.get(c[0]), ensp.get(c[1])
            if a is None or b is None:
                continue
            if a in anchors:
                out[a].add(b)
            if b in anchors:
                out[b].add(a)
    return out


def build_huri_rows(proteins: Mapping[str, Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
                    partition: Set[str], edges: Sequence[Tuple[str, str]],
                    string_nbrs: Mapping[str, Set[str]], closure_size: Mapping[str, int],
                    seed: int = FOLD_SEED, n_neg: int = 20, max_draws: int = 200) -> Dict[str, Any]:
    """§3d HuRI cohort rows (main file + ``grade1_unrestricted``) and their counts."""
    from go_ppi_domain.data_splitter import _build_record, _slot
    adj: Dict[str, Set[str]] = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    observed = sorted(adj)
    deg_bin = decile_bins(observed, [len(adj[g]) for g in observed])
    ann_bin = decile_bins(observed, [closure_size.get(g, 0) for g in observed])
    cells: Dict[Tuple[int, int], List[str]] = collections.defaultdict(list)
    for g in observed:
        if g in partition:
            cells[(deg_bin[g], ann_bin[g])].append(g)
    weak: Dict[str, Set[str]] = collections.defaultdict(set)
    strong: Set[Tuple[str, str]] = set()
    for pr in pairs:
        if int(pr["grade"]) == 1:
            weak[pr["anchor_id"]].add(pr["partner_id"])
        elif int(pr["grade"]) == 2:
            strong.add((pr["anchor_id"], pr["partner_id"]))
    anchors = sorted(a for a in partition if a in proteins and adj.get(a, set()) & partition)
    main, unres = [], []
    unmatched, g1_counts, hard = 0, {"huri_observed_only": 0, "unrestricted": 0}, {"huri_observed_only": 0, "unrestricted": 0}
    n_pos, string_grade2 = 0, []
    for a in anchors:
        pos = sorted(adj[a] & partition)
        forbidden = {a} | adj[a] | set(string_nbrs.get(a, ()))
        rng = random.Random(f"{seed}|huri|{a}")
        chosen: List[str] = []
        for i in range(n_neg):
            p = pos[i % len(pos)]
            pool = cells.get((deg_bin[p], ann_bin[p]), [])
            pick = None
            for _ in range(max_draws if pool else 0):
                c = rng.choice(pool)
                if c not in forbidden and c not in chosen:
                    pick = c
                    break
            if pick is None:
                unmatched += 1
                continue
            chosen.append(pick)
        g1_all = sorted(b for b in weak.get(a, ()) if b in partition and b not in adj[a])
        g1_obs = [b for b in g1_all if b in adj]
        slot = _slot(proteins[a], "protein_id")
        for grade1, rows, key in ((g1_obs, main, "huri_observed_only"), (g1_all, unres, "unrestricted")):
            for grade, ids in ((2, pos), (1, grade1), (0, sorted(chosen))):
                for b in ids:
                    rows.append(_build_record(a, slot, proteins[b], grade))
            g1_counts[key] += len(grade1)
            hard[key] += bool(pos and grade1)
        n_pos += len(pos)
        string_grade2 += [[a, b] for b in pos if (a, b) in strong]
    random.Random(f"{seed}|huri|test").shuffle(main)
    random.Random(f"{seed}|huri|test_grade1_unrestricted").shuffle(unres)
    return {"rows": main, "rows_grade1_unrestricted": unres, "anchors": anchors,
            "counts": {"anchors": len(anchors), "grade2_directed_pairs": n_pos, "grade0": sum(
                1 for r in main if r["job"]["grade"] == 0), "unmatched_grade0_draws": unmatched,
                "grade1_counts": g1_counts, "hard_defined_anchors": hard,
                "huri_observed": len(observed), "partition_huri_observed": sum(len(v) for v in cells.values()),
                "huri_positives_that_are_string_grade2": len(string_grade2)},
            "string_grade2_positives": string_grade2}


# ---------------------------------------------------------------------------- mention flags (§4.5)
def mention_flags(records: Sequence[Mapping[str, Any]], unmasked: Mapping[str, Mapping[str, Any]],
                  lexicon) -> np.ndarray:
    """Partner-symbol mention per record: a lexicon entry owned by the partner symbol in the
    unmasked anchor view body, or one owned by the anchor symbol in the partner view body
    (``leak_pruning.entries_in_text``)."""
    from go_ppi_domain import leak_pruning as lp
    owned: Dict[str, Set[str]] = collections.defaultdict(set)
    for entry, owners in lexicon.owner.items():
        for o in owners:
            owned[o].add(entry)
    cache: Dict[str, Set[str]] = {}

    def entries(pid: str) -> Set[str]:
        if pid not in cache:
            p = unmasked[pid]
            cache[pid] = lp.entries_in_text(lp.strip_view_prefix(p["gene"], p["encoder_view"]), lexicon)
        return cache[pid]

    out = np.zeros(len(records), dtype=bool)
    for i, r in enumerate(records):
        a, b = r["resume"]["protein_id"], r["job"]["partner_id"]
        sa, sb = unmasked[a]["gene"], unmasked[b]["gene"]
        out[i] = bool(owned.get(sb, set()) & entries(a)) or bool(owned.get(sa, set()) & entries(b))
    return out


# ---------------------------------------------------------------------------- gate 2 (§5.2)
GATE2_TOLERANCE_ABS = {"continuous": 1e-5, "rank": 1e-4, "rank_per_query": 1e-3}
_SKIP_KEYS = {"binary_classification", "ordinal_classification"}
_RANK_KEYS = {"validation_selection_curve", "ranking", "ranking_macro", "per_query", "global_auc", "macro_query_auc"}
GATE2_LEAF_CLASSES = {
    "path_semantics": ("leaf paths are key tuples, relative to one runs[i]; per_query rows are paired "
                       "by their 'query' field"),
    "order": ["skipped", "identifier", "exact", "rank_per_query", "rank", "continuous"],
    "skipped": "binary_classification or ordinal_classification in P (threshold metrics not computed)",
    "identifier": ("fraction, seed exactly; checkpoint by its last two path components (the published "
                   "JSON holds an absolute Mac path)"),
    "exact": ("P[-1] in {selected_weight, n_queries} or queries_with_both_classes in P or "
              "confusion_matrix in P or (P[-1] == n and tier_stats in P); selected_weight has the "
              "near-tie exception (gap < 1e-4)"),
    "rank_per_query": "per_query in P",
    "rank": ("any of {validation_selection_curve, ranking, ranking_macro, per_query, global_auc, "
             "macro_query_auc} in P, or P[-1] in {kendall_tau_b, spearman_rho}"),
    "continuous": "everything else (validation_stats, tier_stats mean/sd/median, cohens_d)",
}


def leaf_class(P: Tuple[str, ...]) -> str:
    """Gate-2 class of a leaf with key tuple ``P``; ``skipped`` is tested first."""
    P = tuple(str(x) for x in P)
    if _SKIP_KEYS & set(P):
        return "skipped"
    if len(P) == 1 and P[0] in ("fraction", "seed", "checkpoint"):
        return "identifier"
    if (P[-1] in ("selected_weight", "n_queries") or "queries_with_both_classes" in P
            or "confusion_matrix" in P or (P[-1] == "n" and "tier_stats" in P)):
        return "exact"
    if "per_query" in P:
        return "rank_per_query"
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


def _checkpoint_id(path: Any) -> Tuple[str, ...]:
    return tuple(Path(str(path)).parts[-2:])


#: §5.2 2a float-tie exception for per-anchor leaves (§9 A1). Two records of one anchor are
#: float-tied when their raw float32 text scores differ by < ``gap`` (and, in the fusion arm,
#: their φ is identical), chained over adjacent records in score order.
GATE2_FLOAT_TIE = {"gap": 1e-6, "max_orderings": 5040, "offset": 1e-13,
                   "rule": ("a failing rank_per_query leaf passes iff its anchor has >= 1 float-tied "
                            "group and some ordering of the tied records reproduces the published "
                            "value within the rank_per_query tolerance; logged as an event")}


def float_tie_groups(raw_text: np.ndarray, phi: Optional[np.ndarray] = None,
                     gap: float = GATE2_FLOAT_TIE["gap"]) -> List[List[int]]:
    """Float-tied groups (local indices, size >= 2) of one anchor's records."""
    raw = np.asarray(raw_text, dtype=float)
    order = np.argsort(-raw, kind="stable")
    chains, cur = [], [int(order[0])] if len(order) else []
    for a, b in zip(order[:-1], order[1:]):
        if abs(raw[a] - raw[b]) < gap:
            cur.append(int(b))
        else:
            chains.append(cur)
            cur = [int(b)]
    if cur:
        chains.append(cur)
    groups = []
    for c in chains:
        if len(c) < 2:
            continue
        if phi is None:
            groups.append(c)
            continue
        by_phi: Dict[float, List[int]] = collections.OrderedDict()
        for i in c:
            by_phi.setdefault(float(phi[i]), []).append(i)
        groups.extend(g for g in by_phi.values() if len(g) >= 2)
    return groups


def float_tie_explains(scores: np.ndarray, raw_text: np.ndarray, phi: Optional[np.ndarray],
                       grades: np.ndarray, query: str, metric: str, published: float, tol: float,
                       params: Mapping[str, Any] = GATE2_FLOAT_TIE) -> Dict[str, Any]:
    """§9 A1: is a per-anchor ranking-leaf difference explained by a float tie?

    Re-orders every float-tied group of the anchor in every way (members placed at the group's
    top score minus k * ``offset``, which keeps them between the same neighbours) and recomputes
    ``epf.ranking_metrics`` for the anchor. Explained iff some ordering reproduces ``published``
    within ``tol``.
    """
    E = epf()
    scores = np.asarray(scores, dtype=float)
    groups = float_tie_groups(raw_text, phi, params["gap"])
    out: Dict[str, Any] = {"query": query, "metric": metric, "published": published,
                           "n_tied_groups": len(groups), "tied_group_sizes": [len(g) for g in groups],
                           "explained": False}
    if not groups:
        out["reason"] = "no float-tied group"
        return out
    n_orderings = int(np.prod([math.factorial(len(g)) for g in groups]))
    out["n_orderings"] = n_orderings
    if n_orderings > params["max_orderings"]:
        out["reason"] = "too many orderings"
        return out
    q = np.full(len(scores), query, dtype=object)
    g = np.asarray(grades)
    best = None
    for perms in itertools.product(*(itertools.permutations(gr) for gr in groups)):
        s = scores.copy()
        for gr, perm in zip(groups, perms):
            top = float(np.max(scores[list(gr)]))
            for k, i in enumerate(perm):
                s[i] = top - k * params["offset"]
        v = E.ranking_metrics(s, g, q)["per_query"][0][metric]
        d = abs(float(v) - float(published))
        if best is None or d < best:
            best = d
        if d <= tol:
            out.update({"explained": True, "reproduced": float(v), "abs_diff": d})
            return out
    out.update({"reason": "no ordering reproduces the published value", "best_abs_diff": best})
    return out


def compare_gate2_run(published: Mapping[str, Any], computed: Mapping[str, Any],
                      tol: Mapping[str, float] = GATE2_TOLERANCE_ABS,
                      weight_exception: bool = False,
                      per_query_tie_check: Optional[Callable[[Tuple[str, ...], float], Dict[str, Any]]] = None,
                      ) -> Dict[str, Any]:
    """Compare one published ``runs[i]`` with the scorer's equivalent (§5.2 2a tiers).
    ``weight_exception`` = the cell is a pinned near-tie cell and the computed weight is one
    of its two top weights (the event is logged by the caller). ``per_query_tie_check(path,
    published)`` implements the §9 A1 float-tie exception for failing rank_per_query leaves."""
    comp = dict(iter_leaves(computed))
    failures, missing = [], []
    tie_exceptions: List[Dict[str, Any]] = []
    skipped = 0
    max_by_class: Dict[str, float] = collections.defaultdict(float)
    max_by_group: Dict[str, float] = collections.defaultdict(float)
    n_compared: collections.Counter = collections.Counter()
    for P, pv in iter_leaves(published):
        cls = leaf_class(P)
        if cls == "skipped":
            skipped += 1
            continue
        if P not in comp:
            missing.append(P)
            failures.append({"path": list(P), "class": cls, "reason": "not computed"})
            continue
        cv = comp[P]
        n_compared[cls] += 1
        if cls == "identifier":
            same = _checkpoint_id(pv) == _checkpoint_id(cv) if P == ("checkpoint",) else pv == cv
            if not same:
                failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv})
            continue
        if P == ("selected_weight",) and weight_exception:
            continue
        if isinstance(pv, str) or isinstance(cv, str):
            if pv != cv:
                failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv})
            continue
        if pv is None or cv is None:
            if not (pv is None and cv is None):
                failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv})
            continue
        diff = abs(float(pv) - float(cv))
        group = ".".join(x for x in P[:3] if x != "macro")
        max_by_class[cls] = max(max_by_class[cls], diff)
        max_by_group[f"{cls}:{group}"] = max(max_by_group[f"{cls}:{group}"], diff)
        if (cls == "exact" and float(pv) != float(cv)) or (cls != "exact" and diff > tol[cls]):
            if cls == "rank_per_query" and per_query_tie_check is not None:
                ev = per_query_tie_check(P, float(pv))
                ev.update({"path": list(P), "computed": cv, "abs_diff_computed": diff})
                if ev.get("explained"):
                    tie_exceptions.append(ev)
                    continue
                failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv,
                                 "abs_diff": diff, "float_tie_check": ev})
                continue
            failures.append({"path": list(P), "class": cls, "published": pv, "computed": cv, "abs_diff": diff})
    return {"passed": not failures, "failures": failures[:50], "n_failures": len(failures),
            "missing": [list(p) for p in missing][:50], "skipped": skipped, "n_compared": dict(n_compared),
            "max_abs_by_class": dict(max_by_class), "max_abs_by_group": dict(max_by_group),
            "float_tie_exceptions": tie_exceptions}


def near_tie_cells(published_by_rule: Mapping[str, Mapping[str, Any]], gap: float = GATE2_NEAR_TIE_GAP) -> List[List[Any]]:
    """Cells whose published validation top-two gap is < ``gap`` (derived from the curves)."""
    out = []
    for rule, pub in published_by_rule.items():
        for r in pub["runs"]:
            g = top_two_gap(r["validation_selection_curve"].values())
            if g < gap:
                out.append([rule, int(r["fraction"]), int(r["seed"]), g])
    return sorted(out, key=lambda x: (x[0], x[1], x[2]))


def top_two_weights(curve: Mapping[str, float]) -> List[float]:
    """The two best grid weights of a selection curve (ties -> smaller w first)."""
    items = sorted(((float(w), float(v)) for w, v in curve.items()), key=lambda x: (-x[1], x[0]))
    return [items[0][0], items[1][0]] if len(items) > 1 else [items[0][0]]


# ---------------------------------------------------------------------------- gate 3 (§5.3)
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
                   gpu_name: Optional[str], original_caches_ok: bool) -> List[str]:
    """§5.3 run checks for one training run."""
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
        unexpected = (m.get("single_factor_audit") or {}).get("unexpected_fields")
        if unexpected != list(expected_unexpected):
            fails.append(f"single_factor_audit.unexpected_fields={unexpected} != {list(expected_unexpected)}")
        attach = m.get("attach") or {}
        cc = attach.get("cache_check") or {}
        if not (cc.get("expected_text_dim") == EMBEDDING_DIM and list(cc.get("dims", {})) == [f"({EMBEDDING_DIM},)"]):
            fails.append(f"attach.cache_check not passing: {cc}")
        if "matcher" not in attach or attach.get("matcher") is not None:
            fails.append(f"attach.matcher={attach.get('matcher', '<absent>')!r}; no GO may reach training")
        fails.extend(check_device_block(m.get("device"), gpu_name, "cuda"))
    if not original_caches_ok:
        fails.append("an original embedding cache sha256 changed")
    return fails


# ---------------------------------------------------------------------------- pins chain (§7, trials §7)
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
    """Changed non-whitelisted fields as dotted paths at depth 2; ``protocol.sha256`` is whitelisted."""
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


def load_verified_pins(pins_path: os.PathLike | str, allow_non_latest: bool = False) -> Dict[str, Any]:
    """Load ``--pins``, verify the whole supersedes chain and that it is the latest revision."""
    path = absp(pins_path)
    pins = json.loads(path.read_text(encoding="utf-8"))
    revs = pins_revision_files(path.parent)
    if not allow_non_latest and revs and max(revs) != pins.get("revision"):
        raise ProtocolError(f"{rel(path)} is revision {pins.get('revision')}, latest present is {max(revs)}")
    cur, cur_path = pins, path
    while cur.get("supersedes"):
        prev_path = absp(cur["supersedes"]["path"])
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        verify_revision(prev, cur, rel(prev_path), file_sha256(prev_path))
        cur, cur_path = prev, prev_path
    if cur.get("revision") != 1 or cur_path.name != "pins.json":
        raise ProtocolError(f"chain does not end at revision 1 pins.json ({rel(cur_path)})")
    return pins


def verify_pinned_hashes(pins: Mapping[str, Any],
                         sections: Sequence[str] = ("inputs", "code", "chunks", "configs", "derived", "masked")) -> List[str]:
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
        for split, spec in (pins.get("published_prepared_cache") or {}).items():
            for name, h in (spec.get("chunks") or {}).items():
                p = absp(spec["dir"]) / name
                if not p.exists() or file_sha256(p) != h:
                    bad.append(f"chunk:{split}/{name}")
    if "configs" in sections:
        cfg = pins.get("configs") or {}
        base = cfg.get("base") or {}
        if base and (not absp(base["path"]).exists() or file_sha256(base["path"]) != base["sha256"]):
            bad.append(f"configs:{base['path']}")
        for p, d in (cfg.get("derived") or {}).items():
            if not absp(p).exists() or file_sha256(p) != d["sha256"]:
                bad.append(f"configs:{p}")
    if "derived" in sections:
        for p, h in ((pins.get("derived") or {}).get("files") or {}).items():
            if not absp(p).exists() or file_sha256(p) != h:
                bad.append(f"derived:{p}")
    if "masked" in sections:
        me = pins.get("masked_encodes") or {}
        if me and (not absp(me["dir"]).is_dir() or manifest_digest(me["dir"]) != me["digest"]):
            bad.append(f"masked_encodes:{me.get('dir')}")
    return bad


# ---------------------------------------------------------------------------- run ledger (§6.3)
class Ledger:
    """Append-only ``run_ledger.json`` (trials §6.3 semantics).

    Serialisation: ``[\\n`` + one ``json.dumps(entry, sort_keys=True)`` per line joined by
    ``,\\n`` + ``\\n]\\n``. Each append rewrites atomically (temp + ``os.replace``) and
    asserts the previous entry lines are byte-identical in the new file."""

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
        new_lines = [ln[:-1] if ln.endswith(",") else ln for ln in text.splitlines()[1:-1]]
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
                "failed": sum(e.get("status") == "failed" for e in es), "entries": es}


def run_with_resume(ledger: Ledger, run: Mapping[str, Any],
                    attempt: Callable[[Mapping[str, Any], int], Dict[str, Any]], max_failures: int = 2) -> str:
    """Resume semantics for one run: skip passed/excluded; an output dir without a passing
    entry is a failed attempt (renamed ``.failed{n}``); two failures -> excluded."""
    rid, out = run["run"], absp(run["output_dir"])
    st = ledger.status(rid)
    if st["passed"]:
        return "skipped-passed"
    if st["excluded"]:
        return "skipped-excluded"
    failures = st["failed"]

    def rename_out() -> Optional[str]:
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


# ---------------------------------------------------------------------------- environment, git, code
def gpu_record(must_contain: str = "L40S") -> Dict[str, Any]:
    """``make_pins.py`` GPU check (run without the empty CUDA mask): exactly one visible CUDA
    device whose name contains ``must_contain``; records the exact name and ``nvidia-smi``."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise ProtocolError('make_pins.py must run WITHOUT CUDA_VISIBLE_DEVICES="" (it reads the GPU name; '
                            "it touches no data on the GPU)")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProtocolError(f"exactly one visible CUDA device required, found "
                            f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    name = torch.cuda.get_device_name(0)
    if must_contain not in name:
        raise ProtocolError(f"GPU {name!r} does not contain {must_contain!r}")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        smi = r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # pragma: no cover - host specific
        smi = None
    return {"gpu_name": name, "gpu_count": 1, "nvidia_smi": smi, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def blas_info() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        return "; ".join(f"{k}:{(deps.get(k) or {}).get('name')} {(deps.get(k) or {}).get('version')}"
                         for k in ("blas", "lapack"))
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc!r})"


def environment_record(gpu: Mapping[str, Any], host_label: Optional[str] = None,
                       mpnet_snapshot: Optional[str] = None) -> Dict[str, Any]:
    import scipy
    import sentence_transformers
    import torch
    return {"host": host_label or socket.gethostname(), "hostname": socket.gethostname(),
            "platform": platform.platform(), "machine": platform.machine(), **dict(gpu),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "python": platform.python_version(), "sentence_transformers": sentence_transformers.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "blas": blas_info(),
            "torch_blas": "; ".join(ln.strip() for ln in torch.__config__.show().splitlines()
                                    if "BLAS" in ln or "LAPACK" in ln)[:500],
            "mpnet_snapshot": mpnet_snapshot,
            "device_policy": {"training": "cuda", "scoring": "cpu", "fresh_encode": "cpu", "analysis": "cpu",
                              "gate2": "cpu", "data_builds": "cpu", "make_pins": "gpu name only"},
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}


def git_record() -> Dict[str, Any]:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain") or ""
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_paths": sorted(line[3:] for line in status.splitlines() if line.strip())}


#: Code pinned in ``pins.code``: the new study files plus every reused module (§6.1, §7).
CODE_GLOBS = ("go_ppi_domain/fusion_confirm.py", "go_ppi_domain/tests/test_fusion_confirm.py",
              "go_ppi_domain/tests/_leak_fixtures.py", "scripts/go_ppi_fusion_confirm/*.py",
              "go_ppi_fusion/*.py", "go_ppi_domain/*.py", "scripts/evaluate_publication_fusion.py",
              "scripts/run_learning_curve_point.py", "scripts/go_independent_signal_diagnosis.py",
              "run_phase1_embedding_evaluation.py", "trials_domain/*.py", "contrastive_learning/*.py",
              "scripts/trials_fusion_confirm/requirements-remote.txt")


def code_hashes() -> Dict[str, str]:
    files = sorted({p for g in CODE_GLOBS for p in ROOT.glob(g) if p.is_file()})
    return {rel(p): file_sha256(p) for p in files}


def compare_code_hashes(expected: Mapping[str, str], got: Mapping[str, str]) -> List[Dict[str, Any]]:
    """Step −1 code-tree check: any missing, extra or differing file is a failure."""
    out = []
    for p in sorted(set(expected) | set(got)):
        if p not in got:
            out.append({"path": p, "reason": "missing on remote"})
        elif p not in expected:
            out.append({"path": p, "reason": "not in the Mac manifest"})
        elif expected[p] != got[p]:
            out.append({"path": p, "reason": "sha256 differs", "expected": expected[p], "sha256": got[p]})
    return out


# ---------------------------------------------------------------------------- analysis params (§7)
ANALYSIS_PARAMS: Dict[str, Any] = {
    "weight_grid": list(WEIGHT_GRID),
    "fusion_primary": {"id": "AZ-X", "normaliser": "within_anchor_z", "phi": "X", "select_on": "global_pooled_auc"},
    "fusion_secondary": {"id": "GZ-P", "normaliser": "validation_file_global_z", "phi": "P", "select_on": "ndcg@10"},
    "controls_form": "same_as_controlled_fusion",
    "sensitivity_select_on": {"AZ-X": "ndcg@10", "GZ-P": "global_pooled_auc"},
    "sensitivity_normaliser": "GZ-X",
    "tie_break": "smaller_w",
    "c5_grid": "w_g+w_b<=1, tie smaller w_g then w_b",
    "c5_select_on": "global_pooled_auc",
    "c1_fixed_w": 0.4,
    "phi_variants": ["X", "P", "U", "UX", "N", "T", "S", "S_P", "Bz"],
    "arms": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in ARMS.items()},
    "exploration_reference": {"test_results": BEST_TEST_RESULTS_REL,
                              "finalists": "results/go_ppi_fusion_best/finalists.json",
                              "amendment": "results/go_ppi_fusion_best/test_plan_amendment.json"},
    "B": B_DEFAULT,
    "rng": dict(RNG),
    "rng_stream_order": "c*, a*, then fraction-stratified c* (cohort A), then A-pub c* (cohort A only)",
    "cohortB_primary_population": "Bclean",
    "shuffle": {"seed": SHUFFLE_SEED, "strata": 10, "max_draws": 10000,
                "forbidden": ["self", "candidate", "same_set", "partner_set", "simgic>=0.5"]},
    "fold_seed": FOLD_SEED,
    "holm_family": ["H1", "H2", "H3"],
    "holm_tie_order": ["H1", "H2", "H3"],
    "alpha": 0.05,
    "equivalence": {"bound": 0.01, "bound_de": 0.005, "ci": 0.90},
    "c2_rule": "go_pair_specific_v4",
    "c2_pass_ub": 0.005,
    "c2_fail_pairspecific_ub": 0.005,
    "c2_fail_requires_absorption": True,
    "c2_trials_state_reported": True,
    "c5_rule": "absorbed_share_v3",
    "c5_fail_ub": 0.005,
    "final_label_rule": "ordered_7_label_v4",
    "bclean_min_hard_defined": 2400,
    "huri_grade1": "huri_observed_only",
    "huri_sensitivities": ["huri_new_labels", "huri_grade1_unrestricted"],
    "huri_expected_anchors": [769, 5],
    "min_checkpoints_A": 17,
    "min_complete_folds_B": 4,
    "gate2_tolerance_abs": dict(GATE2_TOLERANCE_ABS),
    "gate2_selected_weight": "exact_except_near_tie<1e-4",
    "gate2_float_tie": dict(GATE2_FLOAT_TIE),
    "gate2_near_tie_cells": [list(c) for c in GATE2_NEAR_TIE_CELLS],
    "gate2_leaf_classes": GATE2_LEAF_CLASSES,
    "gate2_zero_shot": dict(GATE2_ZERO_SHOT),
    "training_order": list(TRAINING_ORDER),
    "gpu_name_must_contain": "L40S",
    "embedding_dim": EMBEDDING_DIM,
    "manifest_digest_rule": "sha256 of sorted '<sha256>  <basename>\\n' lines over chunk_*.npz",
    "per_anchor_metrics": "epf.rank_auc / epf.ranking_metrics per anchor",
}
