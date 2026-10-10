"""Cached-embedding evaluator for the GO/PPI withholding study (design 9).

Subcommands:

``build-cache`` (design 9.1, gate G2)
    Checks the test sha against ``frozen_hashes.json``, takes the unique test
    proteins in record order and their texts via ``content_to_text``, sets the
    frozen MPNet's ``max_seq_length = 512`` (the value ``trainer.py`` sets),
    asserts ``max_tok < 512`` (exit 2 otherwise: "text would be truncated"),
    encodes (batch 64, float32) and writes ``test_<sha16>.npz`` (``protein_ids``,
    ``emb``, ``rec_anchor``, ``rec_partner``, ``grade``, ``query``) and
    ``test_<sha16>.meta.json``; ``append_pin("G2-cache", ...)`` (hashes only).

``score`` (design 9.2-9.5)
    Resolves the final checkpoint and its sha from ``lc_manifest.json``, loads
    ``CareerAwareContrastiveModel(768, 128, 0.3, False)`` strictly, projects the
    cache and writes ``eval/scores.npy`` and ``eval/metrics.json`` (global,
    per-target, collateral). ``--relation`` adds ``eval/relation.json`` (9.3).
    ``--fusion-weight-file`` adds the exploratory simGIC fusion reference (9.4).
    ``--encoder-crosscheck --config <cell config>`` runs the G6 trainer-encoder
    cross-check first (9.2 steps 1-6). NaN/inf scores, a checkpoint/epoch/sha
    mismatch, a target without grade-2 or grade-1 records, or an imported repo
    module outside the code pin are fatal: ``eval/eval_error.json``, exit 2.

Usage (protocol section 10)::

    "$PY" "$S/evaluate_withholding.py" build-cache --test "$R/preprocess/go_withholding/splits/test.jsonl" \\
      --frozen-hashes "$R/preprocess/go_withholding/splits/frozen_hashes.json" \\
      --out "$R/results/go_withholding/eval_cache"
    "$PY" "$S/evaluate_withholding.py" score --run-dir <run dir> --cache <npz> \\
      --target-table "$R/preprocess/go_withholding/targets/target_table.json" --relation
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from go_ppi_domain import withholding as wh  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402
from evaluate_publication_fusion import (  # noqa: E402
    json_safe, rank_auc, ranking_metrics, z_apply, z_stats)

logger = logging.getLogger("evaluate_withholding")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_SEQ_LENGTH = 512
ENCODE_BATCH = 64
INPUT_DIM, PROJECTION_DIM, DROPOUT = 768, 128, 0.3
N_GLOBAL_PAIRS = 20000
N_TARGET_PAIRS = 5000
N_RANDOM_REF = 5000
N_CROSSCHECK = 50
CROSSCHECK_MIN_COS = 0.9999


class EvalError(RuntimeError):
    """Fatal for the run: ``eval/eval_error.json``, exit 2."""

    def __init__(self, msg: str, ids: Sequence[str] = ()):
        super().__init__(msg)
        self.ids = list(ids)[:5]


# ================================================================ helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(wc.relativize_json(json_safe(obj)), indent=1, sort_keys=True)
                   + "\n", encoding="utf-8")
    tmp.replace(path)


def load_encoder(model_name: str, device: Optional[str] = None):
    """The frozen MPNet (tests replace this with a stub)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _content_to_text():
    from run_phase1_embedding_evaluation import content_to_text

    return content_to_text


def cache_paths(out_dir: Path, test_sha256: str) -> Tuple[Path, Path]:
    stem = f"test_{test_sha256[:16]}"
    return Path(out_dir) / f"{stem}.npz", Path(out_dir) / f"{stem}.meta.json"


def meta_path_for(npz: Path) -> Path:
    npz = Path(npz)
    return npz.with_name(npz.name[:-len(".npz")] + ".meta.json")


def load_cache(npz: Path) -> Dict[str, Any]:
    with np.load(npz, allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    cache["protein_ids"] = [str(p) for p in cache["protein_ids"]]
    cache["query"] = np.asarray([str(q) for q in cache["query"]], dtype=object)
    cache["grade"] = np.asarray(cache["grade"], dtype=int)
    cache["rec_anchor"] = np.asarray(cache["rec_anchor"], dtype=np.int64)
    cache["rec_partner"] = np.asarray(cache["rec_partner"], dtype=np.int64)
    cache["emb"] = np.asarray(cache["emb"], dtype=np.float32)
    meta = meta_path_for(npz)
    cache["meta"] = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else None
    return cache


def read_test_records(test: Path) -> Dict[str, Any]:
    """Unique proteins in record order, their texts, and per-record indices."""
    content_to_text = _content_to_text()
    ids: List[str] = []
    texts: Dict[str, str] = {}
    pos: Dict[str, int] = {}
    anchors, partners, grades, queries = [], [], [], []
    with open(test, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec["metadata"]
            a, b = str(meta["protein_id"]), str(meta["partner_id"])
            for pid, slot, kind in ((a, rec["resume"], "resume"), (b, rec["job"], "job")):
                text = content_to_text(slot, kind)
                if not text.strip():
                    raise EvalError(f"blank text for {pid}", [pid])
                if pid not in pos:
                    pos[pid] = len(ids)
                    ids.append(pid)
                    texts[pid] = text
                elif texts[pid] != text:
                    raise EvalError(f"protein {pid} has two different texts in {test.name}",
                                    [pid])
            anchors.append(pos[a])
            partners.append(pos[b])
            grades.append(int(meta["go_grade"]))
            queries.append(str(meta.get("resume_id", a)))
    return {"ids": ids, "texts": [texts[p] for p in ids],
            "rec_anchor": np.asarray(anchors, dtype=np.int64),
            "rec_partner": np.asarray(partners, dtype=np.int64),
            "grade": np.asarray(grades, dtype=np.int64),
            "query": np.asarray(queries, dtype=str)}


def max_token_length(tokenizer, texts: Iterable[str]) -> int:
    return max((len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts),
               default=0)


# ------------------------------------------------------------------ simGIC
class SimGic:
    """Pairwise simGIC over the pruned BP index closures (the GoRelationTable formula)."""

    def __init__(self, index) -> None:
        self.index = index
        self._c: Dict[str, frozenset] = {}
        self._s: Dict[str, float] = {}

    def _closure(self, pid: str) -> frozenset:
        c = self._c.get(pid)
        if c is None:
            c = frozenset(self.index.closure_for(pid))
            self._c[pid] = c
            self._s[pid] = float(sum(float(self.index.ic.get(t, 0.0)) for t in c))
        return c

    def __call__(self, a: str, b: str) -> float:
        ca, cb = self._closure(a), self._closure(b)
        inter = float(sum(float(self.index.ic.get(t, 0.0)) for t in ca & cb))
        union = self._s[a] + self._s[b] - inter
        return min(1.0, max(0.0, inter / union)) if union > 0 else 0.0


def load_measurement_index(converted: Path):
    from go_ppi_domain.go_ontology import GoIndex

    path = Path(converted) / "go_index_P.pkl"
    if not path.is_file():
        raise EvalError(f"measurement index {wc.rel_to_root(path)} does not exist")
    return GoIndex.load(path)


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    from scipy.stats import spearmanr

    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    r = spearmanr(x, y).statistic
    return float(r) if math.isfinite(float(r)) else None


def global_pairs(n: int, n_pairs: int = N_GLOBAL_PAIRS, seed: int = 0) -> List[Tuple[int, int]]:
    """``n_pairs`` index pairs (i != j) drawn once with ``random.Random(seed)``."""
    if n < 2:
        return []
    rng = random.Random(seed)
    return [tuple(rng.sample(range(n), 2)) for _ in range(n_pairs)]


# ================================================================ build-cache
def build_cache(test: Path, frozen_hashes: Path, out: Path, converted: Optional[Path],
                model_name: str = MODEL_NAME, device: Optional[str] = None) -> Dict[str, Any]:
    frozen = json.loads(Path(frozen_hashes).read_text(encoding="utf-8"))
    test_sha = wc.sha256_file(test)
    if test_sha != frozen.get("test_sha256"):
        raise EvalError(f"test sha256 {test_sha} != frozen_hashes[test_sha256] "
                        f"{frozen.get('test_sha256')}")
    wc.assert_writable_target(out)
    npz, meta_path = cache_paths(out, test_sha)
    if npz.exists() or meta_path.exists():
        raise EvalError(f"refusing to overwrite {wc.rel_to_root(npz)}")
    data = read_test_records(test)
    encoder = load_encoder(model_name, device)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    if int(encoder.max_seq_length) != MAX_SEQ_LENGTH:
        raise EvalError(f"encoder.max_seq_length {encoder.max_seq_length} != {MAX_SEQ_LENGTH}")
    max_tok = max_token_length(encoder.tokenizer, data["texts"])
    if max_tok >= MAX_SEQ_LENGTH:
        raise EvalError(f"text would be truncated: max_tok {max_tok} >= {MAX_SEQ_LENGTH}")
    emb = np.asarray(encoder.encode(data["texts"], batch_size=ENCODE_BATCH,
                                    convert_to_numpy=True, show_progress_bar=False),
                     dtype=np.float32)
    if emb.shape != (len(data["ids"]), INPUT_DIM) or not np.isfinite(emb).all():
        raise EvalError(f"bad embedding matrix shape {emb.shape} or non-finite values")

    frozen_rho = None
    if converted is not None and (Path(converted) / "go_index_P.pkl").is_file():
        sim = SimGic(load_measurement_index(converted))
        unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        pairs = global_pairs(len(data["ids"]))
        cos = [float(unit[i] @ unit[j]) for i, j in pairs]
        sg = [sim(data["ids"][i], data["ids"][j]) for i, j in pairs]
        frozen_rho = _spearman(cos, sg)

    out.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(f".{npz.stem}.tmp.npz")
    np.savez(tmp, protein_ids=np.asarray(data["ids"], dtype=str), emb=emb,
             rec_anchor=data["rec_anchor"], rec_partner=data["rec_partner"],
             grade=data["grade"], query=data["query"])
    tmp.replace(npz)
    meta = {"test_sha256": test_sha, "n_records": int(len(data["grade"])),
            "n_proteins": len(data["ids"]), "model": model_name,
            "max_seq_length": MAX_SEQ_LENGTH, "max_tok": int(max_tok), "created": _now(),
            "frozen_global_spearman": frozen_rho}
    _write_json(meta_path, meta)
    pin = {"cache_npz_sha256": wc.sha256_file(npz), "meta_sha256": wc.sha256_file(meta_path),
           "test_sha256": test_sha, "n_proteins": meta["n_proteins"],
           "n_records": meta["n_records"], "max_tok": meta["max_tok"]}
    wc.append_pin("G2-cache", pin)
    return {**meta, "cache": npz.name, **{k: pin[k] for k in ("cache_npz_sha256",)}}


# ================================================================ score
def resolve_checkpoint(run_dir: Path) -> Dict[str, Any]:
    """Design 9.2 step 1: the selected final checkpoint, verified against lc_manifest."""
    import torch

    lc = json.loads((Path(run_dir) / "lc_manifest.json").read_text(encoding="utf-8"))
    epochs = int(lc["epochs"])
    wh_block = lc.get("withholding") or {}
    name = wh_block.get("selected_checkpoint")
    expected = f"checkpoint_epoch_{epochs - 1}.pt"
    if name != expected:
        raise EvalError(f"selected_checkpoint {name!r} != {expected!r} (epochs {epochs})")
    path = Path(run_dir) / name
    if not path.is_file():
        raise EvalError(f"checkpoint {name} missing in {wc.rel_to_root(run_dir)}")
    sha = wc.sha256_file(path)
    if sha != wh_block.get("selected_checkpoint_sha256"):
        raise EvalError(f"{name} sha256 {sha} != lc_manifest "
                        f"{wh_block.get('selected_checkpoint_sha256')}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("epoch") != epochs - 1:
        raise EvalError(f"{name} records epoch {ckpt.get('epoch')} != {epochs - 1}")
    return {"lc": lc, "epochs": epochs, "path": path, "sha256": sha,
            "state": ckpt.get("model_state_dict", ckpt)}


def project(cache_emb: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    import torch
    from run_phase1_embedding_evaluation import CareerAwareContrastiveModel

    if cache_emb.shape[1] != INPUT_DIM:
        raise EvalError(f"cache embedding dim {cache_emb.shape[1]} != {INPUT_DIM}")
    model = CareerAwareContrastiveModel(input_dim=INPUT_DIM, projection_dim=PROJECTION_DIM,
                                        dropout=DROPOUT, use_structured_features=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(cache_emb, dtype=torch.float32)).numpy()


def _macro_query_auc(scores, grades, queries) -> Tuple[Optional[float], int]:
    vals = []
    for q in np.unique(queries):
        m = queries == q
        pos, neg = scores[m & (grades == 2)], scores[m & (grades == 1)]
        if len(pos) and len(neg):
            vals.append(rank_auc(pos, neg))
    return (float(np.mean(vals)) if vals else None), len(vals)


def _mrr(scores, grades, queries) -> Optional[float]:
    if not len(scores):
        return None
    return ranking_metrics(scores, grades, queries)["macro"]["reciprocal_rank"]


def global_metrics(scores, grades, queries) -> Dict[str, Any]:
    mq, _ = _macro_query_auc(scores, grades, queries)
    return {"hard_auc": rank_auc(scores[grades == 2], scores[grades == 1]),
            "easy_auc": rank_auc(scores[grades == 2], scores[grades == 0]),
            "pooled_auc": rank_auc(scores[grades == 2], scores[grades != 2]),
            "macro_query_hard_auc": mq, "macro_mrr": _mrr(scores, grades, queries),
            "n_queries": int(len(np.unique(queries)))}


def per_target_metrics(scores, grades, queries, table: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for t in table["targets"]:
        qs = set(t["test_anchors"])
        m = np.fromiter((q in qs for q in queries), dtype=bool, count=len(queries))
        pos, neg = scores[m & (grades == 2)], scores[m & (grades == 1)]
        if not len(pos) or not len(neg):
            raise EvalError(f"target {t['go_id']} has {len(pos)} grade-2 and {len(neg)} "
                            f"grade-1 test records (need >= 1 each)", [t["go_id"]])
        mq, _ = _macro_query_auc(scores[m], grades[m], queries[m])
        out[t["go_id"]] = {"queries": int(len(np.unique(queries[m]))), "n_pos": int(len(pos)),
                           "n_weak": int(len(neg)), "hard_auc": rank_auc(pos, neg),
                           "macro_query_hard_auc": mq,
                           "macro_mrr": _mrr(scores[m], grades[m], queries[m])}
    return out


def collateral_exclusion(table: Dict[str, Any], pack_id: str) -> set:
    excl: set = set()
    for t in table["targets"]:
        if t["pack_id"] == pack_id:
            excl |= set(t["test_anchors"]) | set(t.get("nbr_test_proteins_sib", []))
    return excl


def collateral_metrics(scores, grades, queries, anchors: Sequence[str],
                       table: Dict[str, Any], cond_id: str) -> Dict[str, Any]:
    """Queries whose anchor is outside the pack's P_test and NP_test_sib (design 9.2)."""
    pack, _k = wh.parse_cond_id(cond_id)
    packs = [p["pack_id"] for p in table["packs"]] if pack == "all" else [pack]
    out = {}
    for p in packs:
        excl = collateral_exclusion(table, p)
        m = np.fromiter((a not in excl for a in anchors), dtype=bool, count=len(anchors))
        mq, _ = _macro_query_auc(scores[m], grades[m], queries[m])
        out[p] = {"queries": int(len(np.unique(queries[m]))),
                  "hard_auc": rank_auc(scores[m & (grades == 2)], scores[m & (grades == 1)]),
                  "macro_query_hard_auc": mq}
    return out


def relation_probe(proj: np.ndarray, ids: List[str], table: Dict[str, Any],
                   simgic) -> Dict[str, Any]:
    """Design 9.3: global and target x neighbour relation preservation."""
    pos = {p: i for i, p in enumerate(ids)}
    pairs = global_pairs(len(ids))
    cos = [float(proj[i] @ proj[j]) for i, j in pairs]
    sg = [simgic(ids[i], ids[j]) for i, j in pairs]
    out = {"global": {"spearman": _spearman(cos, sg), "n_pairs": len(pairs)}, "per_target": {}}
    for t in table["targets"]:
        P = sorted(set(t["test_anchors"]) & set(pos))
        NP = sorted(set(t.get("nbr_test_proteins_sib", [])) & set(pos))
        row: Dict[str, Any] = {"n_P": len(P), "n_NP": len(NP), "n_pairs": 0, "spearman": None,
                               "target_nbr_mean_cos": None, "random_ref_mean_cos": None,
                               "excess": None}
        cross = [(a, b) for a in P for b in NP if a != b]
        if cross:
            if len(cross) > N_TARGET_PAIRS:
                cross = random.Random(0).sample(cross, N_TARGET_PAIRS)
            c = [float(proj[pos[a]] @ proj[pos[b]]) for a, b in cross]
            s = [simgic(a, b) for a, b in cross]
            ref_pool = sorted(set(ids) - set(NP) - set(P))
            ref = random.Random(0).sample(ref_pool, min(N_RANDOM_REF, len(ref_pool)))
            I = sorted({a for a, _ in cross})
            nbr = float(np.mean(c))
            row.update({"n_pairs": len(cross), "spearman": _spearman(c, s),
                        "target_nbr_mean_cos": nbr})
            if ref:
                rm = float(np.mean(proj[[pos[a] for a in I]] @ proj[[pos[r] for r in ref]].T))
                row.update({"random_ref_mean_cos": rm, "excess": nbr - rm, "n_ref": len(ref)})
        out["per_target"][t["go_id"]] = row
    return out


def fusion_reference(scores, cache, table, simgic, w: float) -> Dict[str, Any]:
    """Design 9.4 (exploratory): z(text) + w * z(simGIC), weight fixed upstream."""
    ids = cache["protein_ids"]
    sg = np.asarray([simgic(ids[a], ids[b])
                     for a, b in zip(cache["rec_anchor"], cache["rec_partner"])], dtype=float)
    fused = z_apply(scores.astype(float), z_stats(scores.astype(float))) + \
        float(w) * z_apply(sg, z_stats(sg))
    g, q = cache["grade"], cache["query"]
    per = {go: v["hard_auc"] for go, v in per_target_metrics(fused, g, q, table).items()}
    return {"w": float(w), "exploratory": True,
            "global_hard_auc": rank_auc(fused[g == 2], fused[g == 1]), "per_target_hard_auc": per}


# ------------------------------------------------------- encoder cross-check
def encoder_crosscheck(trainer, cache: Dict[str, Any], proteins_path: Path,
                       eval_dir: Path) -> Dict[str, Any]:
    """Design 9.2 ``--encoder-crosscheck`` steps 1-6 (G6 only)."""
    content_to_text = _content_to_text()
    enc = trainer.text_encoder
    if int(getattr(enc, "max_seq_length", -1)) != MAX_SEQ_LENGTH:
        raise EvalError(f"trainer.text_encoder.max_seq_length {enc.max_seq_length} != "
                        f"{MAX_SEQ_LENGTH}")
    views: Dict[str, str] = {}
    with open(proteins_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                views[str(row["protein_id"])] = row["encoder_view"]
    ids = cache["protein_ids"]
    missing = [p for p in ids if p not in views]
    if missing:
        raise EvalError(f"{len(missing)} cache proteins have no encoder_view", missing)
    to_text = trainer.batch_encoder._content_to_text
    tok = enc.tokenizer
    mismatch, n_differs = [], 0
    trainer_texts: Dict[str, str] = {}
    for p in ids:
        slot = {"encoder_view": views[p]}
        tt = to_text(slot, "resume")
        et = content_to_text(slot, "resume")
        trainer_texts[p] = tt
        if tt != et:
            n_differs += 1
        if tok(tt)["input_ids"] != tok(et)["input_ids"]:
            mismatch.append(p)
    if mismatch:
        raise EvalError(f"trainer and evaluator text functions give different token ids "
                        f"for {len(mismatch)} cache proteins", mismatch)
    sample = random.Random(0).sample(sorted(ids), min(N_CROSSCHECK, len(ids)))
    re_emb = np.asarray(enc.encode([trainer_texts[p] for p in sample], batch_size=len(sample),
                                   convert_to_numpy=True), dtype=np.float64)
    pos = {p: i for i, p in enumerate(ids)}
    cached = cache["emb"][[pos[p] for p in sample]].astype(np.float64)
    cos = np.sum(re_emb * cached, axis=1) / np.maximum(
        np.linalg.norm(re_emb, axis=1) * np.linalg.norm(cached, axis=1), 1e-12)
    bad = [p for p, c in zip(sample, cos) if not c >= CROSSCHECK_MIN_COS]
    if bad:
        raise EvalError(f"encoder cross-check: {len(bad)} of {len(sample)} proteins below "
                        f"cos {CROSSCHECK_MIN_COS} (min {float(np.min(cos)):.6f})", bad)
    import hashlib

    result = {"n": len(sample), "min_cos": float(np.min(cos)) if len(cos) else None,
              "max_seq_length": MAX_SEQ_LENGTH,
              "ids_sha256": hashlib.sha256("\n".join(sample).encode()).hexdigest(),
              "n_token_checked": len(ids), "n_token_mismatch": 0, "n_text_differs": n_differs}
    _write_json(Path(eval_dir) / "encoder_crosscheck.json", result)
    return result


def build_crosscheck_trainer(config_path: Path, eval_dir: Path, device: Optional[str]):
    import go_ppi_domain.record_adapter  # noqa: F401  registers "go_ppi"
    from contrastive_learning.trainer import ContrastiveLearningTrainer
    from scripts.run_learning_curve_point import device_scope, resolve_device

    config = wc.load_study_config(config_path)
    resolved = resolve_device(device or getattr(config, "training_device", "auto"))
    with device_scope(resolved):
        return ContrastiveLearningTrainer(config=config,
                                          output_dir=str(Path(eval_dir) / "crosscheck_tmp"))


def check_import_coverage() -> None:
    imported = wc.imported_repo_modules()
    files = set(wc.code_sha256()[1])
    uncovered = sorted(set(imported) - files)
    if uncovered:
        raise EvalError(f"imported repo modules outside the code pin: {uncovered}", uncovered)


def score(run_dir: Path, cache_path: Path, table_path: Path, relation: bool = False,
          fusion_weight_file: Optional[Path] = None, crosscheck_config: Optional[Path] = None,
          converted: Optional[Path] = None, device: Optional[str] = None,
          trainer_factory=None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    converted = Path(converted or wc.CONVERTED)
    table = json.loads(Path(table_path).read_text(encoding="utf-8"))
    cache = load_cache(cache_path)
    if cache["meta"] is None:
        raise EvalError(f"cache meta {meta_path_for(cache_path).name} missing")

    if crosscheck_config is not None:
        factory = trainer_factory or build_crosscheck_trainer
        trainer = factory(Path(crosscheck_config), eval_dir, device)
        encoder_crosscheck(trainer, cache, converted / "proteins.jsonl", eval_dir)

    ck = resolve_checkpoint(run_dir)
    proj = project(cache["emb"], ck["state"])
    scores = np.sum(proj[cache["rec_anchor"]] * proj[cache["rec_partner"]], axis=1)
    scores = scores.astype(np.float32)
    if not np.isfinite(scores).all():
        raise EvalError(f"{int((~np.isfinite(scores)).sum())} non-finite scores")
    grades, queries = cache["grade"], cache["query"]
    ids = cache["protein_ids"]
    anchors = [ids[i] for i in cache["rec_anchor"]]
    lc = ck["lc"]
    cond_id = (lc.get("withholding") or {}).get("cond_id")
    if cond_id is None:
        raise EvalError("lc_manifest.withholding.cond_id missing")
    np.save(eval_dir / "scores.npy", scores)
    metrics: Dict[str, Any] = {
        "run_id": run_dir.name, "cond_id": cond_id,
        "test_sha256": cache["meta"]["test_sha256"], "checkpoint_sha256": ck["sha256"],
        "checkpoint": ck["path"].name, "scores_sha256": wc.sha256_file(eval_dir / "scores.npy"),
        "epochs": ck["epochs"], "n_records": int(len(scores)),
        "global": global_metrics(scores, grades, queries),
        "per_target": per_target_metrics(scores, grades, queries, table),
        "collateral": collateral_metrics(scores, grades, queries, anchors, table, cond_id)}
    simgic = None
    if relation or fusion_weight_file is not None:
        simgic = SimGic(load_measurement_index(converted))
    if fusion_weight_file is not None:
        fw = json.loads(Path(fusion_weight_file).read_text(encoding="utf-8"))
        metrics["fusion_reference"] = fusion_reference(scores, cache, table, simgic,
                                                       float(fw.get("w", fw.get("weight"))))
    if relation:
        rel = relation_probe(proj.astype(np.float64), ids, table, simgic)
        rel.update({"run_id": run_dir.name, "cond_id": cond_id})
        _write_json(eval_dir / "relation.json", rel)
    check_import_coverage()
    _write_json(eval_dir / "metrics.json", metrics)
    return metrics


def _write_eval_error(run_dir: Path, exc: BaseException) -> None:
    payload = {"error": wc.scrub_root(f"{type(exc).__name__}: {exc}"),
               "ids": getattr(exc, "ids", []),
               "traceback": wc.scrub_root("".join(traceback.format_exception(exc))),
               "time": _now()}
    try:
        _write_json(Path(run_dir) / "eval" / "eval_error.json", payload)
    except OSError:
        logger.error("could not write eval_error.json")


# ================================================================ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-cache")
    b.add_argument("--test", type=Path, default=None)
    b.add_argument("--frozen-hashes", type=Path, default=None)
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--converted", type=Path, default=None,
                   help="for frozen_global_spearman (default: the study converted dir)")
    b.add_argument("--model", default=MODEL_NAME)
    b.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    s = sub.add_parser("score")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--cache", type=Path, required=True)
    s.add_argument("--target-table", type=Path, required=True)
    s.add_argument("--relation", action="store_true")
    s.add_argument("--fusion-weight-file", type=Path, default=None)
    s.add_argument("--encoder-crosscheck", action="store_true")
    s.add_argument("--config", type=Path, default=None,
                   help="cell config (required with --encoder-crosscheck)")
    s.add_argument("--converted", type=Path, default=None)
    s.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    args = ap.parse_args(argv)
    if args.cmd == "score" and args.encoder_crosscheck and args.config is None:
        ap.error("--encoder-crosscheck requires --config")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    if args.cmd == "build-cache":
        device = None if args.device in (None, "auto") else args.device
        try:
            summary = build_cache(
                Path(args.test or (Path(wc.SPLITS) / "test.jsonl")),
                Path(args.frozen_hashes or (Path(wc.SPLITS) / "frozen_hashes.json")),
                Path(args.out or (Path(wc.STUDY_RES) / "eval_cache")),
                Path(args.converted or wc.CONVERTED), args.model, device)
        except Exception as exc:  # noqa: BLE001 - every build-cache failure is exit 2
            logger.error("build-cache failed: %s: %s", type(exc).__name__, exc)
            return 2
        summary["wall_seconds"] = round(time.time() - t0, 1)
        print(json.dumps(json_safe(summary), indent=1, sort_keys=True))
        return 0
    try:
        m = score(args.run_dir, args.cache, args.target_table, relation=args.relation,
                  fusion_weight_file=args.fusion_weight_file,
                  crosscheck_config=args.config if args.encoder_crosscheck else None,
                  converted=args.converted, device=args.device)
    except Exception as exc:  # noqa: BLE001 - design 9.5: fatal for the run
        logger.error("score failed for %s: %s: %s", args.run_dir.name, type(exc).__name__, exc)
        _write_eval_error(args.run_dir, exc)
        return 2
    print(json.dumps(json_safe({"run_id": m["run_id"], "global_hard_auc": m["global"]["hard_auc"],
                                "wall_seconds": round(time.time() - t0, 1)}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
