"""Inputs of the fusion study (protocol §4, §5).

* Tier A bases: DGZ's released ``mlp.th`` / ``deepgozero.th`` loaded strictly with
  ``go_function.model.DgzCompatModel`` on InterPro features built from the split's
  DGZ pickle (``go_function.features.interpro_csr``); logits from
  ``go_function.model.predict_probs(..., sigmoid=False)``.
* Tier B bases: TP-trained ``all`` checkpoints (``state_dict`` written by
  ``go_function.train.run``) rebuilt with ``go_function.model.build_model`` from the
  run's materialized spec and ``config/go_function/study.json``.
* TP labels: ``go_function.labels.load_labels`` when the test split is requested
  (never in search mode); otherwise only the train/valid members of the npz are
  read (``go_function.pins.load_csr``), so a search process never materialises test
  labels.
* Evaluation sets (§5): proteins with >= 1 TP label in the aspect; terms with
  0 < n_pos < n, root excluded, in confirmatory mode also alt-id duplicate columns
  and columns without a primary id excluded.

Every file read goes through :mod:`go_function_fusion.guard`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from go_function_fusion import ROOT, guard

DGZ = "dataset/go_function/dgz"
PRE = "preprocess/go_function"
MODES = ("confirmatory", "pilot")
TIER_A = ("mlp", "deepgozero")
BASE_NAMES = {"mlp": "MLP", "deepgozero": "DGZ"}


def rp(rel: str) -> Path:
    return ROOT / rel


# ------------------------------------------------------------------ labels
def labels_path(aspect: str, regime: str = "TP") -> Path:
    return rp(f"{PRE}/{aspect}/labels_{regime}.npz")


def load_tp_labels(aspect: str, splits: Sequence[str] = ("train", "valid"),
                   path: Optional[Path] = None) -> Dict[str, Any]:
    """TP labels of ``splits`` (CSR, rows = the split's proteins in DGZ order),
    ``proteins_<split>`` and ``vocab``. With ``test`` in ``splits`` the whole file is
    read through ``go_function.labels.load_labels`` (refused in search mode)."""
    path = Path(path) if path is not None else labels_path(aspect)
    for s in splits:
        guard.check_split(s)
    guard.check_path(path)
    if "test" in splits:
        from go_function.labels import load_labels
        return load_labels(path)
    from go_function.pins import load_csr
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as z:
        out["vocab"] = [str(x) for x in z["vocab"]]
        for s in splits:
            out[s] = load_csr(z, s)
            out[f"proteins_{s}"] = [str(x) for x in z[f"proteins_{s}"]]
    return out


def verify_tp_labels(aspect: str) -> Dict[str, str]:
    """The TP label npz hash equals its manifest's, and the manifest equals the GF
    G1-labels pin (the npz is verified as a whole file, §10.3)."""
    from go_function.pins import sha256_file
    man_rel = f"{PRE}/{aspect}/labels_TP.json"
    guard.check_path(rp(man_rel))
    man_sha = sha256_file(rp(man_rel))
    pins_p = rp("results/go_function/protocol_pins.json")
    guard.check_path(pins_p)
    want = json.loads(pins_p.read_text())["pins"]["G1-labels"][f"{aspect}/labels_TP.json"]
    want = want.get("sha256") if isinstance(want, dict) else want
    if want != man_sha:
        raise ValueError(f"{man_rel}: sha256 differs from GF pin G1-labels")
    man = json.loads(rp(man_rel).read_text())
    npz_sha = sha256_file(labels_path(aspect))
    if npz_sha != man["npz_sha256"]:
        raise ValueError(f"labels_TP.npz ({aspect}) differs from its manifest")
    return {man_rel: man_sha, man["npz"]: npz_sha}


def fusion_study() -> Dict[str, Any]:
    p = rp("config/go_function_fusion/study.json")
    guard.check_path(p)
    return json.loads(p.read_text())


def n_train(y_train: sp.spmatrix) -> np.ndarray:
    """TP train positives per column (int64)."""
    return np.asarray(sp.csr_matrix(y_train).sum(axis=0)).ravel().astype(np.int64)


# ------------------------------------------------------------------ evaluation sets
def eval_rows(y: sp.spmatrix) -> np.ndarray:
    """Rows (proteins) with >= 1 label."""
    return np.flatnonzero(np.asarray(sp.csr_matrix(y).sum(axis=1)).ravel() > 0).astype(np.int64)


def root_column(terms: Sequence[str], aspect: str) -> int:
    from go_function.constants import ROOTS
    return list(terms).index(ROOTS[aspect]) if ROOTS[aspect] in terms else -1


def candidate_columns(vocab, root: int, mode: str) -> np.ndarray:
    """Columns that may be evaluated before the 0 < n_pos < n filter."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    keep = np.ones(len(vocab), dtype=bool)
    if root >= 0:
        keep[root] = False
    if mode == "confirmatory":
        for c in range(len(vocab)):
            if vocab.primary[c] is None or c in vocab.duplicate_cols:
                keep[c] = False
    return np.flatnonzero(keep)


def eval_terms(y_rows: sp.spmatrix, vocab, root: int, mode: str) -> np.ndarray:
    """Evaluated columns: candidates with 0 < n_pos < n on ``y_rows`` (eval rows)."""
    y = sp.csr_matrix(y_rows)
    npos = np.asarray(y.sum(axis=0)).ravel()
    cand = candidate_columns(vocab, root, mode)
    return cand[(npos[cand] > 0) & (npos[cand] < y.shape[0])]


# ------------------------------------------------------------------ aspect context
@dataclass
class AspectContext:
    aspect: str
    terms: list
    ont: Any
    vocab: Any
    anc: Mapping[str, frozenset]
    desc: Mapping[str, frozenset]
    root: int
    y_train: sp.csr_matrix
    ntr: np.ndarray
    labels: Dict[str, Any]


def aspect_context(aspect: str, splits: Sequence[str] = ("train", "valid"),
                   data_dir: Optional[Path] = None, labels_file: Optional[Path] = None
                   ) -> AspectContext:
    from go_function.labels import build_vocab, read_terms
    from go_function.ontology import ancestors, descendants, parse_obo
    d = Path(data_dir) if data_dir is not None else rp(DGZ)
    guard.check_path(d / aspect / "terms.pkl")
    terms = read_terms(d / aspect / "terms.pkl")
    lab = load_tp_labels(aspect, splits, labels_file)
    if lab["vocab"] != list(terms):
        raise ValueError("labels vocabulary differs from terms.pkl order")
    guard.check_path(d / "go.obo")
    ont = parse_obo(d / "go.obo")
    vocab = build_vocab(terms, ont)
    anc = ancestors(ont)
    ytr = sp.csr_matrix(lab["train"])
    return AspectContext(aspect=aspect, terms=list(terms), ont=ont, vocab=vocab, anc=anc,
                         desc=descendants(anc), root=root_column(terms, aspect), y_train=ytr,
                         ntr=n_train(ytr), labels=lab)


# ------------------------------------------------------------------ DGZ split frames, features
def read_split_frame(aspect: str, split: str, data_dir: Optional[Path] = None):
    import pandas as pd
    guard.check_split(split)
    d = Path(data_dir) if data_dir is not None else rp(DGZ)
    p = d / aspect / f"{split}_data.pkl"
    guard.check_path(p, split)
    return pd.read_pickle(p)


def interpro_features(aspect: str, frame, data_dir: Optional[Path] = None) -> sp.csr_matrix:
    from go_function.features import interpro_csr, read_interpro_vocab
    d = Path(data_dir) if data_dir is not None else rp(DGZ)
    guard.check_path(d / aspect / "interpros.pkl")
    ipr = read_interpro_vocab(d / aspect / "interpros.pkl")
    x, _ = interpro_csr(frame.interpros.tolist(), {g: i for i, g in enumerate(ipr)})
    return x


# ------------------------------------------------------------------ base models
def load_released(aspect: str, kind: str, n_in: int, terms: Sequence[str],
                  data_dir: Optional[Path] = None, device: str = "cpu"):
    """Tier A: DGZ's released checkpoint, strict load (GF G3)."""
    from go_function.model import DgzCompatModel
    if kind not in TIER_A:
        raise ValueError(f"Tier A kind must be one of {TIER_A}")
    d = Path(data_dir) if data_dir is not None else rp(DGZ)
    nf = None
    if kind == "deepgozero":
        from go_function.normforms import load_normal_forms
        guard.check_path(d / "go.norm")
        nf = load_normal_forms(d / "go.norm", {g: i for i, g in enumerate(terms)})
    guard.check_path(d / aspect / f"{kind}.th")
    return DgzCompatModel.load(d / aspect / f"{kind}.th", kind=kind, n_in=n_in,
                               n_terms=len(terms), nf=nf, device=device)


def base_logits(model, x, device: str = "cpu") -> np.ndarray:
    from go_function.model import predict_probs
    return predict_probs(model, x, device=device, sigmoid=False)


def tier_a_logits(aspect: str, kind: str, split: str, terms: Sequence[str],
                  proteins: Sequence[str], data_dir: Optional[Path] = None,
                  device: str = "cpu") -> np.ndarray:
    """Logits (n_split x |V|, float32) of a released checkpoint on a DGZ split; the
    split pickle's protein order must equal the label file's. Proteins without
    InterPro get the zero vector (as DGZ)."""
    import gc
    df = read_split_frame(aspect, split, data_dir)
    if list(df.proteins) != list(proteins):
        raise ValueError(f"{split}_data.pkl protein order differs from the label file")
    x = interpro_features(aspect, df, data_dir)
    del df
    model = load_released(aspect, kind, x.shape[1], terms, data_dir, device)
    L = base_logits(model, x, device)
    del model
    gc.collect()
    return L


def load_study_config() -> Dict[str, Any]:
    p = rp("config/go_function/study.json")
    guard.check_path(p)
    return json.loads(p.read_text())


def tier_b_features(spec: Mapping[str, Any], split: str, proteins: Sequence[str]):
    """Features of a TP-trained base for ``split`` (InterPro CSR or ESM view)."""
    guard.check_split(split)
    from go_function.pins import load_csr
    a = spec["aspect"]
    if spec["feature"] == "interpro":
        p = rp(f"{PRE}/{a}/features_interpro.npz")
        guard.check_path(p)
        with np.load(p, allow_pickle=False) as z:
            if [str(q) for q in z[f"proteins_{split}"]] != list(proteins):
                raise ValueError(f"InterPro {split} rows differ from the label rows")
            return load_csr(z, split)
    from go_function.features import esm_view
    ep, pp = rp(f"{PRE}/esm2_650m/embeddings.npy"), rp(f"{PRE}/esm2_650m/proteins.txt")
    guard.check_path(ep)
    guard.check_path(pp)
    emb = np.load(ep, allow_pickle=False, mmap_mode="r")
    return esm_view(emb, pp.read_text().splitlines(), list(proteins))


def load_tier_b(spec: Mapping[str, Any], checkpoint: Path, vocab, n_in: int,
                device: str = "cpu"):
    """Rebuild a TP-trained base (V0/V4, InterPro or ESM) and load its state dict
    strictly. ``spec`` is the GF materialized run spec (read only)."""
    import torch
    from go_function.model import build_model
    study = load_study_config()
    nf = None
    if spec["variant"] != "V0":
        from go_function import normforms as NF
        guard.check_path(rp(f"{DGZ}/go.norm"))
        nf = NF.select(NF.load_normal_forms(rp(f"{DGZ}/go.norm"), dict(vocab.index)),
                       spec["variant"])
    model = build_model(spec["variant"], n_in, len(vocab), nf, hidden_dim=int(study["dim"]),
                        norm=study["norm"],
                        dropout=float(spec.get("dropout") if spec.get("dropout") is not None
                                      else study["dropout"]),
                        margin=float(study["margin"]),
                        nf4_radius=spec.get("nf4_radius", study.get("nf4_radius", "dgz")))
    guard.check_path(checkpoint)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
