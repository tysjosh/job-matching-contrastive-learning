"""Gate G2b: DGZ's released zero-shot model (protocol §10 G2b, §9.3; plan item 13).

For one aspect, on DGZ's own files:

  L1  deepgozero_zero.th loads strictly into our V4 port with terms_zero.pkl and
      go.norm's zero classes (DGZ order). A failure writes "loaded": false and the
      status "not checkable" (BP) / "not reproduced" (MF).
  L2  for each withheld class (the zero_<GO>_auc.pkl files; = terms.pkl minus
      terms_zero.pkl), our zero-shot AUC (DGZ ``predict_zero``: sigmoid of
      <f(x), e_t + r_hasFunc> + |rho_t|, float32; ``roc_curve`` + ``auc``; DGZ-rule
      labels = prop_annotations) on each evaluation set DGZ used - test
      (zero_<GO>_auc.pkl) and train+valid+test (zero_<GO>_auc_all.pkl) - must be
      within 0.01 of the AUC of the released curve. Criterion: BP >= 6 of 7, MF >= 4
      of 5 classes matching on both sets.
  L3  (BP only; TP test labels) for each withheld class t with a regulates-family
      filler X (logical definition) that is a trained column: AUC_{X->t} = class AUC
      of the zero model's X logit for t's TP test labels. A class passes if
      AUC_{X->t} >= AUC_zero(t) - 0.02 and >= 30% of X's DGZ-rule train positives
      carry t. Confirmed if >= 5 of 7 classes pass.

  BP status: confirmed = L1 and L2 and L3; not checkable = not L1 or not L2;
  otherwise not confirmed. MF: "reproduced" (L1 and L2) or "not reproduced".

Descriptive (no criterion): the AUC under zero_annotations labels on both sets;
the trained model's AUC for t from predictions_deepgozero.pkl vs
zero_<GO>_auc_train.pkl (DGZ evaluate_terms.py writes that file from a trained
model's predictions); and predictions_deepgozero_zero.pkl scored with our
DGZ-faithful evaluator when its column count matches a released terms file.

Writes the aspect section of results/go_function/gates/gate_dgz_zero.json and pin
G2b entry "<aspect>". G2b never blocks: exit 0 whatever the status.

Exit codes: 0 status written; 2 fatal (inputs, differing existing section/pin).

Usage:
  .venv/bin/python scripts/go_function/gate_dgz_zero.py --aspect bp --device cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function.constants import EXIT_FATAL, EXIT_OK  # noqa: E402
from go_function.pins import (canonical_json, record_pin, sha256_bytes,  # noqa: E402
                              verify_inputs, write_json_section)
from go_function.runtime import DEVICES, FatalError, resolve_device, setup_logging  # noqa: E402

DGZ = "dataset/go_function/dgz"
PRE = "preprocess/go_function"
GATE_OUT = ROOT / "results" / "go_function" / "gates" / "gate_dgz_zero.json"
AUC_TOL = 0.01
L2_NEED = {"bp": 6, "mf": 4}
L3_NEED = 5
L3_AUC_SLACK = 0.02
L3_SHARE = 0.30
REG_FAMILY = ("regulates", "positively_regulates", "negatively_regulates")
SETS = {"test": "", "all": "_all"}
#: load_normal_forms keyword overrides (tests on fixtures only)
NF_KW: dict = {}


# ------------------------------------------------------------------ decisions
def l2_decision(aspect: str, per_class: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    n_match = sum(1 for c in per_class.values() if c.get("match"))
    return {"n_classes": len(per_class), "n_match": n_match, "need": L2_NEED[aspect],
            "passed": n_match >= L2_NEED[aspect]}


def l3_class_pass(auc_x: Optional[float], auc_zero: Optional[float], share: Optional[float]) -> bool:
    if auc_x is None or auc_zero is None or share is None:
        return False
    return auc_x >= auc_zero - L3_AUC_SLACK and share >= L3_SHARE


def status_of(aspect: str, l1: bool, l2: Optional[bool], l3: Optional[bool]) -> str:
    """Protocol §10 G2b / §9.3."""
    if aspect == "mf":
        return "reproduced" if (l1 and l2) else "not reproduced"
    if not l1 or not l2:
        return "not checkable"
    return "confirmed" if l3 else "not confirmed"


# ------------------------------------------------------------------ helpers
def released_auc(path: Path) -> float:
    import pandas as pd
    from sklearn.metrics import auc
    df = pd.read_pickle(path)
    return float(auc(df["fpr"].values, df["tpr"].values))


def roc_auc(y, s) -> Optional[float]:
    import numpy as np
    from sklearn.metrics import auc, roc_curve
    y = np.asarray(y, dtype=np.float32)
    if not 0 < y.sum() < len(y):
        return None
    fpr, tpr, _ = roc_curve(y, np.asarray(s))
    return float(auc(fpr, tpr))


def withheld_classes(data_dir: Path, aspect: str, terms: Sequence[str],
                     terms_zero: Sequence[str]) -> List[str]:
    files = sorted(p.name for p in (data_dir / aspect).glob("zero_GO:*_auc.pkl"))
    from_files = sorted(re.match(r"zero_(GO:\d{7})_auc\.pkl", f).group(1) for f in files)
    dropped = sorted(set(terms) - set(terms_zero))
    if from_files != dropped:
        raise FatalError(f"withheld classes from zero_<GO>_auc.pkl files ({len(from_files)}) "
                         f"differ from terms.pkl minus terms_zero.pkl ({len(dropped)})")
    return from_files


def default_l3_labels(aspect: str) -> Dict[str, Any]:
    """TP test labels and DGZ-rule train labels (G1 outputs, pinned)."""
    from go_function import labels as L
    from go_function.pins import sha256_file, verify_pin
    from go_function.runtime import repo_path
    out: Dict[str, Any] = {}
    for regime in ("TP", "DGZ"):
        rel = f"{PRE}/{aspect}/labels_{regime}.json"
        mb = repo_path(rel).read_bytes()
        verify_pin("G1-labels", {f"{aspect}/labels_{regime}.json": sha256_bytes(mb)})
        man = json.loads(mb)
        if sha256_file(repo_path(man["npz"])) != man["npz_sha256"]:
            raise FatalError(f"{man['npz']} differs from its manifest")
        lab = L.load_labels(repo_path(man["npz"]))
        out[regime] = {"vocab": lab["vocab"], "test": lab["test"], "train": lab["train"],
                       "proteins_test": lab["proteins_test"]}
    return out


# ------------------------------------------------------------------ the gate
def evaluate_aspect(aspect: str, data_dir: Path, device: str, log,
                    l3_labels: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    from go_function import evaluate as E
    from go_function.features import interpro_csr, read_interpro_vocab
    from go_function.labels import read_terms
    from go_function.model import CheckpointMismatch, DgzCompatModel, predict_probs
    from go_function.normforms import load_normal_forms
    from go_function.ontology import DgzRule, logical_definition, parse_obo
    terms = read_terms(data_dir / aspect / "terms.pkl")
    terms_zero = read_terms(data_dir / aspect / "terms_zero.pkl")
    zdict = {g: i for i, g in enumerate(terms_zero)}
    classes = withheld_classes(data_dir, aspect, terms, terms_zero)
    ipr = read_interpro_vocab(data_dir / aspect / "interpros.pkl")
    ipr_index = {g: i for i, g in enumerate(ipr)}
    nf = load_normal_forms(data_dir / "go.norm", zdict, **NF_KW)
    out: Dict[str, Any] = {"withheld_classes": classes, "n_terms_zero": len(terms_zero),
                           "n_zero_classes": len(nf.class_index) - nf.n_vocab}
    # L1
    try:
        model = DgzCompatModel.load(data_dir / aspect / "deepgozero_zero.th", kind="deepgozero",
                                    n_in=len(ipr), n_terms=len(terms_zero), nf=nf, device=device)
        out["L1"] = {"loaded": True, "norm": model.norm_type}
    except CheckpointMismatch as e:
        log.warning("L1: deepgozero_zero.th does not load: %s", e)
        out["L1"] = {"loaded": False, "error": str(e)[:2000]}
        out["L2"] = {"passed": False, "reason": "L1 failed"}
        out["L3"] = None if aspect == "mf" else {"passed": False, "reason": "L1 failed"}
        out["status"] = status_of(aspect, False, False, False)
        return out
    missing = [t for t in classes if t not in nf.class_index]
    idx = torch.as_tensor([nf.class_index[t] for t in classes if t in nf.class_index],
                          dtype=torch.long)
    present = [t for t in classes if t in nf.class_index]

    frames = {}
    for split in ("train", "valid", "test"):
        df = pd.read_pickle(data_dir / aspect / f"{split}_data.pkl")
        frames[split] = {"proteins": [str(p) for p in df["proteins"].tolist()],
                         "x": interpro_csr(df["interpros"].tolist(), ipr_index)[0],
                         "prop": [set(x) for x in df["prop_annotations"].tolist()],
                         "zero": [set(x) for x in df["zero_annotations"].tolist()]}
        del df
        gc.collect()
    scores = {s: predict_probs(model, frames[s]["x"], device=device, idx=idx)
              for s in ("train", "valid", "test")}
    sets = {"test": ("test",), "all": ("train", "valid", "test")}

    def labels_for(t: str, splits, col: str):
        return np.concatenate([np.asarray([t in r for r in frames[s][col]], dtype=np.float32)
                               for s in splits])

    per_class: Dict[str, Any] = {}
    for j, t in enumerate(present):
        e: Dict[str, Any] = {}
        ok = True
        for name, splits in sets.items():
            s = np.concatenate([scores[sp][:, j] for sp in splits])
            ours = roc_auc(labels_for(t, splits, "prop"), s)
            ours_zero = roc_auc(labels_for(t, splits, "zero"), s)
            rel = released_auc(data_dir / aspect / f"zero_{t}_auc{SETS[name]}.pkl")
            match = ours is not None and abs(ours - rel) <= AUC_TOL
            ok = ok and match
            e[name] = {"ours": ours, "released": rel,
                       "diff": None if ours is None else ours - rel, "match": match,
                       "ours_zero_annotations_labels": ours_zero,
                       "n_pos": int(labels_for(t, splits, "prop").sum()),
                       "n": int(sum(len(frames[sp]["prop"]) for sp in splits))}
        e["match"] = ok
        per_class[t] = e
    for t in missing:
        per_class[t] = {"match": False, "reason": "not a class of go.norm with terms_zero.pkl"}
    out["L2"] = {**l2_decision(aspect, per_class), "per_class": per_class, "tolerance": AUC_TOL,
                 "labels": "prop_annotations (DGZ rule)",
                 "sets": {"test": "zero_<GO>_auc.pkl", "all": "zero_<GO>_auc_all.pkl"}}
    log.info("%s L1 loaded; L2 %d/%d classes match (need %d)", aspect, out["L2"]["n_match"],
             len(classes), L2_NEED[aspect])

    # L3 (BP): channel X -> t under TP test labels
    if aspect == "bp":
        lab = (l3_labels or default_l3_labels)(aspect)
        if lab["TP"]["proteins_test"] != frames["test"]["proteins"]:
            raise FatalError("TP test label rows differ from test_data.pkl")
        vidx = {g: i for i, g in enumerate(lab["TP"]["vocab"])}
        ont = parse_obo(data_dir / "go.obo")
        zlogit = predict_probs(model, frames["test"]["x"], device=device, sigmoid=False)
        tlogit = predict_probs(model, frames["test"]["x"], device=device, idx=idx, sigmoid=False)
        l3: Dict[str, Any] = {}
        n_pass = 0
        for j, t in enumerate(present):
            y_t = lab["TP"]["test"][:, vidx[t]].toarray().ravel()
            auc_zero = roc_auc(y_t, tlogit[:, j])
            _, diff = logical_definition(ont, t)
            fillers = sorted({c for r, c in diff if r in REG_FAMILY})
            rows = []
            for X in fillers:
                if X not in zdict:
                    rows.append({"X": X, "trained_column": False})
                    continue
                auc_x = roc_auc(y_t, zlogit[:, zdict[X]])
                dg = lab["DGZ"]["train"]
                xp = dg[:, vidx[X]].toarray().ravel().astype(bool)
                tp = dg[:, vidx[t]].toarray().ravel().astype(bool)
                share = float((xp & tp).sum() / xp.sum()) if xp.sum() else None
                rows.append({"X": X, "trained_column": True, "auc_x_to_t": auc_x,
                             "share_x_train_positives_with_t": share,
                             "n_x_train_positives": int(xp.sum()),
                             "passed": l3_class_pass(auc_x, auc_zero, share)})
            passed = any(r.get("passed") for r in rows)
            n_pass += int(passed)
            l3[t] = {"auc_zero_tp": auc_zero, "n_pos_tp": int(y_t.sum()), "fillers": rows,
                     "passed": passed}
        out["L3"] = {"per_class": l3, "n_pass": n_pass, "need": L3_NEED,
                     "passed": n_pass >= L3_NEED, "labels": "TP test labels (labels_TP.npz)",
                     "rule": f"AUC_X->t >= AUC_zero(t) - {L3_AUC_SLACK} and share >= {L3_SHARE}"}
        log.info("bp L3 %d/%d classes pass (need %d)", n_pass, len(classes), L3_NEED)
        del zlogit, tlogit
    else:
        out["L3"] = None
    del model, scores
    gc.collect()

    # descriptive: trained model's AUC for t vs zero_<GO>_auc_train.pkl
    trained = {}
    p = data_dir / aspect / "predictions_deepgozero.pkl"
    if p.exists():
        df = pd.read_pickle(p)
        tidx = {g: i for i, g in enumerate(terms)}
        preds = np.stack([np.asarray(r, dtype=np.float32)[[tidx[t] for t in classes]]
                          for r in df["preds"].tolist()])
        prop = [set(x) for x in df["prop_annotations"].tolist()]
        del df
        for j, t in enumerate(classes):
            y = np.asarray([t in r for r in prop], dtype=np.float32)
            f = data_dir / aspect / f"zero_{t}_auc_train.pkl"
            trained[t] = {"predictions_deepgozero": roc_auc(y, preds[:, j]),
                          "released_auc_train": released_auc(f) if f.exists() else None}
        del preds
    out["trained_model_auc"] = trained
    gc.collect()

    # descriptive: released zero-shot predictions with our DGZ-faithful evaluator
    p = data_dir / aspect / "predictions_deepgozero_zero.pkl"
    rz: Dict[str, Any] = {"file": p.name}
    if p.exists():
        df = pd.read_pickle(p)
        width = len(df["preds"].iloc[0])
        rz["n_columns"] = int(width)
        known = {"terms.pkl": terms, "terms_zero.pkl": terms_zero}
        for extra in ("terms_zero_10.pkl",):
            if (data_dir / aspect / extra).exists():
                known[extra] = read_terms(data_dir / aspect / extra)
        match = [k for k, v in known.items() if len(v) == width]
        rz["matching_terms_files"] = match
        if len(match) == 1:
            cols = known[match[0]]
            preds = np.stack(df["preds"].tolist())
            test_prop = [list(x) for x in df["prop_annotations"].tolist()]
            del df
            dgz = DgzRule(data_dir / "go.obo")
            tv = [list(r) for s in ("train", "valid") for r in frames[s]["prop"]]
            ic = E.dgz_ic(dgz, tv + test_prop)
            pm = E.dgz_protein_metrics(preds, cols, test_prop, E.dgz_namespace_terms(dgz, aspect),
                                       ic, device=device)
            cm = E.dgz_class_auc(preds, cols, test_prop)
            rz.update({"scored": True, "terms_file": match[0], **pm, "auc": cm["auc"],
                       "n_auc_terms": cm["n_terms"]})
            del preds
        else:
            rz.update({"scored": False, "reason": f"preds have {width} columns; no released "
                       f"terms file has that many ({ {k: len(v) for k, v in known.items()} })"})
            del df
    else:
        rz.update({"scored": False, "reason": "file absent"})
    out["released_zero_predictions"] = rz
    out["status"] = status_of(aspect, True, out["L2"]["passed"],
                              None if out["L3"] is None else out["L3"]["passed"])
    return out


def main(argv=None, *, data_dir: Path = None, out_path: Path = None, pins_path: Path = None,
         l3_labels=None, verify: bool = True) -> int:
    """Keyword overrides are for tests only (synthetic data outside the repo)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--device", choices=DEVICES, default="cpu")
    args = ap.parse_args(argv)
    data_dir = Path(data_dir) if data_dir is not None else ROOT / DGZ
    out_path = Path(out_path) if out_path is not None else GATE_OUT
    log = setup_logging("gate_dgz_zero")
    t0 = time.time()
    try:
        try:
            device = resolve_device(args.device)
        except ValueError as e:
            raise FatalError(str(e)) from None
        a = args.aspect
        hashes = {}
        if verify:
            d = f"{DGZ}/{a}"
            rels = [f"{DGZ}/go.obo", f"{DGZ}/go.norm", f"{d}/terms.pkl", f"{d}/terms_zero.pkl",
                    f"{d}/terms_zero_10.pkl", f"{d}/interpros.pkl", f"{d}/deepgozero_zero.th",
                    f"{d}/predictions_deepgozero.pkl", f"{d}/predictions_deepgozero_zero.pkl"]
            rels += [f"{d}/{s}_data.pkl" for s in ("train", "valid", "test")]
            rels += sorted(f"{d}/{p.name}" for p in (data_dir / a).glob("zero_GO:*_auc*.pkl"))
            hashes = verify_inputs(rels)
            log.info("verified %d input hashes", len(hashes))
        res = evaluate_aspect(a, data_dir, device, log, l3_labels=l3_labels)
        section = {"inputs": dict(sorted(hashes.items())), "results": res,
                   "status": res["status"], "loaded": res["L1"]["loaded"],
                   "mode": "DGZ-faithful (design §8.3) for L1/L2; TP test labels for L3"}
        status = write_json_section(out_path, {"schema": 1, "gate": "G2b"}, a, section)
        pin = record_pin("G2b", {a: {"section_sha256": sha256_bytes(canonical_json(section)),
                                     "status": res["status"]}}, pins_path=pins_path)
        log.info("G2b %s status: %s; %s: %s; pin G2b: %s (%.0f s)", a, res["status"],
                 out_path.name, status, pin, time.time() - t0)
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
