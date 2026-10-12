"""Gate G2: our evaluator in DGZ-faithful mode reproduces DGZ's published numbers
from their released predictions (protocol §10 G2, design §8.3, plan item 7).

For one aspect, scores dataset/go_function/dgz/<aspect>/predictions_{mlp,
deepgozero}.pkl as given (DGZ-rule labels = the pickles' prop_annotations,
terms.pkl columns, no re-propagation):
  - protein-centric Fmax, Smin, AUPR exactly as DGZ evaluate.py (IC from train +
    valid + test prop_annotations, namespace label set, 101 thresholds);
  - class-centric average AUC over terms with 0 < n_pos < n (evaluate_terms.py).
Every metric must be within +-0.005 of config/go_function/dgz_published.json.

Writes the aspect's section of results/go_function/gates/gate_dgz_eval.json
(an existing section must be identical) and pin G2 entry "<aspect>". A miss
writes gates/gate_dgz_eval_<aspect>.FAILED.json with every number and exits 3
(protocol stop: ask the user; do not continue to G3).

Exit codes: 0 pass; 2 fatal (input hashes, differing existing section/pin); 3 miss.

Usage:
  .venv/bin/python scripts/go_function/gate_dgz_eval.py --aspect mf --device cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function import evaluate as E  # noqa: E402
from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_STOP  # noqa: E402
from go_function.labels import read_terms  # noqa: E402
from go_function.ontology import DgzRule  # noqa: E402
from go_function.pins import (assert_writable_target, canonical_json, record_pin,  # noqa: E402
                              sha256_bytes, verify_inputs)
from go_function.runtime import (DEVICES, FatalError, ProtocolStop,  # noqa: E402
                                 resolve_device, setup_logging)

DGZ = "dataset/go_function/dgz"
PUBLISHED = ROOT / "config" / "go_function" / "dgz_published.json"
GATE_OUT = ROOT / "results" / "go_function" / "gates" / "gate_dgz_eval.json"
MODELS = ("mlp", "deepgozero")
METRICS = ("fmax", "smin", "aupr", "auc")
TOL = 0.005


def evaluate_aspect(aspect: str, data_dir: Path, device: str, log) -> dict:
    import numpy as np
    import pandas as pd
    dgz = DgzRule(data_dir / "go.obo")
    terms = read_terms(data_dir / aspect / "terms.pkl")
    tv = []
    for split in ("train", "valid"):
        df = pd.read_pickle(data_dir / aspect / f"{split}_data.pkl")
        tv.extend(list(x) for x in df["prop_annotations"].tolist())
        del df
    go_set = E.dgz_namespace_terms(dgz, aspect)
    out = {}
    for model in MODELS:
        t0 = time.time()
        df = pd.read_pickle(data_dir / aspect / f"predictions_{model}.pkl")
        preds = np.stack(df["preds"].tolist())
        test_prop = [list(x) for x in df["prop_annotations"].tolist()]
        n_rows = len(df)
        del df
        gc.collect()
        if preds.shape != (n_rows, len(terms)):
            raise FatalError(f"predictions_{model}.pkl: preds shape {preds.shape}, expected "
                             f"({n_rows}, {len(terms)})")
        ic = E.dgz_ic(dgz, tv + test_prop)
        pm = E.dgz_protein_metrics(preds, terms, test_prop, go_set, ic, device=device)
        cm = E.dgz_class_auc(preds, terms, test_prop)
        out[model] = {**pm, "auc": cm["auc"], "n_auc_terms": cm["n_terms"],
                      "n_test_rows": n_rows, "preds_dtype": str(preds.dtype)}
        log.info("%s %s: Fmax %.4f (t=%.2f) Smin %.4f AUPR %.4f AUC %.4f (%.0f s)", aspect, model,
                 pm["fmax"], pm["tmax"], pm["smin"], pm["aupr"], cm["auc"], time.time() - t0)
        del preds
        gc.collect()
    return out


def compare(aspect: str, results: dict, published: dict, tol: float = TOL):
    rows = []
    for model in MODELS:
        for metric in METRICS:
            pub = published["values"][aspect][model][metric]
            ours = float(results[model][metric])
            diff = ours - float(pub["value"])
            rows.append({"model": model, "metric": metric, "ours": ours,
                         "published": float(pub["value"]), "cell": pub["cell"],
                         "diff": diff, "within_tolerance": abs(diff) <= tol + 1e-12})
    return rows, all(r["within_tolerance"] for r in rows)


def main(argv=None, *, data_dir: Path = None, published_path: Path = None,
         out_path: Path = None, pins_path: Path = None, verify: bool = True) -> int:
    """Keyword overrides are for tests only (synthetic data outside the repo)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--device", choices=DEVICES, default="cpu")
    args = ap.parse_args(argv)
    data_dir = Path(data_dir) if data_dir is not None else ROOT / DGZ
    published_path = Path(published_path) if published_path is not None else PUBLISHED
    out_path = Path(out_path) if out_path is not None else GATE_OUT
    log = setup_logging("gate_dgz_eval")
    t0 = time.time()
    try:
        try:
            device = resolve_device(args.device)
        except ValueError as e:
            raise FatalError(str(e)) from None
        a = args.aspect
        hashes = {}
        if verify:
            rels = [f"{DGZ}/go.obo", f"{DGZ}/{a}/terms.pkl", f"{DGZ}/{a}/train_data.pkl",
                    f"{DGZ}/{a}/valid_data.pkl"] + [f"{DGZ}/{a}/predictions_{m}.pkl" for m in MODELS]
            hashes = verify_inputs(rels)
            log.info("verified %d input hashes", len(hashes))
        published = json.loads(published_path.read_text())
        results = evaluate_aspect(a, data_dir, device, log)
        rows, passed = compare(a, results, published)
        for r in rows:
            log.info("%s %-10s %-4s ours %.4f published %.3f diff %+.4f %s", a, r["model"],
                     r["metric"], r["ours"], r["published"], r["diff"],
                     "ok" if r["within_tolerance"] else "MISS")
        section = {"inputs": dict(sorted(hashes.items())), "results": results,
                   "comparison": rows, "tolerance": TOL, "passed": passed,
                   "mode": "DGZ-faithful (design §8.3)"}
        assert_writable_target(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not passed:
            fail = out_path.with_name(f"{out_path.stem}_{a}.FAILED.json")
            fail.write_bytes(canonical_json(section))
            raise ProtocolStop(f"G2 miss for {a}: "
                               + "; ".join(f"{r['model']} {r['metric']} ours {r['ours']:.4f} vs "
                                           f"{r['published']:.3f}" for r in rows
                                           if not r["within_tolerance"])
                               + f" (details in {fail.name})")
        doc = json.loads(out_path.read_text()) if out_path.exists() else \
            {"schema": 1, "gate": "G2", "aspects": {}}
        norm = json.loads(canonical_json(section))
        if a in doc["aspects"] and doc["aspects"][a] != norm:
            raise FatalError(f"{out_path.name}: existing {a} section differs from this run")
        status = "unchanged" if a in doc["aspects"] else "written"
        doc["aspects"][a] = norm
        if status == "written":
            tmp = out_path.with_name(f".{out_path.name}.tmp")
            tmp.write_bytes(canonical_json(doc))
            tmp.replace(out_path)
        pin = record_pin("G2", {a: {"section_sha256": sha256_bytes(canonical_json(norm)),
                                    "passed": True}}, pins_path=pins_path)
        log.info("G2 %s passed; %s: %s; pin G2: %s (%.0f s)", a, out_path.name, status, pin,
                 time.time() - t0)
    except ProtocolStop as e:
        log.error("%s", e)
        return EXIT_STOP
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
