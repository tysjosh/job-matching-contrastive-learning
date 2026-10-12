"""Gate G3: DGZ's released mlp.th and deepgozero.th load strictly into our port and
reproduce the released predictions (protocol §10 G3, design §6.1/§6.6, plan item 13).

For one aspect:
  1. the norm type of the protein tower is read from the checkpoint keys
     (LayerNorm: weight/bias only; BatchNorm1d: running statistics) and must agree
     between mlp.th and deepgozero.th;
  2. mlp.th -> V0, deepgozero.th -> V4 (go.norm parsed with terms.pkl, DGZ order),
     ``strict=True``; a key/shape mismatch is fatal (CheckpointMismatch, exit 2);
  3. forward on the test proteins (InterPro features built from the predictions
     pickle's ``interpros`` column with interpros.pkl, as DGZ ``get_data``), sigmoid,
     then DGZ-rule score propagation (deepgozero.py / deepgopro.py: every column takes
     the max score over the columns whose DGZ closure contains it);
  4. max |ours - released preds| <= 1e-4 for both models.

Writes the aspect section of results/go_function/gates/gate_checkpoints.json, the
``norm`` field of config/go_function/study.json, and pin G3 entry "<aspect>". A miss
writes gates/gate_checkpoints_<aspect>.FAILED.json and exits 3 (protocol stop).

Exit codes: 0 pass; 2 fatal; 3 miss.

Usage:
  .venv/bin/python scripts/go_function/gate_checkpoints.py --aspect mf --device cpu
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_STOP  # noqa: E402
from go_function.pins import (assert_writable_target, canonical_json, record_pin,  # noqa: E402
                              sha256_bytes, verify_inputs, write_json_section)
from go_function.runtime import (DEVICES, FatalError, ProtocolStop,  # noqa: E402
                                 resolve_device, setup_logging)

DGZ = "dataset/go_function/dgz"
GATE_OUT = ROOT / "results" / "go_function" / "gates" / "gate_checkpoints.json"
MODELS = ("mlp", "deepgozero")
TOL = 1e-4
#: load_normal_forms keyword overrides (tests on fixtures only; real runs keep the G1 pins)
NF_KW: dict = {}


def compare(ours, released, chunk: int = 256) -> dict:
    """max / mean |ours - released| (float64), chunked by rows."""
    import numpy as np
    if ours.shape != released.shape:
        raise FatalError(f"shape {ours.shape} vs released {released.shape}")
    mx, total, loc, n_over = 0.0, 0.0, (0, 0), 0
    for a in range(0, ours.shape[0], chunk):
        d = np.abs(ours[a:a + chunk].astype(np.float64) - released[a:a + chunk])
        m = float(d.max()) if d.size else 0.0
        if m > mx:
            r, c = np.unravel_index(int(d.argmax()), d.shape)
            mx, loc = m, (a + int(r), int(c))
        total += float(d.sum())
        n_over += int((d > TOL).sum())
    return {"max_abs_diff": mx, "mean_abs_diff": total / max(1, ours.size),
            "argmax_row": loc[0], "argmax_col": loc[1], "n_entries_over_tol": n_over}


def evaluate_aspect(aspect: str, data_dir: Path, device: str, log) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from go_function.evaluate import propagate_max
    from go_function.features import interpro_csr, read_interpro_vocab
    from go_function.labels import read_terms
    from go_function.model import (DgzCompatModel, detect_norm, dgz_propagation_pairs,
                                   predict_probs)
    from go_function.normforms import load_normal_forms
    from go_function.ontology import DgzRule
    terms = read_terms(data_dir / aspect / "terms.pkl")
    ipr = read_interpro_vocab(data_dir / aspect / "interpros.pkl")
    ipr_index = {g: i for i, g in enumerate(ipr)}
    dgz = DgzRule(data_dir / "go.obo")
    src, dst = dgz_propagation_pairs(terms, dgz)
    nf = load_normal_forms(data_dir / "go.norm", {g: i for i, g in enumerate(terms)}, **NF_KW)
    norms = {}
    out = {"n_terms": len(terms), "n_interpro": len(ipr), "n_propagation_pairs": int(len(src)),
           "n_zero_classes": len(nf.class_index) - nf.n_vocab,
           "n_relations": len(nf.rel_index), "models": {}}
    for kind in MODELS:
        t0 = time.time()
        sd = torch.load(data_dir / aspect / f"{kind}.th", map_location="cpu", weights_only=True)
        norms[kind] = detect_norm(sd)
        model = DgzCompatModel.load(None, kind=kind, n_in=len(ipr), n_terms=len(terms),
                                    nf=nf if kind == "deepgozero" else None, device=device,
                                    state_dict=sd)
        del sd
        gc.collect()
        df = pd.read_pickle(data_dir / aspect / f"predictions_{kind}.pkl")
        x, unknown = interpro_csr(df["interpros"].tolist(), ipr_index)
        released = np.stack(df["preds"].tolist())
        n_rows = len(df)
        del df
        gc.collect()
        if released.shape != (n_rows, len(terms)):
            raise FatalError(f"predictions_{kind}.pkl preds shape {released.shape}")
        raw = predict_probs(model, x, device=device)
        del model
        gc.collect()
        raw_cmp = compare(raw, released)
        prop = propagate_max(torch.as_tensor(raw), src, dst, device=device).cpu().numpy()
        del raw
        gc.collect()
        cmp = compare(prop, released)
        cmp["max_abs_diff_before_propagation"] = raw_cmp["max_abs_diff"]
        cmp.update({"n_test_rows": n_rows, "unknown_interpro_dropped": unknown,
                    "norm": norms[kind], "strict_load": True,
                    "passed": cmp["max_abs_diff"] <= TOL})
        out["models"][kind] = cmp
        log.info("%s %s: strict load ok (norm %s); max |diff| %.3g (before propagation %.3g), "
                 "mean %.3g, %d entries > %.0e (%.0f s)", aspect, kind, norms[kind],
                 cmp["max_abs_diff"], raw_cmp["max_abs_diff"], cmp["mean_abs_diff"],
                 cmp["n_entries_over_tol"], TOL, time.time() - t0)
        del prop, released
        gc.collect()
    if len(set(norms.values())) != 1:
        raise FatalError(f"norm types differ between checkpoints: {norms}")
    out["norm"] = norms["mlp"]
    return out


def main(argv=None, *, data_dir: Path = None, out_path: Path = None, pins_path: Path = None,
         study_path: Path = None, verify: bool = True) -> int:
    """Keyword overrides are for tests only (synthetic data outside the repo)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--device", choices=DEVICES, default="cpu")
    args = ap.parse_args(argv)
    from go_function.train import set_study_field
    data_dir = Path(data_dir) if data_dir is not None else ROOT / DGZ
    out_path = Path(out_path) if out_path is not None else GATE_OUT
    log = setup_logging("gate_checkpoints")
    t0 = time.time()
    try:
        try:
            device = resolve_device(args.device)
        except ValueError as e:
            raise FatalError(str(e)) from None
        a = args.aspect
        hashes = {}
        if verify:
            rels = [f"{DGZ}/go.obo", f"{DGZ}/go.norm", f"{DGZ}/{a}/terms.pkl",
                    f"{DGZ}/{a}/interpros.pkl"]
            rels += [f"{DGZ}/{a}/{m}.th" for m in MODELS]
            rels += [f"{DGZ}/{a}/predictions_{m}.pkl" for m in MODELS]
            hashes = verify_inputs(rels)
            log.info("verified %d input hashes", len(hashes))
        res = evaluate_aspect(a, data_dir, device, log)
        passed = all(m["passed"] for m in res["models"].values())
        section = {"inputs": dict(sorted(hashes.items())), "results": res, "tolerance": TOL,
                   "passed": passed, "norm": res["norm"],
                   "rule": "forward + DGZ-rule score propagation vs released predictions"}
        assert_writable_target(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not passed:
            fail = out_path.with_name(f"{out_path.stem}_{a}.FAILED.json")
            fail.write_bytes(canonical_json(section))
            raise ProtocolStop(f"G3 miss for {a}: " + "; ".join(
                f"{k} max |diff| {m['max_abs_diff']:.3g}" for k, m in res["models"].items()
                if not m["passed"]) + f" (tolerance {TOL}; details in {fail.name})")
        status = write_json_section(out_path, {"schema": 1, "gate": "G3"}, a, section)
        norm_status = set_study_field("norm", res["norm"], study_path)
        pin = record_pin("G3", {a: {"section_sha256": sha256_bytes(canonical_json(section)),
                                    "passed": True, "norm": res["norm"]}}, pins_path=pins_path)
        log.info("G3 %s passed (norm %s); %s: %s; study.json norm: %s; pin G3: %s (%.0f s)", a,
                 res["norm"], out_path.name, status, norm_status, pin, time.time() - t0)
    except ProtocolStop as e:
        log.error("%s", e)
        return EXIT_STOP
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
