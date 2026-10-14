"""Gate G-R (protocol §9): reproduction.

  --part a  GF G2 numbers recomputed from the released predictions with
            ``go_function.evaluate`` in DGZ-faithful mode (import of
            scripts/go_function/gate_dgz_eval.py ``evaluate_aspect``); every value must
            equal results/go_function/gates/gate_dgz_eval.json to 1e-9. Reads the released
            test predictions and their labels (the already-pinned published metrics
            only), in a process that never enters search mode and computes no fusion.
            Needs --allow-g2-test-read to make that exposure explicit.
  --part b  Pilot numbers in pilot-compat mode (all vocabulary columns, root removed;
            the pilot's float32 zcol for b_valid_pilot.py variants; H_sim): every value
            of the seven pilot JSONs (§13) reproduced (counts exactly; means, band means
            and fmax to <= 5e-5 and equal at the pilot's rounding). Validation only,
            search mode. Copies the pilot snapshot to <results>/pilot_snapshot/ and checks
            §13 hashes; copies dgz_evaluate.py to config/go_function_fusion/dgz_evaluate_ref.py.txt.
  --part c  Confirmatory mode on validation (float64, primary columns): base and
            pilot-equivalent variants, differences from (b) and changed-set counts
            reported, not gated. Search mode.

A miss in (a) or (b) exits 3 (stop and ask the user). Output:
<results>/gates/gate_repro.json (one section per part/aspect/base); pin G-R.

Usage (Mac, CPU):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_function_fusion/gate_repro.py --part b \\
      --aspect mf --base mlp --pilot-dir /tmp/gofunc_fusion --results results/go_function_fusion_dev
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import make_pins as MP  # noqa: E402
from go_function_fusion import data as D  # noqa: E402
from go_function_fusion import fuse as FU  # noqa: E402
from go_function_fusion import guard  # noqa: E402
from go_function_fusion import metrics as M  # noqa: E402
from go_function_fusion import priors as PR  # noqa: E402

STRUCT_PAIRS = (("bp", "mlp"), ("bp", "deepgozero"), ("mf", "mlp"), ("mf", "deepgozero"))
PILOT_PAIRS = (("bp", "mlp"), ("mf", "mlp"), ("mf", "deepgozero"))
GF_G2 = ROOT / "results/go_function/gates/gate_dgz_eval.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ shared validation inputs
def valid_inputs(aspect: str, kind: str, device: str):
    ctx = D.aspect_context(aspect, ("train", "valid"))
    lab = ctx.labels
    L_all = D.tier_a_logits(aspect, kind, "valid", ctx.terms, lab["proteins_valid"], device=device)
    Yv = sp.csr_matrix(lab["valid"]).astype(bool)
    rows = D.eval_rows(Yv)
    return ctx, L_all, Yv, rows


# ------------------------------------------------------------------ part (b): c_struct.py parity
def struct_compat(aspect: str, kind: str, sim: Path, device: str) -> Dict[str, Any]:
    ctx, L_all, Yv, rows = valid_inputs(aspect, kind, device)
    L = L_all[rows].astype(np.float64)
    del L_all
    gc.collect()
    terms, vocab, root = ctx.terms, ctx.vocab, ctx.root
    Yd = Yv[rows].toarray()
    npos = Yd.sum(0)
    keep = np.arange(len(terms)) != root
    eval_cols = np.where(keep & (npos > 0) & (npos < len(rows)))[0]
    ytr32 = sp.csr_matrix(ctx.y_train).astype(np.float32)
    ntr32 = np.asarray(ytr32.sum(0)).ravel()
    nb = PR.neighbour_tables(ctx.ont, vocab, ctx.anc, ctx.desc, ntr32, root, "pilot",
                             keys=("pa_isa", "pa_tp", "pa_any", "anc", "sib", "ch", "reg"))
    perm = PR.decile_permutation(ntr32)
    nb["pa_tp_shuf"] = PR.permute_sets(nb["pa_tp"], perm)
    vnames = list(np.asarray(ctx.labels["proteins_valid"])[rows])
    W = PR.sim_matrix(sim, vnames, ctx.labels["proteins_train"])
    z = FU.z_cols

    def agg(key, C, how="mean"):
        return FU.prior_z(L, nb[key], C, how, method="loop")

    def adapt(tau, C):
        return (2.0 * tau / (tau + ntr32[C]))[None, :]

    V = {
        "own": lambda C, x: z(L[:, C]),
        "pa_isa": lambda C, x: z(L[:, C]) + agg("pa_isa", C),
        "pa_tp": lambda C, x: z(L[:, C]) + x["pa_tp"],
        "pa_tp_w0.5": lambda C, x: z(L[:, C]) + 0.5 * x["pa_tp"],
        "pa_tp_w2": lambda C, x: z(L[:, C]) + 2 * x["pa_tp"],
        "pa_any": lambda C, x: z(L[:, C]) + agg("pa_any", C),
        "anc_tp": lambda C, x: z(L[:, C]) + agg("anc", C),
        "F_protocol": lambda C, x: z(L[:, C]) + x["pa_tp"] - agg("sib", C),
        "ch_max": lambda C, x: z(L[:, C]) + agg("ch", C, "max"),
        "pa_tp+reg": lambda C, x: z(L[:, C]) + x["pa_tp"] + agg("reg", C),
        "pa_tp+H": lambda C, x: z(L[:, C]) + x["pa_tp"] + 0.5 * x["H"],
        "own+H": lambda C, x: z(L[:, C]) + 0.5 * x["H"],
        "pa_tp_shuffled": lambda C, x: z(L[:, C]) + agg("pa_tp_shuf", C),
        "pa_tp_adapt_tau10": lambda C, x: z(L[:, C]) + adapt(10, C) * x["pa_tp"],
        "pa_tp_adapt_tau30": lambda C, x: z(L[:, C]) + adapt(30, C) * x["pa_tp"],
        "pa_tp_adapt_tau100": lambda C, x: z(L[:, C]) + adapt(100, C) * x["pa_tp"],
        "pa_tp_adapt_tau30+H": lambda C, x: z(L[:, C]) + adapt(30, C) * x["pa_tp"] + 0.5 * x["H"],
    }

    def ctx_fn(C):
        return {"pa_tp": agg("pa_tp", C),
                "H": z(np.asarray((W @ ytr32[:, C]).todense(), dtype=np.float64))}

    res = FU.score_variants(L.shape[0], eval_cols, V, ctx_fn, lambda C: Yd[:, C],
                            {"auc": M.rank_auc_pilot}, log=log)["auc"]
    has_reg = np.array([len(nb["reg"][c]) > 0 for c in eval_cols])
    nt = ntr32[eval_cols]
    out = {"aspect": aspect, "model": kind, "n_rows": int(len(rows)), "n_terms": int(len(eval_cols)),
           "n_terms_with_reg_edge": int(has_reg.sum()), "variants": {}}
    bands = [(0, 0), (1, 10), (11, 30), (31, 100), (101, 1000), (1001, 10 ** 9)]
    for k, v in res.items():
        r = {"mean": float(v.mean())}
        for lo, hi in bands:
            m = (nt >= lo) & (nt <= hi)
            r[f"{lo}-{hi}"] = float(v[m].mean())
        r["reg_subset"] = float(v[has_reg].mean()) if has_reg.any() else None
        d = v - res["own"]
        r["delta_vs_own"] = float(d.mean())
        r["share_terms_improved"] = float((d > 0).mean())
        out["variants"][k] = r
    out["band_counts"] = {f"{lo}-{hi}": int(((nt >= lo) & (nt <= hi)).sum()) for lo, hi in bands}
    return out


# ------------------------------------------------------------------ part (b): b_valid_pilot.py parity
def pilot_compat(aspect: str, kind: str, sim: Path, device: str, data_dir: Path) -> Dict[str, Any]:
    import torch
    from go_function.evaluate import propagate_max, tp_ancestor_pairs
    from go_function.model import dgz_propagation_pairs
    from go_function.ontology import DgzRule, parents
    ctx, L, Yv, rows = valid_inputs(aspect, kind, device)
    terms, vocab, root = ctx.terms, ctx.vocab, ctx.root
    P = (1.0 / (1.0 + np.exp(-L))).astype(np.float32)
    ytr32 = sp.csr_matrix(ctx.y_train).astype(np.float32)
    ntr = np.asarray(ytr32.sum(0)).ravel()
    cols_keep = np.ones(len(terms), bool)
    if root >= 0:
        cols_keep[root] = False
    Yd = Yv[rows].toarray()
    npos = Yd.sum(0)
    eval_cols = np.where(cols_keep & (npos > 0) & (npos < len(rows)))[0]
    bands = [(0, 0), (1, 10), (11, 30), (31, 100), (101, 1000), (1001, 10 ** 9)]
    out: Dict[str, Any] = {"aspect": aspect, "model": kind, "n_valid_rows": int(len(rows)),
                           "variants": {}}

    def record(name, S, do_fmax=False):
        S2 = S[rows][:, eval_cols]
        v = np.empty(len(eval_cols))
        for a in range(0, len(eval_cols), 2000):
            v[a:a + 2000] = M.rank_auc_pilot(S2[:, a:a + 2000], Yd[:, eval_cols[a:a + 2000]])
        r: Dict[str, Any] = {"mean": float(v.mean()), "n": int(len(v))}
        for lo, hi in bands:
            m = (ntr[eval_cols] >= lo) & (ntr[eval_cols] <= hi)
            if m.any():
                r[f"{lo}-{hi}"] = [float(v[m].mean()), int(m.sum())]
        if do_fmax:
            r.update(M.pilot_fmax(S[rows][:, cols_keep], Yd[:, cols_keep]))
        out["variants"][name] = r
        log(f"{name} {r['mean']:.4f}")

    record("B_raw", P, True)
    anc = {**dict(ctx.anc), None: frozenset()}
    src, dst = tp_ancestor_pairs(list(range(len(terms))), vocab, anc)
    B_tp = propagate_max(torch.as_tensor(P), src, dst).numpy()
    record("B_tp", B_tp, True)
    dsrc, ddst = dgz_propagation_pairs(terms, DgzRule(data_dir / "go.obo"))
    B_dgz = propagate_max(torch.as_tensor(P), dsrc, ddst).numpy()
    record("B_dgz", B_dgz, True)
    del B_dgz
    gc.collect()
    rng = np.random.default_rng(PR.SHUFFLE_GRAPH_SEED)
    perm = PR.decile_permutation(ntr, rng=rng)
    B_shuf = propagate_max(torch.as_tensor(P), perm[src], perm[dst]).numpy()
    record("B_tp_shuffledgraph", B_shuf)
    del B_shuf
    gc.collect()
    par = parents(ctx.ont)
    pcols = []
    for c in range(len(terms)):
        pr = vocab.primary[c]
        pcols.append([j for p in (par.get(pr, ()) if pr else ())
                      for j in vocab.primary_to_cols.get(p, ()) if j != root])
    has_pa = np.array([len(p) > 0 for p in pcols])
    PM = np.zeros_like(L)
    for c in np.where(has_pa)[0]:
        PM[:, c] = L[:, pcols[c]].mean(1)
    F = FU.z_cols_pilot32(L, rows)
    F[:, has_pa] += FU.z_cols_pilot32(PM, rows)[:, has_pa]
    record("F_pa", F)
    PMs = np.zeros_like(L)
    for c in np.where(has_pa)[0]:
        PMs[:, c] = L[:, perm[pcols[c]]].mean(1)
    Fs = FU.z_cols_pilot32(L, rows)
    Fs[:, has_pa] += FU.z_cols_pilot32(PMs, rows)[:, has_pa]
    record("F_pa_shuffled", Fs)
    del F, Fs, PM, PMs
    gc.collect()
    W = PR.sim_matrix(sim, ctx.labels["proteins_valid"], ctx.labels["proteins_train"])
    tot = np.asarray(W.astype(bool).sum(1)).ravel()
    H = np.asarray((W @ ytr32).todense(), dtype=np.float32)
    out["valid_with_sim_hit"] = int((tot[rows] > 0).sum())
    record("H_sim", H, True)
    for a in (0.2, 0.4, 0.5, 0.6):
        record(f"convex_a{a}", (1 - a) * B_tp + a * H, True)
    zB, zH = FU.z_cols_pilot32(B_tp, rows), FU.z_cols_pilot32(H, rows)
    for w in (0.25, 0.5, 1.0):
        record(f"zsum_w{w}", zB + w * zH)
    del zB, zH
    Hs = np.asarray((W @ ytr32[rng.permutation(ytr32.shape[0])]).todense(), dtype=np.float32)
    record("convex_a0.4_shuffledhits", 0.6 * B_tp + 0.4 * Hs)
    return out


# ------------------------------------------------------------------ comparison
def _cmp_val(ours, ref, tol, digits=4) -> bool:
    if ref is None or ours is None:
        return ref is ours
    return abs(float(ours) - float(ref)) <= tol + 1e-12 and round(float(ours), digits) == round(float(ref), digits)


def compare_struct(ours: Dict[str, Any], ref: Dict[str, Any], tol: float) -> List[Dict[str, Any]]:
    rows = []
    for k in ("n_rows", "n_terms", "n_terms_with_reg_edge"):
        rows.append({"field": k, "ours": ours[k], "ref": ref[k], "ok": ours[k] == ref[k]})
    rows.append({"field": "band_counts", "ours": ours["band_counts"], "ref": ref["band_counts"],
                 "ok": ours["band_counts"] == ref["band_counts"]})
    for v, rr in ref["variants"].items():
        oo = ours["variants"].get(v)
        for f, rv in rr.items():
            ov = None if oo is None else oo.get(f)
            if f == "share_terms_improved":     # pilot rounds a share to 3 decimals
                ok = ov is not None and round(float(ov), 3) == round(float(rv), 3)
            else:
                ok = _cmp_val(ov, rv, tol, 4)
            rows.append({"field": f"{v}.{f}", "ours": ov, "ref": rv, "ok": ok})
    return rows


def compare_pilot(ours: Dict[str, Any], ref: Dict[str, Any], tol: float) -> List[Dict[str, Any]]:
    rows = []
    for k in ("n_valid_rows", "valid_with_sim_hit"):
        rows.append({"field": k, "ours": ours.get(k), "ref": ref.get(k), "ok": ours.get(k) == ref.get(k)})
    for v, rr in ref["variants"].items():
        oo = ours["variants"].get(v, {})
        for f, rv in rr.items():
            ov = oo.get(f)
            if isinstance(rv, list):
                ok = ov is not None and ov[1] == rv[1] and _cmp_val(ov[0], rv[0], tol)
            elif f == "n":
                ok = ov == rv
            elif f == "tmax":
                ok = ov == rv
            else:
                ok = _cmp_val(ov, rv, tol)
            rows.append({"field": f"{v}.{f}", "ours": ov, "ref": rv, "ok": bool(ok)})
    return rows


# ------------------------------------------------------------------ part (a)
def part_a(aspect: str, device: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("gf_gate_dgz_eval",
                                                  ROOT / "scripts/go_function/gate_dgz_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import logging
    lg = logging.getLogger("gate_repro_a")
    res = mod.evaluate_aspect(aspect, ROOT / D.DGZ, device, lg)
    ref = json.loads(GF_G2.read_text())["aspects"][aspect]["results"]
    rows = []
    for model in ("mlp", "deepgozero"):
        for metric in ("fmax", "smin", "aupr", "auc"):
            o, r = float(res[model][metric]), float(ref[model][metric])
            rows.append({"model": model, "metric": metric, "ours": o, "ref": r,
                         "ok": abs(o - r) <= 1e-9})
    return {"comparison": rows, "passed": all(r["ok"] for r in rows)}


# ------------------------------------------------------------------ part (c)
def part_c(aspect: str, kind: str, device: str, pilot_struct: Dict[str, Any]) -> Dict[str, Any]:
    ctx, L_all, Yv, rows = valid_inputs(aspect, kind, device)
    L = L_all[rows].astype(np.float64)
    del L_all
    Yr = Yv[rows]
    cols = D.eval_terms(Yr, ctx.vocab, ctx.root, "confirmatory")
    keys = ("pa_tp", "reg", "sib", "desc")
    conf = PR.neighbour_tables(ctx.ont, ctx.vocab, ctx.anc, ctx.desc, ctx.ntr, ctx.root,
                               "confirmatory", keys)
    pil = PR.neighbour_tables(ctx.ont, ctx.vocab, ctx.anc, ctx.desc, ctx.ntr, ctx.root, "pilot",
                              keys)
    changed = PR.changed_terms(conf, pil, cols, keys)
    perm = PR.decile_permutation(ctx.ntr)
    shuf = PR.permute_sets(conf["pa_tp"], perm)
    w30 = FU.adaptive_w(ctx.ntr, 2, 30)
    z = FU.z_cols
    V = {
        "own": lambda C, x: z(L[:, C]),
        "pa_tp": lambda C, x: z(L[:, C]) + x["pa"],
        "pa_tp_adapt_tau30": lambda C, x: FU.ap_t(z(L[:, C]), x["pa"], w30[C]),
        "pa_tp_shuffled": lambda C, x: z(L[:, C]) + FU.prior_z(L, shuf, C),
        "F_protocol": lambda C, x: FU.protocol_f(z(L[:, C]), x["pa"], FU.prior_z(L, conf["sib"], C),
                                                 np.ones(len(C))),
    }
    res = FU.score_variants(L.shape[0], cols, V, lambda C: {"pa": FU.prior_z(L, conf["pa_tp"], C)},
                            lambda C: Yr[:, C].toarray(), {"auc": M.rank_auc}, log=log)["auc"]
    reg_mask = np.array([len(conf["reg"][c]) > 0 for c in cols])
    out = {"n_rows": int(len(rows)), "n_terms": int(len(cols)),
           "n_alt_or_unprimary_removed": int(pilot_struct["n_terms"] - len(cols))
           if pilot_struct else None, "changed_sets": changed,
           "permutation_report": PR.permutation_report(conf["pa_tp"], perm, cols, ctx.vocab, ctx.root),
           "variants": {}}
    for k, v in res.items():
        s = M.summary(v, ctx.ntr[cols], reg_mask, res["own"])
        if pilot_struct and k in pilot_struct.get("variants", {}):
            s["diff_vs_pilot_compat_mean"] = s["mean"] - pilot_struct["variants"][k]["mean"]
        out["variants"][k] = s
    return out


# ------------------------------------------------------------------ main
def snapshot(pilot_dir: Path, results: Path, study: Dict[str, Any]) -> Dict[str, Any]:
    from go_function.pins import sha256_file
    if not pilot_dir.exists():
        raise MP.Refused(f"pilot directory {pilot_dir.name} is gone: G-R stops and asks the user")
    dest = results / "pilot_snapshot"
    MP.assert_writable(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    names = list(study["pilot_snapshot_sha256"]) + list(study["pilot_perterm_npz"])
    for n in names:
        src = pilot_dir / n
        if not src.exists():
            raise MP.Refused(f"pilot file {n} is missing: G-R stops and asks the user")
        got = sha256_file(src)
        want = study["pilot_snapshot_sha256"].get(n)
        rows[n] = {"sha256": got, "pinned": want, "ok": want is None or want == got}
        if n != "dgz_evaluate.py" and not (dest / n).exists():
            shutil.copyfile(src, dest / n)
    ref_txt = ROOT / "config/go_function_fusion/dgz_evaluate_ref.py.txt"
    if not ref_txt.exists():
        MP.assert_writable(ref_txt)
        shutil.copyfile(pilot_dir / "dgz_evaluate.py", ref_txt)
    rows["dgz_evaluate_ref.py.txt"] = {"sha256": sha256_file(ref_txt),
                                       "pinned": study["pilot_snapshot_sha256"]["dgz_evaluate.py"]}
    rows["dgz_evaluate_ref.py.txt"]["ok"] = rows["dgz_evaluate_ref.py.txt"]["sha256"] == \
        rows["dgz_evaluate_ref.py.txt"]["pinned"]
    pub = ROOT / "config/go_function_fusion/dgz_published_ds.json"
    return {"files": rows, "ok": all(r["ok"] for r in rows.values()),
            "dgz_published_ds_present": pub.exists()}


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=("a", "b", "c"), required=True)
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--base", choices=("mlp", "deepgozero"))
    ap.add_argument("--kind", choices=("struct", "pilot", "both"), default="both")
    ap.add_argument("--pilot-dir", default="/tmp/gofunc_fusion")
    ap.add_argument("--sim", default=None, help="swissprot_exp.sim (default <pilot-dir>/swissprot_exp.sim)")
    ap.add_argument("--results", default=MP.RESULTS_CHOICES[0])
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--allow-g2-test-read", action="store_true")
    a = ap.parse_args(argv)
    results = MP.results_dir(a.results)
    study = D.fusion_study()
    out_p = results / "gates" / "gate_repro.json"
    doc = json.loads(out_p.read_text()) if out_p.exists() else {"schema": 1, "gate": "G-R", "sections": {}}
    pilot_dir = Path(a.pilot_dir)
    if a.part == "a":
        if not a.allow_g2_test_read:
            raise MP.Refused("part a reads the released test predictions and labels (GF G2 "
                             "exposure only); rerun with --allow-g2-test-read")
        sec = part_a(a.aspect, a.device)
        key = f"a/{a.aspect}"
    else:
        guard.enter_search_mode()
        if a.base is None:
            raise MP.Fatal("--base is required for parts b and c")
        from go_function.pins import sha256_file
        sim = Path(a.sim) if a.sim else pilot_dir / "swissprot_exp.sim"
        ih = json.loads((ROOT / "config/go_function_fusion/input_hashes.json").read_text())
        if sha256_file(sim) != ih["files"]["swissprot_exp.sim"]["sha256"]:
            raise MP.Fatal("swissprot_exp.sim differs from config/go_function_fusion/input_hashes.json")
        snap = snapshot(pilot_dir, results, study)
        doc["snapshot"] = snap
        if not snap["ok"]:
            raise MP.Fatal("pilot snapshot hashes differ from protocol §13")
        tol = float(study["pilot_tolerance"])
        key = f"{a.part}/{a.aspect}/{a.base}"
        sec: Dict[str, Any] = {}
        ref_s = pilot_dir / f"struct_{a.aspect}_{a.base}.json"
        if a.part == "b":
            if a.kind in ("struct", "both") and (a.aspect, a.base) in STRUCT_PAIRS:
                ours = struct_compat(a.aspect, a.base, sim, a.device)
                cmp = compare_struct(ours, json.loads(ref_s.read_text()), tol)
                sec["struct"] = {"ours": ours, "misses": [r for r in cmp if not r["ok"]],
                                 "n_checked": len(cmp)}
                gc.collect()
            if a.kind in ("pilot", "both") and (a.aspect, a.base) in PILOT_PAIRS:
                ours = pilot_compat(a.aspect, a.base, sim, a.device, ROOT / D.DGZ)
                ref = json.loads((pilot_dir / f"pilot_{a.aspect}_{a.base}.json").read_text())
                cmp = compare_pilot(ours, ref, tol)
                sec["pilot"] = {"ours": ours, "misses": [r for r in cmp if not r["ok"]],
                                "n_checked": len(cmp)}
            sec["passed"] = all(not v["misses"] for v in sec.values() if isinstance(v, dict))
        else:
            ref = json.loads(ref_s.read_text()) if ref_s.exists() else None
            sec = part_c(a.aspect, a.base, a.device, ref)
            sec["passed"] = True
    doc["sections"][key] = sec
    MP.write_json(out_p, doc)
    log(f"{key}: passed={sec['passed']}")
    if not sec["passed"]:
        log("G-R miss: stop and ask the user (protocol §9)")
        return MP.EXIT_STOP
    MP.record_pin(results, "G-R", {key: {"passed": True, "gate_repro_section_keys": sorted(sec)}})
    return MP.EXIT_OK


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
