"""Gates G-V (Mac, Tier A) and G-V2 (remote): validation search in search mode (protocol §9).

For each (aspect, base) on the DGZ validation split (proteins with >= 1 TP label,
confirmatory term set, float64):

pass 1  AP-T grid (c x tau, 9 cells), fixed-w (3), parent-only, and every AP-T-form
        control on the AP-T grid plus the null cell (shuffled graph, CoA, CoA⁺, Pa_any,
        Desc-max, protocol F adaptive), weight-free GF arm F, pure prop_TP(sigma(l)).
        AP-T (c*, tau*): largest mean class AUC over all evaluated terms; ties ->
        smaller c, then larger tau (1e-12).
pass 2  at (c*, tau*) (with --diamond-valid: the G-F frozen cell from
        finalists_tierA.json, §3.2; outputs then go to search/<aspect>_<base>_dv/): AP-T+R over w_R (selected on the reg-subset mean) and the
        shuffled-Reg control; with --sim the exploratory H_sim reads (never finalists);
        with --diamond-valid (remote, G-V2) AP-T+H over w_H, shuffled hits, the PC grid
        (Fmax), PC shuffled graph / shuffled hits and the X+DS alpha sensitivity.

Outputs (<results>/search/<aspect>_<base>/): auc_<family>.npz per family (per-term
validation AUC per cell, float32; no dense score matrix), selections.json; and
<results>/search_log.jsonl. A guard violation is fatal (exit 2) and logged.

Usage (Mac, Tier A, CPU):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/go_function_fusion/search_valid.py \\
      --aspect mf --base mlp --results results/go_function_fusion_dev --sim /tmp/gofunc_fusion/swissprot_exp.sim
Remote (G-V2): add --diamond-valid (uses the G-D train-only run) and/or --base V0IPR etc.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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

TIER_B = ("V0IPR", "V4IPR", "V0ESM")
AUC_CONTROL_FAMILIES = ("shuf_graph", "coa", "coap", "pa_any", "desc_max", "protocol_F")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ inputs
def tier_b_entry(results: Path, aspect: str, base: str) -> Dict[str, Any]:
    slot = MP.require_pin(results, "G-TP")
    e = slot.get(f"{aspect}/{base}")
    if e is None:
        raise MP.Refused(f"G-TP has no {aspect}/{base} entry")
    return e


def base_logits(aspect: str, base: str, split: str, ctx, results: Path, device: str) -> np.ndarray:
    """Logits (n_split x |V|) of a Tier A or Tier B base."""
    proteins = ctx.labels[f"proteins_{split}"]
    if base in D.TIER_A:
        return D.tier_a_logits(aspect, base, split, ctx.terms, proteins, device=device)
    from go_function.pins import sha256_file
    e = tier_b_entry(results, aspect, base)
    ck = ROOT / e["checkpoint"]
    if sha256_file(ck) != e["checkpoint_sha256"]:
        raise MP.Fatal(f"{e['checkpoint']}: sha256 differs from pin G-TP")
    spec = json.loads((ROOT / e["spec"]).read_text())
    x = D.tier_b_features(spec, split, proteins)
    model = D.load_tier_b(spec, ck, ctx.vocab, x.shape[1], device)
    return D.base_logits(model, x, device)


def frozen_star(results: Path, aspect: str, base: str) -> Dict[str, float]:
    """AP-T (c*, tau*) frozen at G-F (protocol §3.2): the w_H grid of the remote Tier A
    rerun is built at this cell, not at the rerun's own pass-1 selection."""
    from go_function.pins import sha256_file
    gf = MP.require_pin(results, "G-F")
    p = results / "finalists_tierA.json"
    if not p.exists() or sha256_file(p) != gf["finalists_tierA.json"]:
        raise MP.Fatal("finalists_tierA.json is missing or differs from the G-F pin")
    fin = json.loads(p.read_text())["bases"].get(f"{aspect}/{base}")
    if fin is None:
        raise MP.Refused(f"finalists_tierA.json has no {aspect}/{base} entry")
    return {k: float(v) for k, v in fin["apt"].items()}


def search_dir(results: Path, aspect: str, base: str, diamond_valid: bool) -> Path:
    """search/<aspect>_<base>/ (G-V); the remote --diamond-valid rerun writes to
    search/<aspect>_<base>_dv/ so the G-F inputs are never overwritten."""
    return results / "search" / f"{aspect}_{base}{'_dv' if diamond_valid else ''}"


def diamond_paths(results: Path, aspect: str) -> Dict[str, Path]:
    e = MP.require_pin(results, "G-D").get(aspect)
    if e is None:
        raise MP.Refused(f"G-D has no {aspect} entry")
    return {k: ROOT / e[k]["path"] for k in ("valid_res", "test_res")}


def ic_and_pairs(ctx, pc_cols: np.ndarray):
    """IC from TP train labels (``ic_from_labels``) and TP propagation pairs over pc_cols."""
    from go_function.evaluate import ic_from_labels
    from go_function.ontology import parents
    par = parents(ctx.ont)
    pcols = [() if p is None else tuple(sorted(ctx.vocab.index[q] for q in par[p]
                                               if q in ctx.vocab.index))
             for p in ctx.vocab.primary]
    ic = ic_from_labels(ctx.y_train, pcols)
    return ic[pc_cols], FU.prop_pairs(pc_cols, ctx.vocab, ctx.anc)


# ------------------------------------------------------------------ helpers
def family_cells(study: Mapping[str, Any]) -> Dict[str, List[Dict[str, float]]]:
    g = study["grids"]
    return {"apt": FU.apt_cells(g["apt_c"], g["apt_tau"]),
            "fixed": FU.fixed_cells(g["fixed_c"]),
            "parent_only": [{}], "F_weightfree": [{}], "base": [{}],
            **{f: FU.apt_cells(g["apt_c"], g["apt_tau"], null=True) for f in AUC_CONTROL_FAMILIES}}


def run_families(space: FU.Space, cols: np.ndarray, Yr: sp.csr_matrix,
                 fam_cells: Mapping[str, List[Dict[str, float]]]) -> Dict[str, np.ndarray]:
    """{family: (n_cells x n_terms) validation AUC}."""
    variants = {}
    for fam, cells in fam_cells.items():
        for i, cell in enumerate(cells):
            variants[f"{fam}#{i}"] = (lambda C, x, f=fam, c=cell: space.score(f, c, C, x))
    res = FU.score_variants(space.L.shape[0], cols, variants, space.ctx,
                            lambda C: Yr[:, C].toarray(), {"auc": M.rank_auc}, log=log)["auc"]
    return {fam: np.stack([res[f"{fam}#{i}"] for i in range(len(cells))])
            for fam, cells in fam_cells.items()}


def selection(cells, A: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    vals = [float(np.nanmean(a[mask] if mask is not None else a)) for a in A]
    i, cell = FU.select(cells, vals)
    return {"cells": [dict(c) for c in cells], "mean_auc": vals, "selected_index": int(i),
            "selected": dict(cell), "null_selected": FU.is_null(cell)}


def save_family(out: Path, fam: str, cells, A: np.ndarray, cols, ntr) -> str:
    return MP.write_npz(out / f"auc_{fam}.npz",
                        {"cols": np.asarray(cols, dtype=np.int64),
                         "ntr": np.asarray(ntr, dtype=np.int64),
                         "cells": np.asarray([FU.cell_key(c) for c in cells]),
                         "auc": np.asarray(A, dtype=np.float32)})


def jsonable_cell(c: Mapping[str, float]) -> Dict[str, Any]:
    return {k: ("inf" if np.isinf(v) else v) for k, v in c.items()}


# ------------------------------------------------------------------ PC (remote)
def pc_search(ctx, L: np.ndarray, Yr: sp.csr_matrix, sets, study, ds_full, ds_shuf_full,
              device: str) -> Dict[str, Any]:
    """PC grid on validation Fmax, PC shuffled graph (re-selected over the grid), PC
    shuffled hits (alpha grid at frozen c_PC*, tau_PC*), X+DS alpha sensitivity."""
    g = study["grids"]
    pc_cols = D.candidate_columns(ctx.vocab, ctx.root, "confirmatory")
    ic, pairs = ic_and_pairs(ctx, pc_cols)
    y = Yr[:, pc_cols]
    Lp = L[:, pc_cols]
    ntr = ctx.ntr[pc_cols]
    pz = {"real": FU.prior_z(L, sets["pa_tp"], pc_cols), "shuf": FU.prior_z(L, sets["pa_shuf"], pc_cols)}
    cells = FU.pc_cells(g["pc_c"], g["pc_tau"], g["pc_alpha"])

    def fmax(S):
        return M.protein_centric(S, y, ic, device=device)["fmax"]

    out: Dict[str, Any] = {}
    cache: Dict[Any, np.ndarray] = {}
    for which in ("real", "shuf"):
        vals = []
        for cell in cells:
            key = (which, cell["c"], cell["tau"])
            if key not in cache:
                w = FU.adaptive_w(ntr, cell["c"], cell["tau"])
                cache.clear()
                cache[key] = FU.prop_tp(FU.pc_q(Lp, pz[which], w), pairs, device)
            vals.append(fmax(FU.pc_score(cache[key], ds_full, cell["alpha"])))
            log(f"PC {which} {cell} fmax {vals[-1]:.4f}")
        i, sel = FU.select(cells, vals)
        out["pc" if which == "real" else "pc_shuf_graph"] = {
            "cells": cells, "fmax": vals, "selected": dict(sel), "selected_index": int(i)}
    star = out["pc"]["selected"]
    w = FU.adaptive_w(ntr, star["c"], star["tau"])
    qp = FU.prop_tp(FU.pc_q(Lp, pz["real"], w), pairs, device)
    acells = [{"alpha": float(a)} for a in sorted(g["pc_alpha"])]
    vals = [fmax(FU.pc_score(qp, ds_shuf_full, c["alpha"])) for c in acells]
    i, sel = FU.select(acells, vals)
    out["pc_shuf_hits"] = {"cells": acells, "fmax": vals, "selected": dict(sel),
                           "frozen_alpha_fmax": fmax(FU.pc_score(qp, ds_shuf_full, star["alpha"]))}
    b_tp = FU.prop_tp(FU.sigmoid(Lp), pairs, device)
    xcells = [{"alpha": float(a)} for a in sorted(g["xds_alpha"])]
    vals = [fmax(FU.convex(b_tp, ds_full, c["alpha"])) for c in xcells]
    pref = sorted(range(len(xcells)), key=lambda j: (abs(xcells[j]["alpha"] - 0.5) > 1e-12,
                                                       xcells[j]["alpha"]))
    i, sel = FU.select([xcells[j] for j in pref], [vals[j] for j in pref])
    out["xds_alpha_sensitivity"] = {"cells": xcells, "fmax": vals, "selected": dict(sel),
                                    "tie_rule": "0.5 first, then smaller"}
    return out


# ------------------------------------------------------------------ main
def search_one(aspect: str, base: str, a, results: Path, study, slog: guard.SearchLog) -> Dict[str, Any]:
    from go_function.constants import NAMESPACES
    t0 = time.time()
    ctx = D.aspect_context(aspect, ("train", "valid"))
    hashes = D.verify_tp_labels(aspect)
    L_all = base_logits(aspect, base, "valid", ctx, results, a.device)
    Yv = sp.csr_matrix(ctx.labels["valid"]).astype(bool)
    rows = D.eval_rows(Yv)
    L = L_all[rows].astype(np.float64)
    del L_all
    gc.collect()
    Yr = Yv[rows]
    cols = D.eval_terms(Yr, ctx.vocab, ctx.root, "confirmatory")
    sets, info = PR.build_sets(ctx, cols, NAMESPACES[aspect])
    reg_mask = np.array([len(sets["reg"][c]) > 0 for c in cols])
    names_valid = [ctx.labels["proteins_valid"][i] for i in rows]
    hsim = None
    if a.sim:
        W = PR.sim_matrix(Path(a.sim), names_valid, ctx.labels["proteins_train"])
        hsim = lambda C: np.asarray((W @ ctx.y_train[:, C].astype(np.float64)).todense())  # noqa: E731
    ds = ds_shuf = None
    dsp = None
    if a.diamond_valid:
        dsp = diamond_paths(results, aspect)
        PR.check_train_only(dsp["valid_res"], ctx.labels["proteins_valid"], ctx.labels["proteins_train"])
        H = PR.diamond_matrix(dsp["valid_res"], names_valid, ctx.labels["proteins_train"])
        ds = lambda C: PR.ds_scores(H, ctx.y_train, C)  # noqa: E731
        ds_shuf = lambda C: PR.ds_scores(H, ctx.y_train, C, shuffled=True)  # noqa: E731
    space = FU.Space(L, ctx.ntr, sets, ds=ds, ds_shuf=ds_shuf, hsim=hsim)
    out_dir = search_dir(results, aspect, base, bool(a.diamond_valid))
    sel: Dict[str, Any] = {"aspect": aspect, "base": base, "n_rows": int(len(rows)),
                           "n_terms": int(len(cols)), "n_reg_subset": int(reg_mask.sum()),
                           "inputs": hashes, "counts": {k: v for k, v in info.items()
                                                        if k != "coap_overlap_share"},
                           "families": {}, "typed_label": None if base in TIER_B else
                           "confounded (DGZ-rule labels)"}
    ntr_c = ctx.ntr[cols]
    sel["coap_overlap_share_by_band"] = M.summary(info["coap_overlap_share"], ntr_c)
    # pass 1
    fam_cells = family_cells(study)
    A = run_families(space, cols, Yr, fam_cells)
    for fam, cells in fam_cells.items():
        save_family(out_dir, fam, cells, A[fam], cols, ntr_c)
        s = selection(cells, A[fam])
        s["summary_selected"] = M.summary(A[fam][s["selected_index"]], ntr_c, reg_mask, A["base"][0])
        sel["families"][fam] = s
        slog.log("selection", aspect=aspect, base=base, family=fam, selected=jsonable_cell(s["selected"]),
                 mean_auc=s["mean_auc"])
    star = sel["families"]["apt"]["selected"]
    sel["apt_star"] = dict(star)
    if a.diamond_valid:
        # §3.2: pass 2 (w_H, shuffled hits, w_R) at the G-F frozen (c*, tau*); the
        # rerun's own pass-1 selection is recorded for the equality report only.
        frozen = frozen_star(results, aspect, base)
        sel["apt_star_gf_frozen"] = dict(frozen)
        sel["apt_star_equal_gf"] = {k: float(v) for k, v in star.items()} == frozen
        slog.log("frozen_star", aspect=aspect, base=base, rerun=jsonable_cell(star),
                 gf_frozen=jsonable_cell(frozen), equal=sel["apt_star_equal_gf"])
        star = frozen
    sel["pass2_star"] = dict(star)
    for fam in AUC_CONTROL_FAMILIES:
        j = [k for k, c in enumerate(fam_cells[fam]) if c == star][0]
        sel["families"][fam]["frozen_index"] = j
        sel["families"][fam]["frozen_mean_auc"] = sel["families"][fam]["mean_auc"][j]
    # pure propagation (no weight)
    import gc as _gc
    pc_cols = D.candidate_columns(ctx.vocab, ctx.root, "confirmatory")
    _, pairs = ic_and_pairs(ctx, pc_cols)
    pos = {int(c): i for i, c in enumerate(pc_cols)}
    Pp = FU.prop_tp(FU.sigmoid(L[:, pc_cols]), pairs, a.device)
    prop_auc = M.rank_auc(Pp[:, [pos[int(c)] for c in cols]], Yr[:, cols].toarray())
    del Pp
    _gc.collect()
    save_family(out_dir, "prop_only", [{}], prop_auc[None, :], cols, ntr_c)
    sel["families"]["prop_only"] = {"mean_auc": [float(np.nanmean(prop_auc))]}
    # pass 2 at (c*, tau*)
    g = study["grids"]
    p2: Dict[str, List[Dict[str, float]]] = {
        "apt_r": [{**star, "w_R": float(w)} for w in sorted(g["w_R"])],
        "apt_r_shuf": [{**star, "w_R": float(w)} for w in sorted(g["w_R"])]}
    if hsim is not None:
        p2["own_hsim"] = [{"w": w} for w in (0.25, 0.5, 1.0)]
        p2["apt_hsim"] = [{**star, "w": w} for w in (0.25, 0.5, 1.0)]
    if ds is not None:
        p2["apt_h"] = [{**star, "w_H": float(w)} for w in sorted(g["w_H"])]
        p2["apt_h_shufhits"] = [{**star, "w_H": float(w)} for w in sorted(g["w_H"])]
    A2 = run_families(space, cols, Yr, p2)
    for fam, cells in p2.items():
        save_family(out_dir, fam, cells, A2[fam], cols, ntr_c)
        mask = reg_mask if fam.startswith("apt_r") else None
        s = selection(cells, A2[fam], mask)
        if fam in ("own_hsim", "apt_hsim"):
            s["label"] = "exploratory only (H_sim proxy, Mac); never a finalist"
        sel["families"][fam] = s
        slog.log("selection", aspect=aspect, base=base, family=fam,
                 selected=jsonable_cell(s["selected"]), mean_auc=s["mean_auc"])
    wr = sel["families"]["apt_r"]["selected"]["w_R"]
    j = [k for k, c in enumerate(p2["apt_r_shuf"]) if c["w_R"] == wr][0]
    sel["families"]["apt_r_shuf"]["frozen_index"] = j
    if ds is not None:
        wh = sel["families"]["apt_h"]["selected"]["w_H"]
        j = [k for k, c in enumerate(p2["apt_h_shufhits"]) if c["w_H"] == wh][0]
        sel["families"]["apt_h_shufhits"]["frozen_index"] = j
        Hfull = PR.diamond_matrix(dsp["valid_res"], names_valid, ctx.labels["proteins_train"])
        ds_full = PR.ds_scores(Hfull, ctx.y_train, pc_cols)
        ds_shuf_full = PR.ds_scores(Hfull, ctx.y_train, pc_cols, shuffled=True)
        sel["protein_centric"] = pc_search(ctx, L, Yr, sets, study, ds_full, ds_shuf_full, a.device)
        slog.log("selection", aspect=aspect, base=base, family="pc",
                 selected=sel["protein_centric"]["pc"]["selected"])
    sel["wall_seconds"] = round(time.time() - t0, 1)
    MP.write_json(out_dir / "selections.json", json.loads(json.dumps(sel, default=_np)))
    return sel


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isinf(o):
        return "inf"
    raise TypeError(type(o))


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--base", choices=D.TIER_A + TIER_B, required=True)
    ap.add_argument("--results", default=MP.RESULTS_CHOICES[0])
    ap.add_argument("--sim", default=None, help="swissprot_exp.sim for the exploratory H_sim reads")
    ap.add_argument("--diamond-valid", action="store_true", help="use the G-D train-only run (remote)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args(argv)
    results = MP.results_dir(a.results)
    if a.diamond_valid or a.base in TIER_B:
        MP.refuse_on_mac("G-V2 (Diamond / Tier B)")
    if a.diamond_valid and a.base not in D.TIER_A:
        raise MP.Refused("--diamond-valid (w_H, PC) is a Tier A search (protocol §9 G-V2)")
    study = D.fusion_study()
    slog = guard.SearchLog(results / "search_log.jsonl")
    if a.base in TIER_B:
        ck = tier_b_entry(results, a.aspect, a.base)["checkpoint"]
        if ck.startswith(guard.GF_RESULTS):      # source (a); source (b) checkpoints under
            guard.allow_checkpoints([ck])         # tp_bases/ are open by the guard's own rule
    guard.enter_search_mode()
    slog.log("start", aspect=a.aspect, base=a.base, code=MP.code_hashes())
    try:
        search_one(a.aspect, a.base, a, results, study, slog)
    except guard.TestAccessError as e:
        slog.violation(e)
        raise
    slog.log("done", aspect=a.aspect, base=a.base)
    return MP.EXIT_OK


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
