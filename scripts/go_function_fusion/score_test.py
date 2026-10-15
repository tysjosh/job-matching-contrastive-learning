"""Gate G-T (REMOTE ONLY): the single test scoring (protocol §9).

Refuses unless: the host is not macOS and CUDA is available (the remote L40S host);
pins G-F, G-D and G-F2 exist (G-TP when a Tier B base is scored); the code hashes equal
<results>/code_pin.json; <results>/test/ does not exist (a rerun is allowed only with
--rerun-identical, the identical command and code hashes, and is logged).
Writes <results>/test/test_read.json first (timestamp, hashes, command).

Scores every frozen finalist, control (frozen and re-selected cells), comparator and
base from finalists_tierA.json / finalists_remote.json on the test split (proteins with
>= 1 TP label; confirmatory term set):
  per base   test_<aspect>_<base>_terms.npz      per-term AUC / AUPR of every variant
             test_<aspect>_<base>_protein.npz    per-protein threshold tables (Fmax/Smin)
             test_<aspect>_<base>_pboot.json     descriptive protein-bootstrap of P_b
  per aspect dgz_faithful_<aspect>.json          DGZ-faithful X+DS check (does not gate)
Then run report.py.

Usage (remote):
  .venv/bin/python scripts/go_function_fusion/score_test.py --results results/go_function_fusion --device cuda
"""
from __future__ import annotations

import argparse
import gc
import json
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
from go_function_fusion import stats as ST  # noqa: E402

CONTROL_FAMS = ("shuf_graph", "coa", "coap", "pa_any", "desc_max", "protocol_F")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cell(c: Dict[str, Any]) -> Dict[str, float]:
    return {k: (float("inf") if v == "inf" else float(v)) for k, v in c.items()}


# ------------------------------------------------------------------ refusals
def preflight(results: Path, argv: List[str], rerun: bool, need_tier_b: bool) -> Dict[str, Any]:
    MP.refuse_on_mac("score_test.py")
    import torch
    if not torch.cuda.is_available():
        raise MP.Refused("score_test.py needs the remote host (CUDA not available)")
    pins = {k: MP.require_pin(results, k) for k in ("G-F", "G-D", "G-F2")}
    if need_tier_b:
        pins["G-TP"] = MP.require_pin(results, "G-TP")
    from go_function.pins import sha256_file
    for name, key in (("finalists_tierA.json", "G-F"), ("finalists_remote.json", "G-F2")):
        if sha256_file(results / name) != pins[key][name]:
            raise MP.Fatal(f"{name} differs from pin {key}")
    code = MP.code_hashes()
    cmp = MP.compare_code(json.loads((results / "code_pin.json").read_text())["code"], code)
    if not cmp["equal"]:
        raise MP.Refused(f"code differs from the G-F code pin: {cmp['differs']}")
    tdir = results / "test"
    rec = {"utc": guard.utc_now(), "command": argv, "code": code, "protocol_sha256": MP.protocol_sha(),
           "pins": {k: v for k, v in pins.items()}}
    if tdir.exists():
        prev = json.loads((tdir / "test_read.json").read_text()) if (tdir / "test_read.json").exists() else None
        if not rerun or prev is None or prev["command"] != argv or prev["code"] != code:
            raise MP.Refused(f"{MP.rel(tdir)} exists (G-T is scored once; a rerun needs "
                             f"--rerun-identical with the identical command and code)")
        n = 1
        while (tdir / f"test_read.rerun{n}.json").exists():
            n += 1
        MP.write_json(tdir / f"test_read.rerun{n}.json", {**rec, "rerun_of": prev["utc"]})
        log(f"rerun {n} logged")
    else:
        MP.write_json(tdir / "test_read.json", rec)
    return rec


# ------------------------------------------------------------------ scoring
def term_variants(fin_base: Dict[str, Any], remote_base: Dict[str, Any] | None, has_ds: bool
                  ) -> Dict[str, tuple]:
    """name -> (family, cell) for every frozen / re-selected class-centric variant."""
    star = cell(fin_base["apt"])
    v = {"base": ("base", {}), "apt": ("apt", star), "fixed_w": ("fixed", cell(fin_base["fixed_w"])),
         "parent_only": ("parent_only", {}), "F_weightfree": ("F_weightfree", {})}
    for k in CONTROL_FAMS:
        v[f"{k}@frozen"] = (k, star)
        v[f"{k}@reselected"] = (k, cell(fin_base["controls"][k]["reselected"]))
    wr = float(fin_base["apt_r"]["w_R"])
    v["apt_r"] = ("apt_r", {**star, "w_R": wr})
    v["apt_r_shuf@frozen"] = ("apt_r_shuf", {**star, "w_R": wr})
    v["apt_r_shuf@reselected"] = ("apt_r_shuf", cell(fin_base["apt_r_shuf"]["reselected"]))
    if remote_base is not None and has_ds:
        wh = float(remote_base["w_H"])
        v["apt_h"] = ("apt_h", {**star, "w_H": wh})
        v["apt_h_shufhits@frozen"] = ("apt_h_shufhits", {**star, "w_H": wh})
        v["apt_h_shufhits@reselected"] = ("apt_h_shufhits", cell(remote_base["apt_h_shufhits"]["reselected"]))
    return v


def score_base(aspect, base, ctx, L, Yr, cols, sets, H, Hrel, fin_base, remote_base, results,
               device, D_f, D_p, out_dir) -> None:
    ds = (lambda C: PR.ds_scores(H, ctx.y_train, C))
    ds_shuf = (lambda C: PR.ds_scores(H, ctx.y_train, C, shuffled=True))
    space = FU.Space(L, ctx.ntr, sets, ds=ds, ds_shuf=ds_shuf)
    variants = term_variants(fin_base, remote_base, True)
    fns = {name: (lambda C, x, f=f, c=c: space.score(f, c, C, x)) for name, (f, c) in variants.items()}
    if remote_base is not None:
        wh = float(remote_base["w_H"])
        rel_space = FU.Space(L, ctx.ntr, sets, ds=lambda C: PR.ds_scores(Hrel, ctx.y_train, C))
        star = cell(fin_base["apt"])
        fns["apt_h@releasedDS"] = (lambda C, x: rel_space.score("apt_h", {**star, "w_H": wh}, C,
                                                               rel_space.ctx(C)))
    res = FU.score_variants(L.shape[0], cols, fns, space.ctx, lambda C: Yr[:, C].toarray(),
                            {"auc": M.rank_auc, "aupr": M.average_precision}, log=log)
    pc_cols = D.candidate_columns(ctx.vocab, ctx.root, "confirmatory")
    pos = {int(c): i for i, c in enumerate(pc_cols)}
    from search_valid import ic_and_pairs
    ic, pairs = ic_and_pairs(ctx, pc_cols)
    Lp = L[:, pc_cols]
    b_tp = FU.prop_tp(FU.sigmoid(Lp), pairs, device)
    res["auc"]["prop_only"] = M.rank_auc(b_tp[:, [pos[int(c)] for c in cols]], Yr[:, cols].toarray())
    arrays = {"cols": np.asarray(cols), "ntr": ctx.ntr[cols],
              "reg_mask": np.array([len(sets["reg"][c]) > 0 for c in cols])}
    for m in ("auc", "aupr"):
        for k, v in res[m].items():
            arrays[f"{m}__{k}"] = np.asarray(v, dtype=np.float64)
    if base == "deepgozero":
        from go_function.model import dgz_propagation_pairs
        from go_function.ontology import DgzRule
        dsrc, ddst = dgz_propagation_pairs(ctx.terms, DgzRule(ROOT / D.DGZ / "go.obo"))
        Pd = FU.prop_tp(FU.sigmoid(L), (dsrc, ddst), device)
        arrays["auc__dgz_propagated"] = M.rank_auc(Pd[:, cols], Yr[:, cols].toarray())
        del Pd
    MP.write_npz(out_dir / f"test_{aspect}_{base}_terms.npz", arrays)
    # descriptive protein bootstrap of P_b (second protein draw)
    star = cell(fin_base["apt"])
    S_f = np.empty((L.shape[0], len(cols)))
    S_b = np.empty_like(S_f)
    for a0 in range(0, len(cols), 1500):
        C = cols[a0:a0 + 1500]
        x = space.ctx(C)
        S_f[:, a0:a0 + len(C)] = space.score("apt", star, C, x)
        S_b[:, a0:a0 + len(C)] = x["own"]
    pb = ST.protein_boot_P(S_f, S_b, Yr[:, cols].toarray(), D_p)
    MP.write_json(out_dir / f"test_{aspect}_{base}_pboot.json", pb)
    del S_f, S_b
    gc.collect()
    # protein-centric tables
    y = Yr[:, pc_cols]
    ds_full = PR.ds_scores(H, ctx.y_train, pc_cols)
    tabs: Dict[str, Any] = {}
    pc_summaries: Dict[str, Any] = {}

    def tab(name, S):
        tabs[name] = M.protein_tables(S, y, ic)
        pc_summaries[name] = M.protein_centric(S, y, ic, device=device)

    tab("b_TP", b_tp)
    xa = 0.5
    tab("XDS@0.5", FU.convex(b_tp, ds_full, xa))
    if remote_base is not None:
        xs = float(remote_base["xds_alpha_sensitivity"]["alpha"])
        tab("XDS@sensitivity", FU.convex(b_tp, ds_full, xs))
        ds_rel = PR.ds_scores(Hrel, ctx.y_train, pc_cols)
        tab("XDS@0.5_releasedDS", FU.convex(b_tp, ds_rel, xa))
        pc = cell(remote_base["pc"])
        ntr = ctx.ntr[pc_cols]
        pz = FU.prior_z(L, sets["pa_tp"], pc_cols)
        qp = FU.prop_tp(FU.pc_q(Lp, pz, FU.adaptive_w(ntr, pc["c"], pc["tau"])), pairs, device)
        tab("PC@alpha*", FU.pc_score(qp, ds_full, pc["alpha"]))
        tab("PC@alpha0", qp)
        ds_sh = PR.ds_scores(H, ctx.y_train, pc_cols, shuffled=True)
        tab("PC_shuf_hits@frozen", FU.pc_score(qp, ds_sh, pc["alpha"]))
        rs = cell(remote_base["pc_shuf_hits"]["reselected"])
        tab("PC_shuf_hits@reselected", FU.pc_score(qp, ds_sh, rs["alpha"]))
        del qp
        pzs = FU.prior_z(L, sets["pa_shuf"], pc_cols)
        for tag, cc in (("frozen", pc), ("reselected", cell(remote_base["pc_shuf_graph"]["reselected"]))):
            q = FU.prop_tp(FU.pc_q(Lp, pzs, FU.adaptive_w(ntr, cc["c"], cc["tau"])), pairs, device)
            tab(f"PC_shuf_graph@{tag}", FU.pc_score(q, ds_full, cc["alpha"]))
        del pzs, q
    arr = {}
    for k, t in tabs.items():
        for f in ("precision", "recall", "ru", "mi", "has_prediction"):
            arr[f"{k}__{f}"] = t[f]
        arr[f"{k}__rows"] = t["rows"]
    MP.write_npz(out_dir / f"test_{aspect}_{base}_protein.npz", arr)
    MP.write_json(out_dir / f"test_{aspect}_{base}_protein_summary.json",
                  {k: summ for k, summ in pc_summaries.items()})


def dgz_faithful(aspect: str, ctx, L_by_base: Dict[str, np.ndarray], out_dir: Path, device: str) -> None:
    """X+DS at alpha = 0.5 with DGZ's rule (blast_preds from the released
    test_diamond.res, train + valid source, DGZ propagation), DGZ-faithful evaluator."""
    import pandas as pd
    from go_function import evaluate as E
    from go_function.model import dgz_propagation_pairs
    from go_function.ontology import DgzRule
    dgz = DgzRule(ROOT / D.DGZ / "go.obo")
    tr = D.read_split_frame(aspect, "train")
    va = D.read_split_frame(aspect, "valid")
    te = D.read_split_frame(aspect, "test")
    src_names = list(tr.proteins) + list(va.proteins)
    src_ann = [list(x) for x in tr.prop_annotations] + [list(x) for x in va.prop_annotations]
    blast = PR.dgz_blast_preds(list(te.proteins), ROOT / D.DGZ / aspect / "test_diamond.res",
                               src_names, src_ann, ctx.terms, dgz)
    test_prop = [list(x) for x in te.prop_annotations]
    go_set = E.dgz_namespace_terms(dgz, aspect)
    ic = E.dgz_ic(dgz, src_ann + test_prop)
    dsrc, ddst = dgz_propagation_pairs(ctx.terms, dgz)
    out: Dict[str, Any] = {"alpha": 0.5, "label": "descriptive; does not gate"}
    pub_p = ROOT / "config/go_function_fusion/dgz_published_ds.json"
    pub = json.loads(pub_p.read_text()) if pub_p.exists() else None
    for base, Lfull in L_by_base.items():
        P = FU.prop_tp(FU.sigmoid(Lfull), (dsrc, ddst), device).astype(np.float32)
        preds = blast * 0.5 + P * 0.5
        pm = E.dgz_protein_metrics(preds, ctx.terms, test_prop, go_set, ic, device=device)
        cm = E.dgz_class_auc(preds, ctx.terms, test_prop)
        r = {**pm, "auc": cm["auc"]}
        if pub is not None:
            ref = pub.get("values", {}).get(aspect, {}).get(base, {})
            r["published"] = ref
            r["misses_0.005"] = sorted(k for k, v in ref.items()
                                       if k in r and abs(float(r[k]) - float(v["value"])) > 0.005)
        out[base] = r
    MP.write_json(out_dir / f"dgz_faithful_{aspect}.json", out)


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=MP.RESULTS_CHOICES[0])
    ap.add_argument("--aspects", default="bp,mf")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--rerun-identical", action="store_true")
    a = ap.parse_args(argv)
    MP.refuse_on_mac("score_test.py")
    raw = list(argv if argv is not None else sys.argv[1:])
    results = MP.results_dir(a.results)
    fin = json.loads((results / "finalists_tierA.json").read_text())
    rem = json.loads((results / "finalists_remote.json").read_text())
    preflight(results, [x for x in raw if x != "--rerun-identical"], a.rerun_identical, bool(rem["tierB"]))
    out_dir = results / "test"
    study = D.fusion_study()
    for asp in a.aspects.split(","):
        ctx = D.aspect_context(asp, ("train", "valid", "test"))
        Yt = sp.csr_matrix(ctx.labels["test"]).astype(bool)
        rows = D.eval_rows(Yt)
        Yr = Yt[rows]
        cols = D.eval_terms(Yr, ctx.vocab, ctx.root, "confirmatory")
        from go_function.constants import NAMESPACES
        sets, info = PR.build_sets(ctx, cols, NAMESPACES[asp])
        names = [ctx.labels["proteins_test"][i] for i in rows]
        dpaths = {k: ROOT / v["path"] for k, v in MP.require_pin(results, "G-D")[asp].items()
                  if isinstance(v, dict) and "path" in v}
        PR.check_train_only(dpaths["test_res"], ctx.labels["proteins_test"], ctx.labels["proteins_train"])
        H = PR.diamond_matrix(dpaths["test_res"], names, ctx.labels["proteins_train"])
        Hrel = PR.diamond_matrix(ROOT / D.DGZ / asp / "test_diamond.res", names, ctx.labels["proteins_train"])
        D_f, D_p = ST.protein_draws(len(rows), int(study["bootstrap"]["B"]), int(study["bootstrap"]["B_P"]))
        MP.write_json(out_dir / f"meta_{asp}.json", {"n_rows": int(len(rows)), "n_terms": int(len(cols)),
                                                     "counts": {k: v for k, v in info.items()
                                                                if k != "coap_overlap_share"}})
        from search_valid import base_logits
        full: Dict[str, np.ndarray] = {}
        bases = [(b, fin["bases"][f"{asp}/{b}"], rem["tierA"].get(f"{asp}/{b}")) for b in D.TIER_A]
        bases += [(b, rem["tierB"][f"{asp}/{b}"], None) for b in ("V0IPR", "V4IPR", "V0ESM")
                  if f"{asp}/{b}" in rem["tierB"]]
        for base, fb, rb in bases:
            log(f"{asp} {base}")
            L_all = base_logits(asp, base, "test", ctx, results, a.device)
            if base in D.TIER_A:
                full[base] = L_all
            L = L_all[rows].astype(np.float64)
            score_base(asp, base, ctx, L, Yr, cols, sets, H, Hrel, fb, rb, results, a.device,
                       D_f, D_p, out_dir)
            del L
            gc.collect()
        dgz_faithful(asp, ctx, full, out_dir, a.device)
        del full
        gc.collect()
    MP.record_pin(results, "G-T", {"test_read_utc": json.loads((out_dir / "test_read.json")
                                                               .read_text())["utc"]})
    log("G-T done; run report.py")
    return MP.EXIT_OK


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
