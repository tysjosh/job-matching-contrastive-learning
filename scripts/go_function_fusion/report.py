"""After G-T: contrasts, bootstraps, Holm and readings (protocol §6-§8, §12).

Reads <results>/test/ (score_test.py outputs) and the pinned finalists; writes
<results>/analysis/contrasts.json and <results>/analysis/decisions.json. Opens no
label and computes no score: every number comes from the per-term / per-protein
tables written by score_test.py.

Usage:
  .venv/bin/python scripts/go_function_fusion/report.py --results results/go_function_fusion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import make_pins as MP  # noqa: E402
from go_function_fusion import data as D  # noqa: E402
from go_function_fusion import metrics as M  # noqa: E402
from go_function_fusion import stats as ST  # noqa: E402

NAME = {"mlp": "MLP", "deepgozero": "DGZ", "V0IPR": "V0IPR", "V4IPR": "V4IPR", "V0ESM": "V0ESM"}
CONTROLS = ("shuf_graph", "coa", "coap", "pa_any", "desc_max", "protocol_F")


def load_terms(tdir: Path, aspect: str, base: str) -> Optional[Dict[str, np.ndarray]]:
    p = tdir / f"test_{aspect}_{base}_terms.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_tabs(tdir: Path, aspect: str, base: str) -> Dict[str, Dict[str, np.ndarray]]:
    p = tdir / f"test_{aspect}_{base}_protein.npz"
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            name, f = k.split("__")
            out.setdefault(name, {})[f] = z[k]
    return out


def bands(d: np.ndarray, dr: ST.TermDraws) -> Dict[str, Any]:
    return {b: ST.contrast_band(d, dr, b) for b in M.BAND_NAMES}


def comparison_reading(c: Dict[str, Any]) -> str:
    if not c.get("computed"):
        return "not computed"
    lo, hi = c["ci95"]
    if lo > 0:
        return "exceeds"
    if -0.01 <= lo and hi <= 0.01:
        return "matches"
    return "neither"


def analyse_aspect(aspect: str, tdir: Path, fin: Dict[str, Any], rem: Dict[str, Any], study) -> Dict[str, Any]:
    B = int(study["bootstrap"]["B"])
    T = {b: load_terms(tdir, aspect, b) for b in D.TIER_A + ("V0IPR", "V4IPR", "V0ESM")}
    T = {b: v for b, v in T.items() if v is not None}
    ref = T["mlp"]
    ntr, reg = ref["ntr"], ref["reg_mask"].astype(bool)
    for b, v in T.items():
        if not np.array_equal(v["cols"], ref["cols"]):
            raise MP.Fatal(f"{aspect}/{b}: evaluated term set differs between bases")
    dr = ST.term_draws(len(ntr), M.band_index(ntr), reg, B, ST.TERM_SEED)
    n_prot = load_tabs(tdir, aspect, "mlp")["b_TP"]["recall"].shape[1]
    D_f, _ = ST.protein_draws(n_prot, B, int(study["bootstrap"]["B_P"]), ST.PROTEIN_SEED)
    out: Dict[str, Any] = {"confirmatory": {}, "descriptive": {}, "n_terms": int(len(ntr)),
                           "n_reg_subset": int(reg.sum()),
                           "band_counts": {b: int(len(dr.band_members[b])) for b in M.BAND_NAMES}}
    conf, desc = out["confirmatory"], out["descriptive"]
    for b, t in T.items():
        nm = NAME[b]
        auc = lambda k: t[f"auc__{k}"]  # noqa: E731
        d_p = auc("apt") - auc("base")
        conf[f"P_{nm}"] = ST.contrast_all(d_p, dr)
        conf[f"Sc0_{nm}"] = ST.contrast_scarcity(d_p, dr, "0")
        conf[f"Sc1_{nm}"] = ST.contrast_scarcity(d_p, dr, "1-10")
        dd: Dict[str, Any] = {"P_by_band": bands(d_p, dr)}
        fb = fin["bases"].get(f"{aspect}/{b}") or rem["tierB"].get(f"{aspect}/{b}")
        dd["D_struct"] = ST.d_struct(auc("apt"), auc("shuf_graph@frozen"), dr)
        wr = float(fb["apt_r"]["w_R"])
        ty = ST.weighted_or_null(wr, auc("apt_r") - auc("apt"), dr, reg=True)
        dd["D_typed"] = ST.d_typed(wr, auc("apt_r"), auc("apt_r_shuf@frozen"), dr)
        if b in D.TIER_A:
            dd["Ty_confounded"] = {**ty, "label": "confounded (DGZ-rule labels)"}
        else:
            conf[f"Ty_{nm}"] = ty
        for k in ("fixed_w", "parent_only", "F_weightfree", "prop_only"):
            dd[f"{k}-base"] = ST.contrast_all(auc(k) - auc("base"), dr)
        for k in CONTROLS:
            for tag in ("frozen", "reselected"):
                v = auc(f"{k}@{tag}")
                dd[f"{k}@{tag}-base"] = ST.contrast_all(v - auc("base"), dr)
                dd[f"apt-{k}@{tag}"] = ST.contrast_all(auc("apt") - v, dr)
            dd[f"{k}@frozen-base_by_band"] = bands(auc(f"{k}@frozen") - auc("base"), dr)
        dd["aupr_apt-base"] = ST.contrast_all(t["aupr__apt"] - t["aupr__base"], dr)
        pb = tdir / f"test_{aspect}_{b}_pboot.json"
        if pb.exists():
            dd["P_protein_bootstrap"] = json.loads(pb.read_text())
        rb = rem["tierA"].get(f"{aspect}/{b}")
        if b in D.TIER_A and rb is not None:
            wh = float(rb["w_H"])
            conf[f"Hc_{nm}"] = ST.weighted_or_null(wh, auc("apt_h") - auc("apt"), dr)
            tabs = load_tabs(tdir, aspect, b)
            conf[f"Hf_{nm}"] = ST.hf_or_null(float(rb["pc"]["alpha"]), tabs["PC@alpha*"],
                                             tabs["PC@alpha0"], D_f)
            dd["PCp"] = ST.fmax_contrast(tabs["PC@alpha0"], tabs["b_TP"], D_f)
            dd["PCt"] = ST.fmax_contrast(tabs["PC@alpha*"], tabs["b_TP"], D_f)
            dd["XDS"] = ST.fmax_contrast(tabs["XDS@0.5"], tabs["b_TP"], D_f)
            for k in ("XDS@sensitivity", "XDS@0.5_releasedDS", "PC_shuf_graph@frozen",
                      "PC_shuf_graph@reselected", "PC_shuf_hits@frozen", "PC_shuf_hits@reselected"):
                if k in tabs:
                    dd[f"{k}-b_TP"] = ST.fmax_contrast(tabs[k], tabs["b_TP"], D_f)
            dd["apt_h_shufhits@frozen-apt"] = ST.contrast_all(auc("apt_h_shufhits@frozen") - auc("apt"), dr)
            dd["apt_h@releasedDS-apt"] = ST.contrast_all(auc("apt_h@releasedDS") - auc("apt"), dr)
            sp = tdir / f"test_{aspect}_{b}_protein_summary.json"
            if sp.exists():
                dd["protein_centric_points"] = json.loads(sp.read_text())
        desc[nm] = dd
    if "mlp" in T and "deepgozero" in T:
        m, g = T["mlp"], T["deepgozero"]
        for k, v in (("APT(MLP)-DGZ", m["auc__apt"] - g["auc__base"]),
                     ("APT(MLP)-APT(DGZ)", m["auc__apt"] - g["auc__apt"]),
                     ("APT(MLP)-DGZpropagated", m["auc__apt"] - g["auc__dgz_propagated"])):
            bb = bands(v, dr)
            desc[k] = {"all": ST.contrast_all(v, dr), "by_band": bb,
                       "reading_by_band": {b: comparison_reading(c) for b, c in bb.items()}}
        desc["pooled_tierA_P"] = ST.two_level(np.stack([m["auc__apt"] - m["auc__base"],
                                                        g["auc__apt"] - g["auc__base"]]))
    tb = [b for b in ("V0IPR", "V4IPR", "V0ESM") if b in T]
    if len(tb) >= 2:
        desc["pooled_tierB_P"] = ST.two_level(np.stack([T[b]["auc__apt"] - T[b]["auc__base"] for b in tb]))
    # Holm and classification
    fam = study["holm_families"]
    floors = study["floors"]
    readings: Dict[str, Any] = {}
    for fname, names in fam.items():
        if fname == "F-T" and aspect != "bp":
            continue
        pv = {n: (conf[n]["p"] if n in conf and conf[n].get("computed") else None) for n in names}
        padj = ST.holm(names, pv)
        for n in names:
            if n not in conf:
                continue
            conf[n]["p_holm"] = padj[n]
            conf[n]["family"] = fname
            conf[n]["class"] = ST.classify(conf[n], floors[n.split("_")[0].rstrip("01")], padj[n])
    if aspect != "bp":
        for n in [k for k in conf if k.startswith("Ty_")]:
            conf[n]["family"] = "descriptive (MF typed reads)"
            conf[n]["class"] = ST.classify(conf[n], floors["Ty"], None)
    for b, nm in ((b, NAME[b]) for b in T):
        dd = desc[nm]
        r: Dict[str, Any] = {}
        p = conf.get(f"P_{nm}", {})
        r["primary"] = p.get("class")
        r["structural"] = ST.ci_low_positive(dd["D_struct"])
        r["scarcity"] = conf.get(f"Sc0_{nm}", {}).get("class") == "supported" and \
            conf.get(f"Sc1_{nm}", {}).get("class") == "supported"
        if f"Ty_{nm}" in conf:
            r["typed"] = conf[f"Ty_{nm}"].get("class")
            r["typed_structure"] = ST.ci_low_positive(dd["D_typed"])
        coa = dd.get("apt-coa@frozen", {})
        r["coannotation_suffices"] = bool(coa.get("computed") and coa["estimate"] <= 0)
        readings[nm] = r
    if "MLP" in readings:
        out["s1_primary_success"] = readings["MLP"]["primary"] == "supported"
        out["adds_to_dgz_axioms"] = out["s1_primary_success"] and readings.get("DGZ", {}).get("primary") == "supported"
    out["readings"] = readings
    return out


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=MP.RESULTS_CHOICES[0])
    ap.add_argument("--aspects", default="bp,mf")
    a = ap.parse_args(argv)
    results = MP.results_dir(a.results)
    MP.require_pin(results, "G-T")
    tdir = results / "test"
    fin = json.loads((results / "finalists_tierA.json").read_text())
    rem = json.loads((results / "finalists_remote.json").read_text())
    study = D.fusion_study()
    allc = {asp: analyse_aspect(asp, tdir, fin, rem, study) for asp in a.aspects.split(",")}
    MP.write_json(results / "analysis" / "contrasts.json", json.loads(json.dumps(allc, default=_np)))
    dec = {asp: {"s1_primary_success": v.get("s1_primary_success"),
                 "adds_to_dgz_axioms": v.get("adds_to_dgz_axioms"), "readings": v["readings"],
                 "confirmatory_classes": {k: c.get("class") for k, c in v["confirmatory"].items()}}
           for asp, v in allc.items()}
    MP.write_json(results / "analysis" / "decisions.json", json.loads(json.dumps(dec, default=_np)))
    print(json.dumps(dec, indent=2, default=_np))
    return MP.EXIT_OK


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
