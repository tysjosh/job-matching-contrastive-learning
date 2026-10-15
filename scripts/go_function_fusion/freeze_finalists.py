"""Gates G-F (Mac, Tier A) and G-F2 (remote, --remote): freeze the finalists (protocol §9, §11.2).

G-F writes <results>/finalists_tierA.json: per (aspect, base) the AP-T (c*, tau*), the
fixed-w c*, every control's frozen cell (= AP-T's) and re-selected cell, the Tier A
typed read's w_R* (confounded), the S2 parameters (Tier A released-MLP AP-T per aspect),
code hashes of go_function_fusion/ and scripts/go_function_fusion/, input hashes; writes
<results>/code_pin.json; pins G-F. From here every confirmatory code file is frozen.

G-F2 (--remote) writes <results>/finalists_remote.json: Tier A w_H* and PC
(c_PC*, tau_PC*, alpha*), Tier B AP-T / fixed-w / AP-T+R / controls, the X+DS alpha
sensitivity; verifies the code hashes equal G-F's; records gf_grid_snapshot (listing
only) for the §11.2 rule; pins G-F2.

Usage:
  .venv/bin/python scripts/go_function_fusion/freeze_finalists.py --results results/go_function_fusion
  .venv/bin/python scripts/go_function_fusion/freeze_finalists.py --remote --results results/go_function_fusion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_pins as MP  # noqa: E402
from go_function_fusion import data as D  # noqa: E402
from go_function_fusion import guard  # noqa: E402
from go_function_fusion import stats as ST  # noqa: E402

CONTROL_FAMS = ("shuf_graph", "coa", "coap", "pa_any", "desc_max", "protocol_F")
TIER_B = ("V0IPR", "V4IPR", "V0ESM")


def load_sel(results: Path, aspect: str, base: str, suffix: str = "") -> Dict[str, Any]:
    p = results / "search" / f"{aspect}_{base}{suffix}" / "selections.json"
    if not p.exists():
        raise MP.Refused(f"{MP.rel(p)} missing: run search_valid.py first")
    return json.loads(p.read_text())


def base_block(sel: Dict[str, Any], tier_b: bool) -> Dict[str, Any]:
    f = sel["families"]
    out = {"apt": f["apt"]["selected"], "fixed_w": f["fixed"]["selected"],
           "controls": {k: {"frozen": f["apt"]["selected"], "reselected": f[k]["selected"],
                            "reselected_null": f[k]["null_selected"]} for k in CONTROL_FAMS},
           "apt_r": {"w_R": f["apt_r"]["selected"]["w_R"],
                     "label": None if tier_b else "confounded (DGZ-rule labels)"},
           "apt_r_shuf": {"reselected": f["apt_r_shuf"]["selected"]},
           "n_terms_valid": sel["n_terms"]}
    return out


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=MP.RESULTS_CHOICES[0])
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--aspects", default="bp,mf")
    a = ap.parse_args(argv)
    results = MP.results_dir(a.results)
    guard.enter_search_mode()
    aspects = a.aspects.split(",")
    code = MP.code_hashes()
    if not a.remote:
        if not (ROOT / "config/go_function_fusion/dgz_published_ds.json").exists():
            raise MP.Refused("config/go_function_fusion/dgz_published_ds.json (DGZ Tables 3-4 DS "
                             "cells) must be transcribed at G-R, before G-F (§6.3)")
        if (results / "finalists_tierA.json").exists():
            raise MP.Refused("finalists_tierA.json exists (G-F is written once)")
        fin: Dict[str, Any] = {"schema": 1, "gate": "G-F", "protocol_sha256": MP.protocol_sha(),
                               "code": code, "bases": {}, "s2": {}}
        for asp in aspects:
            for base in D.TIER_A:
                fin["bases"][f"{asp}/{base}"] = base_block(load_sel(results, asp, base), False)
            fin["s2"][asp] = {"source": f"{asp}/mlp (Tier A released MLP AP-T)",
                              **fin["bases"][f"{asp}/mlp"]["apt"]}
        inputs = {}
        for asp in aspects:
            inputs.update(D.verify_tp_labels(asp))
        fin["inputs"] = inputs
        sha = MP.write_json(results / "finalists_tierA.json", fin)
        csha = MP.write_json(results / "code_pin.json", {"code": code})
        MP.record_pin(results, "G-F", {"finalists_tierA.json": sha, "code_pin.json": csha,
                                       "utc": guard.utc_now()})
        print(f"G-F pinned: finalists_tierA.json {sha[:12]}")
        return MP.EXIT_OK
    MP.refuse_on_mac("G-F2")
    gf = MP.require_pin(results, "G-F")
    cp = json.loads((results / "code_pin.json").read_text())["code"]
    cmp = MP.compare_code(cp, code)
    if not cmp["equal"]:
        raise MP.Fatal(f"code differs from the G-F pin (a §14 amendment is needed): {cmp['differs']}")
    if (results / "finalists_remote.json").exists():
        raise MP.Refused("finalists_remote.json exists (G-F2 is written once)")
    fin = {"schema": 1, "gate": "G-F2", "protocol_sha256": MP.protocol_sha(),
           "finalists_tierA_sha256": gf["finalists_tierA.json"], "tierA": {}, "tierB": {}}
    from go_function.pins import sha256_file
    if sha256_file(results / "finalists_tierA.json") != gf["finalists_tierA.json"]:
        raise MP.Fatal("finalists_tierA.json differs from the G-F pin")
    fin_a = json.loads((results / "finalists_tierA.json").read_text())["bases"]
    for asp in aspects:
        for base in D.TIER_A:
            s = load_sel(results, asp, base, "_dv")
            if "protein_centric" not in s or "apt_h" not in s["families"]:
                raise MP.Refused(f"{asp}/{base}: run search_valid.py --diamond-valid first")
            frozen = {k: float(v) for k, v in fin_a[f"{asp}/{base}"]["apt"].items()}
            got = {k: float(v) for k, v in s.get("pass2_star", {}).items()}
            if got != frozen:   # §3.2: w_H must be selected at the G-F (c*, tau*)
                raise MP.Fatal(f"{asp}/{base}: w_H grid cell {got} differs from the G-F AP-T {frozen}")
            f = s["families"]
            pc = s["protein_centric"]
            fin["tierA"][f"{asp}/{base}"] = {
                "w_H_at": frozen, "rerun_apt_equal_gf": bool(s["apt_star_equal_gf"]),
                "w_H": f["apt_h"]["selected"]["w_H"],
                "apt_h_shufhits": {"reselected": f["apt_h_shufhits"]["selected"]},
                "pc": pc["pc"]["selected"], "pc_shuf_graph": {"reselected": pc["pc_shuf_graph"]["selected"]},
                "pc_shuf_hits": {"reselected": pc["pc_shuf_hits"]["selected"]},
                "xds_alpha_sensitivity": pc["xds_alpha_sensitivity"]["selected"]}
        for base in TIER_B:
            p = results / "search" / f"{asp}_{base}" / "selections.json"
            if p.exists():
                fin["tierB"][f"{asp}/{base}"] = base_block(json.loads(p.read_text()), True)
    snap = ST.grid_snapshot(ROOT / "results/go_function/runs")
    import time
    snap["utc"] = guard.utc_now()
    snap["epoch_seconds"] = time.time()
    fin["gf_grid_snapshot"] = snap
    sha = MP.write_json(results / "finalists_remote.json", fin)
    MP.record_pin(results, "G-F2", {"finalists_remote.json": sha, "gf_grid_snapshot": snap,
                                    "code_pin_equal": True})
    print(f"G-F2 pinned: finalists_remote.json {sha[:12]}; snapshot {snap}")
    return MP.EXIT_OK


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
