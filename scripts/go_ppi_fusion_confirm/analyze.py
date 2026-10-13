#!/usr/bin/env python3
"""Confirmatory analysis (protocol_go_ppi_fusion.md §1, §2.5-2.7, §4). CPU only, run once.

Reads the ``score.py`` export, re-derives every fused score (each validation-selected weight is
recomputed and must equal ``selected_w.json``), computes per-anchor Δ = metric(fused) −
metric(text) on the same text model, and the two-level bootstrap with common random numbers
per cohort (§4.1). Then: Holm(3) for H1-H3 (trained, AZ-X), H1z-H3z (zero-shot) and H1g-H3g
(GZ-P); the C2 (``go_pair_specific_v4``) and C5 (``absorbed_share_v3``) states and the trials C2
state; TOST and four-state labels; the DE slope state; the ordered 7-label rule and headline;
the HuRI cohort; every §4.5 sensitivity; the pooled A + Bclean estimate. Writes
``analysis.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go_ppi_domain import fusion_confirm as fc  # noqa: E402

HYPS = ("H1", "H2", "H3")
ALL_METRICS = fc.METRICS


# --------------------------------------------------------------------------- loading
class Group:
    """One score group: per-file φ / grades / flags, per-model text, recorded selections."""

    def __init__(self, gdir: Path):
        self.dir = gdir
        self.meta = json.loads((gdir / "selected_w.json").read_text(encoding="utf-8"))
        self.files, self.ai = {}, {}
        for name in self.meta["files"]:
            with np.load(gdir / f"phis_{name}.npz", allow_pickle=False) as z:
                self.files[name] = {k: z[k] for k in z.files}
            f = self.files[name]
            self.ai[name] = fc.AnchorIndex(f["anchors"].astype(object), f["grades"])
        self.text = {}
        for m in self.meta["models"]:
            with np.load(gdir / f"{m}.npz", allow_pickle=False) as z:
                self.text[m] = {k: z[k] for k in z.files}

    def view(self, file: str, model: str, idx: Optional[np.ndarray] = None,
             ai: Optional[fc.AnchorIndex] = None) -> Dict[str, Any]:
        f = self.files[file]
        text = self.text[model][f"{file}_text"]
        phi = {k[4:]: v for k, v in f.items() if k.startswith("phi_")}
        grades, queries = f["grades"], self.ai[file].queries
        if idx is not None:
            text, grades, queries = text[idx], grades[idx], queries[idx]
            phi = {k: v[idx] for k, v in phi.items()}
        return {"text": text, "phi": phi, "grades": grades, "queries": queries,
                "ai": ai if ai is not None else self.ai[file]}

    def anchor_any(self, file: str, flag: str) -> np.ndarray:
        """Per anchor (np.unique order): any record of the anchor carries ``flag``."""
        fl = self.files[file][f"flag_{flag}"]
        return np.asarray([bool(fl[i].any()) for i in self.ai[file].idx])


# --------------------------------------------------------------------------- per model
def model_deltas(val, tst, arms: Sequence[str], recorded: Optional[Dict[str, Any]]):
    """Per-anchor Δ for every arm (fit on ``val``, applied to ``tst``) plus the contrasts."""
    text_m = fc.anchor_metrics(tst["text"], tst["ai"])
    D: Dict[str, Dict[str, np.ndarray]] = {}
    states = {}
    for arm in arms:
        if not (fc.arm_available(arm, val["phi"]) and fc.arm_available(arm, tst["phi"])):
            continue
        st = fc.fit_arm(arm, val)
        if recorded is not None:
            rw = (recorded.get(arm) or {}).get("w")
            if rw != st["w"]:
                raise fc.ProtocolError(f"recomputed w {st['w']} != selected_w.json {rw} ({arm})")
        fm = fc.anchor_metrics(fc.apply_arm(arm, st, tst), tst["ai"])
        D[arm] = {m: fm[m] - text_m[m] for m in ALL_METRICS}
        states[arm] = {"w": st["w"], "top2_gap": st.get("top2_gap")}
    for name, combo in fc.CONTRASTS.items():
        if all(a in D for a in combo):
            D[name] = {m: sum(c * D[a][m] for a, c in combo.items()) for m in ALL_METRICS}
    return D, states, text_m


def rescore_azx(view: Dict[str, Any], w: float) -> Dict[str, np.ndarray]:
    """AZ-X Δ on a (reduced) scored file at a fixed, already-selected w (z recomputed within anchors)."""
    fused = fc.apply_arm("AZ-X", {"w": w}, view)
    fm, tm = fc.anchor_metrics(fused, view["ai"]), fc.anchor_metrics(view["text"], view["ai"])
    return {m: fm[m] - tm[m] for m in ALL_METRICS}


def place(values: Dict[str, np.ndarray], sub_anchors: np.ndarray, anchors: np.ndarray) -> Dict[str, np.ndarray]:
    pos = {a: i for i, a in enumerate(anchors)}
    cols = np.asarray([pos[a] for a in sub_anchors], dtype=int)
    out = {}
    for m, v in values.items():
        arr = np.full(len(anchors), np.nan)
        arr[cols] = v
        out[m] = arr
    return out


def huri_supported(G: "Group", models: Sequence[str], states, anchors: np.ndarray) -> Tuple[np.ndarray, int]:
    """AZ-X hard-AUC Δ on the HuRI-supported-positive stratum of one group's test file,
    placed on ``anchors`` (NaN elsewhere)."""
    f = G.files["test"]
    sup = np.asarray([bool(((f["grades"][i] == 2) & f["flag_huri_edge"][i]).any()) for i in G.ai["test"].idx])
    anchors_sup = set(G.ai["test"].anchors[sup])
    keep = np.asarray([q in anchors_sup for q in G.ai["test"].queries]) & ((f["grades"] != 2) | f["flag_huri_edge"])
    idx = np.flatnonzero(keep)
    D = np.full((len(models), len(anchors)), np.nan)
    if len(idx):
        ai_s = fc.AnchorIndex(G.ai["test"].queries[idx], f["grades"][idx])
        for j, m in enumerate(models):
            D[j] = place(rescore_azx(G.view("test", m, idx, ai_s), states[m]["AZ-X"]["w"]), ai_s.anchors, anchors)["hard"]
    return D, int(sup.sum())


def stack(per_model: List[Dict[str, Dict[str, np.ndarray]]]) -> Dict[str, Dict[str, np.ndarray]]:
    names = set(per_model[0])
    for d in per_model[1:]:
        names &= set(d)
    return {n: {m: np.stack([d[n][m] for d in per_model]) for m in ALL_METRICS} for n in sorted(names)}


# --------------------------------------------------------------------------- estimation
def estimate(D: Dict[str, Dict[str, np.ndarray]], draws: fc.Draws, mask=None,
             names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    return {n: {m: fc.summarize(D[n][m], draws, mask) for m in ALL_METRICS}
            for n in (names or D) if n in D}


def companion(s: Dict[str, Any], level: float) -> List[float]:
    key = {round(fc.HOLM_LEVELS[0], 6): "ci9833", 0.975: "ci975", 0.95: "ci95"}[round(level, 6)]
    return list(s[key])


def decisions(est: Dict[str, Any], arm: str, names: Sequence[str], a: Dict[str, Any]) -> Dict[str, Any]:
    """Holm(3), TOST, companion intervals, p/CI disagreements, and (AZ-X) the C2/C5 states."""
    metric = dict(zip(names, ("pooled", "ap", "hard")))
    ps = {n: est[arm][metric[n]]["p"] for n in names}
    pts = {n: est[arm][metric[n]]["point"] for n in names}
    h = fc.holm3(ps, pts, a["alpha"], names)
    bound = a["equivalence"]["bound"]
    dis = []
    for n in names:
        s = est[arm][metric[n]]
        lo, hi = companion(s, h[n]["companion_level"])
        h[n]["companion_interval"] = [lo, hi]
        h[n]["tost"] = fc.tost_state(s["ci90"][0], s["ci90"][1], bound)
        if (lo > 0 or hi < 0) != h[n]["rejected"]:
            dis.append(f"{arm} {n}: Holm {'rejected' if h[n]['rejected'] else 'not rejected'} but "
                       f"{h[n]['companion_level']:.2%} interval [{lo:+.4f}, {hi:+.4f}]")
    out: Dict[str, Any] = {"holm": h, "p_ci_disagreements": dis}
    if arm == "AZ-X" and "AZ-S" in est and "X-S" in est:
        out["c2"] = {n: fc.c2_states_of(est["AZ-X"][metric[n]], est["AZ-S"][metric[n]], est["X-S"][metric[n]])
                     for n in names}
    if arm == "AZ-X" and "C5" in est:
        out["c5"] = {n: fc.c5_state_of(est["C5"][metric[n]], est["X-C5"][metric[n]]) for n in names}
    if arm == "GZ-P" and "GZ-S_P" in est:
        out["c2_descriptive"] = {n: fc.c2_states_of(est["GZ-P"][metric[n]], est["GZ-S_P"][metric[n]],
                                                    est["GZ:P-S_P"][metric[n]]) for n in names}
    return out


def four_states(est: Dict[str, Any], names: Sequence[str], metrics: Sequence[str] = ALL_METRICS,
                bound: float = 0.01) -> Dict[str, Dict[str, str]]:
    return {n: {m: fc.four_state_of(est[n][m], bound) for m in metrics} for n in names if n in est}


def cohort_block(D_tr, D_zs, draws: fc.Draws, draws_zs: fc.Draws, a, mask=None, mask_zs=None) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    if D_tr:
        est = estimate(D_tr, draws, mask)
        res["trained"] = {"estimates": est, "AZ-X": decisions(est, "AZ-X", HYPS, a),
                          "GZ-P": decisions(est, "GZ-P", ("H1g", "H2g", "H3g"), a),
                          "four_state": four_states(est, list(est))}
        for arm, names in (("AZ-X|ndcg", HYPS), ("GZ-P|pooled", ("H1g", "H2g", "H3g")), ("GZ-X", HYPS)):
            if arm in est:
                res["trained"][f"sensitivity:{arm}"] = decisions(est, arm, names, a)
    if D_zs:
        est = estimate(D_zs, draws_zs, mask_zs)
        res["zeroshot"] = {"estimates": est, "AZ-X": decisions(est, "AZ-X", ("H1z", "H2z", "H3z"), a),
                           "GZ-P": decisions(est, "GZ-P", ("H1g", "H2g", "H3g"), a),
                           "four_state": four_states(est, list(est))}
    return res


def label_inputs(block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tr = block["trained"]
    out = {}
    for h, m in fc.HYP_METRIC.items():
        ds, xs = tr["estimates"]["AZ-S"][m], tr["estimates"]["X-S"][m]
        out[h] = {"positive": tr["AZ-X"]["holm"][h]["positive"], "harm": tr["AZ-X"]["holm"][h]["harm"],
                  "tost": tr["AZ-X"]["holm"][h]["tost"], "c2": tr["AZ-X"]["c2"][h]["state"],
                  "c5": tr["AZ-X"]["c5"][h], "ds": (ds["point"], *ds["ci95"]), "xs": (xs["point"], *xs["ci95"])}
    return out


# --------------------------------------------------------------------------- cohorts
def cohort_a(scores: Path, pins, arms, B: int):
    A = Group(scores / "A" / "all")
    trained = [f"f{r['fraction']}_s{r['seed']}" for r in pins["cohortA"]["new_runs"]
               if f"f{r['fraction']}_s{r['seed']}" in A.meta["models"]]
    fractions = [int(n.split("_")[0][1:]) for n in trained]
    per, states, text_level = {}, {}, {}
    for m in trained + ["zeroshot"]:
        rec = A.meta["models"][m]["arms"]
        per[m], states[m], tm = model_deltas(A.view("validation", m), A.view("test", m), arms, rec)
        text_level[m] = {k: float(np.nanmean(v)) for k, v in tm.items()}
    return {"group": A, "trained": trained, "fractions": fractions, "per_model": per, "states": states,
            "text_level": text_level, "anchors": A.ai["test"].anchors}


def cohort_b(scores: Path, pins, arms):
    seeds = [f"s{s}" for s in pins["cohortB"]["seeds"]]
    folds, complete = {}, []
    for k in range(pins["cohortB"]["n_folds"]):
        gdir = scores / "B" / f"fold_{k}"
        if not gdir.exists():
            continue
        G = Group(gdir)
        per, st = {}, {}
        for m in G.meta["models"]:
            per[m], st[m], _ = model_deltas(G.view("validation", m), G.view("test", m), arms, G.meta["models"][m]["arms"])
        folds[k] = {"group": G, "per_model": per, "states": st, "anchors": G.ai["test"].anchors}
        if all(s in per for s in seeds):
            complete.append(k)
    return {"folds": folds, "complete": complete, "seeds": seeds}


def b_matrix(cb, names: Sequence[str], keep_folds: Sequence[int]):
    """``{name: {metric: D[model, anchor]}}`` over the anchors of ``keep_folds`` (each anchor
    uses its own fold's models), plus the anchor list."""
    anchors = [a for k in keep_folds for a in cb["folds"][k]["anchors"]]
    pos = {a: i for i, a in enumerate(anchors)}
    common = None
    for k in keep_folds:
        for n in names:
            s = set(cb["folds"][k]["per_model"][n])
            common = s if common is None else common & s
    out = {c: {m: np.full((len(names), len(anchors)), np.nan) for m in ALL_METRICS} for c in sorted(common or [])}
    for k in keep_folds:
        f = cb["folds"][k]
        cols = np.asarray([pos[a] for a in f["anchors"]], dtype=int)
        for j, n in enumerate(names):
            for c in out:
                for m in ALL_METRICS:
                    out[c][m][j, cols] = f["per_model"][n][c][m]
    return out, np.asarray(anchors, dtype=object)


def pooled(estA: Dict[str, Any], estB: Dict[str, Any], arm: str = "AZ-X") -> Dict[str, Any]:
    out = {}
    for h, m in fc.HYP_METRIC.items():
        a, b = estA[arm][m], estB[arm][m]
        n1, n2 = a["n_defined"], b["n_defined"]
        T = (n1 * a["_T"] + n2 * b["_T"]) / (n1 + n2)
        out[h] = fc.summarize_T(T, (n1 * a["point"] + n2 * b["point"]) / (n1 + n2), [n1, n2])
    return out


def tertile_masks(values: np.ndarray) -> List[np.ndarray]:
    cuts = np.quantile(values, [1 / 3, 2 / 3])
    t = np.searchsorted(cuts, values, side="right")
    return [t == i for i in range(3)]


def stratified(D, draws, values, label) -> Dict[str, Any]:
    """§4.5 study-bias tertiles (cut points from the cohort's test anchors) + top − bottom."""
    masks = tertile_masks(values)
    out = {"variable": label, "cut_points": [float(x) for x in np.quantile(values, [1 / 3, 2 / 3])], "tertiles": []}
    for i, mk in enumerate(masks):
        out["tertiles"].append({h: fc.summarize(D["AZ-X"][m], draws, mk) for h, m in fc.HYP_METRIC.items()})
        out["tertiles"][-1]["n_anchors"] = int(mk.sum())
    out["top_minus_bottom"] = {}
    for h in HYPS:
        top, bot = out["tertiles"][2][h], out["tertiles"][0][h]
        out["top_minus_bottom"][h] = fc.summarize_T(top["_T"] - bot["_T"], top["point"] - bot["point"])
    return out


# --------------------------------------------------------------------------- report
def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in ("_T",)}
    if isinstance(obj, (list, tuple)):
        return [strip(v) for v in obj]
    return obj


def fmt(s: Dict[str, Any]) -> str:
    lo, hi = s["ci95"]
    n = s.get("n_defined")
    return f"{s['point']:+.4f} [{lo:+.4f}, {hi:+.4f}] p={s['p']:.4f}" + (f" n={n}" if n is not None else "")


def render(res: Dict[str, Any]) -> str:
    L = [f"**Headline:** {res['headline']}", "", res.get("components_line", ""), "",
         res.get("secondary_line", ""), "", "# GO/PPI fusion confirmation (AZ-X primary, GZ-P secondary): report", ""]
    for w in res.get("warnings", []):
        L.append(f"**WARNING:** {w}")
    L += ["", f"Pins: `{res['pins']['path']}` sha256 `{res['pins']['sha256']}`", "",
          "## Prior exposure (§0.3, disclosed, not corrected)", ""] + [f"- {d}" for d in DISCLOSURES] + [""]
    L += ["## Final labels (trained, AZ-X)", ""] + [f"- {h}: {res['labels'][h]['label']} (label {res['labels'][h]['label_id']})"
                                                    for h in HYPS if h in res.get("labels", {})]
    L += ["", "## Co-primary, zero-shot and secondary-fusion Holm triples (Δ fused − text)", "",
          "| hyp | cohort | arm | Δ [95% CI] | companion | Holm | positive | harm | TOST |", "|---|---|---|---|---|---|---|---|---|"]
    for cname in ("A", "Bclean", "all_B"):
        c = res["cohorts"].get(cname, {})
        for kind, arm, names in (("trained", "AZ-X", HYPS), ("zeroshot", "AZ-X", ("H1z", "H2z", "H3z")),
                                 ("trained", "GZ-P", ("H1g", "H2g", "H3g"))):
            if kind not in c:
                continue
            dec = c[kind][arm]["holm"]
            for n, m in zip(names, ("pooled", "ap", "hard")):
                s = c[kind]["estimates"][arm][m]
                d = dec[n]
                L.append(f"| {n} | {cname} | {arm} | {fmt(s)} | {d['companion_level']:.2%} "
                         f"[{d['companion_interval'][0]:+.4f}, {d['companion_interval'][1]:+.4f}] | "
                         f"{'rejected' if d['rejected'] else 'not rejected'} | {d['positive']} | {d['harm']} | {d['tost']} |")
    L += ["", "## Secondary metrics S1-S5 (AZ-X; uncorrected 95% CIs) and controls", "",
          "| quantity | cohort | model | " + " | ".join(ALL_METRICS) + " |", "|---|---|---|" + "---|" * len(ALL_METRICS)]
    for cname in ("A", "Bclean", "huri"):
        c = res["cohorts"].get(cname, {})
        for kind in ("trained", "zeroshot"):
            if kind not in c:
                continue
            for q in ("AZ-X", "AZ-S", "X-S", "C1:P-X", "C1a:P-N", "C1a:N-X", "C1t:P-T", "C1ii:P-X@0.4", "C3:UX-X",
                      "AZ-Bz", "C5", "X-C5", "GZ-P", "AZX-GZP", "GZ:U-P", "GZ:P-S_P"):
                e = c[kind]["estimates"].get(q)
                if e is None:
                    continue
                L.append(f"| {q} | {cname} | {kind} | " + " | ".join(
                    f"{e[m]['point']:+.4f} [{e[m]['ci95'][0]:+.4f}, {e[m]['ci95'][1]:+.4f}] {c[kind]['four_state'].get(q, {}).get(m, '')}"
                    for m in ALL_METRICS) + " |")
    L += ["", "## Control states (trained)", ""]
    for cname in ("A", "Bclean", "all_B"):
        c = res["cohorts"].get(cname, {}).get("trained")
        if c and "c2" in c["AZ-X"]:
            L.append(f"- {cname}: C2 {{{', '.join(f'{h}: {c['AZ-X']['c2'][h]['state']} (trials: {c['AZ-X']['c2'][h]['trials_state']})' for h in HYPS)}}}; "
                     f"C5 {c['AZ-X']['c5']}")
    if "de" in res:
        de = res["de"]
        L += ["", "## Data efficiency (cohort A, AZ-X hard AUC)", "",
              f"- state: **{de['state']}**; β_DE = {fmt(de['hard'])}" if "hard" in de else f"- state: {de['state']}"]
        for k in ("pooled", "ap", "gzp_hard"):
            if k in de:
                L.append(f"- β_DE {k}: {fmt(de[k])}")
        for F, s in de.get("by_fraction", {}).items():
            L.append(f"- F={F}: Δ hard {fmt(s)}; text hard {de['text_level_hard'][F]:.4f}")
        if "endpoint_F10_minus_F100" in de:
            L.append(f"- F10 − F100: {fmt(de['endpoint_F10_minus_F100'])}")
    L += ["", "## Sensitivities (none changes a label)", ""]
    for name, s in res.get("sensitivities", {}).items():
        if isinstance(s, dict) and all(h in s for h in HYPS):
            L.append(f"- {name}: " + "; ".join(f"{h} {fmt(s[h])}" for h in HYPS))
        elif isinstance(s, dict) and "point" in s:
            L.append(f"- {name}: {fmt(s)}")
        elif isinstance(s, dict) and "tertiles" in s:
            for i, t in enumerate(s["tertiles"]):
                L.append(f"- {name} tertile {i + 1} (n={t['n_anchors']}): " + "; ".join(f"{h} {fmt(t[h])}" for h in HYPS))
            L.append(f"- {name} top − bottom: " + "; ".join(f"{h} {fmt(s['top_minus_bottom'][h])}" for h in HYPS))
        elif isinstance(s, dict) and "estimates" in s:
            for arm in ("AZ-X", "AZ-UX", "GZ-P", "GZ-U", "GZ-U|pooled"):
                e = s["estimates"].get(arm)
                if e:
                    L.append(f"- {name} {arm}: " + "; ".join(f"{h} {fmt(e[m])}" for h, m in fc.HYP_METRIC.items()))
            for arm, d in s["retrained_A_minus_apub"].items():
                L.append(f"- {name} retrained A − A-pub {arm}: " + "; ".join(f"{h} {fmt(d[h])}" for h in HYPS))
        elif isinstance(s, dict) and all(isinstance(v, dict) and "H3" in v for v in s.values()):
            L.append(f"- {name}: " + "; ".join(f"{k} (n={v['n_anchors']}) H3 {fmt(v['H3'])}" for k, v in s.items()))
        else:
            L.append(f"- {name}: see analysis.json")
    L += ["", "## Pooled A + Bclean (descriptive)", ""]
    for kind, p in res.get("pooled", {}).items():
        L.append(f"- {kind}: " + "; ".join(f"{h} {fmt(p[h])}" for h in HYPS))
    L += ["", "## p / CI disagreements", ""]
    L += [f"- {d}" for d in res.get("p_ci_disagreements", [])] or ["- none"]
    L += ["", "## Near ties, exclusions and counts", ""]
    L += [f"- {e}" for e in res.get("exclusions", [])] or ["- no exclusions"]
    for n in res.get("near_ties", []):
        L.append(f"- near tie: {n}")
    for k, v in res.get("counts", {}).items():
        L.append(f"- {k}: {v}")
    return "\n".join(L) + "\n"


DISCLOSURES = (
    "Cohort A test and validation anchors were all scored under fusion before (publication run, F§5); "
    "AZ-X itself was chosen on cohort A's validation file and scored once on its test file (§0.1b).",
    "Older 3-checkpoint fusion probes on subsamples of the same files (results/ontology_ceiling/fusion_probe_*).",
    "Zero-shot text-only metrics on the published files; masked-text smoke models without φ (go_withholding smoke).",
    "φ-only simGIC ceilings on the whole interactome (HuRI 0.548 vs Lit-BM 0.674), label-level counts, "
    "fold simulations under seed 2026 (confirmatory seed 2027), label-free φ statistics.",
    "The best-fusion search (208 variants, validation only) and its one test scoring; no cohort-B fold, "
    "HuRI file or train-partition anchor was read, so Bclean is unaffected.",
)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    fc.require_hashseed()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pins", required=True, type=Path)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    fc.require_cpu(args.device, "analysis")
    out, report = fc.refuse_existing(args.out), fc.refuse_existing(args.report)
    pins = fc.load_verified_pins(args.pins)
    if not pins.get("confirmatory"):
        raise fc.ProtocolError("non-confirmatory pins")
    bad = fc.verify_pinned_hashes(pins, ("inputs", "code", "configs", "derived"))
    if bad:
        raise fc.ProtocolError(f"pinned hash mismatch, refusing to analyse: {bad[:10]}")
    a = pins["analysis"]
    B = a["B"]
    scores = fc.absp(args.scores)
    sm = json.loads((scores / "score_manifest.json").read_text(encoding="utf-8"))
    pins_ref = {"path": fc.rel(fc.absp(args.pins)), "sha256": fc.file_sha256(args.pins)}
    if sm.get("pins") != pins_ref:
        raise fc.ProtocolError(f"scores were produced under pins {sm.get('pins')}, not {pins_ref}")
    t0 = time.time()
    arms = [x for x in fc.COHORT_ARMS if x != "AZ-T" or a.get("T_computed")]
    res: Dict[str, Any] = {"schema": "go_ppi_fusion_confirm/analysis/v1", "created_utc": fc.utc_now(),
                           "confirmatory": True, "pins": pins_ref,
                           "scores_manifest_sha256": fc.file_sha256(scores / "score_manifest.json"),
                           "cohorts": {}, "sensitivities": {}, "exclusions": [f"run excluded: {r}" for r in sm["excluded_runs"]],
                           "near_ties": [], "counts": {}, "run_env": fc.run_env_record()}

    # ---------------- cohort A
    ca = cohort_a(scores, pins, arms, B)
    if len(ca["trained"]) < a["min_checkpoints_A"]:
        raise fc.ProtocolError(f"cohort A has {len(ca['trained'])} < {a['min_checkpoints_A']} checkpoints; stop and amend")
    DA = stack([ca["per_model"][m] for m in ca["trained"]])
    DZ = stack([ca["per_model"]["zeroshot"]])
    nA = len(ca["anchors"])
    dA = fc.Draws(a["rng"]["A"], len(ca["trained"]), nA, B, strata=ca["fractions"],
                  extra_models=len(pins["published_checkpoints"]))
    res["cohorts"]["A"] = cohort_block(DA, DZ, dA, dA.zero_shot(), a)
    res["cohorts"]["A"]["weights"] = ca["states"]
    for m, st in ca["states"].items():
        for arm, s in st.items():
            if s.get("top2_gap") is not None and s["top2_gap"] < 1e-4:
                res["near_ties"].append(f"A {m} {arm}: w={s['w']} top-two gap {s['top2_gap']:.2e}")
    print(f"cohort A done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- cohort B
    cb = cohort_b(scores, pins, arms)
    incomplete = [k for k in cb["folds"] if k not in cb["complete"]]
    res["exclusions"] += [f"cohort B fold {k} incomplete: its anchors dropped from trained estimates, kept in zero-shot"
                          for k in incomplete]
    if len(cb["complete"]) < a["min_complete_folds_B"]:
        raise fc.ProtocolError(f"only {len(cb['complete'])} complete cohort-B folds; stop and amend")
    DB, anchors_B = b_matrix(cb, cb["seeds"], cb["complete"])
    DBz, anchors_Bz = b_matrix(cb, ["zeroshot"], sorted(cb["folds"]))
    bclean = set(pins["derived"]["bclean"]["anchors"])
    cols = np.asarray([i for i, x in enumerate(anchors_B) if x in bclean], dtype=int)
    cols_z = np.asarray([i for i, x in enumerate(anchors_Bz) if x in bclean], dtype=int)

    def sub(D, c):
        return {n: {m: v[:, c] for m, v in mets.items()} for n, mets in D.items()}

    dBc = fc.Draws(a["rng"]["Bclean"], len(cb["seeds"]), len(cols), B)
    dBcz = dBc.zero_shot() if np.array_equal(anchors_B[cols], anchors_Bz[cols_z]) else \
        fc.Draws(list(a["rng"]["Bclean"]) + [0], 1, len(cols_z), B).zero_shot()
    res["cohorts"]["Bclean"] = cohort_block(sub(DB, cols), sub(DBz, cols_z), dBc, dBcz, a)
    dAll = fc.Draws(a["rng"]["all_B"], len(cb["seeds"]), len(anchors_B), B)
    dAllz = dAll.zero_shot() if np.array_equal(anchors_B, anchors_Bz) else \
        fc.Draws(list(a["rng"]["all_B"]) + [0], 1, len(anchors_Bz), B).zero_shot()
    res["cohorts"]["all_B"] = cohort_block(DB, DBz, dAll, dAllz, a)
    exposed_mask = np.asarray([x not in bclean for x in anchors_B])
    res["sensitivities"]["all_B: exposed anchors alone (descriptive)"] = {
        h: fc.summarize(DB["AZ-X"][m], dAll, exposed_mask) for h, m in fc.HYP_METRIC.items()}
    res["counts"].update({"Bclean anchors (trained)": len(cols), "all_B anchors (trained)": len(anchors_B),
                          "cohort A anchors": nA, "complete folds": cb["complete"]})
    print(f"cohort B done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- HuRI cohort (cohort-A models, cohort-A selections)
    A = ca["group"]
    H = Group(scores / "huri" / "all")
    per_h, per_hu = {}, {}
    for m in ca["trained"] + ["zeroshot"]:
        rec = A.meta["models"][m]["arms"]
        per_h[m], _, _ = model_deltas(A.view("validation", m), H.view("test", m), arms, rec)
        per_hu[m], _, _ = model_deltas(A.view("validation", m), H.view("test_g1u", m), ["AZ-X"], rec)
    nH = len(H.ai["test"].anchors)
    dH = fc.Draws(a["rng"]["huri"], len(ca["trained"]), nH, B)
    DH = stack([per_h[m] for m in ca["trained"]])
    res["cohorts"]["huri"] = cohort_block(DH, stack([per_h["zeroshot"]]), dH, dH.zero_shot(), a)
    huri_state = {h: res["cohorts"]["huri"]["trained"]["four_state"]["AZ-X"][m] for h, m in fc.HYP_METRIC.items()}
    nHu = len(H.ai["test_g1u"].anchors)
    dHu = fc.Draws(a["rng"]["huri"], len(ca["trained"]), nHu, B)
    res["sensitivities"]["huri_grade1_unrestricted"] = {
        h: fc.summarize(np.stack([per_hu[x]["AZ-X"][m] for x in ca["trained"]]), dHu) for h, m in fc.HYP_METRIC.items()}
    keep = ~(H.files["test"]["flag_string_grade2"] & (H.files["test"]["grades"] == 2))
    idx = np.flatnonzero(keep)
    ai_sub = fc.AnchorIndex(H.ai["test"].queries[idx], H.files["test"]["grades"][idx])
    Dn = np.stack([place(rescore_azx(H.view("test", m, idx, ai_sub), ca["states"][m]["AZ-X"]["w"]),
                         ai_sub.anchors, H.ai["test"].anchors)["hard"] for m in ca["trained"]])
    res["sensitivities"]["huri_new_labels (H3)"] = fc.summarize(Dn, dH)
    res["counts"]["HuRI anchors"] = nH
    print(f"HuRI done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- labels, headline, warnings
    LA, LB = label_inputs(res["cohorts"]["A"]), label_inputs(res["cohorts"]["Bclean"])
    labels = {h: fc.final_label(LA[h], LB[h], huri_state[h]) for h in HYPS}
    res["labels"] = labels
    res["headline"] = fc.headline(labels)
    trA = res["cohorts"]["A"]["trained"]["estimates"]
    xs, ux = trA["X-S"], trA["C3:UX-X"]
    res["components_line"] = ("Pair-specific X − S (A): " + "; ".join(f"{h} {fmt(xs[m])} {fc.four_state_of(xs[m])}"
                                                                       for h, m in fc.HYP_METRIC.items())
                              + " | leak UX − X (A): " + "; ".join(f"{h} {fmt(ux[m])} {fc.four_state_of(ux[m])}"
                                                                   for h, m in fc.HYP_METRIC.items()))
    parts = []
    for cname in ("A", "Bclean"):
        tr = res["cohorts"][cname]["trained"]
        hg = tr["GZ-P"]["holm"]
        status = ", ".join(f"{n} {'positive' if hg[n]['positive'] else ('harm' if hg[n]['harm'] else 'not rejected')}"
                           for n in ("H1g", "H2g", "H3g"))
        fs = tr["four_state"]
        parts.append(f"{cname}: Secondary GZ-P: {status}, nDCG@10 {fs['GZ-P']['ndcg10']}; "
                     f"AZ-X − GZ-P: H3 {fs['AZX-GZP']['hard']}, nDCG@10 {fs['AZX-GZP']['ndcg10']}")
    res["secondary_line"] = " | ".join(parts)
    res["warnings"] = fc.exposure_warnings(
        labels, {h: res["cohorts"]["Bclean"]["trained"]["estimates"]["AZ-X"][m]["point"] for h, m in fc.HYP_METRIC.items()},
        {h: res["cohorts"]["all_B"]["trained"]["estimates"]["AZ-X"][m]["point"] for h, m in fc.HYP_METRIC.items()})
    res["p_ci_disagreements"] = [d for c in res["cohorts"].values() for k in ("trained", "zeroshot") if k in c
                                 for arm in ("AZ-X", "GZ-P") for d in c[k][arm]["p_ci_disagreements"]]

    # ---------------- DE (§1.3, §4.4)
    per_frac = collections.Counter(ca["fractions"])
    de: Dict[str, Any] = {}
    if all(per_frac[F] >= 2 for F in fc.FRACTIONS):
        de["hard"] = fc.de_slope(DA["AZ-X"]["hard"], ca["fractions"], dA)
        de["state"] = fc.de_state(de["hard"], a["alpha"], a["equivalence"]["bound_de"])
        de["pooled"] = fc.de_slope(DA["AZ-X"]["pooled"], ca["fractions"], dA)
        de["ap"] = fc.de_slope(DA["AZ-X"]["ap"], ca["fractions"], dA)
        de["gzp_hard"] = fc.de_slope(DA["GZ-P"]["hard"], ca["fractions"], dA)
        de["by_fraction"], de["text_level_hard"] = {}, {}
        groups = collections.OrderedDict()
        for i, F in enumerate(ca["fractions"]):
            groups.setdefault(F, []).append(i)
        start = 0
        for F, members in groups.items():
            colsF = np.arange(start, start + len(members))       # Cs columns of fraction F (strata order)
            start += len(members)
            local = {g: i for i, g in enumerate(members)}
            CF = np.vectorize(local.__getitem__)(dA.Cs[:, colsF])
            de["by_fraction"][str(F)] = fc.summarize(DA["AZ-X"]["hard"][members], dA.with_models(CF, len(members)))
            de["text_level_hard"][str(F)] = float(np.mean([ca["text_level"][ca["trained"][i]]["hard"] for i in members]))
        f10, f100 = de["by_fraction"]["10"], de["by_fraction"]["100"]
        de["endpoint_F10_minus_F100"] = fc.summarize_T(f10["_T"] - f100["_T"], f10["point"] - f100["point"])
    else:
        de["state"] = "not computed (a fraction has < 2 checkpoints)"
    res["de"] = de

    # ---------------- cohort-A sensitivities
    res["sensitivities"]["fraction-stratified c* (A)"] = {h: fc.summarize(DA["AZ-X"][m], dA.stratified())
                                                          for h, m in fc.HYP_METRIC.items()}
    nonempty = ~A.anchor_any("test", "empty_anchor")
    res["sensitivities"]["exclude empty-φ anchors (A)"] = {h: fc.summarize(DA["AZ-X"][m], dA, nonempty)
                                                           for h, m in fc.HYP_METRIC.items()}
    res["counts"]["empty-φ anchors (A test)"] = int((~nonempty).sum())
    ft = A.files["test"]
    keep = ~(ft["flag_mention"] & (ft["grades"] == 2))
    idx = np.flatnonzero(keep)
    ai_m = fc.AnchorIndex(A.ai["test"].queries[idx], ft["grades"][idx])
    Dm = {h: [] for h in HYPS}
    for x in ca["trained"]:
        d = place(rescore_azx(A.view("test", x, idx, ai_m), ca["states"][x]["AZ-X"]["w"]), ai_m.anchors, ca["anchors"])
        for h, m in fc.HYP_METRIC.items():
            Dm[h].append(d[m])
    res["sensitivities"]["mention: drop flagged grade-2 records (A)"] = {
        h: fc.summarize(np.stack(Dm[h]), dA) for h in HYPS}
    n_pmid = json.loads(fc.absp(f"{pins['derived']['phi_dir']}/n_pmid.json").read_text())["n_pmid"]
    degree = collections.Counter(p["anchor_id"] for p in fc.read_jsonl(fc.PAIRS_REL))
    for label, val in (("n_PMID(anchor)", [n_pmid.get(x, 0) for x in ca["anchors"]]),
                       ("STRING degree(anchor)", [degree.get(x, 0) for x in ca["anchors"]])):
        res["sensitivities"][f"study-bias tertiles A: {label}"] = stratified(DA, dA, np.asarray(val, float), label)
    # HuRI-supported-positive strata (A, Bclean, all_B): anchors with a grade-2 HuRI edge; grade-2 rows limited to HuRI edges
    # HuRI-supported-positive strata (A, Bclean, all_B): anchors with a grade-2 record that is a
    # HuRI edge, their grade-2 rows limited to HuRI-supported positives (re-scored at the selected w)
    hs = {}
    Ds, n_sup = huri_supported(A, ca["trained"], ca["states"], ca["anchors"])
    hs["A"] = {"n_anchors": n_sup, "H3": fc.summarize(Ds, dA)}
    allD = np.full((len(cb["seeds"]), len(anchors_B)), np.nan)
    for k in cb["complete"]:
        Dk, _ = huri_supported(cb["folds"][k]["group"], cb["seeds"], cb["folds"][k]["states"], anchors_B)
        allD = np.where(np.isfinite(Dk), Dk, allD)
    for cname, colset, draws in (("Bclean", cols, dBc), ("all_B", np.arange(len(anchors_B)), dAll)):
        Dsub = allD[:, colset]
        hs[cname] = {"n_anchors": int(np.isfinite(Dsub[0]).sum()), "H3": fc.summarize(Dsub, draws)}
    res["sensitivities"]["HuRI-supported-positive strata (H3, descriptive)"] = hs

    # ---------------- A-pub (descriptive)
    P = Group(scores / "apub" / "all")
    pub_models = [f"f{r['fraction']}_s{r['seed']}" for r in pins["published_checkpoints"]]
    if not np.array_equal(P.ai["test"].anchors, ca["anchors"]):
        raise fc.ProtocolError("A-pub anchors differ from cohort A anchors")
    per_p = [model_deltas(P.view("validation", m), P.view("test", m), list(fc.APUB_ARMS),
                          P.meta["models"][m]["arms"])[0] for m in pub_models]
    DP = stack(per_p)
    dP = dA.with_models(dA.C_extra, len(pub_models))
    estP = estimate(DP, dP)
    apub = {"estimates": estP, "retrained_A_minus_apub": {}}
    for arm in ("AZ-X", "GZ-P"):
        apub["retrained_A_minus_apub"][arm] = {
            h: fc.summarize_T(trA[arm][m]["_T"] - estP[arm][m]["_T"], trA[arm][m]["point"] - estP[arm][m]["point"])
            for h, m in fc.HYP_METRIC.items()}
    res["sensitivities"]["A-pub (15 published checkpoints, descriptive)"] = apub

    # ---------------- pooled A + Bclean (descriptive)
    res["pooled"] = {"trained": pooled(trA, res["cohorts"]["Bclean"]["trained"]["estimates"]),
                     "zeroshot": pooled(res["cohorts"]["A"]["zeroshot"]["estimates"],
                                        res["cohorts"]["Bclean"]["zeroshot"]["estimates"])}
    res["seconds"] = round(time.time() - t0, 1)
    clean = fc.json_safe(strip(res))
    fc.write_json(out, clean)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(clean), encoding="utf-8")
    print(res["headline"])
    print(f"wrote {fc.rel(out)} and {fc.rel(report)} ({res['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
