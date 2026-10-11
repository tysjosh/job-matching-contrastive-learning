#!/usr/bin/env python3
"""Confirmatory analysis (protocol_trials_fusion.md §1, §2.4-2.6, §4). CPU only, run once.

Reads the per-pair score export of ``score.py``, re-derives every fused score (the
validation-nDCG weight is recomputed and must equal ``selected_w.json``), computes
per-patient Δ = metric(fused) − metric(text) and the two-level bootstrap with common
random numbers per cohort, then the Holm decision, C1-C3, S4, the C2 states, the
ordered final labels and headline, the sensitivities and the pooled estimate. Writes
``analysis.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trials_domain import fusion_confirm as fc  # noqa: E402

CONDITIONS = {  # name: (variant, normaliser, fixed w or None)
    "primary": ("hierarchy", "global_z", None),
    "C1_exact": ("exact", "global_z", None),
    "C2_shuffled": ("shuffled", "global_z", None),
    "C3ii_hier_w02": ("hierarchy", "global_z", 0.2),
    "C3ii_exact_w02": ("exact", "global_z", 0.2),
    "sens_within_patient_z": ("hierarchy", "within_patient_z", None),
}
HYP = {"H1": "pooled", "H2": "ap"}


# --------------------------------------------------------------------------- per model
def model_deltas(d, sel: Dict[str, Any], grid, patients: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-patient Δ for every condition (+ the C3(i) / hier−C2 contrasts) for one model."""
    vg, vq = d["validation_grades"], d["validation_patients"]
    tg, tq = d["test_grades"], d["test_patients"]
    vt, tt = d["validation_text"], d["test_text"]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    fused_m: Dict[str, Dict[str, np.ndarray]] = {}
    text_m: Dict[str, Dict[str, np.ndarray]] = {}
    for name, (variant, norm, fixed) in CONDITIONS.items():
        vp, tp = d[f"validation_phi_{variant}"], d[f"test_phi_{variant}"]
        if norm == "global_z":
            ts, ps = fc.zstats(vt), fc.zstats(vp)
            vtz, ttz, vpz, tpz = fc.zapply(vt, ts), fc.zapply(tt, ts), fc.zapply(vp, ps), fc.zapply(tp, ps)
        else:
            vtz, ttz = fc.within_patient_z(vt, vq), fc.within_patient_z(tt, tq)
            vpz, tpz = fc.within_patient_z(vp, vq), fc.within_patient_z(tp, tq)
        if fixed is None:
            w, _ = fc.select_weight(vtz, vpz, vg, vq, grid)
            recorded = sel["variants"][variant][norm]["w"]
            if w != recorded:
                raise fc.ProtocolError(f"recomputed w {w} != selected_w.json {recorded} ({variant}, {norm})")
        else:
            w = fixed
        if norm not in text_m:
            text_m[norm] = fc.per_patient_metrics(ttz, tg, tq, patients)
        fused_m[name] = fc.per_patient_metrics(fc.fuse(ttz, tpz, w), tg, tq, patients)
        out[name] = {m: fused_m[name][m] - text_m[norm][m] for m in fc.METRICS}
        out[name]["_w"] = w
    out["hier_minus_C2"] = {m: fused_m["primary"][m] - fused_m["C2_shuffled"][m] for m in fc.METRICS}
    out["C3i_hier_minus_exact"] = {m: fused_m["primary"][m] - fused_m["C1_exact"][m] for m in fc.METRICS}
    out["C3ii_hier_minus_exact_w02"] = {m: fused_m["C3ii_hier_w02"][m] - fused_m["C3ii_exact_w02"][m] for m in fc.METRICS}
    return out


def load_group(gdir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sel = json.loads((gdir / "selected_w.json").read_text(encoding="utf-8"))
    data = {}
    for name in sel["models"]:
        with np.load(gdir / f"{name}.npz", allow_pickle=False) as z:
            data[name] = {k: z[k] for k in z.files}
    return sel, data


# --------------------------------------------------------------------------- cohort assembly
def assemble_2022(scores: Path, grid) -> Dict[str, Any]:
    sel, data = load_group(scores / "2022" / "all")
    any_d = next(iter(data.values()))
    patients = fc.sort_topics(set(any_d["test_patients"]))
    trained = [n for n in sel["models"] if sel["models"][n]["kind"] != "zeroshot"]
    deltas = {n: model_deltas(data[n], sel["models"][n], grid, patients) for n in sel["models"]}
    cond = patient_condition(any_d, patients)
    return {"patients": patients, "trained": trained, "zeroshot": ["zeroshot"], "deltas": deltas,
            "fractions": [sel["models"][n]["fraction"] for n in trained], "condition_signal": cond,
            "weights": {n: {c: deltas[n][c]["_w"] for c in CONDITIONS} for n in sel["models"]},
            "shuffle_draws": {"all": sel["shuffle_draws"]}, "trained_patients": patients,
            "zeroshot_patients": patients}


def patient_condition(d, patients) -> Dict[str, bool]:
    out = {}
    for p, c in zip(d["test_patients"], d["test_condition_signal"]):
        out.setdefault(str(p), bool(c))
    return {p: out[p] for p in patients}


def assemble_2021(scores: Path, grid, seeds, n_folds: int) -> Dict[str, Any]:
    per_fold, complete, all_patients, cond = {}, [], [], {}
    weights, draws = {}, {}
    for k in range(n_folds):
        gdir = scores / "2021" / f"fold_{k}"
        if not gdir.exists():
            continue
        sel, data = load_group(gdir)
        any_d = next(iter(data.values()))
        pats = fc.sort_topics(set(any_d["test_patients"]))
        all_patients += pats
        cond.update(patient_condition(any_d, pats))
        deltas = {n: model_deltas(data[n], sel["models"][n], grid, pats) for n in sel["models"]}
        per_fold[k] = {"patients": pats, "deltas": deltas}
        weights[f"fold_{k}"] = {n: {c: deltas[n][c]["_w"] for c in CONDITIONS} for n in sel["models"]}
        draws[f"fold_{k}"] = sel["shuffle_draws"]
        if all(f"s{s}" in deltas for s in seeds):
            complete.append(k)
    all_patients = fc.sort_topics(all_patients)
    trained_patients = fc.sort_topics(p for k in complete for p in per_fold[k]["patients"])

    def stack(model_names: List[str], patients: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """``{condition: {metric: D[model, patient]}}`` taking each patient's own fold."""
        pos = {p: i for i, p in enumerate(patients)}
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for k, f in per_fold.items():
            for j, name in enumerate(model_names):
                if name not in f["deltas"]:
                    continue
                for c, mets in f["deltas"][name].items():
                    for m, v in mets.items():
                        if m == "_w":
                            continue
                        arr = out.setdefault(c, {}).setdefault(m, np.full((len(model_names), len(patients)), np.nan))
                        for i, p in enumerate(f["patients"]):
                            if p in pos:
                                arr[j, pos[p]] = v[i]
        return out

    trained_names = [f"s{s}" for s in seeds]
    return {"patients": all_patients, "trained_patients": trained_patients, "zeroshot_patients": all_patients,
            "complete_folds": complete, "folds_present": sorted(per_fold),
            "D_trained": stack(trained_names, trained_patients) if trained_patients else {},
            "D_zeroshot": stack(["zeroshot"], all_patients), "condition_signal": cond,
            "weights": weights, "shuffle_draws": draws, "trained": trained_names}


def D_2022(c: Dict[str, Any], names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for cond in c["deltas"][names[0]]:
        out[cond] = {m: np.stack([c["deltas"][n][cond][m] for n in names])
                     for m in fc.METRICS}
    return out


# --------------------------------------------------------------------------- estimation
def estimate_block(D: Dict[str, Dict[str, np.ndarray]], C, P, cols: Optional[np.ndarray] = None) -> Dict[str, Any]:
    out = {}
    for cond, mets in D.items():
        out[cond] = {}
        for m, arr in mets.items():
            a = arr if cols is None else arr[:, cols]
            out[cond][m] = fc.summarize(a, C, P)
    return out


def decisions(est: Dict[str, Any], alpha: float, names=("H1", "H2"), c2_ub=0.005, bound=0.01) -> Dict[str, Any]:
    pr = est["primary"]
    h = fc.holm(pr["pooled"]["p"], pr["ap"]["p"], pr["pooled"]["point"], pr["ap"]["point"], alpha, names)
    disagreements = []
    for n, m in zip(names, ("pooled", "ap")):
        lvl = h[n]["companion_level"]
        lo, hi = pr[m]["ci975"] if lvl == 0.975 else pr[m]["ci95"]
        excludes = lo > 0 or hi < 0
        h[n]["companion_interval"] = [lo, hi]
        if excludes != h[n]["rejected"]:
            disagreements.append(f"{n}: Holm {'rejected' if h[n]['rejected'] else 'not rejected'} but "
                                 f"{lvl:.1%} interval [{lo:+.4f}, {hi:+.4f}] {'excludes' if excludes else 'contains'} 0")
    c2, diff = est["C2_shuffled"]["pooled"], est["hier_minus_C2"]["pooled"]
    state = fc.c2_state(c2["ci95"][0], c2["ci95"][1], diff["ci95"][0], pr["pooled"]["point"], c2["point"], c2_ub)
    s4 = fc.equivalence_state(pr["hard"]["ci90"][0], pr["hard"]["ci90"][1], bound)
    return {"holm": h, "c2_state": state, "s4_state": s4, "p_ci_disagreements": disagreements}


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k != "_T"}
    if isinstance(obj, (list, tuple)):
        return [strip(v) for v in obj]
    return obj


def cohort_analysis(year: str, D_tr, D_zs, tr_patients, zs_patients, models_meta, pins, B, cond_signal,
                    fractions=None) -> Dict[str, Any]:
    a = pins["analysis"]
    seed = a["rng"][year]
    rng = np.random.default_rng(seed)
    res: Dict[str, Any] = {"rng": seed, "B": B, "n_patients_trained": len(tr_patients),
                           "n_patients_zeroshot": len(zs_patients), "models": models_meta}
    if D_tr:
        n_c = next(iter(next(iter(D_tr.values())).values())).shape[0]
        C, P = fc.draw_indices(rng, n_c, len(tr_patients), B)
        res["trained"] = estimate_block(D_tr, C, P)
        res["trained_decisions"] = decisions(res["trained"], a["alpha"], ("H1", "H2"), a["c2_pass_ub"], a["equivalence_bound"])
    else:
        C = P = None
    if zs_patients == tr_patients and P is not None:
        Pz = P
    else:
        Pz = rng.integers(0, len(zs_patients), size=(B, len(zs_patients)))
        res["zeroshot_patient_draws"] = "separate (patient set differs from trained)"
    Cz = np.zeros((B, 1), dtype=int)
    res["zeroshot"] = estimate_block(D_zs, Cz, Pz)
    res["zeroshot_decisions"] = decisions(res["zeroshot"], a["alpha"], ("H1z", "H2z"), a["c2_pass_ub"], a["equivalence_bound"])

    # sensitivity: drop condition_signal_present == false patients (fresh generator, same seed)
    sens = {}
    for label, pats, D, is_zs in (("trained", tr_patients, D_tr, False), ("zeroshot", zs_patients, D_zs, True)):
        if not D:
            continue
        keep = np.array([cond_signal[p] for p in pats])
        r2 = np.random.default_rng(seed)
        n_c = 1 if is_zs else next(iter(next(iter(D.values())).values())).shape[0]
        C2_, P2_ = fc.draw_indices(r2, n_c, int(keep.sum()), B)
        if is_zs:
            C2_ = np.zeros_like(C2_)
        sens[f"exclude_no_condition_signal_{label}"] = {
            "excluded": [p for p, k in zip(pats, keep) if not k],
            "primary": estimate_block({"primary": D["primary"]}, C2_, P2_, np.flatnonzero(keep))["primary"]}
    if fractions is not None and D_tr:
        Cs = fc.draw_stratified_models(np.random.default_rng(a["rng_fraction_stratified_2022"]), fractions, B)
        sens["fraction_stratified_trained"] = estimate_block({"primary": D_tr["primary"]}, Cs, P)["primary"]
    res["sensitivity"] = sens
    return res


def subset_analysis(D, patients, subset, pins, B, n_models, zeroshot: bool, holm_names) -> Dict[str, Any]:
    cols = np.array([i for i, p in enumerate(patients) if p in set(subset)], dtype=int)
    rng = np.random.default_rng(pins["analysis"]["rng"]["2021"])
    C, P = fc.draw_indices(rng, n_models, len(cols), B)
    if zeroshot:
        C = np.zeros_like(C)
    est = estimate_block({"primary": D["primary"]}, C, P, cols)["primary"]
    pr = est
    h = fc.holm(pr["pooled"]["p"], pr["ap"]["p"], pr["pooled"]["point"], pr["ap"]["point"],
                pins["analysis"]["alpha"], holm_names)
    return {"patients": [patients[i] for i in cols], "n": len(cols), "primary": est, "holm": h}


def pooled(est22, est21, kind: str) -> Dict[str, Any]:
    out = {}
    for m in fc.METRICS:
        a, b = est22["primary"][m], est21["primary"][m]
        n1, n2 = a["n_defined"], b["n_defined"]
        T = (n1 * a["_T"] + n2 * b["_T"]) / (n1 + n2)
        pt = (n1 * a["point"] + n2 * b["point"]) / (n1 + n2)
        out[m] = {"patient_weighted": {"point": pt, "ci95": fc.ci(T, 0.95), "n": [n1, n2]}}
        if a["sd"] > 0 and b["sd"] > 0:
            w1, w2 = 1 / a["sd"] ** 2, 1 / b["sd"] ** 2
            Ti = (w1 * a["_T"] + w2 * b["_T"]) / (w1 + w2)
            out[m]["inverse_variance"] = {"point": (w1 * a["point"] + w2 * b["point"]) / (w1 + w2),
                                          "ci95": fc.ci(Ti, 0.95)}
        else:   # e.g. w = 0 selected for every model: Δ ≡ 0, bootstrap SD 0
            out[m]["inverse_variance"] = {"point": None, "ci95": None,
                                          "note": "undefined: a cohort's bootstrap SD is 0"}
    return {"model_type": kind, "metrics": out, "note": "descriptive only (§4.2)"}


# --------------------------------------------------------------------------- report
def fmt(s: Dict[str, Any], lvl: str = "ci95") -> str:
    lo, hi = s[lvl]
    return f"{s['point']:+.4f} [{lo:+.4f}, {hi:+.4f}] p={s['p']:.4f} n={s['n_defined']}"


def render_report(res: Dict[str, Any]) -> str:
    L = [f"# TREC-CT MeSH fusion confirmation: report", ""]
    if not res.get("confirmatory", True):
        L += ["**NON-CONFIRMATORY (dev-local smoke). These numbers are not study results.**", ""]
    L += [f"**Headline:** {res['headline']}", ""]
    for w in res.get("warnings", []):
        L.append(f"**WARNING:** {w}")
    L += ["", f"Pins: `{res['pins']['path']}` sha256 `{res['pins']['sha256']}`", ""]
    if res.get("disclosures"):
        L += ["## Prior-exposure disclosures (§9)", ""] + [f"- {d}" for d in res["disclosures"]] + [""]
    L += ["## Final labels (trained)", ""]
    for h in ("H1", "H2"):
        if h in res.get("labels", {}):
            L.append(f"- {h}: {res['labels'][h]['label']}")
    order = [("H1", "trained", "pooled"), ("H2", "trained", "ap"), ("H1z", "zeroshot", "pooled"),
             ("H2z", "zeroshot", "ap")]
    L += ["", "## Co-primary and zero-shot (Δ fused − text, two-level bootstrap)", "",
          "| hyp | cohort | Δ [95% CI] | companion interval | Holm | positive | harm |", "|---|---|---|---|---|---|---|"]
    for h, kind, m in order:
        for year in ("2022", "2021"):
            c = res["cohorts"].get(year, {})
            if kind not in c:
                continue
            dec = c[f"{kind}_decisions"]["holm"][h]
            L.append(f"| {h} | {year} | {fmt(c[kind]['primary'][m])} | {dec['companion_level']:.1%} "
                     f"[{dec['companion_interval'][0]:+.4f}, {dec['companion_interval'][1]:+.4f}] | "
                     f"{'rejected' if dec['rejected'] else 'not rejected'} | {dec['positive']} | {dec['harm']} |")
    L += ["", "## Secondary S1-S4 (uncorrected 95% CIs; S4 90% CI vs ±0.01)", "",
          "| metric | cohort | model | Δ [95% CI] | 90% CI | state |", "|---|---|---|---|---|---|"]
    for m in ("ndcg10", "p10", "g1g0", "hard"):
        for year in ("2022", "2021"):
            for kind in ("trained", "zeroshot"):
                c = res["cohorts"].get(year, {})
                if kind not in c:
                    continue
                s = c[kind]["primary"][m]
                st = c[f"{kind}_decisions"]["s4_state"] if m == "hard" else ""
                L.append(f"| {fc.METRIC_LABEL[m]} | {year} | {kind} | {fmt(s)} | [{s['ci90'][0]:+.4f}, {s['ci90'][1]:+.4f}] | {st} |")
    L += ["", "## Controls C1-C3 (pooled AUC and AP)", "",
          "| control | cohort | model | pooled Δ | AP Δ | C2 state |", "|---|---|---|---|---|---|"]
    for cond in ("C1_exact", "C2_shuffled", "hier_minus_C2", "C3i_hier_minus_exact", "C3ii_hier_minus_exact_w02"):
        for year in ("2022", "2021"):
            for kind in ("trained", "zeroshot"):
                c = res["cohorts"].get(year, {})
                if kind not in c:
                    continue
                st = c[f"{kind}_decisions"]["c2_state"] if cond == "C2_shuffled" else ""
                L.append(f"| {cond} | {year} | {kind} | {fmt(c[kind][cond]['pooled'])} | {fmt(c[kind][cond]['ap'])} | {st} |")
    L += ["", "C3 is labelled suggestive whatever the outcome. Zero-shot controls are descriptive.", "",
          "## Sensitivities", ""]
    for year in ("2022", "2021"):
        c = res["cohorts"].get(year, {})
        for kind in ("trained", "zeroshot"):
            if kind in c:
                s = c[kind]["sens_within_patient_z"]
                L.append(f"- {year} {kind} within-patient z: pooled {fmt(s['pooled'])}; AP {fmt(s['ap'])}")
        for name, s in c.get("sensitivity", {}).items():
            pr = s.get("primary", s)
            L.append(f"- {year} {name}: pooled {fmt(pr['pooled'])}; AP {fmt(pr['ap'])}")
    for name, s in res.get("sensitivity_2021", {}).items():
        for kind, v in s.items():
            L.append(f"- 2021 {name} ({kind}, n={v['n']}): pooled {fmt(v['primary']['pooled'])}; AP {fmt(v['primary']['ap'])}"
                     + (f"; Holm {v['holm']['order']}" if "holm" in v else ""))
    L += ["", "## Pooled (descriptive)", ""]
    for kind, p in res.get("pooled", {}).items():
        for m in ("pooled", "ap"):
            pw = p["metrics"][m]["patient_weighted"]
            L.append(f"- {kind} {m}: {pw['point']:+.4f} [{pw['ci95'][0]:+.4f}, {pw['ci95'][1]:+.4f}] (n={pw['n']})")
    L += ["", "## p / CI disagreements", ""]
    dis = [d for c in res["cohorts"].values() for k in ("trained_decisions", "zeroshot_decisions")
           for d in c.get(k, {}).get("p_ci_disagreements", [])]
    L += [f"- {d}" for d in dis] or ["- none"]
    L += ["", "## Exclusions", ""] + ([f"- {e}" for e in res.get("exclusions", [])] or ["- none"])
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    fc.require_hashseed()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pins", required=True, type=Path)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--device", choices=["cpu"], default="cpu")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dev-local", action="store_true")
    ap.add_argument("--B", type=int, default=None, help="--dev-local only")
    ap.add_argument("--seeds", default=None, help="--dev-local only: comma-separated 2021 seeds")
    args = ap.parse_args(argv)
    fc.set_root(args.root)
    fc.require_cpu_scoring(args.device)
    out, report = fc.refuse_existing(args.out), fc.refuse_existing(args.report)
    pins = fc.load_verified_pins(args.pins)
    if bool(pins.get("confirmatory")) == bool(args.dev_local) or (
            (args.B is not None or args.seeds is not None) and not args.dev_local):
        raise fc.ProtocolError("dev pins / --B only with --dev-local")
    if args.dev_local:
        fc.require_dev_local(out)
        fc.require_dev_local(report)
    bad = fc.verify_pinned_hashes(pins, ("inputs", "code", "configs"))
    if bad:
        raise fc.ProtocolError(f"pinned hash mismatch, refusing to analyse: {bad[:10]}")
    a = pins["analysis"]
    B = args.B or a["B"]
    grid = a["weight_grid"]
    scores = fc.absp(args.scores)
    sm = json.loads((scores / "score_manifest.json").read_text(encoding="utf-8"))
    pins_ref = {"path": fc.rel(fc.absp(args.pins)), "sha256": fc.file_sha256(args.pins)}
    if sm.get("pins") != pins_ref:
        raise fc.ProtocolError(f"scores were produced under pins {sm.get('pins')}, not --pins {pins_ref}")

    res: Dict[str, Any] = {"schema": "trials_fusion_confirm/analysis/v1", "created_utc": fc.utc_now(),
                           "confirmatory": pins["confirmatory"],
                           "pins": pins_ref,
                           "disclosures": list(fc.DEV_SMOKE_DISCLOSURES),
                           "scores_manifest_sha256": fc.file_sha256(scores / "score_manifest.json"),
                           "scores_pins": sm["pins"],
                           "cohorts": {}, "exclusions": [f"run excluded: {r}" for r in sm.get("excluded_runs", [])],
                           "run_env": fc.run_env_record()}
    est_types: Dict[str, Dict[str, Any]] = {}
    if (scores / "2022" / "all").exists():
        c22 = assemble_2022(scores, grid)
        if len(c22["trained"]) < a["min_checkpoints_2022"] and not args.dev_local:
            raise fc.ProtocolError(f"cohort 2022 has {len(c22['trained'])} < {a['min_checkpoints_2022']} checkpoints; stop and amend")
        res["cohorts"]["2022"] = cohort_analysis(
            "2022", D_2022(c22, c22["trained"]), D_2022(c22, ["zeroshot"]), c22["patients"], c22["patients"],
            {"trained": c22["trained"], "weights": c22["weights"], "shuffle_draws": c22["shuffle_draws"]},
            pins, B, c22["condition_signal"], fractions=c22["fractions"])
    if (scores / "2021").exists():
        cv = pins["cohort2021"]
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cv["seeds"]
        c21 = assemble_2021(scores, grid, seeds, cv["n_folds"])
        incomplete = [k for k in c21["folds_present"] if k not in c21["complete_folds"]]
        res["exclusions"] += [f"2021 fold {k} incomplete: its patients dropped from trained estimates, kept in zero-shot"
                              for k in incomplete]
        if len(c21["complete_folds"]) < a["min_complete_folds_2021"] and not args.dev_local:
            raise fc.ProtocolError(f"only {len(c21['complete_folds'])} complete 2021 folds; stop and amend")
        res["cohorts"]["2021"] = cohort_analysis(
            "2021", c21["D_trained"], c21["D_zeroshot"], c21["trained_patients"], c21["zeroshot_patients"],
            {"trained": c21["trained"], "complete_folds": c21["complete_folds"], "weights": c21["weights"],
             "shuffle_draws": c21["shuffle_draws"]}, pins, B, c21["condition_signal"])
        closed, exposed = set(cv["exposed_topics_closed"]), set(cv["exposed_topics"])
        partners = closed - exposed
        dev_smoke = set(cv.get("dev_smoke_exposed_topics", fc.DEV_SMOKE_EXPOSED_2021))
        sens21: Dict[str, Dict[str, Any]] = {}
        for name, pick in (("unexposed_closed", lambda p: p not in closed),
                           # §9 A0, descriptive: the 52 minus the 13 the Mac dev smoke scored
                           ("unexposed_closed_minus_dev_smoke", lambda p: p not in closed and p not in dev_smoke),
                           ("unexposed60", lambda p: p not in exposed),
                           ("exposed15", lambda p: p in exposed), ("partner8", lambda p: p in partners)):
            sens21[name] = {}
            for kind, D, pats, n_m, names in (("trained", c21["D_trained"], c21["trained_patients"], len(seeds), ("H1", "H2")),
                                              ("zeroshot", c21["D_zeroshot"], c21["zeroshot_patients"], 1, ("H1z", "H2z"))):
                sub = [p for p in pats if pick(p)]
                if D and sub:
                    sens21[name][kind] = subset_analysis(D, pats, sub, pins, B, n_m, kind == "zeroshot", names)
                    if name != "unexposed_closed":
                        sens21[name][kind].pop("holm")
        res["sensitivity_2021"] = sens21

    # final labels, headline, warnings (trained); zero-shot reported without labels
    cs = res["cohorts"]
    if "2022" in cs and "2021" in cs and "trained" in cs["2022"] and "trained" in cs["2021"]:
        d22, d21 = cs["2022"]["trained_decisions"], cs["2021"]["trained_decisions"]
        labels = {h: fc.final_label(d22["holm"][h]["positive"], d22["holm"][h]["harm"],
                                    d21["holm"][h]["positive"], d21["holm"][h]["harm"],
                                    d22["c2_state"], d21["c2_state"]) for h in ("H1", "H2")}
        res["labels"] = labels
        res["headline"] = fc.headline(labels)
        warnings = []
        uc = res.get("sensitivity_2021", {}).get("unexposed_closed", {})
        for h, m in HYP.items():
            if labels[h]["label_id"] in (4, 5) and "trained" in uc and uc["trained"]["primary"][m]["point"] <= 0:
                warnings.append(f"{h}: replication driven by previously exposed patients or their paraphrases")
        for h, m in (("H1z", "pooled"), ("H2z", "ap")):
            if cs["2021"]["zeroshot_decisions"]["holm"][h]["positive"] and "zeroshot" in uc \
                    and uc["zeroshot"]["primary"][m]["point"] <= 0:
                warnings.append(f"{h}: replication driven by previously exposed patients or their paraphrases")
        res["warnings"] = warnings
        res["pooled"] = {"trained": pooled(cs["2022"]["trained"], cs["2021"]["trained"], "trained"),
                         "zeroshot": pooled(cs["2022"]["zeroshot"], cs["2021"]["zeroshot"], "zeroshot")}
    else:
        res["headline"] = "incomplete analysis (one cohort missing): no final label"
        res["warnings"] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc.json_safe(strip(res)), indent=2) + "\n", encoding="utf-8")
    report.write_text(render_report(strip(res)), encoding="utf-8")
    print(res["headline"])
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
