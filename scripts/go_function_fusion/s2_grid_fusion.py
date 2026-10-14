"""S2: descriptive fusion add-on to the go_function grid (protocol §11).

For every GF run (V0 / V4 / V3 / V3c ..., any rule) with a ``scores.npz`` under
``--runs`` and each pack target t with stored columns:

  AP-T_S2(p,t)  = z(l_t) + w_t z(mean Pa(t)),  w_t = c* tau* / (tau* + k)
                  (k = the condition's retained count; ``all`` -> TP train count)
  parent-only   = z(l_t) + z(mean Pa(t))
  F             = go_function.fusion.fusion_scores (GF arm F, recomputed, U6)
  AP-T_S2 + R   (BP, descriptive): R block = X(t) ∩ cols(Reg(t)) with Reg(t) from the
                  S1 confirmatory builder; Reg columns not stored in X(t) are skipped
                  and counted (the stored X_reg channel is not used)

Parameters: S1-frozen Tier A released-MLP AP-T (finalists_tierA.json, per aspect);
remote parameters (finalists_remote.json) only under the §11.2 snapshot rule
(``go_function_fusion.stats.s2_remote_eligible``); placeholder (c, tau) = (2, 30),
flagged, if G-F has not run (dry-run only).

--smoke (Mac, G-S2dry): results/go_function/smoke/runs. Reads no test label: (a) F
parity is checked on the label-free columns of fusion_F.csv (n_pa, n_sib,
n_sib_candidates, pa_dropped, sib_dropped; auc_V0 against each run's eval.json) and
at score level against an independent ``go_function.fusion.fuse`` call (1e-9); the
auc_F to 1e-9 check needs the test labels and runs only with --with-labels (remote);
(b) AP-T_S2 and parent-only computed for every V0 run; (c) output schema complete.
Outputs: <out>/ (default results/go_function_fusion/s2_dryrun), each labelled
"smoke, 2 epochs, non-informative".

Full S2 (remote, after GF G10): --runs results/go_function/runs --with-labels
--out results/go_function_fusion/s2 ; reads per k (two-way pack x seed bootstrap,
B = 10,000, default_rng(20261011), go_function.vendored.two_way_bootstrap).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import make_pins as MP  # noqa: E402
from go_function_fusion import data as D  # noqa: E402
from go_function_fusion import fuse as FU  # noqa: E402
from go_function_fusion import priors as PR  # noqa: E402
from go_function_fusion import stats as ST  # noqa: E402

SMOKE_RUNS = "results/go_function/smoke/runs"
SMOKE_F = "results/go_function/smoke/analysis/fusion_F.csv"
SMOKE_LABEL = "smoke, 2 epochs, non-informative"
OUT_CHOICES = ("results/go_function_fusion/s2_dryrun", "results/go_function_fusion_dev/s2_dryrun",
               "results/go_function_fusion/s2")
F_INFO_COLS = ("n_pa", "n_sib", "n_sib_candidates", "pa_dropped", "sib_dropped")
SCHEMA_COLS = ("aspect", "feature", "variant", "regime", "rule", "draw", "pack", "target", "k",
               "seed", "source_run", "w_t", "n_pa", "n_reg_used", "n_reg_skipped",
               "auc_base", "auc_APT_S2", "auc_parent_only", "auc_F", "auc_APT_S2_R", "label")


# ------------------------------------------------------------------ inputs
def discover(runs_dir: Path) -> List[Path]:
    return sorted(p.parent for p in runs_dir.rglob("scores.npz")
                  if not any(s in part for part in p.parent.parts
                             for s in (".attempt", ".interrupted", ".refused")))


def load_run(d: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    rj = json.loads((d / "run.json").read_text())
    ev = json.loads((d / "eval.json").read_text())
    with np.load(d / "scores.npz", allow_pickle=False) as z:
        sc = {k: z[k] for k in z.files}
    return rj, ev, sc


def condition_counts(spec: Dict[str, Any]) -> np.ndarray:
    from go_function.pins import load_csr
    base = f"{D.PRE}/{spec['aspect']}/conditions/{spec['regime']}"
    rel = f"{base}/all" if spec["pack"] is None else (
        f"{base}/{spec['rule']}/{spec['pack']}_k{spec['k']}"
        + ("_d2" if spec.get("retention_draw", 1) == 2 else ""))
    with np.load(ROOT / rel / "train_labels.npz", allow_pickle=False) as z:
        y = load_csr(z, "train")
    return np.bincount(y.indices, minlength=y.shape[1])


def parameters(fusion_results: Path, aspect: str, runs_dir: Path) -> Dict[str, Any]:
    """S1-frozen S2 parameters with the §11.2 eligibility record."""
    study = D.fusion_study()
    fin_p = fusion_results / "finalists_tierA.json"
    out: Dict[str, Any] = {"tierB_in_s2": False, "wH_in_s2": False}
    if fin_p.exists():
        s2 = json.loads(fin_p.read_text())["s2"][aspect]
        out.update(c=float(s2["c"]), tau=float(s2["tau"]), source="finalists_tierA.json", placeholder=False)
    else:
        ph = study["s2"]["placeholder_params"]
        out.update(c=float(ph["c"]), tau=float(ph["tau"]), source="placeholder (G-F not run)",
                   placeholder=True)
    pins = MP.load_pins(fusion_results)["pins"].get("G-F2")
    snap = pins.get("gf_grid_snapshot") if pins else None
    ok, why = ST.s2_remote_eligible(snap, runs_dir, snap.get("epoch_seconds") if snap else None)
    out["remote_eligible"] = ok
    out["remote_ineligible_reason"] = why
    if ok:
        rem = json.loads((fusion_results / "finalists_remote.json").read_text())
        mlp = rem["tierA"].get(f"{aspect}/mlp")
        if mlp is not None:
            out["w_H"] = float(mlp["w_H"])
            out["wH_in_s2"] = True
        out["tierB_in_s2"] = bool(rem["tierB"])
    return out


# ------------------------------------------------------------------ per run
def s2_run(d: Path, table: Dict[str, Any], vocab_index: Dict[str, int], root_id: str,
           reg_sets: List[List[int]], params: Dict[str, Any], labels: Optional[Any]
           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from go_function.fusion import fuse as gf_fuse, fusion_scores
    rj, ev, sc = load_run(d)
    spec = rj["spec"]
    pos = {int(c): i for i, c in enumerate(np.asarray(sc["columns"]).tolist())}
    L = np.asarray(sc["logits"], dtype=np.float64)
    counts = condition_counts(spec)
    targets = sorted(ev.get("targets") or {})
    tg = []
    for t in targets:
        info = table["targets"][t]
        pa = [vocab_index[g] for g in info["channels"]["Pa"] if g != root_id and g in vocab_index]
        sib = [vocab_index[g] for g in info["channels"]["siblings"] if g in vocab_index]
        tg.append((t, int(info["column"]), pa, sib))
    F = fusion_scores(sc, tg, counts) if spec["variant"] == "V0" else {}
    rows, parity = [], []
    for t, col, pa, sib in tg:
        k = counts[col] if spec["k"] == "all" or spec["pack"] is None else int(spec["k"])
        w = float(FU.adaptive_w(np.asarray([k]), params["c"], params["tau"])[0])
        own = FU.z_cols(L[:, pos[col]])[:, 0]
        paz = FU.z_cols(L[:, [pos[c] for c in pa]].mean(axis=1))[:, 0] if pa else np.zeros(len(own))
        apt = own + w * paz
        par = own + paz
        xcols = {vocab_index[g] for g in table["targets"][t]["channels"]["X"] if g in vocab_index}
        reg = [c for c in reg_sets[col] if c in xcols and c in pos]
        skipped = len(reg_sets[col]) - len(reg)
        regz = FU.z_cols(L[:, [pos[c] for c in reg]].mean(axis=1))[:, 0] if reg else np.zeros(len(own))
        aptr = apt + regz if reg else apt
        row = {"aspect": spec["aspect"], "feature": spec["feature"], "variant": spec["variant"],
               "regime": spec["regime"], "rule": spec["rule"], "draw": spec.get("retention_draw", 1),
               "pack": spec["pack"] or "all", "target": t, "k": spec["k"], "seed": spec["seed"],
               "source_run": MP.rel(d), "w_t": w, "n_pa": len(pa), "n_reg_used": len(reg),
               "n_reg_skipped": int(skipped), "auc_base": ev["targets"][t].get("auc"),
               "auc_APT_S2": None, "auc_parent_only": None, "auc_F": None, "auc_APT_S2_R": None,
               "label": params.get("label")}
        if t in F:
            s_f, info = F[t]
            ref, _ = gf_fuse(L[:, pos[col]], L[:, [pos[c] for c in pa]] if pa else None,
                             L[:, [pos[c] for c in sib if counts[c] >= 10]]
                             if any(counts[c] >= 10 for c in sib) else None)
            parity.append({"run": MP.rel(d), "target": t, "info": {**info, "n_sib_candidates": len(sib)},
                           "score_maxdiff_vs_fuse": float(np.max(np.abs(s_f - ref))),
                           "auc_V0_eval": ev["targets"][t].get("auc")})
        if labels is not None:
            from go_function.evaluate import class_metrics
            y = labels(spec, col, np.asarray(sc["eval_rows"]))
            row["auc_APT_S2"] = class_metrics(apt, y)["auc"]
            row["auc_parent_only"] = class_metrics(par, y)["auc"]
            row["auc_APT_S2_R"] = class_metrics(aptr, y)["auc"] if spec["aspect"] == "bp" else None
            if t in F:
                row["auc_F"] = class_metrics(F[t][0], y)["auc"]
                parity[-1]["auc_F"] = row["auc_F"]
        rows.append(row)
    return rows, parity


def check_f_parity(parity: List[Dict[str, Any]], f_csv: Path) -> Dict[str, Any]:
    """Label-free parity with fusion_F.csv (and auc_F when computed)."""
    ref = {(r["source_run"], r["target"]): r for r in csv.DictReader(io.StringIO(f_csv.read_text()))}
    misses, n = [], 0
    for p in parity:
        r = ref.get((p["run"], p["target"]))
        if r is None:
            misses.append({"run": p["run"], "target": p["target"], "why": "absent from fusion_F.csv"})
            continue
        n += 1
        for c in F_INFO_COLS:
            ours = p["info"][c]
            want = (r[c] == "True") if c.endswith("dropped") else int(r[c])
            if ours != want:
                misses.append({"run": p["run"], "target": p["target"], "field": c, "ours": ours, "ref": want})
        if abs(float(r["auc_V0"]) - float(p["auc_V0_eval"])) > 1e-9:
            misses.append({"run": p["run"], "target": p["target"], "field": "auc_V0"})
        if p["score_maxdiff_vs_fuse"] > 1e-9:
            misses.append({"run": p["run"], "target": p["target"], "field": "score"})
        if "auc_F" in p and abs(float(r["auc_F"]) - p["auc_F"]) > 1e-9:
            misses.append({"run": p["run"], "target": p["target"], "field": "auc_F"})
    return {"n_compared": n, "n_reference_rows": len(ref), "misses": misses,
            "passed": not misses and n == len(ref),
            "auc_F_checked": any("auc_F" in p for p in parity)}


def reads(rows: List[Dict[str, Any]], table: Dict[str, Any], B: int) -> Dict[str, Any]:
    """Δ(AP-T(V) − V) per (feature, rule, k) and AP-T(V0) − V4 at k = 0, two-way bootstrap."""
    from go_function import stats as GS
    from go_function import vendored as V
    from go_function.constants import SEEDS, UNSTABLE_SHARE
    units = GS.units_from_table(table)
    idx = {(r["feature"], r["variant"], r["rule"], r["pack"], str(r["k"]), int(r["seed"]), r["target"]): r
           for r in rows if r["auc_APT_S2"] is not None}

    def mat(fn):
        X = np.full((len(units.targets), len(SEEDS)), np.nan)
        for u, t in enumerate(units.targets):
            for j, s in enumerate(SEEDS):
                v = fn(t, units.pack_of[t], s)
                if v is not None:
                    X[u, j] = v
        return X

    def boot(X):
        if np.isnan(X).all():
            return None
        return V.two_way_bootstrap(X, units.unit_pack, len(units.packs), B, seed=ST.S2_SEED,
                                   unstable_share=UNSTABLE_SHARE)

    out: Dict[str, Any] = {"label": "descriptive"}
    feats = sorted({r["feature"] for r in rows})
    for f in feats:
        for rule in ("W2", "W1"):
            for k in (0, 1, 5, 10, 25):
                for var in ("V0", "V4"):
                    def g(t, p, s, var=var, key="auc_APT_S2"):
                        r = idx.get((f, var, rule, p, str(k), s, t))
                        return None if r is None or r["auc_base"] is None else r[key] - r["auc_base"]
                    res = boot(mat(g))
                    if res is not None:
                        out[f"{f}/{rule}/k{k}/APT({var})-{var}"] = GS.classify_descriptive(res)

                def h(t, p, s):
                    a = idx.get((f, "V0", rule, p, str(k), s, t))
                    b = idx.get((f, "V4", rule, p, str(k), s, t))
                    return None if a is None or b is None or b["auc_base"] is None else \
                        a["auc_APT_S2"] - b["auc_base"]
                if k == 0:
                    res = boot(mat(h))
                    if res is not None:
                        out[f"{f}/{rule}/k0/APT(V0)-V4"] = GS.classify_descriptive(res)
    return out


def csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(SCHEMA_COLS), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("" if r.get(c) is None else r[c]) for c in SCHEMA_COLS})
    return buf.getvalue().encode()


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None, choices=OUT_CHOICES)
    ap.add_argument("--fusion-results", default=MP.RESULTS_CHOICES[0], choices=MP.RESULTS_CHOICES)
    ap.add_argument("--with-labels", action="store_true", help="read GF test labels (remote only)")
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(argv)
    if a.with_labels:
        MP.refuse_on_mac("s2_grid_fusion.py --with-labels (reads test labels)")
    runs_dir = ROOT / (a.runs or (SMOKE_RUNS if a.smoke else "results/go_function/runs"))
    out = ROOT / (a.out or (OUT_CHOICES[0] if a.smoke else OUT_CHOICES[2]))
    fres = MP.results_dir(a.fusion_results)
    from go_function.constants import ROOTS
    from go_function.labels import read_terms
    run_dirs = discover(runs_dir)
    by_aspect: Dict[str, List[Path]] = {}
    for d in run_dirs:
        by_aspect.setdefault(json.loads((d / "run.json").read_text())["spec"]["aspect"], []).append(d)
    all_rows, all_par, meta = [], [], {"label": SMOKE_LABEL if a.smoke else "descriptive",
                                       "runs_dir": MP.rel(runs_dir), "aspects": {}}
    for asp, dirs in sorted(by_aspect.items()):
        ctx = D.aspect_context(asp, ("train", "valid"))
        terms = read_terms(ROOT / D.DGZ / asp / "terms.pkl")
        vidx = {g: i for i, g in enumerate(terms)}
        reg = PR.neighbour_tables(ctx.ont, ctx.vocab, ctx.anc, ctx.desc, ctx.ntr, ctx.root,
                                  "confirmatory", keys=("reg",))["reg"]
        table = json.loads((ROOT / D.PRE / asp / "targets/target_table.json").read_text())
        params = parameters(fres, asp, runs_dir)
        params["label"] = SMOKE_LABEL if a.smoke else (
            "placeholder parameters" if params["placeholder"] else "descriptive")
        labels = None
        if a.with_labels:
            from go_function.evaluate import load_eval_labels
            cache: Dict[str, Any] = {}

            def labels(spec, col, erows, cache=cache):
                er = "TPnoIPI" if spec["regime"] == "TPnoIPI" else "TP"
                if er not in cache:
                    cache[er] = load_eval_labels(er, spec["aspect"])
                ev = cache[er]
                if not np.array_equal(np.asarray(ev["rows"]), erows):
                    raise MP.Fatal("scores.npz eval_rows differ from the evaluation set")
                return ev["y"][:, col].toarray().ravel()
        for d in dirs:
            rows, par = s2_run(d, table, vidx, ROOTS[asp], reg, params, labels)
            all_rows += rows
            all_par += par
        meta["aspects"][asp] = {"params": params, "n_runs": len(dirs)}
        if a.with_labels and not a.smoke:
            meta["aspects"][asp]["reads"] = reads([r for r in all_rows if r["aspect"] == asp], table, a.B)
    meta["schema_complete"] = all(set(SCHEMA_COLS) <= set(r) for r in all_rows) and bool(all_rows)
    meta["n_rows"] = len(all_rows)
    meta["n_v0_rows_with_apt"] = sum(1 for r in all_rows if r["variant"] == "V0")
    meta["reg_skipped_pairs"] = int(sum(r["n_reg_skipped"] for r in all_rows))
    if a.smoke:
        meta["f_parity"] = check_f_parity(all_par, ROOT / SMOKE_F)
    MP.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s2_rows.csv").write_bytes(csv_bytes(all_rows))
    MP.write_json(out / "s2_summary.json", json.loads(json.dumps(meta, default=_np)))
    MP.write_json(out / "f_parity_detail.json", json.loads(json.dumps(all_par, default=_np)))
    print(json.dumps({k: meta[k] for k in ("n_rows", "schema_complete")}, default=_np))
    if a.smoke and not meta["f_parity"]["passed"]:
        print("F parity miss", json.dumps(meta["f_parity"]["misses"][:5], default=_np))
        return MP.EXIT_STOP
    return MP.EXIT_OK


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(MP.main_wrapper(run))
