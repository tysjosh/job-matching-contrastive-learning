"""Analysis of the GO-function grid (G11) and of the G9 smoke (design §10-12,
protocol §1, §7-9; plan item 16).

  .venv/bin/python scripts/go_function/analyze.py --runs results/go_function/runs \
      --out results/go_function/analysis --seed 20261011 --B 10000
  .venv/bin/python scripts/go_function/analyze.py --runs results/go_function/smoke/runs \
      --out results/go_function/smoke/analysis --smoke

Reads every completed run under --runs (``DONE.json``; ``*.attempt<n>``,
``*.interrupted<n>`` and ``*.refused<n>`` dirs are ignored). A DONE.json or run.json whose hashes
disagree with the files is fatal (exit 2). Then:

- long_table.csv keyed by (aspect, feature, variant, regime, rule, draw, pack,
  target, k, seed) with class_auc, class_aupr, n_pos, fmax, smin, micro_aupr
  (micro_aupr = the collateral micro-AUPR, ``aupr_micro`` in eval.json) and
  class_auc_dgz_labels (the protocol §7 sensitivity read: the same logits scored
  against DGZ-rule test labels on the run's evaluation rows; model runs only). Pack
  ``all`` is the pack-free ``all`` condition. Arm F (weight-free fusion of V0 under
  W2 at every k and W1 at k in {0, 1, 5}) and arm DS (DiamondScore per condition,
  no seed) are rows of the same table;
- fusion_F.csv and baseline_DS.csv: the F and DS per-target details;
- contrasts.json: per aspect and feature the preregistered family (C1-C7, Holm,
  classification, S1 beside each), Vbest (D11), and the descriptive reads (per-k
  deltas, ladder steps, C6/C7 for every arm, the C3 decomposition with its 1e-12
  identity, the channel reads C3_leak_X/desc/anc, W1 - W2, F - V0, DS, InterPro,
  noIPI, re-draw, collateral harm, and C1-C5 / Delta^small vs V0 under DGZ-rule
  test labels, ``dgz_test_labels``);
- decisions.json: classifications, the exploratory flag (smoke, or > 10% of the
  main block's (t, k, s, arm) values missing) and the §9.3 reading of C3.

The esm2_650m family is confirmatory; the InterPro family is descriptive.

Exit codes: 0 ok; 2 fatal (hash disagreement, identity violation, bad inputs);
4 refused (no completed run, or the code pin is missing outside --smoke).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from go_function import stats as ST  # noqa: E402
from go_function.constants import (ANALYSIS_SEED, ASPECTS, EXIT_FATAL, EXIT_OK,  # noqa: E402
                                   EXIT_REFUSED, ROOTS)
from go_function.pins import assert_no_abs_path, assert_writable_target, canonical_json, sha256_bytes  # noqa: E402
from go_function.runtime import FatalError, ProtocolStop, repo_path, repo_rel, setup_logging  # noqa: E402

PRE = "preprocess/go_function"
DGZ = "dataset/go_function/dgz"
FEATURES = ("esm2_650m", "interpro")
CONFIRMATORY_FEATURE = "esm2_650m"
MISSING_LIMIT = 0.10
LONG_COLS = ("aspect", "feature", "variant", "regime", "rule", "draw", "pack", "target", "k",
             "seed", "class_auc", "class_aupr", "n_pos", "fmax", "smin", "micro_aupr",
             "class_auc_dgz_labels")
SKIP_PARTS = (".attempt", ".interrupted", ".refused")


class Refused(RuntimeError):
    pass


# ------------------------------------------------------------------ runs
class RunRec:
    def __init__(self, d: Path, run_json: Dict[str, Any], ev: Dict[str, Any]):
        self.dir = d
        self.run = run_json
        self.spec = run_json["spec"]
        self.eval = ev

    @property
    def pack(self) -> str:
        return self.spec["pack"] or "all"

    @property
    def k(self):
        return ST.norm_k(self.spec["k"])

    def scores(self) -> Dict[str, np.ndarray]:
        with np.load(self.dir / "scores.npz", allow_pickle=False) as z:
            return {n: z[n] for n in z.files}


def _rel(p: Path) -> str:
    try:
        return repo_rel(p)
    except ValueError:
        return p.name


def discover(runs_dir: Path) -> Tuple[List[RunRec], Dict[str, int]]:
    """Completed runs (hash-verified) and counts of failed/incomplete dirs."""
    done, counts = [], {"done": 0, "failed": 0, "incomplete": 0}
    marker_dirs = sorted({p.parent for name in ("DONE.json", "FAILED.json", "run.json",
                                                "epochs.jsonl")
                          for p in runs_dir.rglob(name)})
    for d in marker_dirs:
        if any(s in part for part in d.relative_to(runs_dir).parts for s in SKIP_PARTS):
            continue
        if not (d / "DONE.json").exists():
            counts["failed" if (d / "FAILED.json").exists() else "incomplete"] += 1
            continue
        db = (d / "DONE.json").read_bytes()
        rb = (d / "run.json").read_bytes() if (d / "run.json").exists() else None
        if rb is None:
            raise FatalError(f"{_rel(d)}: DONE.json without run.json")
        dj, rj = json.loads(db), json.loads(rb)
        if dj.get("run_json_sha256") != sha256_bytes(rb):
            raise FatalError(f"{_rel(d)}: DONE.json run_json_sha256 differs from run.json")
        for name, sha in rj.get("outputs", {}).items():
            got = sha256_bytes((d / name).read_bytes()) if (d / name).exists() else None
            if got != sha or dj.get(f"{name}_sha256") != sha:
                raise FatalError(f"{_rel(d)}: {name} hash disagrees between the file, "
                                 f"run.json and DONE.json")
        ev = json.loads((d / "eval.json").read_text())
        done.append(RunRec(d, rj, ev))
        counts["done"] += 1
    return done, counts


def long_rows(r: RunRec, dgz_auc: Optional[Dict[str, Optional[float]]] = None
              ) -> List[Dict[str, Any]]:
    s = r.spec
    col = r.eval.get("collateral") or {}
    out = []
    for t, m in sorted((r.eval.get("targets") or {}).items()):
        out.append({"aspect": s["aspect"], "feature": s["feature"], "variant": s["variant"],
                    "regime": s["regime"], "rule": s["rule"], "draw": s.get("retention_draw", 1),
                    "pack": r.pack, "target": t, "k": r.k, "seed": s["seed"],
                    "class_auc": m.get("auc"), "class_aupr": m.get("aupr"), "n_pos": m.get("n_pos"),
                    "fmax": col.get("fmax"), "smin": col.get("smin"),
                    "micro_aupr": col.get("aupr_micro"),
                    "class_auc_dgz_labels": (dgz_auc or {}).get(t)})
    return out


def dgz_label_aucs(r: RunRec, y_dgz_test, target_col: Dict[str, int]
                   ) -> Dict[str, Optional[float]]:
    """Sensitivity read (protocol §7, 'AUC with DGZ-rule test labels'): class AUC of
    each target's logit column against the DGZ-rule test labels, on the run's own
    evaluation rows (scores.npz ``eval_rows``). ``y_dgz_test`` is the full DGZ-rule
    test matrix (sparse, test rows x vocabulary; same rows and vocabulary as the TP
    file); ``target_col`` maps target -> vocabulary column. None for a column that
    is degenerate (no positive or no negative) on those rows."""
    from go_function.evaluate import class_metrics
    targets = r.eval.get("targets") or {}
    if not targets:
        return {}
    sc = r.scores()
    pos = {int(c): j for j, c in enumerate(np.asarray(sc["columns"]))}
    rows = np.asarray(sc["eval_rows"], dtype=np.int64)
    out: Dict[str, Optional[float]] = {}
    for t in sorted(targets):
        col = int(target_col[t])
        if col not in pos:
            raise FatalError(f"{_rel(r.dir)}: no scores.npz column for target {t}")
        y = np.asarray(y_dgz_test[:, col].toarray()).ravel()[rows]
        n_pos = int(y.sum())
        out[t] = None if n_pos in (0, len(y)) else class_metrics(sc["logits"][:, pos[col]], y)["auc"]
    return out


def condition_dir(s: Dict[str, Any]) -> str:
    base = f"{PRE}/{s['aspect']}/conditions/{s['regime']}"
    if s["pack"] is None:
        return f"{base}/all"
    return (f"{base}/{s['rule']}/{s['pack']}_k{s['k']}"
            + ("_d2" if s.get("retention_draw", 1) == 2 else ""))


# ------------------------------------------------------------------ per-aspect inputs
class AspectInputs:
    """Target table, vocabulary, evaluation labels and condition labels (cached)."""

    def __init__(self, aspect: str, log):
        self.aspect = aspect
        self.log = log
        tt = repo_path(f"{PRE}/{aspect}/targets/target_table.json")
        b = tt.read_bytes()
        self._check_pin("G6", f"{aspect}/targets/target_table.json", sha256_bytes(b))
        self.table = json.loads(b)
        self.units = ST.units_from_table(self.table)
        self._labels: Dict[str, Dict[str, Any]] = {}
        self._cond: Dict[str, Tuple[Any, np.ndarray]] = {}
        self._H: Dict[str, Any] = {}
        self._dgz = None

    def _check_pin(self, key: str, name: str, sha: str) -> None:
        from go_function.pins import PINS_PATH
        if not PINS_PATH.exists():
            self.log.warning("no protocol pins; %s not verified", name)
            return
        want = json.loads(PINS_PATH.read_text())["pins"].get(key, {}).get(name)
        want = want.get("sha256") if isinstance(want, dict) else want
        if want is not None and want != sha:
            raise FatalError(f"{name}: sha256 differs from pin {key}")

    def labels(self, regime: str) -> Dict[str, Any]:
        """Evaluation labels: TPnoIPI for the noIPI block, TP otherwise."""
        from go_function.evaluate import eval_label_source
        from go_function.labels import load_labels
        er = "TPnoIPI" if regime == "TPnoIPI" else "TP"
        if er not in self._labels:
            src = eval_label_source(er, self.aspect)
            lab = load_labels(repo_path(src.label_file))
            rows = np.asarray(lab[src.eval_set], dtype=np.int64)
            vocab = list(lab["vocab"])
            self._labels[er] = {"y": lab["test"][rows].tocsc(), "rows": rows,
                                "test_names": [lab["proteins_test"][i] for i in rows],
                                "train_names": list(lab["proteins_train"]), "vocab": vocab,
                                "index": {g: i for i, g in enumerate(vocab)}}
            del lab
        return self._labels[er]

    def dgz_test(self):
        """DGZ-rule test labels (CSC, every test row) for the protocol §7 sensitivity
        read; rows and vocabulary must equal the TP file's."""
        if self._dgz is None:
            from go_function.labels import load_labels
            tp = self.labels("TP")
            lab = load_labels(repo_path(f"{PRE}/{self.aspect}/labels_DGZ.npz"))
            if list(lab["vocab"]) != tp["vocab"] or \
                    [lab["proteins_test"][i] for i in tp["rows"]] != tp["test_names"]:
                raise FatalError(f"labels_DGZ ({self.aspect}): test rows or vocabulary differ "
                                 f"from labels_TP")
            self._dgz = lab["test"].tocsc()
            del lab
        return self._dgz

    def target_col_map(self) -> Dict[str, int]:
        return {t: int(v["column"]) for t, v in self.table["targets"].items()}

    def y_col(self, regime: str, col: int) -> np.ndarray:
        return self.labels(regime)["y"][:, col].toarray().ravel()

    def condition(self, rel: str):
        """(train labels CSR, mask pairs) of a condition dir."""
        if rel not in self._cond:
            from go_function.pins import load_csr
            d = repo_path(rel)
            with np.load(d / "train_labels.npz", allow_pickle=False) as z:
                y = load_csr(z, "train")
            with np.load(d / "train_mask.npz", allow_pickle=False) as z:
                pairs = np.asarray(z["pairs"]).reshape(-1, 2)
            self._cond = {rel: (y, pairs)}       # keep one condition in memory
        return self._cond[rel]

    def diamond(self, regime: str):
        from go_function.baselines import diamond_matrix, read_diamond
        from go_function.pins import verify_inputs
        er = "TPnoIPI" if regime == "TPnoIPI" else "TP"
        if er not in self._H:
            rel = f"{DGZ}/{self.aspect}/test_diamond.res"
            verify_inputs([rel])
            lab = self.labels(regime)
            self._H[er] = diamond_matrix(read_diamond(repo_path(rel)), lab["test_names"],
                                         lab["train_names"])
        return self._H[er]

    def target_cols(self, regime: str, t: str) -> Tuple[int, List[int], List[int]]:
        info = self.table["targets"][t]
        idx = self.labels(regime)["index"]
        root = ROOTS[self.aspect]
        pa = [idx[g] for g in info["channels"]["Pa"] if g != root and g in idx]
        sib = [idx[g] for g in info["channels"]["siblings"] if g in idx]
        return int(info["column"]), pa, sib


# ------------------------------------------------------------------ F and DS
def fusion_rows(runs: List[RunRec], inp: AspectInputs) -> Tuple[List[dict], List[dict]]:
    from go_function.evaluate import class_metrics
    from go_function.fusion import fusion_scores
    rows, details = [], []
    for r in runs:
        s = r.spec
        if s["variant"] != "V0" or not (s["rule"] == "W2" or (s["rule"] == "W1" and r.k in (0, 1, 5))):
            continue
        sc = r.scores()
        lab = inp.labels(s["regime"])
        if not np.array_equal(np.asarray(sc["eval_rows"]), lab["rows"]):
            raise FatalError(f"{_rel(r.dir)}: scores.npz eval_rows differ from the evaluation set")
        y_tr, _ = inp.condition(condition_dir(s))
        counts = np.bincount(y_tr.indices, minlength=y_tr.shape[1])
        tgts = [(t, *inp.target_cols(s["regime"], t)) for t in sorted(r.eval.get("targets", {}))]
        fused = fusion_scores(sc, tgts, counts)
        for t, (score, info) in fused.items():
            m = class_metrics(score, inp.y_col(s["regime"], int(inp.table["targets"][t]["column"])))
            base = {"aspect": s["aspect"], "feature": s["feature"], "variant": "F",
                    "regime": s["regime"], "rule": s["rule"], "draw": s.get("retention_draw", 1),
                    "pack": r.pack, "target": t, "k": r.k, "seed": s["seed"]}
            rows.append({**base, "class_auc": m["auc"], "class_aupr": m["aupr"],
                         "n_pos": m["n_pos"], "fmax": None, "smin": None, "micro_aupr": None})
            details.append({**base, "source_run": _rel(r.dir), "auc_F": m["auc"],
                            "auc_V0": r.eval["targets"][t]["auc"], **info})
    return rows, details


def ds_rows(runs: List[RunRec], inp: AspectInputs) -> List[dict]:
    from go_function.baselines import diamond_score
    from go_function.evaluate import class_metrics
    conds: Dict[tuple, Dict[str, Any]] = {}
    for r in runs:
        s = r.spec
        key = (s["feature"], s["regime"], s["rule"], s.get("retention_draw", 1), r.pack, r.k)
        e = conds.setdefault(key, {"spec": s, "targets": set()})
        e["targets"] |= set(r.eval.get("targets", {}))
    out = []
    for key in sorted(conds, key=str):
        feature, regime, rule, draw, pack, k = key
        e = conds[key]
        y_tr, mask = inp.condition(condition_dir(e["spec"]))
        lab = inp.labels(regime)
        if y_tr.shape[0] != len(lab["train_names"]):
            raise FatalError(f"{condition_dir(e['spec'])}: train rows differ from the label file")
        tg = sorted(e["targets"])
        cols = [int(inp.table["targets"][t]["column"]) for t in tg]
        S = diamond_score(inp.diamond(regime), y_tr, cols, mask)
        for j, t in enumerate(tg):
            m = class_metrics(S[:, j], inp.y_col(regime, cols[j]))
            out.append({"aspect": inp.aspect, "feature": feature, "variant": "DS",
                        "regime": regime, "rule": rule, "draw": draw, "pack": pack,
                        "target": t, "k": k, "seed": None, "class_auc": m["auc"],
                        "class_aupr": m["aupr"], "n_pos": m["n_pos"], "fmax": None,
                        "smin": None, "micro_aupr": None,
                        "condition": condition_dir(e["spec"])})
    return out


def channel_cells(runs: List[RunRec], inp: AspectInputs) -> Dict[str, Dict[tuple, dict]]:
    """Per feature: {(t, seed): {X, Desc, Pa}} from V4 WDGZrel/W2drop k = 0 pairs."""
    by = {}
    for r in runs:
        s = r.spec
        if s["variant"] == "V4" and s["regime"] == "DGZ" and r.k == 0 \
                and s["rule"] in ("WDGZrel", "W2drop"):
            by[(s["feature"], s["rule"], r.pack, s["seed"])] = r
    out: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for (feature, rule, pack, seed), a in sorted(by.items(), key=str):
        if rule != "WDGZrel" or (feature, "W2drop", pack, seed) not in by:
            continue
        b = by[(feature, "W2drop", pack, seed)]
        ts = sorted(set(a.eval.get("targets", {})) & set(b.eval.get("targets", {})))
        y = {t: inp.y_col("TP", int(inp.table["targets"][t]["column"])) for t in ts}
        reads = ST.channel_reads(a.scores(), b.scores(), inp.units.channels, y)
        for t, v in reads.items():
            out[feature][(t, seed)] = v
    return out


# ------------------------------------------------------------------ contrasts
def dgz_label_reads(at_dgz: ST.AucTable, units: ST.Units, feature: str, B: int, seed: int
                    ) -> Dict[str, Any]:
    """Descriptive sensitivity read (protocol §7): C1-C5 and every arm's
    Delta^small vs V0, re-scored against DGZ-rule test labels (two-way bootstrap,
    no Holm, never classified)."""
    def arm(v, regime="TP", rule="W2"):
        return (feature, v, regime, rule, 1)

    def dk(A, Bm, k):
        return ST.v_diff(ST.v_auc(at_dgz, A, k), ST.v_auc(at_dgz, Bm, k))

    def dsmall(A, Bm):
        return ST.v_small(lambda k: dk(A, Bm, k))

    def est(fn):
        return ST.estimate(ST.matrix(units, fn), units, B, seed)

    out = {"C1": est(dsmall(arm("V3"), arm("V3c"))), "C2": est(dk(arm("V4"), arm("V0"), 0)),
           "C3": est(ST.v_diff(ST.v_auc(at_dgz, arm("V4", "DGZ", "WDGZrel"), 0),
                               ST.v_auc(at_dgz, arm("V4"), 0))),
           "C4": est(dsmall(arm("V5"), arm("V0"))), "C5": est(dsmall(arm("V6"), arm("V5")))}
    out.update({f"{v}-V0_small": est(dsmall(arm(v), arm("V0"))) for v in ST.ARMS_BEST})
    out["n_values"] = sum(1 for key in at_dgz.v if key[0] == feature)
    return out


def analyse_feature(at: ST.AucTable, units: ST.Units, aspect: str, feature: str, B: int,
                    seed: int, channels: Dict[tuple, dict], collateral: Dict[tuple, dict],
                    at_dgz: Optional[ST.AucTable] = None) -> Dict[str, Any]:
    def arm(v, regime="TP", rule="W2", draw=1):
        return (feature, v, regime, rule, draw)

    def M(fn, **kw):
        return ST.matrix(units, fn, **kw)

    def est(X, strata=None, s1=False):
        return ST.estimate(X, units, B, seed, strata, s1)

    def dk(A, Bm, k):
        return ST.v_diff(ST.v_auc(at, A, k), ST.v_auc(at, Bm, k))

    def dsmall(A, Bm):
        return ST.v_small(lambda k: dk(A, Bm, k))

    fam: Dict[str, Any] = {}
    fam["C1"] = est(M(dsmall(arm("V3"), arm("V3c"))), s1=True)
    fam["C2"] = est(M(dk(arm("V4"), arm("V0"), 0)), s1=True)
    fam["C3"] = est(M(ST.v_diff(ST.v_auc(at, arm("V4", "DGZ", "WDGZrel"), 0),
                                ST.v_auc(at, arm("V4"), 0))), s1=True)
    fam["C4"] = est(M(dsmall(arm("V5"), arm("V0"))), s1=True)
    fam["C5"] = est(M(dsmall(arm("V6"), arm("V5"))), s1=True)
    small_pts = {v: ST.point(M(dsmall(arm(v), arm("V0")))) for v in ST.ARMS_BEST}
    vbest = ST.choose_vbest(small_pts)
    Xb = (M(dsmall(arm(vbest), arm("V0"))) if vbest else
          np.full((len(units.targets), len(ST.SEEDS)), np.nan))
    fam["C6"] = est(Xb, units.S, s1=True)
    if aspect == "bp":
        fam["C7"] = est(Xb, units.LD, s1=True)
    ST.classify_family(fam)

    d: Dict[str, Any] = {"delta_small_vs_V0_points": small_pts}
    d["C6_C7_every_arm"] = {}
    for v in ST.ARMS_BEST:
        X = M(dsmall(arm(v), arm("V0")))
        d["C6_C7_every_arm"][v] = {"C6": est(X, units.S)}
        if aspect == "bp":
            d["C6_C7_every_arm"][v]["C7"] = est(X, units.LD)
    d["per_k"] = {f"{v}-V0@k{k}": est(M(dk(arm(v), arm("V0"), k)))
                  for v in ST.ARMS_BEST for k in ST.ALL_KS}
    d["per_k"].update({f"V3-V3c@k{k}": est(M(dk(arm("V3"), arm("V3c"), k))) for k in ST.ALL_KS})
    d["ladder"] = {f"{a}-{b}": est(M(dsmall(arm(a), arm(b))))
                   for a, b in (("V2", "V1"), ("V3", "V2"), ("V4", "V3"))}
    d["C3_decomposition"] = ST.decomposition(at, units, feature, B, seed)
    d["channel_reads"] = ST.channel_estimates(channels, units, B, seed)
    d["regime_DGZ_minus_TP"] = {f"{v}@k{k}": est(M(dk(arm(v, "DGZ"), arm(v), k)))
                                for v in ("V0", "V4") for k in ST.ALL_KS}
    d["W1_minus_W2"] = {}
    for v in ("V0", "V4", "V5"):
        d["W1_minus_W2"][v] = {"small": est(M(dsmall(arm(v, rule="W1"), arm(v))))}
        d["W1_minus_W2"][v].update({f"k{k}": est(M(dk(arm(v, rule="W1"), arm(v), k)))
                                    for k in ST.SMALL_KS})
    d["F_minus_V0"] = {"W2_small": est(M(dsmall(arm("F"), arm("V0"))))}
    d["F_minus_V0"].update({f"W2@k{k}": est(M(dk(arm("F"), arm("V0"), k))) for k in ST.ALL_KS})
    d["F_minus_V0"].update({f"W1@k{k}": est(M(dk(arm("F", rule="W1"), arm("V0", rule="W1"), k)))
                            for k in ST.SMALL_KS})
    d["DS"] = {}
    for k in ST.ALL_KS:
        ds_auc = ST.v_auc(at, arm("DS"), k, seedless=True)
        X1 = M(ds_auc, seeds=(None,))
        d["DS"][f"k{k}"] = {"mean_auc": ST.point(X1), "n_targets": int((~np.isnan(X1)).sum()),
                            "DS_minus_V0": est(M(ST.v_diff(ds_auc, ST.v_auc(at, arm("V0"), k))))}
    noipi_only = lambda t: units.noipi_kept.get(t, True)  # noqa: E731
    d["noIPI"] = {f"{v}-V0_small": est(M(dsmall(arm(v, "TPnoIPI"), arm("V0", "TPnoIPI")),
                                         only=noipi_only)) for v in ("V4", "V5")}
    d["noIPI"].update({f"{v}-V0@k{k}": est(M(dk(arm(v, "TPnoIPI"), arm("V0", "TPnoIPI"), k),
                                             only=noipi_only))
                       for v in ("V4", "V5") for k in (0, 1, 5, "all")})
    d["redraw"] = {f"k{k}": ST.redraw(at, units, feature, k, B, seed) for k in (1, 5)}
    d["collateral"] = {}
    for v in ST.ARMS_BEST:
        for k in (0, 1, 5, 10, 25):
            for metric in ("fmax", "micro_aupr"):
                d["collateral"][f"{v}-V0@k{k}:{metric}"] = ST.collateral_contrast(
                    collateral, feature, v, "TP", "W2", k, units.packs, metric, B, seed)
    d["collateral_harm"] = sorted(k for k, c in d["collateral"].items()
                                  if c["classification"] == "harm")
    if at_dgz is not None:
        d["dgz_test_labels"] = dgz_label_reads(at_dgz, units, feature, B, seed)
    return {"family": fam, "vbest": vbest, "descriptive": d}


def reading_9_3(fam_c3: Dict[str, Any], dec: Dict[str, Any], g2b: Optional[str]) -> str:
    """Protocol §9.3 row for C3 (BP)."""
    leak = dec["C3_decomposition"]["C3_leak"]
    leak_x = dec["channel_reads"]["C3_leak_X"]
    pos = lambda c: c.get("ci_low") is not None and c["ci_low"] > 0  # noqa: E731
    if fam_c3.get("estimate") is None:
        return "undefined: no C3 cell (no complete V4 DGZ/WDGZrel and TP/W2 k = 0 pair)"
    if g2b != "confirmed":
        return "secondary methodological note; C3 reported as a contrast only"
    if fam_c3["classification"] != "supported":
        return "central result on DGZ's released model; controlled re-test does not reproduce an inflation"
    if not pos(leak):
        return "central result on DGZ's released model; C3 reported with its decomposition, not attributed to the leak"
    if pos(leak_x):
        return "central result: label-propagation leak through conflated relation types (C3_leak and C3_leak_X CIs > 0)"
    return "central result: protocol leak (C3_leak CI > 0); no typed-relation attribution"


def g2b_status() -> Optional[str]:
    p = ROOT / "results" / "go_function" / "gates" / "gate_dgz_zero.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["aspects"].get("bp", {}).get("status")


# ------------------------------------------------------------------ output
def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def csv_bytes(rows: List[dict], cols) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in cols])
    data = buf.getvalue().encode()
    assert_no_abs_path(data, "CSV")
    return data


def _write(path: Path, data: bytes) -> None:
    assert_writable_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def smoke_checks(runs: List[RunRec]) -> Dict[str, Any]:
    """G9a: every run has DONE.json, eval.json, scores.npz, epochs.jsonl with
    grad-norm records."""
    bad = []
    for r in runs:
        names = {p.name for p in r.dir.iterdir()}
        need = {"DONE.json", "eval.json", "scores.npz", "epochs.jsonl", "run.json"}
        ok = need <= names
        if ok:
            ep = [json.loads(x) for x in (r.dir / "epochs.jsonl").read_text().splitlines() if x]
            ok = bool(ep) and all(e.get("grad_norms") for e in ep)
        if not ok:
            bad.append(_rel(r.dir))
    if bad:
        raise FatalError(f"smoke runs with missing outputs: {bad}")
    return {"n_runs_checked": len(runs), "all_outputs_present": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="runs dir (repo-relative)")
    ap.add_argument("--out", required=True, help="output dir (repo-relative)")
    ap.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true", help="G9 smoke: outputs labelled exploratory")
    args = ap.parse_args(argv)
    log = setup_logging("analyze")
    try:
        from go_function.train import Refused as RunRefused, check_import_coverage
        try:
            info_code = check_import_coverage("smoke" if args.smoke else "study", log=log)
        except RunRefused as e:
            raise Refused(str(e)) from None
        runs_dir = repo_path(args.runs) if not Path(args.runs).is_absolute() else Path(args.runs)
        out_dir = repo_path(args.out) if not Path(args.out).is_absolute() else Path(args.out)
        runs, counts = discover(runs_dir)
        log.info("runs: %s", counts)
        if not runs:
            raise Refused(f"no completed run under {args.runs}")
        smoke = smoke_checks(runs) if args.smoke else None
        long, fdet, ds = [], [], []
        contrasts: Dict[str, Any] = {}
        decisions: Dict[str, Any] = {"schema": 1, "smoke": bool(args.smoke), "aspects": {},
                                     "seed": args.seed, "B": args.B}
        for aspect in ASPECTS:
            ar = [r for r in runs if r.spec["aspect"] == aspect]
            if not ar:
                continue
            inp = AspectInputs(aspect, log)
            tcol = inp.target_col_map()
            rows = [x for r in ar
                    for x in long_rows(r, dgz_label_aucs(r, inp.dgz_test(), tcol)
                                       if r.eval.get("targets") else None)]
            at_dgz = ST.AucTable(rows, metric="class_auc_dgz_labels")
            frows, fd = fusion_rows(ar, inp)
            drows = ds_rows(ar, inp)
            chans = channel_cells(ar, inp)
            coll = {}
            for r in ar:
                c = r.eval.get("collateral")
                if c:
                    s = r.spec
                    coll[(s["feature"], s["variant"], s["regime"], s["rule"],
                          s.get("retention_draw", 1), r.pack, r.k, s["seed"])] = {
                        "fmax": c.get("fmax"), "micro_aupr": c.get("aupr_micro"),
                        "smin": c.get("smin")}
            at = ST.AucTable(rows + frows + drows)
            features = [f for f in FEATURES if any(r.spec["feature"] == f for r in ar)]
            miss = ST.missing_share(at, inp.units, CONFIRMATORY_FEATURE)
            reasons = (["smoke run"] if args.smoke else []) + (
                [f"{miss['share_missing']:.1%} of the main block's values missing (> 10%)"]
                if miss["share_missing"] > MISSING_LIMIT else [])
            contrasts[aspect] = {}
            dec_a = {"exploratory": bool(reasons), "exploratory_reasons": reasons,
                     "main_block_missing": miss, "features": {}}
            for f in features:
                res = analyse_feature(at, inp.units, aspect, f, args.B, args.seed,
                                      chans.get(f, {}), coll, at_dgz)
                contrasts[aspect][f] = res
                fam = res["family"]
                dec_f = {"role": "confirmatory" if f == CONFIRMATORY_FEATURE else "descriptive",
                         "vbest": res["vbest"],
                         "classification": {k: c["classification"] for k, c in fam.items()},
                         "estimate": {k: c["estimate"] for k, c in fam.items()},
                         "p_holm": {k: c["p_holm"] for k, c in fam.items()},
                         "collateral_harm": res["descriptive"]["collateral_harm"]}
                if aspect == "bp":
                    dec_f["C3_reading_9_3"] = reading_9_3(fam["C3"], res["descriptive"],
                                                          g2b_status())
                dec_a["features"][f] = dec_f
            decisions["aspects"][aspect] = dec_a
            long += rows + frows + drows
            fdet += fd
            ds += drows
            del inp
        decisions["g2b_status_bp"] = g2b_status()
        decisions["run_counts"] = counts
        decisions["code"] = info_code
        # repeated after fusion, baselines and evaluate were imported (review finding 1)
        try:
            decisions["code_at_finish"] = check_import_coverage("smoke" if args.smoke else "study")
        except RunRefused as e:
            raise Refused(str(e)) from None
        if smoke:
            decisions["smoke_checks"] = smoke
        decisions["outputs"] = ["long_table.csv", "fusion_F.csv", "baseline_DS.csv",
                                "contrasts.json", "decisions.json"]
        long.sort(key=lambda r: tuple(str(r.get(c)) for c in LONG_COLS[:10]))
        fcols = list(LONG_COLS[:10]) + ["source_run", "auc_F", "auc_V0", "n_pa", "n_sib",
                                        "n_sib_candidates", "pa_dropped", "sib_dropped"]
        _write(out_dir / "long_table.csv", csv_bytes(long, LONG_COLS))
        _write(out_dir / "fusion_F.csv", csv_bytes(fdet, fcols))
        _write(out_dir / "baseline_DS.csv", csv_bytes(ds, list(LONG_COLS[:13]) + ["condition"]))
        _write(out_dir / "contrasts.json", canonical_json(_jsonable(contrasts)))
        _write(out_dir / "decisions.json", canonical_json(_jsonable(decisions)))
        log.info("wrote %s: %d long rows, %d F rows, %d DS rows", _rel(out_dir), len(long),
                 len(fdet), len(ds))
    except Refused as e:
        log.error("%s", e)
        return EXIT_REFUSED
    except (FatalError, ProtocolStop) as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
