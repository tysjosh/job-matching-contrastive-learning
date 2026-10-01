#!/usr/bin/env python3
"""Consolidate every ontology-signal ceiling measurement into one table.

Reads results/ontology_ceiling/*.json (written by go_bulk_signal_diagnosis.py,
mesh_hierarchy_vs_set_diagnosis.py and isco_encoding_diagnosis.py) plus the
career/trials numbers measured earlier in the project, and reports:

  1. Headline ceilings, one row per ontology, using each ontology's best
     defensible configuration.
  2. The GO decomposition: how the n=42 Jaccard probe's 0.7628 relates to the
     bulk measurement, factor by factor.
  3. The ISCO encoding question and its oracle bound.
  4. Cross-split stability warnings where a number is not reproducible.

Nothing here is compared ACROSS datasets. Each row is an ontology measured
against its own dataset's own label; the table exists so each row can be checked
against chance on its own task.

Usage
  .venv/bin/python3 scripts/ontology_ceiling_report.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CEIL = ROOT / "results" / "ontology_ceiling"


def load(name):
    p = CEIL / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def auc(d, measure, contrast):
    if not d:
        return None
    m = d.get("measures", {}).get(measure)
    if not m:
        return None
    c = m.get("contrasts", {}).get(contrast)
    return c.get("rank_auc") if c else None


def ci(d, measure, contrast):
    if not d:
        return None
    m = d.get("measures", {}).get(measure)
    if not m:
        return None
    c = m.get("contrasts", {}).get(contrast)
    return c.get("auc_ci95") if c else None


def fmt(v, w=7):
    return f"{v:{w}.4f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"


def fmtci(v):
    if not v or not all(isinstance(x, (int, float)) for x in v):
        return ""
    return f"[{v[0]:.4f},{v[1]:.4f}]"


def main():
    print("=" * 100)
    print("ONTOLOGY SIGNAL CEILINGS -- consolidated")
    print("=" * 100)
    print()
    print("Each row: how well that ontology's similarity alone ranks its OWN dataset's")
    print("OWN graded label. No model, no training -- this bounds what any model relying")
    print("on that ontology could extract from it. Rows are NOT comparable to each other.")
    print()
    print("  easy contrast = clear positive vs clear negative")
    print("  hard contrast = the distinction negative selection must get right")
    print("                  (good_fit vs potential_fit / eligible vs relevant-but-not /")
    print("                   established interaction vs weak-evidence interaction)")
    print()

    go_abs_P = load("go_bulk_P_experimental_absolute.json")
    go_abs_A = load("go_bulk_A_experimental_absolute.json")
    go_abs_C = load("go_bulk_C_experimental_absolute.json")
    go_abs_noguard = load("go_bulk_A_experimental_absolute_keepIPI_keepIEA_keepBinding.json")
    go_tert_P = load("go_bulk_P_experimental.json")
    go_repro = load("go_bulk_A_combined_score_ego_keepIPI_keepIEA_keepBinding_noProp.json")
    mesh_test = load("mesh_hier_vs_set_test.json")
    mesh_val = load("mesh_hier_vs_set_validation.json")
    isco = load("isco_encoding_data.json")
    cpc = load("cpc_signal.json")
    cpc_rows = load("cpc_signal_rows.json")

    # ---------------------------------------------------------------- headline
    print("-" * 100)
    print("1. HEADLINE CEILINGS")
    print("-" * 100)
    print(f"  {'ontology':<16} {'domain':<7} {'measure':<14} {'easy':>8} {'hard':>8}  {'hard 95% CI':<19} label / split")
    print(f"  {'-'*16} {'-'*7} {'-'*14} {'-'*8} {'-'*8}  {'-'*19} {'-'*30}")

    rows = []
    mset = "set: ontology_set_similarity"
    if mesh_val:
        rows.append(("MeSH", "trials", "set-sim", auc(mesh_val, mset, "easy"),
                     auc(mesh_val, mset, "hard"), ci(mesh_val, mset, "hard"),
                     "eligibility / validation 2021"))
    if mesh_test:
        rows.append(("MeSH", "trials", "set-sim", auc(mesh_test, mset, "easy"),
                     auc(mesh_test, mset, "hard"), ci(mesh_test, mset, "hard"),
                     "eligibility / TEST 2022"))
    rows.append(("ESCO skill", "career", "graph-dist", 0.6275, 0.5581, None,
                 "potential_fit / train sample"))
    if isco:
        fb = isco["measures"]["fixed bands (training code)"]["contrasts"]
        rows.append(("ISCO occupation", "career", "fixed bands", fb["easy"]["rank_auc"],
                     fb["hard"]["rank_auc"], fb["hard"]["auc_ci95"],
                     "potential_fit / train sample"))
    if cpc:
        cm = "set: simGIC (IC-weighted)"
        rows.append(("CPC", "patents", "simGIC", auc(cpc, cm, "easy"), auc(cpc, cm, "hard"),
                     ci(cpc, cm, "hard"), "examiner X vs A / PatentMatch test"))
        cw = "hier: Wu-Palmer BMA (depth-norm)"
        rows.append(("CPC", "patents", "Wu-Palmer", auc(cpc, cw, "easy"), auc(cpc, cw, "hard"),
                     ci(cpc, cw, "hard"), "examiner X vs A / PatentMatch test"))
    rows.append(("GO (n=42 probe)", "PPI", "Jaccard", 0.7959, 0.7628, None,
                 "STRING combined, 4 ego-nets"))
    if go_abs_P:
        rows.append(("GO", "PPI", "simGIC", auc(go_abs_P, "simGIC", "easy"),
                     auc(go_abs_P, "simGIC", "hard"), ci(go_abs_P, "simGIC", "hard"),
                     "STRING experimental, bulk BP"))
    if go_abs_C:
        rows.append(("GO", "PPI", "simGIC", auc(go_abs_C, "simGIC", "easy"),
                     auc(go_abs_C, "simGIC", "hard"), ci(go_abs_C, "simGIC", "hard"),
                     "STRING experimental, bulk CC"))
    if go_abs_A:
        rows.append(("GO", "PPI", "simGIC", auc(go_abs_A, "simGIC", "easy"),
                     auc(go_abs_A, "simGIC", "hard"), ci(go_abs_A, "simGIC", "hard"),
                     "STRING experimental, bulk all"))

    for name, dom, meas, e, h, c, lab in rows:
        star = ""
        if isinstance(c, list) and len(c) == 2 and c[0] <= 0.5 <= c[1]:
            star = "  <- chance"
        print(f"  {name:<16} {dom:<7} {meas:<14} {fmt(e, 8)} {fmt(h, 8)}  {fmtci(c):<19} {lab}{star}")

    print()
    print("  Reading: on the HARD contrast, four ontologies across three domains sit")
    print("  between 0.485 and 0.607 -- ESCO, ISCO, MeSH and CPC -- with ISCO's and CPC's")
    print("  CIs spanning 0.5 outright. GO sits near 0.82 with a CI nowhere near chance.")
    print("  So the diagnostic is not simply insensitive: applied to an ontology that")
    print("  genuinely encodes the target relation it returns a large value on the same")
    print("  statistic. The nulls are a property of those ontology/label pairings.")
    print()
    print("  CPC is the sharpest case. Its EASY contrast is 0.977 -- near-perfect at")
    print("  telling cited prior art from uncited documents -- while its HARD contrast is")
    print("  0.527 with the CI containing 0.5. The same ontology that almost perfectly")
    print("  identifies 'same technical field' carries essentially nothing about whether a")
    print("  document in that field destroys novelty. That gap is the whole finding: these")
    print("  ontologies encode topicality, and the hard contrast is not about topicality.")

    # ------------------------------------------------------- GO decomposition
    print()
    print("-" * 100)
    print("2. GO DECOMPOSITION -- reconciling the n=42 probe (0.7628) with bulk data")
    print("-" * 100)
    print("  The probe's 0.7628 was initially contradicted by a bulk run at 0.5284, which")
    print("  looked like the positive control collapsing. It was not. Two separate issues:")
    print()
    print("  (a) TIERING was wrong in the first bulk run. STRING's experimental scores are")
    print("      heavily right-skewed (median 102, p90 292) while STRING itself calls 400")
    print("      medium and 700 high confidence. Confidence TERTILES therefore cut at ~84")
    print("      and ~134, so 'high vs medium' compared weak evidence against slightly less")
    print("      weak evidence -- a proxy for how much assay attention a pair received, not")
    print("      for whether the interaction is real. GO does not predict study effort.")
    t = auc(go_tert_P, "simGIC", "hard")
    a = auc(go_abs_P, "simGIC", "hard")
    if t and a:
        print(f"        tertile bands, BP, GO-independent label : hard {t:.4f}")
        print(f"        STRING confidence bands, same data      : hard {a:.4f}   ({a - t:+.4f})")
    print()
    print("  (b) The probe ALSO had four methodological weaknesses that inflate it. Their")
    print("      combined effect is modest; most of the probe's excess was n=14-vs-14 noise.")
    grid = [
        ("bulk, BP, experimental label, propagated, guards on", "go_bulk_P_experimental_keepBinding.json"),
        ("+ all GO aspects", "go_bulk_A_experimental_keepBinding.json"),
        ("+ restrict to the 4 ego-networks", "go_bulk_A_experimental_ego_keepBinding.json"),
        ("+ combined_score label (circular)", "go_bulk_A_combined_score_ego_keepBinding.json"),
        ("+ direct annots, IPI+IEA+binding kept (= probe's exact recipe)",
         "go_bulk_A_combined_score_ego_keepIPI_keepIEA_keepBinding_noProp.json"),
    ]
    print(f"      {'configuration (tertile bands throughout)':<62} {'Jaccard':>8} {'simGIC':>8}")
    for label, fn in grid:
        d = load(fn)
        print(f"      {label:<62} {fmt(auc(d, 'Jaccard', 'hard'), 8)} {fmt(auc(d, 'simGIC', 'hard'), 8)}")
    print(f"      {'probe as actually reported (n=14 vs 14)':<62} {0.7628:>8.4f} {'-':>8}")
    if go_repro:
        r = auc(go_repro, "Jaccard", "hard")
        if r:
            print()
            print(f"      Reproducing every one of the probe's choices at n=2000 gives {r:.4f}.")
            print(f"      The gap to its reported 0.7628 ({0.7628 - r:+.4f}) is sampling noise:")
            print(f"      at n=14 vs 14 the 95% CI on an AUC is roughly +/-0.16.")
    print()
    print("  Conclusion: the probe pointed the right way but its number was not meaningful.")
    print("  The defensible GO figure is the STRING-confidence-band measurement, which is")
    print("  HIGHER than the probe claimed and rests on 3000 pairs per tier with a")
    print("  GO-independent label and all circularity guards active.")

    if go_abs_A and go_abs_noguard:
        on_h = auc(go_abs_A, "simGIC", "hard")
        off_h = auc(go_abs_noguard, "simGIC", "hard")
        on_e = auc(go_abs_A, "simGIC", "easy")
        off_e = auc(go_abs_noguard, "simGIC", "easy")
        print()
        print("  Circularity guards (IPI evidence, IEA evidence, protein-binding terms),")
        print("  all GO aspects, STRING confidence bands:")
        print(f"      guards ON  (reported)  easy {fmt(on_e)}  hard {fmt(on_h)}")
        print(f"      guards OFF             easy {fmt(off_e)}  hard {fmt(off_h)}")
        if on_h and off_h:
            print(f"      difference             easy {off_e - on_e:+.4f}  hard {off_h - on_h:+.4f}")
            if off_h < on_h:
                print("      Dropping the guards LOWERS the result, so GO's signal here is not")
                print("      an artefact of interaction-derived annotations leaking the label.")
                print("      (IEA adds many low-quality terms, which dilutes more than it leaks.)")
            else:
                print("      Guards cost real AUC, i.e. some of the naive signal IS circular.")

    # ------------------------------------------------------------------ ISCO
    if isco:
        print()
        print("-" * 100)
        print("3. ISCO -- is the ceiling the hierarchy or the encoding?")
        print("-" * 100)
        for name in ("fixed bands (training code)", "Wu-Palmer (depth-normalised)",
                     "linear shared/4", "Resnik IC(LCA prefix)", "Lin IC-normalised"):
            m = isco["measures"].get(name)
            if not m:
                continue
            c = m["contrasts"]
            print(f"  {name:<30} [{m['kind']:<18}] "
                  f"easy {c['easy']['rank_auc']:.4f}  hard {c['hard']['rank_auc']:.4f}  "
                  f"({m['n_distinct_values']} distinct values)")
        o = isco.get("oracle_prefix_length_only", {})
        print(f"  {'ORACLE over the 5 prefix buckets':<30} [{'upper bound':<18}] "
              f"easy {o.get('easy', float('nan')):.4f}  hard {o.get('hard', float('nan')):.4f}")
        print()
        print("  Two results close this question:")
        print("  - Every prefix-length-based distance gives IDENTICAL rank-AUC. ISCO is")
        print("    fixed-depth (4 digits) and single-position, so fixed bands, Wu-Palmer and")
        print("    hop counts are all monotone functions of one integer, and rank-AUC is")
        print("    invariant under monotone transforms. Re-encoding ISCO CANNOT change it.")
        print("  - IC weighting, which can reorder pairs, gains +0.004 and stays below the")
        print(f"    oracle. The oracle itself is only {o.get('hard', float('nan')):.4f}, so no tuning of the five")
        print("    constants in batch_processor.py:1100 can rescue the hard contrast.")
        print("  Actionable: isco_weight=0.0 stands; do not spend a run on retuning bands.")

        if cpc:
            print()
            print("  CPC settles the follow-up question: was ISCO's result about HIERARCHIES")
            print("  or about ISCO's impoverished encoding? CPC is also a classification")
            print("  hierarchy but has none of ISCO's limits --")
            print(f"      ISCO  fixed depth 4, 1 code/entity,  5 distinct distance values")
            print(f"      CPC   variable depth to 14, ~17 codes/entity, 765 distinct values")
            cw = "hier: Wu-Palmer BMA (depth-norm)"
            cl = "hier: LCA depth BMA (raw)"
            print(f"      ISCO hard 0.4851 (oracle 0.5192)")
            print(f"      CPC  hard {auc(cpc, cw, 'hard'):.4f} depth-normalised, "
                  f"{auc(cpc, cl, 'hard'):.4f} raw LCA depth")
            print("  A ~150x richer hierarchy buys ~+0.04 and still cannot clear chance. So")
            print("  the encoding was never the binding constraint; resolution is not the")
            print("  missing ingredient.")

    # ----------------------------------------------------------- MeSH stability
    if mesh_test and mesh_val:
        print()
        print("-" * 100)
        print("4. STABILITY WARNINGS")
        print("-" * 100)
        print("  MeSH hard contrast is NOT stable across the two trials splits, and the")
        print("  hierarchy-vs-set ordering actually reverses:")
        print(f"      {'view':<32} {'validation 2021':>16} {'TEST 2022':>12}")
        for name in (mset, "hier: Wu-Palmer (depth-norm)", "hier: LCA hops (raw)",
                     "hier: top-level branch overlap"):
            v = auc(mesh_val, name, "hard")
            t2 = auc(mesh_test, name, "hard")
            print(f"      {name:<32} {fmt(v, 16)} {fmt(t2, 12)}")
        print()
        print("  Only ~220-270 grade-1 pairs exist per split, giving CIs near +/-0.05. Do")
        print("  not draw a hierarchy-vs-set conclusion from trials. The ISCO oracle above")
        print("  answers the ISCO encoding question without needing this comparison.")
        print()
        print("  Note also that the validation figures (easy 0.7647, hard 0.5256) reproduce")
        print("  the numbers carried in this project's earlier notes exactly, which confirms")
        print("  the reimplementation rather than the stability of the quantity.")

    if cpc and cpc_rows:
        cm = "set: Jaccard(full codes)"
        dh, dci = auc(cpc, cm, "hard"), ci(cpc, cm, "hard")
        rh, rci = auc(cpc_rows, cm, "hard"), ci(cpc_rows, cm, "hard")
        print()
        print("  CPC: unit-of-analysis inflation. PatentMatch rows are claim x paragraph")
        print("  pairs, but CPC codes belong to the DOCUMENT, so one document pair recurs")
        print(f"  ~100 times ({cpc_rows['n_rows']} rows over {cpc['n_pairs_scored']} unique pairs).")
        print(f"      deduped  hard {dh:.4f}  CI {fmtci(dci)}  width {dci[1]-dci[0]:.4f}  spans 0.5")
        print(f"      raw rows hard {rh:.4f}  CI {fmtci(rci)}  width {rci[1]-rci[0]:.4f}  EXCLUDES 0.5")
        print(f"  The interval is {(dci[1]-dci[0])/(rci[1]-rci[0]):.1f}x narrower on raw rows and the")
        print("  conclusion flips to 'significantly above chance'. The deduped figure is the")
        print("  correct one; the repeats are re-measurements, not independent evidence.")

    print()
    print("=" * 100)
    print("files")
    print("=" * 100)
    for p in sorted(CEIL.glob("*.json")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
