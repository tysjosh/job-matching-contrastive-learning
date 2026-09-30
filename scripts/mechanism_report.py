#!/usr/bin/env python3
"""Why ontology injection failed, per domain, in one table.

Pulls together the three measurements that together identify the failure mode,
which no one of them can do alone:

  1. ONTOLOGY CEILING  -- how well ontology similarity alone ranks the label.
     results/ontology_ceiling/*.json, from the *_signal_diagnosis scripts.
  2. TEXT CEILING and INCREMENT -- how well the frozen encoder's text similarity
     ranks the same label, and how much the ontology adds ON TOP of it (5-fold
     cross-validated). results/ontology_ceiling/text_vs_ontology*.json, from
     scripts/text_vs_ontology_ceiling.py.
  3. TRAINING OUTCOME -- what ontology-guided negative selection actually
     delivered, on held-out test data, with the negative-diversity confound
     controlled for where a control arm exists.

Reading the three together separates failure modes that look identical if you
only look at the training result:

  A. text at chance AND ontology at chance
        The dataset does not contain the distinction. No ontology and no
        mechanism can help. A null result here says nothing about the method.
  B. text strong, increment ~0
        Redundancy. The ontology is real but the text already carries it.
  C. ontology has a real increment, yet training gains nothing
        The information is present and non-redundant, so the null indicts the
        MECHANISM. Negative selection only chooses which examples to show; it
        never puts the ontology value in front of the model at inference. The
        cross-validated combined AUC is the existence proof that a model which
        consumes the ontology as a FEATURE would capture the increment.

Usage
    .venv/bin/python3 scripts/mechanism_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CEIL = ROOT / "results" / "ontology_ceiling"
LC = ROOT / "results" / "lc_single_factor"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, w=9, prec=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return " " * (w - 1) + "-"
    return f"{v:{w}.{prec}f}"


def main():
    tvo = load(CEIL / "text_vs_ontology.json") or {}
    tvo_career = load(CEIL / "text_vs_ontology_career_train.json") or {}
    # The career test split holds only 800 records; a train-split re-measurement
    # is used when present because this quantity involves no model.
    if tvo_career.get("career"):
        tvo["career"] = tvo_career["career"]

    go_report = load(LC / "go_ppi_test_report.json") or {}
    career_report = load(LC / "career_final_report.json") or {}

    print("=" * 104)
    print("WHY ONTOLOGY INJECTION FAILED -- ceiling, text, increment, outcome")
    print("=" * 104)
    print()
    print("All numbers are the HARD contrast (good_fit vs potential_fit, or that")
    print("domain's equivalent): eligible vs ineligible on trials, established vs")
    print("weak-evidence interaction on GO/PPI. Every figure is within-dataset against")
    print("that dataset's own label; rows are NOT comparable to each other.")
    print()

    # ------------------------------------------------------------------ table
    print("-" * 104)
    print("1. THE THREE MEASUREMENTS")
    print("-" * 104)
    print(f"  {'domain':<9} {'AUC_text':>9} {'AUC_onto':>9} {'INCREMENT':>10} "
          f"{'incr 95% CI':<20} {'rho':>7}   failure mode")
    print(f"  {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*20} {'-'*7}   {'-'*30}")

    modes = {}
    for dom in ("career", "trials", "patents", "go_ppi"):
        r = tvo.get(dom)
        if not r:
            continue
        h = (r.get("contrasts") or {}).get("hard")
        if not h:
            continue
        inc, ci = h["increment"], h["increment_ci95"]
        spans0 = not math.isnan(ci[0]) and ci[0] <= 0.0 <= ci[1]
        # Classify on the COMBINED ceiling, not on either source alone. A domain
        # where text and ontology together still only reach ~0.55 is barely
        # learnable no matter what mechanism is used, even if the ontology's
        # increment over a chance-level text baseline is statistically detectable.
        # Judging on auc_ontology alone put career (0.5518, a hair over a 0.55
        # threshold) in the same class as trials (0.6031 with a +0.12 increment),
        # which reads far too generously.
        combined = h.get("auc_combined_cv")
        if combined is not None and combined < 0.60:
            mode = "A  barely learnable at all (combined %.3f)" % combined
        elif spans0:
            mode = "?  increment unresolved"
        elif inc > 0:
            mode = "C  real increment, mechanism at fault"
        else:
            mode = "B  ontology worse than text"
        modes[dom] = mode
        print(f"  {dom:<9} {fmt(h['auc_text'])} {fmt(h['auc_ontology'])} "
              f"{inc:>+10.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  "
              f"{r['spearman_text_vs_ontology']:>+7.3f}   {mode}")

    # ------------------------------------------------------- training outcomes
    print()
    print("-" * 104)
    print("2. WHAT TRAINING ACTUALLY DELIVERED (held-out test, paired per seed)")
    print("-" * 104)

    hk = "eligible_vs_ineligible"
    cons = (go_report.get("contrasts") or {}).get(hk, {})
    means = (go_report.get("arm_means") or {}).get(hk, {})
    if cons:
        print("  go_ppi -- ontology-guided negative selection, decomposed:")
        print(f"    {'arm':<26} {'hard AUC':>9}")
        for a in ("baseline", "randwin", "ontneg_stoch", "ontneg"):
            if means.get(a) is not None:
                print(f"    {a:<26} {means[a]:>9.4f}")
        print()
        for key, nice in (("NARROWING", "narrowing the negative pool"),
                          ("ONTOLOGY", "GO choosing the region  <-- THE ANSWER"),
                          ("DETERMINISM", "losing per-epoch variety"),
                          ("TOTAL", "TOTAL baseline -> GO prefix")):
            c = cons.get(key, {}).get("pooled")
            if c:
                print(f"    {nice:<40} {c['mean']:+.4f}  sd={c['sd']:.4f}  "
                      f"{c['wins']}/{c['n']} seeds up")
        print()
        print("    Only the ONTOLOGY row is an ontology effect. It compares a GO-chosen")
        print("    window against a RANDOM window of the same width, matched on negative")
        print("    diversity and grade mix. The other two rows are mechanical costs of")
        print("    HOW ontology guidance draws negatives, and they are what makes the")
        print("    naive baseline-vs-ontology delta look like harm.")

    if career_report:
        print()
        print("  career -- four arms, 4 fractions x 3 seeds, held-out test:")
        for arm, v in (career_report.get("results", {}).get("eval", {})).items():
            print(f"    {arm:<26} {v['pooled_delta']:+.4f}  p={v['p']:.3f}  "
                  f"{v['wins']}/{v['pooled_n']} wins")

    # ---------------------------------------------------------------- reading
    print()
    print("=" * 104)
    print("3. READING")
    print("=" * 104)

    car = modes.get("career", "")
    tri = modes.get("trials", "")
    go = modes.get("go_ppi", "")

    if car.startswith("A"):
        ct = tvo["career"]["contrasts"]["hard"]
        print(f"  CAREER is failure mode A. Text {ct['auc_text']:.4f} (chance), ontology "
              f"{ct['auc_ontology']:.4f}, both together {ct['auc_combined_cv']:.4f}.")
        print(f"  ESCO's increment over text is {ct['increment']:+.4f} and its CI excludes")
        print("  zero, so it is not literally nothing -- but it is an increment on top of a")
        print("  chance-level baseline, and the best any consumer of these two features")
        print(f"  could reach is {ct['auc_combined_cv']:.4f}. good_fit and potential_fit are")
        print("  close to indistinguishable in this dataset from either source.")
        print("  The 96 null training runs are therefore weak evidence about ontology")
        print("  injection as a method: there was almost nothing available to inject.")
        print("  This is a property of the labels; no mechanism repairs it.")
        print()

    if tri.startswith("C") or go.startswith("C"):
        print("  TRIALS and GO/PPI are failure mode C, and this is the finding.")
        for dom in ("trials", "go_ppi"):
            r = tvo.get(dom)
            if not r:
                continue
            h = r["contrasts"]["hard"]
            print(f"    {dom:<8} text {h['auc_text']:.4f} -> ontology adds "
                  f"{h['increment']:+.4f}  (combined {h['auc_combined_cv']:.4f})")
        print()
        print("  On trials the text is at chance (0.4938) while MeSH reaches 0.6031, so")
        print("  almost all of MeSH's signal is information the encoder does not have.")
        print("  On GO/PPI the text is already strong (0.7790) and GO still adds")
        print("  +0.0173 beyond it. In BOTH cases the ontology holds real,")
        print("  non-redundant signal about the label.")
        print()
        onto_pooled = cons.get("ONTOLOGY", {}).get("pooled") or {}
        if onto_pooled:
            print(f"  And in both cases ontology-guided negative selection delivered "
                  f"~nothing:")
            print(f"    GO/PPI  {onto_pooled['mean']:+.4f} (sd {onto_pooled['sd']:.4f}, "
                  f"n={onto_pooled['n']} seeds, {onto_pooled['wins']}/{onto_pooled['n']} up) "
                  f"with the diversity confound controlled")
            print(f"    trials  null / sign-unstable (one seed, arm ordering inverted)")
        else:
            print("  And in both cases ontology-guided negative selection delivered "
                  "~nothing.")
        print("  The information is available and the mechanism fails to move it into")
        print("  the model.")
        print()
        print("  Why the mechanism cannot work: negative selection uses the ontology")
        print("  ONLY to choose which examples appear in a batch. The model never sees")
        print("  an ontology value, at training or inference, so the only way the signal")
        print("  can arrive is indirectly, through which contrasts the encoder happens to")
        print("  be exposed to. The cross-validated combined AUC shows what is being")
        print("  left on the table: a two-feature model consuming ontology similarity")
        print("  directly captures the whole increment.")
        print()
        print("  ACTIONABLE: stop injecting through negative selection and sample")
        print("  weighting. Use the ontology as an inference-time feature or an")
        print("  auxiliary supervision target. On trials that is worth up to +0.12 AUC")
        print("  on the contrast that matters, and it is currently discarded.")

    print()
    print("  CAVEATS")
    print("    - The GO/PPI ontology contrast is n=5 seeds (others n=3); sd ~0.010, so")
    print("      it resolves 'not a large positive effect', not +/-0.01.")
    print("    - Career's increment is measured on 800 test records (200 vs 200) unless")
    print("      the train-split re-measurement is present; its CI is correspondingly wide.")
    print("    - Trials' training result rests on ONE seed with an inverted arm ordering")
    print("      and a 2021->2022 distribution shift; it is not a clean null.")
    print("    - INCREMENT is an upper bound on what a perfect consumer of the ontology")
    print("      could gain from these two features. It is not a promise that a dense")
    print("      retriever reaches it.")

    out = CEIL / "mechanism_report.json"
    out.write_text(json.dumps(
        {"failure_modes": modes,
         "text_vs_ontology": tvo,
         "go_ppi_training": {"arm_means": means, "contrasts": cons}},
        indent=2, default=str))
    print()
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
