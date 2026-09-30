#!/usr/bin/env python3
"""Re-select the fusion weight from saved curves under different criteria.

``probe_fusion_mechanism.py`` stores the full validation and test sweeps per
checkpoint, so the choice of WHICH criterion the validation weight optimises can
be revisited without re-encoding anything (the encoding is ~30 min per domain).

This exists because the first run exposed a flaw in my own selection rule. I chose
the weight by the HARD contrast on validation. On trials seed 13 that drove w to
1.0 -- pure ontology, no text at all -- which lifted the hard contrast (+0.0581)
and destroyed everything else (easy -0.0870, pooled -0.0722). Optimising a
sub-metric can push the weight to a corner that wrecks the metric you actually
report.

The defensible rule is to select on the SAME metric you report. Comparing the
criteria side by side also shows how much the conclusion depends on that choice,
which is worth knowing rather than hiding.

Selection is always on validation; every reported number is test.

Usage
    .venv/bin/python3 scripts/fusion_reselect.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CEIL = ROOT / "results" / "ontology_ceiling"

NEG_SELECTION_BASELINE = {
    # what ontology-guided NEGATIVE SELECTION delivered on the hard contrast,
    # for reference: the mechanism this is meant to replace
    "go_ppi": ("+0.0011", "n=5 seeds, 4/5 up, diversity confound controlled"),
    "trials": ("null", "one seed, arm ordering inverted"),
}


def main() -> int:
    print("=" * 100)
    print("SCORE FUSION vs NEGATIVE SELECTION -- weight selected on validation, "
          "scored on test")
    print("=" * 100)
    print()

    summary = {}
    for domain in ("go_ppi", "trials"):
        path = CEIL / f"fusion_probe_{domain}.json"
        if not path.exists():
            print(f"SKIP {domain}: no {path.name}")
            continue
        data = json.loads(path.read_text())
        pcs = data["per_checkpoint"]
        weights = sorted(float(w) for w in pcs[0]["val_curve"])

        print("-" * 100)
        print(f"{domain}  ({len(pcs)} checkpoints)")
        print("-" * 100)
        print(f"  {'select on':<12} {'w chosen':<14} {'hard':>18} {'easy':>18} {'pooled':>18}")
        print(f"  {'-'*12} {'-'*14} {'-'*18} {'-'*18} {'-'*18}")

        dom_out = {}
        for criterion in ("hard", "easy", "pooled"):
            chosen, deltas = [], {"hard": [], "easy": [], "pooled": []}
            for pc in pcs:
                vc, tc = pc["val_curve"], pc["test_curve"]
                best = max(weights,
                           key=lambda w: (vc[str(w)][criterion]
                                          if not math.isnan(vc[str(w)][criterion]) else -1))
                chosen.append(best)
                for m in deltas:
                    b, s = tc["0.0"][m], tc[str(best)][m]
                    if not (math.isnan(b) or math.isnan(s)):
                        deltas[m].append(s - b)
            cells = []
            for m in ("hard", "easy", "pooled"):
                d = deltas[m]
                if not d:
                    cells.append(f"{'-':>18}")
                    continue
                mean = sum(d) / len(d)
                ups = sum(1 for x in d if x > 0)
                cells.append(f"{mean:>+10.4f} {ups}/{len(d):<5}")
            wtxt = ",".join(f"{w:.1f}" for w in chosen)
            print(f"  {criterion:<12} {wtxt:<14} " + " ".join(cells))
            dom_out[criterion] = {
                "weights": chosen,
                "deltas": {m: (sum(v) / len(v) if v else None) for m, v in deltas.items()},
                "ups": {m: sum(1 for x in v if x > 0) for m, v in deltas.items()},
                "n": {m: len(v) for m, v in deltas.items()},
            }
        summary[domain] = dom_out

        # absolute levels at the pooled-selected weight, the defensible headline
        print()
        best_abs = []
        for pc in pcs:
            vc, tc = pc["val_curve"], pc["test_curve"]
            w = max(weights, key=lambda x: (vc[str(x)]["pooled"]
                                            if not math.isnan(vc[str(x)]["pooled"]) else -1))
            best_abs.append((pc["checkpoint"], w, tc["0.0"], tc[str(w)]))
        print("  absolute test AUC at the pooled-selected weight:")
        for tag, w, b, s in best_abs:
            print(f"    {tag:<28} w={w:.1f}   hard {b['hard']:.4f}->{s['hard']:.4f}   "
                  f"easy {b['easy']:.4f}->{s['easy']:.4f}   "
                  f"pooled {b['pooled']:.4f}->{s['pooled']:.4f}")

        ref, note = NEG_SELECTION_BASELINE.get(domain, ("?", ""))
        hd = dom_out["pooled"]["deltas"]["hard"]
        print()
        print(f"  for comparison, ontology-guided NEGATIVE SELECTION on the hard")
        print(f"  contrast delivered: {ref}  ({note})")
        if hd is not None:
            print(f"  score fusion on the same contrast: {hd:+.4f}")
        print()

    print("=" * 100)
    print("reading")
    print("=" * 100)
    print("  Selecting the weight on the hard contrast is a trap. It can drive w to")
    print("  1.0 (pure ontology, text discarded), which lifts the hard contrast and")
    print("  wrecks the easy contrast and the pooled metric. Select on the metric you")
    print("  report. The 'pooled' row is the defensible one.")
    print()
    print("  The two domains want different things, and the sweep shape says why:")
    print("    go_ppi  text and ontology are complementary WITHIN each contrast, so a")
    print("            single blended score improves hard, easy and pooled together.")
    print("    trials  they are complementary ACROSS contrasts -- ontology owns the")
    print("            hard contrast (0.6031 vs text 0.4938), text owns the easy one")
    print("            (0.8028 vs ontology 0.7012). One global weight must trade them")
    print("            off, so the gain is real but bounded, and pushing w toward the")
    print("            ontology past the optimum costs more than it buys.")
    print()
    print("  Implication for trials: a single fused score is the wrong shape. A")
    print("  two-stage design fits the evidence better -- retrieve with text (which")
    print("  wins the easy contrast), then re-rank the shortlist with the ontology")
    print("  (which wins the hard contrast). That also preserves ANN indexability,")
    print("  because the ontology term never has to be evaluated against the whole")
    print("  corpus.")

    out = CEIL / "fusion_reselect.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
