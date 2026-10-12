"""Materialise the run specs of the grid and audit them (G8; design §12.2, protocol
§6.2; plan item 15).

  .venv/bin/python scripts/go_function/materialize.py --aspect bp --feature esm2_650m

Writes one run spec per protocol §6.2 grid cell of (aspect, feature) into
results/go_function/specs/<aspect>/<feature>/<run_name>.json and an
``index.json`` (ordered spec list with hashes, block counts, the single-factor
audit table, the spec-set sha256). Specs are write-or-verify: a re-run is
byte-identical ("unchanged"); a different existing spec is fatal.

Block counts are asserted per block; over both features they total BP 2,782 and
MF 3,157 runs. The single-factor audit pairs the specs of every preregistered
contrast and descriptive component and requires that they differ in exactly the
declared fields (out_dir always differs; lambda_el differs only as implied by
the variant, null for V0/V5 and study lambda for EL arms). Any other difference
is fatal (exit 2).

Exit codes: 0 ok; 2 fatal; 4 refused (E* for (aspect, feature) not in
config/go_function/study.json: run pilot.py first).

This module is also loaded (importlib) by sweep.py and pilot.py for ``make_spec``;
it imports go_function.train only inside ``main`` (pilot.py must never import the
evaluator).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_function.constants import EXIT_FATAL, EXIT_OK, EXIT_REFUSED, SEEDS  # noqa: E402
from go_function.pins import canonical_json, sha256_bytes, write_or_verify_file  # noqa: E402
from go_function.runtime import FatalError, setup_logging  # noqa: E402

STUDY = ROOT / "config" / "go_function" / "study.json"
SPEC_ROOT = ROOT / "results" / "go_function" / "specs"
RUNS_REL = "results/go_function/runs"
FEATURES = ("esm2_650m", "interpro")
EL_ARMS = ("V1", "V2", "V3", "V3c", "V4", "V6")
PACKS = tuple(f"p{i:02d}" for i in range(1, 9))
ALL = [(None, "all")]
K_ALL5 = [(p, k) for p in PACKS for k in (0, 1, 5, 10, 25)]
K015 = [(p, k) for p in PACKS for k in (0, 1, 5)]
K0 = [(p, 0) for p in PACKS]
K15 = [(p, k) for p in PACKS for k in (1, 5)]


class Block:
    def __init__(self, name, feature, arms, regimes, rule, conditions, seeds=SEEDS, draw=1,
                 aspects=("bp", "mf"), expected=None):
        self.name, self.feature, self.arms, self.regimes = name, feature, tuple(arms), tuple(regimes)
        self.rule, self.conditions, self.seeds, self.draw = rule, list(conditions), tuple(seeds), draw
        self.aspects = aspects
        self.expected = expected

    def cells(self):
        for regime in self.regimes:
            for pack, k in self.conditions:
                for v in self.arms:
                    for s in self.seeds:
                        yield v, regime, pack, k, s


#: protocol §6.2 (order = materialisation order within a feature)
BLOCKS = (
    Block("main", "esm2_650m", ("V0", "V1", "V2", "V3", "V3c", "V4", "V5", "V6"), ("TP",), "W2",
          ALL + K_ALL5, expected=1640),
    Block("propagation_control", "esm2_650m", ("V0", "V4"), ("DGZ",), "W2", ALL + K_ALL5,
          expected=410),
    Block("W1", "esm2_650m", ("V0", "V4", "V5"), ("TP",), "W1", K015, expected=360),
    Block("dgz_protocol", "esm2_650m", ("V4",), ("DGZ",), "WDGZrel", K0, expected=40),
    Block("c3_decomposition", "esm2_650m", ("V4",), ("DGZ",), "W2drop", K0, expected=40),
    Block("interpro_replicate", "interpro", ("V0", "V4"), ("TP", "DGZ"), "W2", ALL + K0,
          expected=180),
    Block("interpro_dgz_protocol", "interpro", ("V4",), ("DGZ",), "WDGZrel", K0, expected=40),
    Block("interpro_c3_decomposition", "interpro", ("V4",), ("DGZ",), "W2drop", K0, expected=40),
    Block("redraw", "esm2_650m", ("V0", "V4"), ("TP",), "W2", K15, seeds=(13,), draw=2,
          expected=32),
    Block("mf_noipi", "esm2_650m", ("V0", "V4", "V5"), ("TPnoIPI",), "W2", ALL + K015,
          aspects=("mf",), expected=375),
)
TOTALS = {"bp": 2782, "mf": 3157}


def run_name(variant, regime, rule, pack, k, seed, draw=1) -> str:
    return (f"{variant}__{regime}__{rule}__{pack or 'all'}__k{k}" + ("__d2" if draw == 2 else "")
            + f"__s{seed}")


def make_spec(study: Mapping[str, Any], *, aspect: str, feature: str, variant: str, regime: str,
              rule: str, pack: Optional[str], k, seed: int, epochs: int, out_dir: str,
              mode: str = "study", draw: int = 1, lambda_el: Optional[float] = None,
              max_train_batches: Optional[int] = None) -> Dict[str, Any]:
    """A run spec (design §7.1) with the frozen study.json fields."""
    lam = None
    if variant in EL_ARMS:
        lam = float(study["lambda_el"]) if lambda_el is None else float(lambda_el)
    s = {"aspect": aspect, "feature": feature, "variant": variant, "regime": regime,
         "rule": rule, "retention_draw": draw, "pack": pack, "k": k, "seed": seed,
         "epochs": int(epochs), "lambda_el": lam, "batch_size": study["batch_size"],
         "lr": study["lr"], "milestones": list(study["milestones"]), "gamma": study["gamma"],
         "selection": "final", "out_dir": out_dir, "mode": mode,
         "nf4_radius": study.get("nf4_radius", "dgz")}
    if max_train_batches is not None:
        s["max_train_batches"] = int(max_train_batches)
    return s


def check_totals() -> None:
    for a, want in TOTALS.items():
        got = sum(b.expected for b in BLOCKS if a in b.aspects)
        if got != want:
            raise FatalError(f"protocol §6.2 blocks total {got} runs for {a}, expected {want}")


def build_specs(study: Mapping[str, Any], aspect: str, feature: str
                ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Dict[str, int]]]:
    """[(block, run_name, spec)] in materialisation order and block counts."""
    check_totals()
    star = (study.get("epochs_star") or {}).get(f"{aspect}_{feature}")
    if star is None:
        raise RefusedNoEstar(f"E* for {aspect}_{feature} is not in study.json (G7: pilot.py)")
    out, counts, seen = [], {}, set()
    for b in BLOCKS:
        if b.feature != feature or aspect not in b.aspects:
            continue
        n = 0
        for v, regime, pack, k, s in b.cells():
            name = run_name(v, regime, b.rule, pack, k, s, b.draw)
            if name in seen:
                raise FatalError(f"duplicate run name {name}")
            seen.add(name)
            spec = make_spec(study, aspect=aspect, feature=feature, variant=v, regime=regime,
                             rule=b.rule, pack=pack, k=k, seed=s, epochs=star, draw=b.draw,
                             out_dir=f"{RUNS_REL}/{aspect}/{feature}/{name}")
            out.append((b.name, name, spec))
            n += 1
        if n != b.expected:
            raise FatalError(f"block {b.name}: {n} specs, expected {b.expected}")
        counts[b.name] = {"n": n, "expected": b.expected}
    return out, counts


class RefusedNoEstar(RuntimeError):
    pass


# ------------------------------------------------------------------ single-factor audit (§12.2)
COORDS = ("variant", "regime", "rule", "retention_draw", "pack", "k", "seed")
Pred = Callable[[Mapping[str, Any]], bool]

#: (contrast, selects arm A, partner coordinates, declared differing fields)
AUDIT: Tuple[Tuple[str, Pred, Dict[str, Any], Set[str]], ...] = (
    ("C1 (V3 - V3c)", lambda s: s["variant"] == "V3", {"variant": "V3c"}, {"variant"}),
    ("C2/C4/C6/C7/ladder (X - V0)",
     lambda s: s["variant"] in ("V1", "V2", "V3", "V4", "V5", "V6") and s["rule"] in ("W2", "W1"),
     {"variant": "V0"}, {"variant"}),
    ("C5 (V6 - V5)", lambda s: s["variant"] == "V6", {"variant": "V5"}, {"variant"}),
    ("C3 (DGZ/WDGZrel - TP/W2)", lambda s: s["rule"] == "WDGZrel",
     {"regime": "TP", "rule": "W2"}, {"regime", "rule"}),
    ("C3_leak (WDGZrel - W2drop)", lambda s: s["rule"] == "WDGZrel", {"rule": "W2drop"}, {"rule"}),
    ("C3_col (W2drop - W2)", lambda s: s["rule"] == "W2drop", {"rule": "W2"}, {"rule"}),
    ("C3_reg / propagation control (DGZ - TP)",
     lambda s: s["regime"] == "DGZ" and s["rule"] == "W2", {"regime": "TP"}, {"regime"}),
    ("W1 - W2", lambda s: s["rule"] == "W1", {"rule": "W2"}, {"rule"}),
    ("re-draw (draw 2 - draw 1)", lambda s: s["retention_draw"] == 2, {"retention_draw": 1},
     {"retention_draw"}),
)


def _coord(s: Mapping[str, Any]) -> tuple:
    return tuple(json.dumps(s.get(c)) for c in COORDS)


def spec_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> Set[str]:
    """Differing fields, without out_dir, and without lambda_el when it is the one
    implied by a variant difference (EL arm vs non-EL arm)."""
    keys = (set(a) | set(b)) - {"out_dir"}
    diff = {k for k in keys if a.get(k) != b.get(k)}
    if "lambda_el" in diff and "variant" in diff:
        if (a["variant"] in EL_ARMS) != (b["variant"] in EL_ARMS):
            diff.discard("lambda_el")
    return diff


def audit(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The single-factor audit table; any undeclared difference or a missing
    partner is fatal."""
    by = {}
    for s in specs:
        c = _coord(s)
        if c in by:
            raise FatalError(f"two specs share the coordinates {c}")
        by[c] = s
    rows = []
    for name, pred, partner, declared in AUDIT:
        n = 0
        for a in specs:
            if not pred(a):
                continue
            want = dict(a)
            want.update(partner)
            b = by.get(_coord(want))
            if b is None:
                raise FatalError(f"audit {name}: no partner spec for {a['out_dir'].split('/')[-1]}")
            diff = spec_diff(a, b)
            if diff != declared:
                raise FatalError(f"audit {name}: {a['out_dir'].split('/')[-1]} vs "
                                 f"{b['out_dir'].split('/')[-1]} differ in {sorted(diff)}, "
                                 f"declared {sorted(declared)}")
            n += 1
        rows.append({"contrast": name, "declared": sorted(declared), "n_pairs": n})
    return rows


def materialize(study: Mapping[str, Any], aspect: str, feature: str, spec_root: Path,
                log) -> Dict[str, Any]:
    from go_function.train import validate_spec
    items, counts = build_specs(study, aspect, feature)
    for _, _, spec in items:
        validate_spec(spec, study)
    table = audit([s for _, _, s in items])
    out_dir = Path(spec_root) / aspect / feature
    entries, status = [], {"written": 0, "unchanged": 0}
    for block, name, spec in items:
        b = canonical_json(spec)
        status[write_or_verify_file(out_dir / f"{name}.json", b)] += 1
        entries.append({"name": name, "block": block, "sha256": sha256_bytes(b),
                        "seed": spec["seed"], "out_dir": spec["out_dir"]})
    set_sha = sha256_bytes("\n".join(f"{e['name']} {e['sha256']}" for e in entries).encode())
    index = {"schema": 1, "aspect": aspect, "feature": feature,
             "epochs_star": study["epochs_star"][f"{aspect}_{feature}"],
             "n_specs": len(entries), "blocks": counts, "audit": table,
             "spec_set_sha256": set_sha, "specs": entries,
             "totals_over_features": TOTALS[aspect]}
    res = write_or_verify_file(out_dir / "index.json", canonical_json(index))
    log.info("%s/%s: %d specs (%s), index %s, spec set %s", aspect, feature, len(entries),
             status, res, set_sha[:16])
    return {"n_specs": len(entries), "status": status, "index": res, "spec_set_sha256": set_sha}


def main(argv=None, *, spec_root: Path = None, study_path: Path = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aspect", choices=("bp", "mf"), required=True)
    ap.add_argument("--feature", choices=FEATURES, required=True)
    args = ap.parse_args(argv)
    log = setup_logging("materialize")
    try:
        study = json.loads(Path(study_path or STUDY).read_text())
        res = materialize(study, args.aspect, args.feature, Path(spec_root or SPEC_ROOT), log)
        print(json.dumps(res, sort_keys=True))
    except RefusedNoEstar as e:
        log.error("%s", e)
        return EXIT_REFUSED
    except FatalError as e:
        log.error("%s", e)
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
