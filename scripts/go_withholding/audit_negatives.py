"""Offline audit of the negatives each cell's production selector draws (design 7).

    audit_negatives.py --cond <cond_id> --cells baseline loss_only neg_only both \\
        --seed 13 --epochs 0 7 14 [--anchors all|N]

Descriptive audit; the binding identity check is the run-time recorder (design
4.6). Per cell: the cell config with ``go_ppi_split_dir = <cond dir>`` and
``training_seed = --seed`` (the configs carry 42 and ``from_config`` reads
``config.training_seed``), the pruned-index preflight of design 4.2 step 2, the
production selector ``GoPpiNegativeSelector.from_config(cfg, "train", matcher=...)``
(asserted to hold ``training_seed == --seed``), and one ``select_batch_negatives``
call per train-record anchor and epoch. A separate measurement matcher
``GoMatcher(GoIndex.load(pruned), "simgic")`` scores simGIC; HuRI and Lit-BM are
mapped ENSG -> symbol via STRING ``Ensembl_gene`` aliases for the
independent-evidence false-negative rate.

Fatal (exit 2): index preflight; selector seed != ``--seed``; any withheld (W)
protein among the negatives of a pack condition (``share_withheld_negatives``
must be exactly 0); the share invariant ``target_annotated == withheld +
retained`` (pack conditions, 1e-12); offline identity baseline == loss_only and
neg_only == both (when both cells are audited).

Output: ``results/go_withholding/audits/neg__<cond>__<cell>__s<seed>.json``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import withholding as wh  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("audit_negatives")

BULK = ROOT / "dataset/go_ppi/bulk"
INDEPENDENT = ROOT / "dataset/go_ppi/independent"
DEFAULT_EPOCHS = (0, 7, 14)
INVARIANT_TOL = 1e-12
IDENTITY_PAIRS = (("baseline", "loss_only"), ("neg_only", "both"))


class AuditError(RuntimeError):
    pass


# --------------------------------------------------------- independent PPI
def ensg_to_symbols(aliases_gz: Path, info_gz: Path) -> Dict[str, Set[str]]:
    """ENSG -> symbols via STRING ``Ensembl_gene`` aliases and ``protein.info`` names."""
    from go_ppi_domain.leak_pruning import _ensp_to_symbol, _iter_aliases

    ensp2sym = _ensp_to_symbol(Path(info_gz))
    out: Dict[str, Set[str]] = {}
    for ensp, alias, sources in _iter_aliases(Path(aliases_gz)):
        if "Ensembl_gene" in sources and alias.startswith("ENSG") and ensp in ensp2sym:
            out.setdefault(alias, set()).add(ensp2sym[ensp])
    return out


def load_independent_pairs(paths: Iterable[Path], ensg2sym: Dict[str, Set[str]]
                           ) -> Set[FrozenSet[str]]:
    """Undirected symbol pairs from ENSG-pair TSVs (HuRI, Lit-BM); unmapped rows skipped."""
    pairs: Set[FrozenSet[str]] = set()
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) < 2 or not c[0].startswith("ENSG"):
                    continue
                for a in ensg2sym.get(c[0].split(".")[0], ()):
                    for b in ensg2sym.get(c[1].split(".")[0], ()):
                        if a != b:
                            pairs.add(frozenset((a, b)))
    return pairs


# ----------------------------------------------------------------- selection
def record_anchors(train_file: Path) -> List[Tuple[str, str]]:
    """Distinct train-record anchors (sorted) with their first positive partner."""
    first: Dict[str, str] = {}
    with open(train_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            a = str(rec["resume"]["protein_id"])
            first.setdefault(a, str(rec["job"]["partner_id"]))
    return sorted(first.items())


def _sample(anchor: str, positive: str) -> SimpleNamespace:
    return SimpleNamespace(metadata={"resume_id": anchor, "protein_id": anchor},
                           resume={"protein_id": anchor, "skill_uris": [], "coarse_uris": []},
                           job={"partner_id": positive, "skill_uris": [], "coarse_uris": []})


def build_cell_selector(cfg, seed: int):
    """The production selector for a cell config (design 7), seed asserted."""
    from go_ppi_domain.negative_selector import GoPpiNegativeSelector
    from go_ppi_domain.run_config import build_go_matcher

    cfg.training_seed = int(seed)
    matcher = build_go_matcher(cfg) if bool(cfg.go_ppi_go_tiered_negatives) else None
    selector = GoPpiNegativeSelector.from_config(cfg, "train", matcher=matcher)
    if selector.training_seed != int(seed):
        raise AuditError(f"selector.training_seed {selector.training_seed} != --seed {seed}")
    return selector


def target_sets(table: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Neighbour, target-annotated, withheld and retained protein sets of a condition."""
    by_id = {t["go_id"]: t for t in table["targets"]}
    if manifest["pack_id"] == "all":
        targets = list(table["targets"])
    else:
        targets = [by_id[mt["go_id"]] for mt in manifest["targets"]]
    nbr: Set[str] = set()
    annotated: Set[str] = set()
    for t in targets:
        nbr |= set(t.get("nbr_train_proteins_sib", []))
        annotated |= set(t["train_annotated_proteins"])
    retained: Set[str] = set()
    for mt in manifest.get("targets", []):
        retained |= set(mt.get("retained", []))
    return {"nbr": nbr, "annotated": annotated,
            "withheld": set(manifest.get("withheld_proteins", [])), "retained": retained}


def _share(n: int, d: int) -> float:
    return n / d if d else 0.0


def audit_cell(selector, anchors: List[Tuple[str, str]], epochs: Sequence[int],
               max_negatives: int, sets: Dict[str, Set[str]], measure,
               indep: Set[FrozenSet[str]], pack_condition: bool) -> Dict[str, Any]:
    """Per-epoch shares and the offline selection hash for one cell."""
    lines: List[str] = []
    by_epoch: Dict[str, Any] = {}
    nbr_anchor_share = _share(sum(a in sets["nbr"] for a, _ in anchors), len(anchors))
    for e in epochs:
        negs: List[Tuple[str, str, int]] = []
        none = 0
        for anchor, pos in anchors:
            out = selector.select_batch_negatives(_sample(anchor, pos), [], max_negatives, e)
            if out is None:
                none += 1
                continue
            chosen = out[0]
            lines.append(f"{e}\t{anchor}\t{pos}\t"
                         + ",".join(f"{v['partner_id']}:{v['grade']}" for v in chosen))
            negs.extend((anchor, str(v["partner_id"]), int(v["grade"])) for v in chosen)
        n = len(negs)
        grades = {"1": [x for x in negs if x[2] == 1], "0": [x for x in negs if x[2] == 0]}
        sims = {g: [measure(a, b) for a, b, _ in xs] for g, xs in grades.items()}
        allsims = sims["1"] + sims["0"]
        counts = {k: sum(b in sets[k] for _, b, _ in negs)
                  for k in ("nbr", "withheld", "annotated", "retained")}
        row = {
            "n": n, "none_returns": none,
            "grade_mix": {g: _share(len(xs), n) for g, xs in grades.items()},
            "simgic_mean": {"1": statistics.fmean(sims["1"]) if sims["1"] else None,
                            "0": statistics.fmean(sims["0"]) if sims["0"] else None,
                            "all": statistics.fmean(allsims) if allsims else None},
            "indep_fn_rate": {g: _share(sum(frozenset((a, b)) in indep for a, b, _ in xs),
                                        len(xs)) for g, xs in grades.items()},
            "share_target_neighbour_negatives": _share(counts["nbr"], n),
            "share_target_neighbour_anchors": nbr_anchor_share,
            "share_withheld_negatives": _share(counts["withheld"], n),
            "share_target_annotated_negatives": _share(counts["annotated"], n),
            "share_retained_negatives": _share(counts["retained"], n),
            "counts": counts,
        }
        if pack_condition:
            if counts["withheld"] != 0:
                bad = sorted({b for _, b, _ in negs if b in sets["withheld"]})[:5]
                raise AuditError(f"epoch {e}: {counts['withheld']} withheld protein(s) drawn "
                                 f"as negatives (first ids: {bad})")
            lhs = row["share_target_annotated_negatives"]
            rhs = row["share_withheld_negatives"] + row["share_retained_negatives"]
            if abs(lhs - rhs) > INVARIANT_TOL:
                raise AuditError(f"epoch {e}: share invariant broken: target_annotated "
                                 f"{lhs!r} != withheld + retained {rhs!r}")
        by_epoch[str(e)] = row
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return {"anchors": len(anchors), "selection_sha256_offline": digest, "by_epoch": by_epoch}


# ---------------------------------------------------------------------- run
def run(cond_id: str, cells: Sequence[str], seed: int, epochs: Sequence[int],
        anchors_arg: str, independent: Optional[Sequence[Path]], aliases: Path,
        info: Path, out_dir: Path) -> Dict[str, Any]:
    from go_ppi_domain.go_ontology import GoIndex, GoMatcher
    from scripts.run_learning_curve_point import preflight_index

    cond_dir = Path(wc.CONDITIONS) / cond_id
    manifest = json.loads((cond_dir / wh.MANIFEST).read_text(encoding="utf-8"))
    if manifest["cond_id"] != cond_id:
        raise AuditError(f"manifest cond_id {manifest['cond_id']!r} != {cond_id!r}")
    table = json.loads((Path(wc.TARGETS) / "target_table.json").read_text(encoding="utf-8"))
    frozen = json.loads((Path(wc.SPLITS) / "frozen_hashes.json").read_text(encoding="utf-8"))
    sets = target_sets(table, manifest)
    pack_condition = manifest["pack_id"] != "all"
    anchors = record_anchors(cond_dir / "train.jsonl")
    if anchors_arg != "all":
        anchors = anchors[:int(anchors_arg)]
    indep: Set[FrozenSet[str]] = set()
    if independent:
        indep = load_independent_pairs(independent, ensg_to_symbols(aliases, info))

    results: Dict[str, Dict[str, Any]] = {}
    measure_matcher = None
    for cell in cells:
        cfg = wc.load_study_config(Path(wc.CONFIG_DIR) / f"{cell}.json")
        cfg.go_ppi_split_dir = str(cond_dir)
        try:
            preflight_index(cfg, frozen)
        except RuntimeError as exc:
            raise AuditError(f"{cell}: {exc}") from exc
        if measure_matcher is None:
            measure_matcher = GoMatcher(GoIndex.load(cfg.go_index_cache),
                                        similarity_mode="simgic")

            def measure(a: str, b: str, _m=measure_matcher) -> float:
                return float(_m.ontology_set_similarity(sorted(_m.terms_for_gene(a)),
                                                        sorted(_m.terms_for_gene(b))))
        selector = build_cell_selector(cfg, seed)
        res = audit_cell(selector, anchors, epochs, int(cfg.max_negatives_per_anchor), sets,
                         measure, indep, pack_condition)
        res = {"cond_id": cond_id, "cell": cell, "seed": int(seed), **res,
               "selector_training_seed": selector.training_seed,
               "independent_pairs": len(indep)}
        results[cell] = res
        out_dir.mkdir(parents=True, exist_ok=True)
        wc.assert_writable_target(out_dir)
        (out_dir / f"neg__{cond_id}__{cell}__s{seed}.json").write_text(
            json.dumps(wc.relativize_json(res), indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
    identity = {}
    for a, b in IDENTITY_PAIRS:
        if a in results and b in results:
            same = (results[a]["selection_sha256_offline"]
                    == results[b]["selection_sha256_offline"])
            identity[f"{a}=={b}"] = same
            if not same:
                raise AuditError(f"offline identity {a} == {b} fails for {cond_id} s{seed}")
    return {"cond_id": cond_id, "seed": seed, "cells": list(results), "identity": identity,
            "anchors": len(anchors)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cond", required=True)
    ap.add_argument("--cells", nargs="+", choices=wc.CELLS, default=list(wc.CELLS))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, nargs="+", default=list(DEFAULT_EPOCHS))
    ap.add_argument("--anchors", default="all", help="'all' or the first N sorted anchors")
    ap.add_argument("--independent", type=Path, nargs="*",
                    default=[INDEPENDENT / "HuRI.tsv", INDEPENDENT / "Lit-BM.tsv"],
                    help="ENSG-pair TSVs; pass no value to skip the FN-rate mapping")
    ap.add_argument("--aliases", type=Path, default=BULK / "9606.protein.aliases.v12.0.txt.gz")
    ap.add_argument("--info", type=Path, default=BULK / "9606.protein.info.v12.0.txt.gz")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.anchors != "all" and not (args.anchors.isdigit() and int(args.anchors) > 0):
        ap.error("--anchors must be 'all' or a positive integer")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    try:
        summary = run(args.cond, args.cells, args.seed, args.epochs, args.anchors,
                      args.independent, args.aliases, args.info,
                      Path(args.out_dir or (Path(wc.STUDY_RES) / "audits")))
    except Exception as exc:  # noqa: BLE001 - every audit failure is fatal for the grid
        logger.error("audit_negatives failed: %s: %s", type(exc).__name__, exc)
        return 2
    summary["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
