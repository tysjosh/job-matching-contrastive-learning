"""Re-measure feature leakage before and after pruning (design 1.5, gate G1).

Per split (train, test) and grade, for the reference data and the new data:

(a) symbol mention: the partner's symbol appears whole-word (case-sensitive) in
    the other side's annotation text (``encoder_view`` minus the self prefix),
    either direction;
(b) symbol or lexicon-alias mention: (a), or any ``mask_lexicon.json`` entry
    owned by one side occurs whole-word in the other side's text;
(c) with/from hit (UniProtKB): a KEPT GAF annotation of one side (any aspect)
    names the other in column 8 through ``human_accession_map``. "Kept" uses the
    guards of the index each dataset was built with (reference: default guards;
    new: the pruned index's guards plus its ``leak_pruning`` rules). ``c_P`` is
    the same restricted to biological_process rows;
(d) ungated residual: kept BP rows whose with/from has ``HGNC:`` /
    ``ENSEMBL_GeneID:`` tokens resolving to the partner (``d_other_db``); pairs
    where either side has a kept BP row with a ``ComplexPortal:`` token
    (``d_complexportal``); descriptive complex-name mentions
    ("component of the ... complex", ``d_complex_name``).

Acceptance (fatal, exit 3): on the new test split, (a), (b) and (c) at grade 2
are each <= 0.5% and exceed the grade-1 rate by <= 0.5 percentage points.
Writes ``leak_audit.json`` and, on pass, ``append_pin("G1-leak", ...)``.
Other failures (bad inputs, hash mismatch, existing output): exit 2.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go_ppi_domain import leak_pruning as lp  # noqa: E402
from go_ppi_domain import withholding_constants as wc  # noqa: E402

logger = logging.getLogger("measure_leakage")

#: Gate thresholds (design 1.5): grade-2 rate cap and grade-2 minus grade-1 cap.
MAX_RATE = 0.005
MAX_EXCESS = 0.005
GATED_METRICS = ("a", "b", "c")
#: Reference sanity values on test (not a gate): grade 2 / 1 / 0.
REFERENCE_TEST_EXPECTED = {"a": (0.1107, 0.0017, 0.0), "c": (0.0592, 0.0041, 0.0)}
_COMPLEX_NAME = re.compile(r"\bcomponent of the\b[^.;]*?\bcomplex\b", re.IGNORECASE)
_ALNUM = re.compile(r"[A-Za-z0-9]+")
_ALNUM_FULL = re.compile(r"[A-Za-z0-9]+")


class LeakError(RuntimeError):
    """Input/consistency failure (exit 2)."""


# ------------------------------------------------------------------ inputs
def _read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_texts(proteins_jsonl: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``({protein_id: gene}, {protein_id: annotation text})``."""
    gene, text = {}, {}
    for row in _read_jsonl(proteins_jsonl):
        pid, g = row["protein_id"], row["gene"]
        gene[pid] = g
        text[pid] = lp.strip_view_prefix(g, row["encoder_view"])
    return gene, text


def gaf_rules(index_meta: dict) -> dict:
    lpm = index_meta.get("leak_pruning") or {}
    return {"excluded_evidence": frozenset(index_meta.get("excluded_evidence", ())),
            "excluded_terms": frozenset(index_meta.get("excluded_terms", ())),
            "with_from_human_protein": bool(lpm.get("with_from_human_protein", False)),
            "excluded_term_roots": tuple(lpm.get("excluded_term_roots", ()))}


def kept_annotation_refs(gaf_gz: Path, rules: dict, acc_map: Dict[str, str],
                         other_map: Dict[str, str], excluded_root_terms: Set[str],
                         taxon: str = "taxon:9606") -> dict:
    """With/from references of the GAF rows kept under ``rules`` (all aspects).

    Same row guards as ``go_ontology.parse_gaf`` (NOT, evidence, excluded terms,
    taxon, then the optional leak-pruning filters), applied to every aspect.
    """
    from go_ppi_domain.go_ontology import uniprot_accession, with_from_names_other_protein

    uni: Dict[str, Dict[str, Set[str]]] = {a: defaultdict(set) for a in "PFC"}
    other_bp: Dict[str, Set[str]] = defaultdict(set)
    complexportal_bp: Set[str] = set()
    kept = Counter()
    with gzip.open(gaf_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 15:
                continue
            sym, qual, go_id, ev, wf, asp, tax = c[2], c[3], c[4], c[6], c[7], c[8], c[12]
            if ("NOT" in qual or ev in rules["excluded_evidence"]
                    or go_id in rules["excluded_terms"] or not tax.startswith(taxon)):
                continue
            if rules["with_from_human_protein"] and with_from_names_other_protein(
                    wf, sym, acc_map):
                continue
            if go_id in excluded_root_terms:
                continue
            kept[asp] += 1
            if not wf:
                continue
            for tok in wf.replace(",", "|").split("|"):
                tok = tok.strip()
                acc = uniprot_accession(tok)
                if acc is not None:
                    s = acc_map.get(acc)
                    if s is not None and s != sym and asp in uni:
                        uni[asp][sym].add(s)
                elif asp == "P":
                    if tok.startswith("HGNC:") or tok.startswith("ENSEMBL_GeneID:"):
                        s = other_map.get(tok)
                        if s is not None and s != sym:
                            other_bp[sym].add(s)
                    elif tok.startswith("ComplexPortal:"):
                        complexportal_bp.add(sym)
    return {"uniprot": uni, "other_bp": other_bp, "complexportal_bp": complexportal_bp,
            "kept_rows_by_aspect": dict(kept)}


# ----------------------------------------------------------------- metrics
def _mentions(symbol: str, text: str, tokens: Set[str]) -> bool:
    if _ALNUM_FULL.fullmatch(symbol):
        return symbol in tokens
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(symbol) + r"(?![A-Za-z0-9])",
                     text) is not None


def measure_dataset(splits: Path, proteins_jsonl: Path, refs: dict,
                    lexicon: lp.MaskLexicon) -> Dict[str, Any]:
    gene, text = load_texts(proteins_jsonl)
    tokens = {p: set(_ALNUM.findall(t)) for p, t in text.items()}
    owned: Dict[str, Set[str]] = defaultdict(set)
    for entry, owners in lexicon.owner.items():
        for o in owners:
            owned[o].add(entry)
    in_text = {p: lp.entries_in_text(t, lexicon) for p, t in text.items()}
    complex_name = {p for p, t in text.items() if _COMPLEX_NAME.search(t)}
    uni, other, cp = refs["uniprot"], refs["other_bp"], refs["complexportal_bp"]
    out: Dict[str, Any] = {}
    for split in ("train", "test"):
        counts: Dict[str, Counter] = defaultdict(Counter)
        for rec in _read_jsonl(splits / f"{split}.jsonl"):
            m = rec["metadata"]
            a, b, g = m["protein_id"], m["partner_id"], str(m["go_grade"])
            if a not in text or b not in text:
                raise LeakError(f"{split}: record protein missing from {proteins_jsonl}: {a}, {b}")
            sa, sb = gene[a], gene[b]
            k = counts[g]
            k["n"] += 1
            hit_a = _mentions(sb, text[a], tokens[a]) or _mentions(sa, text[b], tokens[b])
            hit_b = hit_a or bool(owned.get(sb, set()) & in_text[a]) \
                or bool(owned.get(sa, set()) & in_text[b])
            hit_c_by = {asp: (sb in uni[asp].get(sa, ()) or sa in uni[asp].get(sb, ()))
                        for asp in "PFC"}
            k["a"] += hit_a
            k["b"] += hit_b
            k["c"] += any(hit_c_by.values())
            k["c_P"] += hit_c_by["P"]
            k["d_other_db"] += (sb in other.get(sa, ()) or sa in other.get(sb, ()))
            k["d_complexportal"] += (sa in cp or sb in cp)
            k["d_complex_name"] += (a in complex_name or b in complex_name)
        out[split] = {}
        for g in sorted(counts, reverse=True):
            k = counts[g]
            n = k["n"]
            row = {"n": n}
            for metric in ("a", "b", "c", "c_P", "d_other_db", "d_complexportal",
                           "d_complex_name"):
                row[metric] = k[metric] / n if n else 0.0
            row["counts"] = {m: k[m] for m in sorted(k) if m != "n"}
            out[split][g] = row
    return out


def apply_gate(new: Dict[str, Any]) -> List[str]:
    """The design 1.5 acceptance rule on the new test split."""
    test = new.get("test", {})
    g2, g1 = test.get("2"), test.get("1", {})
    if g2 is None:
        return ["new test split has no grade-2 records"]
    failures = []
    for metric in GATED_METRICS:
        r2, r1 = g2[metric], g1.get(metric, 0.0)
        if r2 > MAX_RATE:
            failures.append(f"test grade-2 {metric} rate {r2:.4%} > {MAX_RATE:.2%}")
        if r2 - r1 > MAX_EXCESS:
            failures.append(f"test grade-2 {metric} rate exceeds grade 1 by "
                            f"{(r2 - r1) * 100:.3f} pp > {MAX_EXCESS * 100:.1f} pp")
    return failures


def reference_sanity(ref: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for metric, expected in REFERENCE_TEST_EXPECTED.items():
        observed = tuple(round(ref["test"].get(g, {}).get(metric, 0.0), 4)
                         for g in ("2", "1", "0"))
        out[metric] = {"expected": list(expected), "observed": list(observed),
                       "match": all(abs(o - e) < 5e-5 for o, e in zip(observed, expected))}
    return out


def _excluded_root_terms(obo: Path, roots: Sequence[str]) -> Set[str]:
    if not roots:
        return set()
    from go_ppi_domain.go_ontology import obo_descendants, parse_obo

    parents = parse_obo(obo)[0]
    out: Set[str] = set()
    for r in roots:
        out |= {r} | set(obo_descendants(parents, r))
    return out


def run(splits: Path, converted: Path, reference_splits: Path,
        reference_converted: Path, bulk_dir: Path, out: Path) -> Tuple[int, dict]:
    from go_ppi_domain.go_ontology import GoIndex

    t0 = time.time()
    wc.assert_writable_target(out)
    if out.exists():
        raise LeakError(f"refusing to overwrite {out}")
    frozen = json.loads((splits / "frozen_hashes.json").read_text(encoding="utf-8"))
    for key, path in (("test_sha256", splits / "test.jsonl"),
                      ("train_sha256", splits / "train.jsonl"),
                      ("proteins_sha256", converted / "proteins.jsonl"),
                      ("go_index_P_sha256", converted / "go_index_P.pkl")):
        got = wc.sha256_file(path)
        if got != frozen[key]:
            raise LeakError(f"{path} sha256 {got} != frozen_hashes[{key}] {frozen[key]}")

    gaf = bulk_dir / "goa_human.gaf.gz"
    aliases, info = (bulk_dir / "9606.protein.aliases.v12.0.txt.gz",
                     bulk_dir / "9606.protein.info.v12.0.txt.gz")
    acc_map = lp.human_accession_map(gaf, aliases, info)
    other_map = lp.other_db_symbol_map(aliases, info)
    lexicon = lp.lexicon_from_json(json.loads(
        (converted / "mask_lexicon.json").read_text(encoding="utf-8")))

    datasets = {}
    rules_used = {}
    for name, sp, cv in (("reference", reference_splits, reference_converted),
                         ("new", splits, converted)):
        meta = GoIndex.load(cv / "go_index_P.pkl").meta
        rules = gaf_rules(meta)
        if name == "new" and not rules["with_from_human_protein"]:
            raise LeakError("new index meta has no leak_pruning.with_from_human_protein")
        roots_terms = _excluded_root_terms(bulk_dir / "go-basic.obo",
                                           rules["excluded_term_roots"])
        refs = kept_annotation_refs(gaf, rules, acc_map, other_map, roots_terms)
        datasets[name] = measure_dataset(sp, cv / "proteins.jsonl", refs, lexicon)
        rules_used[name] = {"excluded_evidence": sorted(rules["excluded_evidence"]),
                            "excluded_terms": sorted(rules["excluded_terms"]),
                            "with_from_human_protein": rules["with_from_human_protein"],
                            "excluded_term_roots": list(rules["excluded_term_roots"]),
                            "kept_rows_by_aspect": refs["kept_rows_by_aspect"]}
    failures = apply_gate(datasets["new"])
    audit = {"schema_version": 1, **datasets, "pass": not failures, "failures": failures,
             "gate": {"metrics": list(GATED_METRICS), "max_rate": MAX_RATE,
                      "max_excess_over_grade1": MAX_EXCESS},
             "reference_sanity": reference_sanity(datasets["reference"]),
             "gaf_rules": rules_used,
             "inputs": {"splits": wc.rel_to_root(splits), "converted": wc.rel_to_root(converted),
                        "reference_splits": wc.rel_to_root(reference_splits),
                        "reference_converted": wc.rel_to_root(reference_converted),
                        "frozen_test_sha256": frozen["test_sha256"]},
             "wall_seconds": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        for f in failures:
            logger.error("leak gate: %s", f)
        return 3, audit
    wc.append_pin("G1-leak", {"leak_audit_sha256": wc.sha256_file(out), "pass": True,
                              "failures": [], "new_test": datasets["new"]["test"]})
    return 0, audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--converted", type=Path, default=None)
    ap.add_argument("--reference-splits", type=Path, default=None)
    ap.add_argument("--reference-converted", type=Path, default=None)
    ap.add_argument("--bulk-dir", type=Path, default=ROOT / "dataset/go_ppi/bulk")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        code, audit = run(
            Path(args.splits or wc.SPLITS).resolve(),
            Path(args.converted or wc.CONVERTED).resolve(),
            Path(args.reference_splits or wc.REFERENCE_SPLITS).resolve(),
            Path(args.reference_converted or wc.REFERENCE_CONVERTED).resolve(),
            Path(args.bulk_dir).resolve(),
            Path(args.out or (Path(wc.STUDY_PRE) / "leak_audit.json")).resolve())
    except Exception as exc:  # noqa: BLE001
        logger.error("measure_leakage failed: %s: %s", type(exc).__name__, exc)
        return 2
    summary = {ds: {sp: {g: {m: round(v[m], 5) for m in ("n", "a", "b", "c", "c_P")}
                         for g, v in audit[ds][sp].items()}
                    for sp in audit[ds]} for ds in ("reference", "new")}
    print(json.dumps({"pass": audit["pass"], "failures": audit["failures"],
                      "rates": summary, "reference_sanity": audit["reference_sanity"],
                      "wall_seconds": audit["wall_seconds"]}, indent=1, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
