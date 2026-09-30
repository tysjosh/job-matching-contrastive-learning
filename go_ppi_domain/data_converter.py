"""Convert STRING + GOA bulk files into protein view records and graded pairs.

Emits, into ``go_ppi_converted_dir``, the direct analogues of trials'
``topics.jsonl`` / ``trials.jsonl`` / ``qrels.jsonl``:

    proteins.jsonl  {protein_id, gene, encoder_view, go_uris, coarse_uris, n_terms}
    pairs.jsonl     {anchor_id, partner_id, grade, grade_name, experimental}

GRADES -- and why they are absolute, not tertiles
-------------------------------------------------
    grade 2  experimental >= 700          established interaction     positive
    grade 1  0 < experimental < 150       weak / ambiguous evidence   hard negative
    grade 0  no STRING edge at all        no interaction recorded     easy negative

Pairs with ``150 <= experimental < 700`` are DROPPED, not assigned a grade. They
are the ambiguous middle, and including them in either tier would blur the one
contrast this dataset exists to test.

The absolute cut points are not a free choice. STRING's experimental scores are
severely right-skewed -- of 5.85M nonzero scores the median is 102 and the 90th
percentile 292, while STRING itself calls 400 "medium confidence" and 700 "high".
Splitting interacting pairs into confidence *tertiles* therefore cuts at ~84 and
~134, so a "high vs medium" contrast compares weak evidence against
slightly-less-weak evidence. That tracks how much assay attention a protein pair
has received -- study bias -- not whether the interaction is real, and GO has no
reason to predict it. Measured that way the GO ceiling reads 0.5284; under these
absolute bands the identical data gives 0.8195. The grading here is the one the
ceiling was measured with, so the ceiling describes the signal training sees.

ENCODER VIEW
------------
STRING's ``protein.info`` annotation column, prefixed with the gene symbol:

    "TP53 — Cellular tumor antigen p53; Acts as a tumor suppressor in many
     tumor types; induces growth arrest or apoptosis depending on ..."

Mean 284 characters over 19,699 human proteins, with 2,171 shorter than 50
characters; those are dropped by ``--min-view-chars``, since a protein whose only
text is its symbol gives the encoder nothing to learn from and would make the
task partly an entity-memorisation problem.

LABEL INDEPENDENCE
------------------
The ``experimental`` channel is used alone, never ``combined_score``. STRING's
combined score fuses a ``database`` channel (curated pathways) and a
``textmining`` channel, both of which absorb the same co-annotation evidence GO
encodes, so grading by combined score and then scoring GO similarity against
those grades is circular. Measured cost of that shortcut on the hard contrast:
+0.03 to +0.06 AUC of pure artefact.

Usage
    .venv/bin/python3 -m go_ppi_domain.data_converter \
        --bulk-dir dataset/go_ppi/bulk --output-dir preprocess/go_ppi
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

GRADE_HIGH = 2
GRADE_WEAK = 1
GRADE_NONE = 0

GRADE_NAMES = {
    GRADE_HIGH: "high_confidence",
    GRADE_WEAK: "weak_evidence",
    GRADE_NONE: "no_interaction",
}

#: Career ordinal vocabulary, so run_ordinal_evaluation.py works unchanged.
ORDINAL_LABEL = {
    GRADE_HIGH: "good_fit",
    GRADE_WEAK: "potential_fit",
    GRADE_NONE: "no_fit",
}

BINARY_LABEL = {GRADE_HIGH: 1, GRADE_WEAK: 0, GRADE_NONE: 0}

DEFAULT_HIGH_CUT = 700
DEFAULT_WEAK_CUT = 150
DEFAULT_MIN_VIEW_CHARS = 50


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def load_protein_info(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (ensp -> gene symbol, gene symbol -> annotation text)."""
    ensp2gene: Dict[str, str] = {}
    gene2text: Dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            ensp, gene, _size, annotation = c[0], c[1], c[2], c[3]
            if not gene:
                continue
            ensp2gene[ensp] = gene
            # Keep the longest annotation when a gene maps from several proteins.
            if len(annotation) > len(gene2text.get(gene, "")):
                gene2text[gene] = annotation
    return ensp2gene, gene2text


def iter_links(path: Path, ensp2gene: Dict[str, str]) -> Iterator[Tuple[str, str, int]]:
    """Yield ``(gene_a, gene_b, experimental)`` for each undirected STRING edge.

    STRING lists both directions; the pair is normalised so a downstream ``set``
    deduplicates correctly.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = next(fh).split()
        col = None
        for cand in ("experimental", "experiments"):
            if cand in header:
                col = header.index(cand)
                break
        if col is None:
            raise SystemExit(
                f"no experimental-evidence column in {path.name}; header={header}")
        for line in fh:
            c = line.split()
            if len(c) <= col:
                continue
            a = ensp2gene.get(c[0])
            b = ensp2gene.get(c[1])
            if not a or not b or a == b:
                continue
            if a > b:
                a, b = b, a
            yield a, b, int(c[col])


def build(
    bulk_dir: Path,
    output_dir: Path,
    aspect: str = "P",
    high_cut: int = DEFAULT_HIGH_CUT,
    weak_cut: int = DEFAULT_WEAK_CUT,
    min_view_chars: int = DEFAULT_MIN_VIEW_CHARS,
    max_anchors: int = 3000,
    positives_per_anchor: int = 4,
    weak_pool_cap: int = 40,
    none_pool_size: int = 40,
    min_weak_per_anchor: int = 4,
    go_index_cache: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Emit ``proteins.jsonl`` and ``pairs.jsonl``, returning a manifest."""
    from go_ppi_domain.go_ontology import GoIndex

    rng = random.Random(seed)
    info_path = bulk_dir / "9606.protein.info.v12.0.txt.gz"
    links_path = bulk_dir / "9606.protein.links.detailed.v12.0.txt.gz"
    obo_path = bulk_dir / "go-basic.obo"
    gaf_path = bulk_dir / "goa_human.gaf.gz"
    for p in (info_path, links_path, obo_path, gaf_path):
        if not p.exists():
            raise SystemExit(
                f"missing {p}. See dataset/go_ppi/README.md for the fetch commands.")

    cache = go_index_cache or (output_dir / f"go_index_{aspect}.pkl")
    index = GoIndex.build_or_load(obo_path, gaf_path, cache, aspect=aspect)

    ensp2gene, gene2text = load_protein_info(info_path)
    logger.info("STRING info: %d proteins, %d gene annotations",
                len(ensp2gene), len(gene2text))

    # Eligible = has GO annotation AND substantive text.
    eligible = {
        g for g in index.closure
        if len(gene2text.get(g, "")) >= min_view_chars
    }
    logger.info("eligible genes (GO + >=%d chars of text): %d",
                min_view_chars, len(eligible))
    if len(eligible) < 500:
        raise SystemExit("too few eligible genes; check the bulk inputs")

    high: Dict[str, List[str]] = defaultdict(list)
    weak: Dict[str, List[str]] = defaultdict(list)
    any_edge: Dict[str, set] = defaultdict(set)
    n_edges = n_mid = 0
    for a, b, exp in iter_links(links_path, ensp2gene):
        if a not in eligible or b not in eligible:
            continue
        n_edges += 1
        any_edge[a].add(b)
        any_edge[b].add(a)
        if exp >= high_cut:
            high[a].append(b)
            high[b].append(a)
        elif 0 < exp < weak_cut:
            weak[a].append(b)
            weak[b].append(a)
        else:
            n_mid += 1
    logger.info(
        "edges among eligible genes: %d (%d dropped as the ambiguous middle "
        "%d<=exp<%d)", n_edges, n_mid, weak_cut, high_cut)

    # Anchors need positives AND a hard pool: an anchor with no grade-1 partners
    # can only ever receive easy negatives, which is exactly the degenerate case
    # this dataset exists to avoid.
    candidates = sorted(
        g for g in eligible
        if len(set(high.get(g, ()))) >= 1
        and len(set(weak.get(g, ()))) >= min_weak_per_anchor
    )
    logger.info("anchor candidates (>=1 grade-2 and >=%d grade-1 partners): %d",
                min_weak_per_anchor, len(candidates))
    if not candidates:
        raise SystemExit("no anchor satisfies the pool requirements")
    rng.shuffle(candidates)
    anchors = sorted(candidates[:max_anchors])

    # ---- protein view records -------------------------------------------
    needed = set(anchors)
    for a in anchors:
        needed.update(set(high.get(a, ())))
        needed.update(set(weak.get(a, ())))
    # Grade-0 partners are drawn from the eligible pool at split time; index the
    # whole eligible set so the splitter can choose freely.
    needed |= eligible

    proteins = []
    for g in sorted(needed):
        text = gene2text.get(g, "")
        if len(text) < min_view_chars:
            continue
        proteins.append({
            "protein_id": g,
            "gene": g,
            "encoder_view": f"{g} — {text}",
            # Direct terms; the matcher propagates. Storing the closure here would
            # multiply the file size ~7x for no gain.
            "go_uris": sorted(index.terms_for(g)),
            "coarse_uris": sorted(index.coarse_for(g)),
            "n_terms": len(index.terms_for(g)),
        })
    view_ids = {p["protein_id"] for p in proteins}

    # ---- graded pairs ----------------------------------------------------
    pairs: List[Dict[str, Any]] = []
    for a in anchors:
        pos = sorted(set(high.get(a, ())) & view_ids)
        if not pos:
            continue
        if len(pos) > positives_per_anchor:
            sub = random.Random(f"{seed}|pos|{a}")
            pos = sorted(sub.sample(pos, positives_per_anchor))
        for b in pos:
            pairs.append({"anchor_id": a, "partner_id": b,
                          "grade": GRADE_HIGH,
                          "grade_name": GRADE_NAMES[GRADE_HIGH]})

        wk = sorted((set(weak.get(a, ())) & view_ids) - set(pos))
        if len(wk) > weak_pool_cap:
            sub = random.Random(f"{seed}|weak|{a}")
            wk = sorted(sub.sample(wk, weak_pool_cap))
        for b in wk:
            pairs.append({"anchor_id": a, "partner_id": b,
                          "grade": GRADE_WEAK,
                          "grade_name": GRADE_NAMES[GRADE_WEAK]})

        # Grade 0: sampled from proteins with NO recorded edge to the anchor.
        # Excluding every edge (not just high/weak) matters -- a pair sitting in
        # the dropped 150-700 band is not a safe negative.
        forbidden = any_edge.get(a, set()) | {a}
        pool = list(view_ids - forbidden)
        sub = random.Random(f"{seed}|none|{a}")
        sub.shuffle(pool)
        for b in pool[:none_pool_size]:
            pairs.append({"anchor_id": a, "partner_id": b,
                          "grade": GRADE_NONE,
                          "grade_name": GRADE_NAMES[GRADE_NONE]})

    n_proteins = _write_jsonl(output_dir / "proteins.jsonl", proteins)
    n_pairs = _write_jsonl(output_dir / "pairs.jsonl", pairs)

    by_grade = defaultdict(int)
    for p in pairs:
        by_grade[p["grade_name"]] += 1
    anchors_used = sorted({p["anchor_id"] for p in pairs})

    manifest = {
        "aspect": aspect,
        "grade_definition": {
            "2": f"STRING experimental >= {high_cut} (high confidence)",
            "1": f"0 < STRING experimental < {weak_cut} (weak evidence)",
            "0": "no STRING edge recorded",
            "dropped": f"{weak_cut} <= experimental < {high_cut} (ambiguous middle)",
        },
        "label_channel": "STRING experimental (GO-independent; combined_score is circular)",
        "seed": seed,
        "min_view_chars": min_view_chars,
        "eligible_genes": len(eligible),
        "anchor_candidates": len(candidates),
        "anchors": len(anchors_used),
        "proteins": n_proteins,
        "pairs": n_pairs,
        "pairs_by_grade": dict(by_grade),
        "positives_per_anchor": positives_per_anchor,
        "weak_pool_cap": weak_pool_cap,
        "none_pool_size": none_pool_size,
        "min_weak_per_anchor": min_weak_per_anchor,
        "go_index": index.meta,
        "go_index_cache": str(cache),
    }
    (output_dir / "convert_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("converted: %d proteins, %d pairs %s",
                n_proteins, n_pairs, dict(by_grade))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bulk-dir", type=Path, default=Path("dataset/go_ppi/bulk"))
    ap.add_argument("--output-dir", type=Path, default=Path("preprocess/go_ppi"))
    ap.add_argument("--aspect", default="P", choices=["P", "F", "C", "A"])
    ap.add_argument("--high-cut", type=int, default=DEFAULT_HIGH_CUT)
    ap.add_argument("--weak-cut", type=int, default=DEFAULT_WEAK_CUT)
    ap.add_argument("--min-view-chars", type=int, default=DEFAULT_MIN_VIEW_CHARS)
    ap.add_argument("--max-anchors", type=int, default=3000)
    ap.add_argument("--positives-per-anchor", type=int, default=4)
    ap.add_argument("--weak-pool-cap", type=int, default=40)
    ap.add_argument("--none-pool-size", type=int, default=40)
    ap.add_argument("--min-weak-per-anchor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build(
        args.bulk_dir, args.output_dir, aspect=args.aspect,
        high_cut=args.high_cut, weak_cut=args.weak_cut,
        min_view_chars=args.min_view_chars, max_anchors=args.max_anchors,
        positives_per_anchor=args.positives_per_anchor,
        weak_pool_cap=args.weak_pool_cap, none_pool_size=args.none_pool_size,
        min_weak_per_anchor=args.min_weak_per_anchor, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
