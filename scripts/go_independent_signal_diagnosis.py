#!/usr/bin/env python3
"""Validate GO biological-process similarity on independent PPI benchmarks.

HuRI and Lit-BM use Ensembl gene IDs. STRING's alias table maps those IDs to
the preferred gene symbols used by GOA. Negatives are non-edges matched to the
positive endpoint on both network-degree and GO-annotation-count deciles.

The GO guards match ``go_bulk_signal_diagnosis.py``: biological_process only,
with IEA, IPI, ND, protein binding, and binding excluded.
"""

import argparse
import gzip
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "dataset" / "go_ppi" / "bulk"
INDEPENDENT = ROOT / "dataset" / "go_ppi" / "independent"


def load_bulk_module():
    path = ROOT / "scripts" / "go_bulk_signal_diagnosis.py"
    spec = importlib.util.spec_from_file_location("go_bulk_signal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_go(go):
    parents, namespace, alt_to_main, _ = go.parse_obo(BULK / "go-basic.obo")
    keep = {term for term, ns in namespace.items() if ns == "biological_process"}
    ancestors = go.build_ancestors(parents, keep)
    direct, gaf_stats = go.parse_gaf(
        BULK / "goa_human.gaf.gz",
        "P",
        go.DEFAULT_EXCLUDED_EVIDENCE,
        go.BLACKLIST_TERMS,
    )
    gene_terms, unmapped = go.propagate(direct, ancestors, alt_to_main)
    ic, _, _ = go.information_content(gene_terms)
    return gene_terms, ic, dict(gaf_stats), len(unmapped)


def load_ensembl_gene_map(gene_terms):
    protein_to_symbol = {}
    with gzip.open(BULK / "9606.protein.info.v12.0.txt.gz", "rt") as handle:
        next(handle)
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            protein_to_symbol[columns[0]] = columns[1]

    ensembl_to_symbols = defaultdict(set)
    with gzip.open(BULK / "9606.protein.aliases.v12.0.txt.gz", "rt") as handle:
        next(handle)
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 3 or columns[2] != "Ensembl_gene":
                continue
            ensembl_gene = columns[1]
            symbol = protein_to_symbol.get(columns[0])
            if ensembl_gene.startswith("ENSG") and symbol in gene_terms:
                ensembl_to_symbols[ensembl_gene].add(symbol)

    # A few Ensembl genes map through multiple protein records. Prefer the
    # symbol with the richest usable GO closure, deterministically on ties.
    return {
        gene: sorted(symbols, key=lambda s: (-len(gene_terms[s]), s))[0]
        for gene, symbols in ensembl_to_symbols.items()
    }


def decile_bins(genes, values):
    cuts = np.quantile(np.asarray(values), np.linspace(0.0, 1.0, 11))
    return {
        gene: int(min(9, np.searchsorted(cuts[1:-1], value, side="right")))
        for gene, value in zip(genes, values)
    }


def bootstrap_auc(go, positives, negatives, seed, repetitions):
    rng = np.random.default_rng(seed)
    positives = np.asarray(positives)
    negatives = np.asarray(negatives)
    estimates = []
    for _ in range(repetitions):
        pos = positives[rng.integers(0, len(positives), len(positives))].tolist()
        neg = negatives[rng.integers(0, len(negatives), len(negatives))].tolist()
        estimates.append(go.rank_auc(pos, neg))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def audit_benchmark(go, path, ensembl_to_symbol, gene_terms, ic, seed, bootstraps):
    raw_edges = []
    raw_genes = set()
    mapped_edges = []
    with path.open() as handle:
        for line in handle:
            left, right = line.rstrip("\n").split("\t")[:2]
            raw_edges.append((left, right))
            raw_genes.update((left, right))
            a = ensembl_to_symbol.get(left)
            b = ensembl_to_symbol.get(right)
            if a and b and a != b:
                mapped_edges.append(tuple(sorted((a, b))))

    raw_unique = {tuple(sorted(edge)) for edge in raw_edges if edge[0] != edge[1]}
    edges = sorted(set(mapped_edges))
    edge_set = set(edges)
    degree = Counter(gene for edge in edges for gene in edge)
    genes = sorted(degree)
    degree_bin = decile_bins(genes, [degree[gene] for gene in genes])
    annotation_bin = decile_bins(genes, [len(gene_terms[gene]) for gene in genes])

    pools = defaultdict(list)
    for gene in genes:
        pools[(degree_bin[gene], annotation_bin[gene])].append(gene)

    rng = random.Random(seed)
    positives = []
    negatives = []
    unmatched = 0
    for a, b in edges:
        if rng.random() < 0.5:
            a, b = b, a
        replacement = None
        for _ in range(200):
            candidate = rng.choice(pools[(degree_bin[b], annotation_bin[b])])
            if candidate != a and tuple(sorted((a, candidate))) not in edge_set:
                replacement = candidate
                break
        if replacement is None:
            unmatched += 1
            continue
        positive = go.sim_gic(gene_terms[a], gene_terms[b], ic)
        negative = go.sim_gic(gene_terms[a], gene_terms[replacement], ic)
        if positive is not None and negative is not None:
            positives.append(positive)
            negatives.append(negative)

    return {
        "benchmark": path.stem,
        "raw_edges": len(raw_edges),
        "raw_unique_nonself_edges": len(raw_unique),
        "raw_genes": len(raw_genes),
        "mapped_unique_edges": len(edges),
        "mapped_genes": len(genes),
        "mapping_edge_coverage": len(edges) / len(raw_unique),
        "n_scored_per_class": len(positives),
        "unmatched": unmatched,
        "positive_mean": float(np.mean(positives)),
        "matched_nonedge_mean": float(np.mean(negatives)),
        "rank_auc": go.rank_auc(positives, negatives),
        "bootstrap_ci95": bootstrap_auc(go, positives, negatives, seed, bootstraps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "ontology_ceiling" / "go_independent_signal.json",
    )
    args = parser.parse_args()

    go = load_bulk_module()
    gene_terms, ic, gaf_stats, unmapped_terms = load_go(go)
    ensembl_to_symbol = load_ensembl_gene_map(gene_terms)
    results = {
        "method": {
            "ontology": "GO biological_process",
            "similarity": "simGIC",
            "excluded_evidence": sorted(go.DEFAULT_EXCLUDED_EVIDENCE),
            "blacklisted_terms": sorted(go.BLACKLIST_TERMS),
            "negative_control": "non-edge matched on endpoint degree and GO-closure-size deciles",
            "seed": args.seed,
            "bootstrap_repetitions": args.bootstraps,
        },
        "goa": {
            "stats": gaf_stats,
            "unmapped_terms": unmapped_terms,
            "mapped_ensembl_genes": len(ensembl_to_symbol),
        },
        "benchmarks": [
            audit_benchmark(
                go,
                INDEPENDENT / f"{name}.tsv",
                ensembl_to_symbol,
                gene_terms,
                ic,
                args.seed,
                args.bootstraps,
            )
            for name in ("HuRI", "Lit-BM")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
