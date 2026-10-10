"""Synthetic raw GO/PPI inputs for the leak-pruning and G1 build tests (no real data).

``write_obo`` / ``write_gaf`` give a tiny ontology and annotation file;
``write_bulk_dir`` adds STRING-format ``protein.info`` / ``protein.aliases``
files; ``write_reference`` writes a reference ``converted`` dir (proteins,
pairs, manifest) and reference splits produced by the real splitter, the same
way ``preprocess/go_ppi{,_splits}`` were made.
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Dict, List

OBO = """format-version: 1.2

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000001
name: process one
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0000002
name: process two
namespace: biological_process
is_a: GO:0000001 ! process one

[Term]
id: GO:0000003
name: process three
namespace: biological_process
is_a: GO:0008150 ! biological_process
relationship: regulates GO:0000001 ! process one

[Term]
id: GO:0000004
name: process four
namespace: biological_process
relationship: part_of GO:0000003 ! process three

[Term]
id: GO:0000005
name: process five
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0000099
name: obsolete process
namespace: biological_process
is_obsolete: true

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0005515
name: protein binding
namespace: molecular_function
is_a: GO:0005488 ! binding

[Term]
id: GO:0000010
name: kinase binding
namespace: molecular_function
is_a: GO:0005515 ! protein binding

[Term]
id: GO:0000011
name: other function
namespace: molecular_function
is_a: GO:0003674 ! molecular_function
"""

BP_TERMS = ("GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004", "GO:0000005")


def write_obo(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OBO, encoding="utf-8")
    return path


def gaf_line(symbol: str, acc: str, go_id: str, evidence: str = "IDA", with_from: str = "",
             aspect: str = "P", qualifier: str = "involved_in",
             taxon: str = "taxon:9606", db: str = "UniProtKB") -> str:
    cols = [db, acc, symbol, qualifier, go_id, "PMID:1", evidence, with_from, aspect,
            f"{symbol} protein", "", "protein", taxon, "20240101", "UniProt", "", ""]
    return "\t".join(cols) + "\n"


def write_gaf(path: Path, lines: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        fh.write("!gaf-version: 2.2\n")
        fh.writelines(lines)
    return path


def _gz_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


def acc_of(i: int) -> str:
    return f"Q{i:05d}"


def sym_of(i: int) -> str:
    return f"PRT{i:02d}"


N_PROTEINS = 60


def write_bulk_dir(bulk: Path, n: int = N_PROTEINS, seed: int = 7) -> Dict[str, Path]:
    """OBO, GAF and STRING info/aliases for ``n`` proteins PRT00..; deterministic.

    Every protein gets 1-2 BP annotations. Every 4th protein has one annotation
    whose with/from names another human protein (dropped by pruning) and one
    self-referencing with/from (kept). Aliases: ``ALT<i>`` (UniProt_GN_Synonyms),
    plus a common alias ``CDS`` on every protein (dropped by the doc-frequency
    filter only if it occurs in texts; see ``write_reference``).
    """
    rnd = random.Random(seed)
    bulk.mkdir(parents=True, exist_ok=True)
    obo = write_obo(bulk / "go-basic.obo")
    lines = []
    for i in range(n):
        s, a = sym_of(i), acc_of(i)
        terms = rnd.sample(BP_TERMS, rnd.randint(1, 2))
        for t in terms:
            lines.append(gaf_line(s, a, t))
        if i % 4 == 0:
            other = (i + 1) % n
            lines.append(gaf_line(s, a, "GO:0000005", "IBA", f"UniProtKB:{acc_of(other)}-2"))
            lines.append(gaf_line(s, a, "GO:0000003", "ISS", f"UniProtKB:{a}"))
        lines.append(gaf_line(s, a, "GO:0000010", "IDA", aspect="F"))
        lines.append(gaf_line(s, a, "GO:0000011", "IGI", f"UniProtKB:{acc_of((i + 2) % n)}",
                              aspect="F"))
    gaf = write_gaf(bulk / "goa_human.gaf.gz", lines)
    info = ["#string_protein_id\tpreferred_name\tprotein_size\tannotation\n"]
    aliases = ["#string_protein_id\talias\tsource\n"]
    for i in range(n):
        ensp = f"9606.ENSP{i:011d}"
        info.append(f"{ensp}\t{sym_of(i)}\t100\tannotation {i}\n")
        aliases.append(f"{ensp}\t{sym_of(i)}\tBioMart_HUGO Ensembl_HGNC\n")
        aliases.append(f"{ensp}\tALT{i:02d}\tUniProt_GN_Synonyms\n")
        aliases.append(f"{ensp}\t{acc_of(i)}\tUniProt_AC\n")
        aliases.append(f"{ensp}\tHGNC:{1000 + i}\tEnsembl_HGNC_hgnc_id\n")
        aliases.append(f"{ensp}\tENSG{i:011d}\tEnsembl_gene\n")
        aliases.append(f"{ensp}\tCDS\tUniProt_GN_Synonyms\n")
        aliases.append(f"{ensp}\tX{i}\tEnsembl_EMBL\n")      # not an alias source
    info_p = _gz_text(bulk / "9606.protein.info.v12.0.txt.gz", "".join(info))
    alias_p = _gz_text(bulk / "9606.protein.aliases.v12.0.txt.gz", "".join(aliases))
    return {"obo": obo, "gaf": gaf, "info": info_p, "aliases": alias_p}


def write_reference(root: Path, bulk: Path, n: int = N_PROTEINS, seed: int = 11
                    ) -> Dict[str, Path]:
    """Reference converted dir (default-guard index) plus splits from the real splitter."""
    from go_ppi_domain.data_splitter import build_splits
    from go_ppi_domain.go_ontology import GoIndex

    rnd = random.Random(seed)
    conv = root / "ref_converted"
    splits = root / "ref_splits"
    conv.mkdir(parents=True, exist_ok=True)
    index = GoIndex.build(bulk / "go-basic.obo", bulk / "goa_human.gaf.gz", "P")
    index.save(conv / "go_index_P.pkl")
    ids = list(range(n))
    pos: Dict[int, List[int]] = {}
    pairs = []
    for i in ids:
        others = [j for j in ids if j != i]
        rnd.shuffle(others)
        p2, p1, p0 = others[:6], others[6:16], others[16:26]
        pos[i] = p2
        for grade, group, name in ((2, p2, "high_confidence"), (1, p1, "weak_evidence"),
                                   (0, p0, "no_interaction")):
            for j in sorted(group):
                pairs.append({"anchor_id": sym_of(i), "partner_id": sym_of(j),
                              "grade": grade, "grade_name": name})
    with open(conv / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(conv / "proteins.jsonl", "w", encoding="utf-8") as fh:
        for i in ids:
            g = sym_of(i)
            a, b = pos[i][0], pos[i][1]
            text = (f"Protein {g}-like factor; binds {sym_of(a)} and {sym_of(b)}-binding "
                    f"partners; also called ALT{i:02d}; CDS region")
            fh.write(json.dumps({"protein_id": g, "gene": g,
                                 "encoder_view": f"{g} — {text}",
                                 "go_uris": sorted(index.terms_for(g)),
                                 "coarse_uris": sorted(index.coarse_for(g)),
                                 "n_terms": len(index.terms_for(g))}, ensure_ascii=False) + "\n")
    (conv / "convert_manifest.json").write_text(json.dumps({"aspect": "P", "seed": 42}),
                                                encoding="utf-8")
    build_splits(conv, splits, seed=42, val_no_interaction_cap=20, test_no_interaction_cap=0)
    return {"converted": conv, "splits": splits}
