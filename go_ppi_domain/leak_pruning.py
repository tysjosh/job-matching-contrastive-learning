"""Leak pruning for the GO/PPI withholding study (design section 1.1).

Two leaks are removed from the features before the study starts:

* **Text.** STRING/UniProt annotation text names interaction partners (11.07% of
  test positives mention the partner's symbol, 0.17% of weak pairs, 0% of
  no-interaction pairs). Every whole-word mention of ANOTHER protein's symbol or
  filtered alias in ``encoder_view`` becomes ``[GENE]``; the protein's own symbol
  prefix and its own aliases are kept.
* **GO annotation.** with/from (GAF column 8) references to another human protein
  (handled by ``go_ontology.parse_gaf(drop_with_from_human_protein=True)``; the
  accession map is built here).

Whole-word means ``(?<![A-Za-z0-9])X(?![A-Za-z0-9])``, case-sensitive (the leak
audit's semantics). Matching runs in two passes over the ORIGINAL text: (a)
maximal ``[A-Za-z0-9]+`` runs looked up in ``alnum``; (b) entries with other
characters (``-``, ``.``, ``/`` ...) found by enumerating whole-word spans inside
each whitespace-free chunk and looking them up in ``special`` (equivalent to one
longest-first regex alternation, but linear in the text). Overlaps resolve
leftmost-longest; a span owned by the protein itself is kept and shadows any
shorter entry inside it.

Pure functions; nothing here reads the study paths.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

ALIAS_SOURCES = frozenset({"Ensembl_HGNC_symbol", "Ensembl_HGNC_alias_symbol",
                           "Ensembl_HGNC_prev_symbol", "UniProt_GN_Name",
                           "UniProt_GN_Synonyms", "BioMart_HUGO"})
MASK_TOKEN = "[GENE]"
VIEW_SEP = " — "          # data_converter writes f"{g} — {text}"

_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")
_ALNUM_FULL = re.compile(r"[A-Za-z0-9]+")
_CHUNK = re.compile(r"\S+")
_HAS_UPPER_OR_DIGIT = re.compile(r"[A-Z0-9]")


@dataclass(frozen=True)
class MaskLexicon:
    owner: Dict[str, FrozenSet[str]]   # entry -> symbols that own it (self-exemption)
    alnum: FrozenSet[str]              # purely [A-Za-z0-9] entries (dict lookup pass)
    special: Tuple[str, ...]           # entries with - . / : , etc., longest first
    params: dict


# ----------------------------------------------------------------- matching
def _is_alnum(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")


def _special_spans(text: str, special: Set[str], max_len: int) -> Iterator[Tuple[int, int]]:
    """Every whole-word occurrence (start, end) of a ``special`` entry in ``text``.

    Entries contain no whitespace, so every occurrence lies inside one ``\\S+``
    chunk. A span is whole-word when the char before it and the char after it
    are not alphanumeric (or are the text boundary).
    """
    if not special:
        return
    for m in _CHUNK.finditer(text):
        s0, e0 = m.start(), m.end()
        chunk = m.group()
        if _ALNUM_FULL.fullmatch(chunk):
            continue          # an all-alnum chunk cannot hold a special entry
        n = len(chunk)
        # The chunk is delimited by whitespace / text boundaries, so only the
        # chunk's own characters decide the whole-word boundaries.
        starts = [i for i in range(n) if i == 0 or not _is_alnum(chunk[i - 1])]
        ends = [j for j in range(n, 0, -1) if j == n or not _is_alnum(chunk[j])]
        for i in starts:
            for j in ends:                      # longest first
                if j <= i:
                    break
                if j - i > max_len:
                    continue
                if chunk[i:j] in special:
                    yield s0 + i, s0 + j


def _entry_occurrences(text: str, alnum: FrozenSet[str] | Set[str], special: Set[str],
                       max_special: int) -> Set[str]:
    """Distinct entries that occur whole-word in ``text`` (doc-frequency helper)."""
    found = {t for t in _ALNUM_RUN.findall(text) if t in alnum}
    for s, e in _special_spans(text, special, max_special):
        found.add(text[s:e])
    return found


# ---------------------------------------------------------------- aliases
def _ensp_to_symbol(info_gz: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with gzip.open(info_gz, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2 and c[1]:
                out[c[0]] = c[1]
    return out


def _iter_aliases(aliases_gz: Path) -> Iterator[Tuple[str, str, Set[str]]]:
    with gzip.open(aliases_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            yield c[0], c[1], set(c[2].split())


def load_symbol_aliases(aliases_gz: Path, info_gz: Path,
                        symbols: Set[str]) -> Dict[str, Set[str]]:
    """``{symbol: aliases}`` for ``symbols``; each symbol always includes itself.

    ENSP -> symbol comes from ``protein.info``; an alias row counts when its
    source field shares a whitespace token with ``ALIAS_SOURCES``.
    """
    ensp2sym = _ensp_to_symbol(Path(info_gz))
    out: Dict[str, Set[str]] = {s: {s} for s in symbols}
    for ensp, alias, sources in _iter_aliases(Path(aliases_gz)):
        sym = ensp2sym.get(ensp)
        if sym in out and sources & ALIAS_SOURCES and alias:
            out[sym].add(alias)
    return out


def build_mask_lexicon(texts: Dict[str, str], aliases: Dict[str, Set[str]],
                       min_alias_len: int = 3, min_symbol_len: int = 2,
                       max_doc_freq: float = 0.01) -> MaskLexicon:
    """Lexicon of maskable entries plus, per entry, the symbols that own it.

    Primary symbols with ``len >= min_symbol_len`` are always eligible. An alias
    needs ``len >= min_alias_len``, >= 1 uppercase letter or digit, no
    whitespace, and a doc frequency (share of ``texts`` containing it whole-word)
    ``<= max_doc_freq``. Ownership is recorded over ALL of a symbol's aliases
    (filtered or not), so a protein never masks a string that names itself.
    """
    owner: Dict[str, Set[str]] = defaultdict(set)
    for sym, al in aliases.items():
        owner[sym].add(sym)
        for a in al:
            owner[a].add(sym)
    symbols = set(aliases)
    symbols_unmaskable = sorted(s for s in symbols if len(s) < min_symbol_len)
    primary = {s for s in symbols if len(s) >= min_symbol_len}

    alias_candidates: Set[str] = set()
    for sym, al in aliases.items():
        for a in al:
            if a in primary:
                continue
            if (len(a) >= min_alias_len and _HAS_UPPER_OR_DIGIT.search(a)
                    and not any(ch.isspace() for ch in a)):
                alias_candidates.add(a)
    cand_alnum = {a for a in alias_candidates if _ALNUM_FULL.fullmatch(a)}
    cand_special = alias_candidates - cand_alnum
    max_special = max((len(a) for a in cand_special), default=0)
    df: Counter = Counter()
    for text in texts.values():
        df.update(_entry_occurrences(text, cand_alnum, cand_special, max_special))
    n_texts = max(1, len(texts))
    dropped = sorted(a for a in alias_candidates if df[a] / n_texts > max_doc_freq)
    entries = primary | (alias_candidates - set(dropped))
    alnum = frozenset(e for e in entries if _ALNUM_FULL.fullmatch(e))
    special = tuple(sorted((e for e in entries if e not in alnum),
                           key=lambda e: (-len(e), e)))
    params = {"min_alias_len": min_alias_len, "min_symbol_len": min_symbol_len,
              "max_doc_freq": max_doc_freq, "n_texts": len(texts),
              "symbols_unmaskable": symbols_unmaskable,
              "aliases_dropped_docfreq": dropped}
    return MaskLexicon(owner={e: frozenset(owner[e]) for e in entries},
                       alnum=alnum, special=special, params=params)


def lexicon_to_json(lexicon: MaskLexicon) -> dict:
    """Serialisable form (``mask_lexicon.json``); ``lexicon_from_json`` inverts it."""
    return {"params": lexicon.params,
            "owner": {e: sorted(lexicon.owner[e]) for e in sorted(lexicon.owner)}}


def lexicon_from_json(data: dict) -> MaskLexicon:
    owner = {e: frozenset(o) for e, o in data["owner"].items()}
    alnum = frozenset(e for e in owner if _ALNUM_FULL.fullmatch(e))
    special = tuple(sorted((e for e in owner if e not in alnum), key=lambda e: (-len(e), e)))
    return MaskLexicon(owner=owner, alnum=alnum, special=special, params=data["params"])


def entries_in_text(text: str, lexicon: MaskLexicon) -> Set[str]:
    """Every lexicon entry occurring whole-word in ``text`` (overlaps included)."""
    special, max_special = _lexicon_special_index(lexicon)
    return _entry_occurrences(text, lexicon.alnum, special, max_special)


def _lexicon_special_index(lexicon: MaskLexicon) -> Tuple[Set[str], int]:
    cached = getattr(lexicon, "_special_index", None)
    if cached is None:
        cached = (set(lexicon.special), max((len(e) for e in lexicon.special), default=0))
        object.__setattr__(lexicon, "_special_index", cached)
    return cached


def find_entry_spans(text: str, lexicon: MaskLexicon) -> List[Tuple[int, int, str]]:
    """Non-overlapping whole-word lexicon spans (start, end, entry), leftmost-longest."""
    special, max_special = _lexicon_special_index(lexicon)
    spans = [(m.start(), m.end()) for m in _ALNUM_RUN.finditer(text)
             if m.group() in lexicon.alnum]
    spans.extend(_special_spans(text, special, max_special))
    spans.sort(key=lambda se: (se[0], -(se[1] - se[0])))
    out: List[Tuple[int, int, str]] = []
    last = 0
    for s, e in spans:
        if s >= last:
            out.append((s, e, text[s:e]))
            last = e
    return out


def mask_view_text(self_symbol: str, text: str, lexicon: MaskLexicon) -> Tuple[str, int]:
    """Replace whole-word entries NOT owned by ``self_symbol`` with ``MASK_TOKEN``.

    Returns ``(masked_text, n_replacements)``.
    """
    parts: List[str] = []
    pos = n = 0
    for s, e, entry in find_entry_spans(text, lexicon):
        if self_symbol in lexicon.owner.get(entry, ()):
            continue
        parts.append(text[pos:s])
        parts.append(MASK_TOKEN)
        pos = e
        n += 1
    if not n:
        return text, 0
    parts.append(text[pos:])
    return "".join(parts), n


# ------------------------------------------------------- accession maps
def human_accession_map(gaf_gz: Path, aliases_gz: Path, info_gz: Path) -> Dict[str, str]:
    """UniProt accession -> human gene symbol.

    GAF ``UniProtKB`` rows (col 2 -> col 3, first occurrence wins), then STRING
    ``UniProt_AC`` aliases (ENSP -> symbol via ``protein.info``) for accessions
    the GAF does not cover.
    """
    out: Dict[str, str] = {}
    with gzip.open(gaf_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 3 or c[0] != "UniProtKB" or not c[1] or not c[2]:
                continue
            out.setdefault(c[1], c[2])
    ensp2sym = _ensp_to_symbol(Path(info_gz))
    for ensp, alias, sources in _iter_aliases(Path(aliases_gz)):
        if "UniProt_AC" in sources and ensp in ensp2sym and alias not in out:
            out[alias] = ensp2sym[ensp]
    return out


def other_db_symbol_map(aliases_gz: Path, info_gz: Path) -> Dict[str, str]:
    """``"HGNC:<n>"`` and ``"ENSEMBL_GeneID:<ENSG>"`` -> symbol (ungated residual, design 1.5)."""
    ensp2sym = _ensp_to_symbol(Path(info_gz))
    out: Dict[str, str] = {}
    for ensp, alias, sources in _iter_aliases(Path(aliases_gz)):
        sym = ensp2sym.get(ensp)
        if sym is None:
            continue
        low = {s.lower() for s in sources}
        if any("hgnc_id" in s for s in low):
            key = alias if alias.startswith("HGNC:") else f"HGNC:{alias}"
            out.setdefault(key, sym)
        if "ensembl_gene" in low:
            out.setdefault(f"ENSEMBL_GeneID:{alias}", sym)
    return out


# ------------------------------------------------------------- rewrite
def _quantiles(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "total": 0}
    v = sorted(values)

    def q(p: float) -> int:
        return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]

    return {"mean": round(sum(v) / len(v), 6), "p50": q(0.5), "p90": q(0.9),
            "p99": q(0.99), "max": v[-1], "total": sum(v)}


def view_prefix(gene: str) -> str:
    return f"{gene}{VIEW_SEP}"


def strip_view_prefix(gene: str, encoder_view: str) -> str:
    """The annotation text after ``"{gene} — "``; ``ValueError`` if the prefix is missing."""
    prefix = view_prefix(gene)
    if not encoder_view.startswith(prefix):
        raise ValueError(f"encoder_view of {gene!r} does not start with {prefix!r}")
    return encoder_view[len(prefix):]


def rewrite_proteins(reference_proteins: Path, out_proteins: Path, annotation_index,
                     view_text_fn: Callable[[str, str], Tuple[str, int]]) -> dict:
    """Reference ``proteins.jsonl`` -> same ids, same order, same key order.

    Only ``encoder_view`` (prefix kept, annotation text passed through
    ``view_text_fn``), ``go_uris``, ``coarse_uris`` and ``n_terms`` (recomputed
    from ``annotation_index``) change. A missing ``"{gene} — "`` prefix is fatal
    (``ValueError``). Written with ``json.dumps(row, ensure_ascii=False) + "\\n"``.
    """
    ids: List[str] = []
    replacements: List[int] = []
    tokens = empty = 0
    lines: List[str] = []
    with open(reference_proteins, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            g = row["gene"]
            raw = strip_view_prefix(g, row["encoder_view"])
            masked, n = view_text_fn(g, raw)
            row["encoder_view"] = view_prefix(g) + masked
            row["go_uris"] = sorted(annotation_index.terms_for(g))
            row["coarse_uris"] = sorted(annotation_index.coarse_for(g))
            row["n_terms"] = len(row["go_uris"])
            ids.append(row["protein_id"])
            replacements.append(int(n))
            tokens += len(_ALNUM_RUN.findall(raw))
            empty += not row["go_uris"]
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    out_proteins = Path(out_proteins)
    out_proteins.parent.mkdir(parents=True, exist_ok=True)
    with open(out_proteins, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return {"n": len(ids),
            "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "replacements": _quantiles(replacements),
            "empty_go_uris": empty,
            "token_mask_rate": (sum(replacements) / tokens) if tokens else 0.0}


def read_protein_texts(proteins_jsonl: Path) -> Dict[str, str]:
    """``{gene: annotation text}`` (prefix stripped) from a proteins file."""
    out: Dict[str, str] = {}
    with open(proteins_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["gene"]] = strip_view_prefix(row["gene"], row["encoder_view"])
    return out
